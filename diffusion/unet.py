"""
UNet model with cross-attention conditioning.
"""

from abc import abstractmethod

import torch as th
import torch.nn.functional as F
from torch import nn

from .blocks import AttentionBlock, Downsample, GaussianFourierProjection, Upsample
from .nn import append_dims, checkpoint, conv_nd, normalization, zero_module


class TimestepBlock(nn.Module):
    """Base class for modules that take a timestep embedding."""

    @abstractmethod
    def forward(self, x, emb):
        """Apply the module given a timestep embedding."""


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """Sequential container that forwards timestep embedding and conditioning."""

    def forward(self, x, emb, cond_emb=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif hasattr(layer, 'use_cross_attention') and layer.use_cross_attention:
                # AttentionBlock supports cross-attention conditioning.
                x = layer(x, cond_emb)
            else:
                x = layer(x)
        return x


class ResBlock(TimestepBlock):
    """Residual block with timestep embedding."""

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        kernel_size=3,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
    ):
        super().__init__()
        out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, out_channels, kernel_size, padding="same"),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * out_channels if use_scale_shift_norm else out_channels,
            ),
        )

        self.out_layers = nn.Sequential(
            normalization(out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, out_channels, out_channels, kernel_size, padding="same")
            ),
        )

        if out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, out_channels, kernel_size, padding="same"
            )
        else:
            self.skip_connection = conv_nd(dims, channels, out_channels, 1)

    def forward(self, x, emb):
        """Forward pass."""
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        emb_out = append_dims(emb_out, h.dim())

        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)

        return self.skip_connection(x) + h


class ConditionCrossAttention(nn.Module):
    """Cross-attention from spatial features (queries) to condition tokens (keys/values)."""

    def __init__(self, channels, cond_dim, num_heads=8, head_dim=None):
        super().__init__()
        self.channels = channels
        self.cond_dim = cond_dim
        self.num_heads = num_heads
        self.head_dim = head_dim or channels // num_heads
        self.scale = self.head_dim ** -0.5

        # Query projection (from spatial features)
        self.to_q = nn.Linear(channels, num_heads * self.head_dim, bias=False)
        # Key/Value projections (from condition tokens)
        self.to_k = nn.Linear(cond_dim, num_heads * self.head_dim, bias=False)
        self.to_v = nn.Linear(cond_dim, num_heads * self.head_dim, bias=False)

        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(num_heads * self.head_dim, channels),
            nn.Dropout(0.1)
        )

    def forward(self, x, cond_emb):
        """Apply cross-attention conditioning."""
        if cond_emb is None:
            return x

        if cond_emb.dim() == 2:
            cond_emb = cond_emb.unsqueeze(1)

        spatial_shape = x.shape[2:]
        N, C = x.shape[:2]
        spatial_dims = 1
        for dim in spatial_shape:
            spatial_dims *= dim
        x_flat = x.view(N, C, -1).transpose(1, 2)  # [N, spatial_dims, C]

        # Query (from spatial features)
        q = self.to_q(x_flat)
        q = q.view(N, spatial_dims, self.num_heads, self.head_dim).transpose(1, 2)

        # Key/Value (from condition embeddings)
        k = self.to_k(cond_emb)
        v = self.to_v(cond_emb)
        k = k.view(N, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(N, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention weights
        attn = th.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        # Apply attention
        out = th.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(N, spatial_dims, -1)

        # Output projection
        out = self.to_out(out)

        # Reshape back to spatial format + residual
        out = out.transpose(1, 2).view(N, C, *spatial_shape)
        return x + out


class AttentionPooling(nn.Module):
    """Attention pooling over a token sequence to produce a global vector."""

    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        # Learnable query vector
        self.query = nn.Parameter(th.randn(1, 1, embed_dim) * 0.02)
        # Multi-head attention
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True, dropout=0.0
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens):
        """Pool tokens into a single vector per batch item."""
        batch_size = tokens.size(0)
        # Expand query to batch size
        query = self.query.expand(batch_size, -1, -1)  # [batch, 1, embed_dim]

        # Attention pooling
        pooled, _ = self.attn(query, tokens, tokens)  # [batch, 1, embed_dim]

        # Normalize and squeeze
        pooled = self.norm(pooled.squeeze(1))  # [batch, embed_dim]
        return pooled


class ConditionTokenEncoder(nn.Module):
    """Encode heterogeneous condition variables into a token sequence."""

    def __init__(
        self,
        model_channels,
        embed_dim,
        cond_configs,
        token_patch_size=4,
        attn_heads=4,
    ):
        super().__init__()
        self.cond_configs = cond_configs
        self.embed_dim = embed_dim
        self.token_patch_size = token_patch_size

        self.vector_encoders = nn.ModuleDict()
        self.scalar_encoders = nn.ModuleDict()
        self.vector_lengths: dict[str, int] = {}
        self.vector_patch_sizes: dict[str, int] = {}
        self.condition_bias = nn.ParameterDict()
        self.position_embeddings = nn.ParameterDict()

        for cond_name, config in cond_configs.items():
            cond_type = config.get("type", "vector")
            if cond_type == "vector":
                length = int(config["length"])
                patch = min(token_patch_size, length)
                patch = max(1, patch)
                num_tokens = (length + patch - 1) // patch

                self.vector_lengths[cond_name] = length
                self.vector_patch_sizes[cond_name] = patch

                self.vector_encoders[cond_name] = nn.Sequential(
                    nn.Linear(patch, embed_dim),
                    nn.SiLU(),
                    nn.Linear(embed_dim, embed_dim),
                )

                # Position embedding
                self.position_embeddings[cond_name] = nn.Parameter(
                    th.randn(1, num_tokens, embed_dim) * 0.02
                )
            else:  # scalar
                self.scalar_encoders[cond_name] = nn.Sequential(
                    nn.Linear(1, embed_dim),
                    nn.SiLU(),
                    nn.Linear(embed_dim, embed_dim),
                )

            # Condition-type bias
            self.condition_bias[cond_name] = nn.Parameter(th.zeros(1, 1, embed_dim))

        # Token mixer (3-layer Transformer encoder)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=attn_heads,
            dim_feedforward=embed_dim * 2,
            batch_first=True,
        )
        self.token_mixer = nn.TransformerEncoder(encoder_layer, num_layers=3)

    def _encode_vector(self, cond_name, x):
        """Encode a vector condition into tokens."""
        length = self.vector_lengths[cond_name]
        patch = self.vector_patch_sizes[cond_name]
        pad = (patch - length % patch) % patch
        if pad:
            x = F.pad(x, (0, pad))
        x = x.view(x.shape[0], -1, patch)  # [batch, num_tokens, patch]
        tokens = self.vector_encoders[cond_name](x)  # [batch, num_tokens, embed_dim]

        # Add position embedding + type bias
        tokens = tokens + self.position_embeddings[cond_name] + self.condition_bias[cond_name]
        return tokens

    def _encode_scalar(self, cond_name, x):
        """Encode a scalar condition into tokens."""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        tokens = self.scalar_encoders[cond_name](x)
        return tokens.unsqueeze(1) + self.condition_bias[cond_name]

    def forward(self, conditions):
        """Encode condition dict into a token sequence."""
        token_list = []
        for cond_name, config in self.cond_configs.items():
            cond_input = conditions[cond_name]
            cond_type = config.get("type", "vector")
            if cond_type == "vector":
                token_list.append(self._encode_vector(cond_name, cond_input))
            else:
                token_list.append(self._encode_scalar(cond_name, cond_input))

        tokens = th.cat(token_list, dim=1)
        tokens = self.token_mixer(tokens)

        return {"tokens": tokens}


class UNetModel(nn.Module):
    """UNet with cross-attention conditioning and CFG support."""

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        cond_configs,
        num_res_blocks,
        attention_resolutions=(8, 16, 32),
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_kernel_size=3,
        conv_resample=True,
        dims=2,
        cond_emb_scale=None,
        use_checkpoint=False,
        num_heads=1,
        use_scale_shift_norm=False,
        flash_attention=True,
        use_cross_attention=True,
    ):
        super().__init__()
        self.use_cross_attention = use_cross_attention

        embed_dim = model_channels * 4
        self.time_embed = GaussianFourierProjection(model_channels)
        self.time_mlp = nn.Sequential(
            nn.Linear(model_channels, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.condition_encoder = ConditionTokenEncoder(
            model_channels,
            embed_dim,
            cond_configs,
        )
        # Unconditional embedding for CFG
        self.uncond_embed = nn.Parameter(th.zeros(1, embed_dim))

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, ch, conv_kernel_size, padding="same")
                )
            ]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        kernel_size=conv_kernel_size,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            flash_attention=flash_attention,
                            use_cross_attention=use_cross_attention,
                            cond_dim=embed_dim if use_cross_attention else None,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                embed_dim,
                dropout,
                kernel_size=conv_kernel_size,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                num_heads=num_heads,
                dims=dims,
                use_checkpoint=use_checkpoint,
                flash_attention=flash_attention,
                use_cross_attention=use_cross_attention,
                cond_dim=embed_dim if use_cross_attention else None,
            ),
            ResBlock(
                ch,
                embed_dim,
                dropout,
                kernel_size=conv_kernel_size,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        kernel_size=conv_kernel_size,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            flash_attention=flash_attention,
                            use_cross_attention=use_cross_attention,
                            cond_dim=embed_dim if use_cross_attention else None,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        Upsample(
                            ch,
                            conv_resample,
                            dims=dims,
                            out_channels=out_ch,
                            kernel_size=conv_kernel_size,
                        )
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(
                conv_nd(dims, input_ch, out_channels, conv_kernel_size, padding="same")
            ),
        )

    def forward(self, x, timesteps, cond=None):
        """Forward pass."""
        # Timestep embedding (kept separate from conditioning)
        time_emb = self.time_mlp(self.time_embed(timesteps))  # [N, embed_dim]

        # Conditioning tokens
        if cond is None:
            # Unconditional branch (for CFG)
            time_emb = time_emb + self.uncond_embed.expand(time_emb.size(0), -1)
            cond_tokens = None
        else:
            cond_outputs = self.condition_encoder(cond)
            cond_tokens = cond_outputs["tokens"]

        # Input blocks
        hs = []
        h = x
        for module in self.input_blocks:
            h = module(h, time_emb, cond_tokens)
            hs.append(h)

        # Middle block
        h = self.middle_block(h, time_emb, cond_tokens)

        # Output blocks
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, time_emb, cond_tokens)

        return self.out(h)
