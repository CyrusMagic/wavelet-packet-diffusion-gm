import pytorch_lightning as pl
import torch as th
import time
import os
from datetime import datetime

from diffusion.nn import append_dims
from diffusion.unet import UNetModel


class EDM:
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: float = 7.0
    sigma_data: float = 0.5
    P_mean: float = -1.2
    P_std: float = 1.2
    S_churn: float = 40
    S_min: float = 0.05
    S_max: float = 50
    S_noise: float = 1.003

    def sigma(self, eps):
        return (eps * self.P_std + self.P_mean).exp()

    def loss_weight(self, sigma):
        return (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

    def skip_scaling(self, sigma):
        return self.sigma_data**2 / (sigma**2 + self.sigma_data**2)

    def out_scaling(self, sigma):
        return sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5

    def in_scaling(self, sigma):
        return 1 / (sigma**2 + self.sigma_data**2) ** 0.5

    def noise_conditioning(self, sigma):
        return 0.25 * sigma.log()

    def sampling_sigmas(self, num_steps, device=None):
        rho_inv = 1 / self.rho
        step_idxs = th.arange(num_steps, dtype=th.float64, device=device)
        sigmas = (
            self.sigma_max**rho_inv
            + step_idxs
            / (num_steps - 1)
            * (self.sigma_min**rho_inv - self.sigma_max**rho_inv)
        ) ** self.rho
        return th.cat([sigmas, th.zeros_like(sigmas[:1])])  # add sigma=0

    def sigma_hat(self, sigma, num_steps):
        gamma = (
            min(self.S_churn / num_steps, 2**0.5 - 1)
            if self.S_min <= sigma <= self.S_max
            else 0
        )
        return sigma + gamma * sigma


class LightningEDM(pl.LightningModule):
    """A Pyth Lightning module of the EDM model [1].

    Parameters
    ----------
    unet_config : dict
        The configuration for the U-Net model.
    optimizer_params : dict
        A dictionary of parameters for the optimizer.
    num_sampling_steps : int, optional
        The number of sampling steps during inference.
    deterministic_sampling : bool, optional
        If True, use deterministic sampling instead of stochastic sampling.
        Stochastic sampling can be more accurate but usually requires more (e.g. 256) steps.
    edm : EDM, optional
        The EDM model parameters.
    References
    ----------
    [1] Elucidating the Design Space of Diffusion-Based Generative Models
    """

    def __init__(
        self,
        modelConfig: dict = None,
        unet_config: dict = None,
        trainer_params: dict = None,
        optimizer_params: dict = None,
        edm: EDM = None,
        # Kept for backward compatibility
        **kwargs
    ):
        super().__init__()

        # Store configs (defaults for backward compatibility)
        self.modelConfig = modelConfig or {}
        self.trainer_params = trainer_params or {}
        
        # Pull frequently-used fields from modelConfig
        self.num_sampling_steps = self.modelConfig.get("num_sampling_steps", 25)
        self.deterministic_sampling = self.modelConfig.get("deterministic_sampling", True)
        self.cfg_dropout_prob = self.modelConfig.get("cfg_dropout_prob", 0.2)
        self.guidance_scale = self.modelConfig.get("guidance_scale", 1.0)
        
        # Initialize modules
        if unet_config is None:
            raise ValueError("unet_config is required")
        
        # Filter deprecated args for backward compatibility
        unet_config_filtered = {k: v for k, v in unet_config.items() 
                              if k not in ['cond_features']}
        self.unet = UNetModel(**unet_config_filtered)
        
        if optimizer_params is None:
            raise ValueError("optimizer_params is required")
        self.optimizer_params = optimizer_params
        
        self.edm = edm or EDM()
        self.save_hyperparameters(ignore=("edm",))
        
        # Timing statistics
        self.train_start_time = None
        self.total_training_steps = 0
        self.total_validation_steps = 0
        self.total_training_time = 0.0
        self.total_validation_time = 0.0
        self.epoch_start_times = []
        self.epoch_training_times = []

    def forward(self, sample, sigma, cond_sample=None, cond=None):
        """Make a forward pass through the network with skip connection."""
        dim = sample.dim()
        sample_in = sample * append_dims(self.edm.in_scaling(sigma), dim)
        input = (
            sample_in
            if cond_sample is None
            else th.cat((sample_in, cond_sample), dim=1)
        )
        noise_cond = self.edm.noise_conditioning(sigma)
        out = self.unet(input, noise_cond, cond=cond)
        skip = append_dims(self.edm.skip_scaling(sigma), dim) * sample
        return out * append_dims(self.edm.out_scaling(sigma), dim) + skip

    def step(self, batch, batch_idx):
        """A single step in the training loop."""
        sample = batch["signal"]  # [batch_size, channels, nodes, width]
        cond_sample = batch["cond_signal"] if "cond_signal" in batch else None
        cond = batch["cond"] if "cond" in batch else None

        # ===== Classifier-Free Guidance (training-time dropout) =====
        # With probability cfg_dropout_prob, drop conditions for the whole batch
        # to let the model learn both conditional and unconditional distributions.
        if self.cfg_dropout_prob > 0:
            drop_flag = False
            # synchronize dropout decision across DDP ranks to avoid unused-parameter mismatch
            try:
                import torch.distributed as dist

                if dist.is_available() and dist.is_initialized():
                    if getattr(self.trainer, "is_global_zero", True):
                        flag = (
                            th.rand((), device=self.device) < self.cfg_dropout_prob
                        ).to(th.uint8)
                    else:
                        flag = th.zeros((), dtype=th.uint8, device=self.device)
                    dist.broadcast(flag, src=0)
                    drop_flag = bool(flag.item())
                else:
                    drop_flag = bool(
                        (th.rand((), device=self.device) < self.cfg_dropout_prob).item()
                    )
            except Exception:
                drop_flag = bool(
                    (th.rand((), device=self.device) < self.cfg_dropout_prob).item()
                )

            if drop_flag:
                cond = None
                cond_sample = None

        eps = th.randn(sample.shape[0], device=self.device)
        sigma = self.edm.sigma(eps)
        noise = th.randn_like(sample) * append_dims(sigma, sample.dim())
        pred = self(sample + noise, sigma, cond_sample, cond)  # call self.forward()

        loss = (pred - sample) ** 2
        loss_weight = append_dims(self.edm.loss_weight(sigma), loss.dim())

        return (loss * loss_weight).mean()

    def training_step(self, batch, batch_idx):
        step_start_time = time.time()
        
        loss = self.step(batch, batch_idx)
        
        # Step duration (minutes)
        step_duration = (time.time() - step_start_time) / 60.0
        
        # Accumulate statistics
        self.total_training_steps += 1
        self.total_training_time += step_duration
        
        # Log epoch-level loss
        self.log("training/loss", loss.item(), sync_dist=True, on_step=False, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        step_start_time = time.time()
        
        loss = self.step(batch, batch_idx)
        
        # Step duration (minutes)
        step_duration = (time.time() - step_start_time) / 60.0
        
        # Accumulate statistics
        self.total_validation_steps += 1
        self.total_validation_time += step_duration
        
        # Log epoch-level loss
        self.log("validation/loss", loss.item(), sync_dist=True, on_step=False, on_epoch=True)
        
        return loss

    @th.no_grad()
    def sample(self, shape, cond_sample=None, cond=None):
        """Sample using Heun's second order method."""
        sigmas = self.edm.sampling_sigmas(self.num_sampling_steps, device=self.device)
        eps = th.randn(shape, device=self.device, dtype=th.float64) * sigmas[0]
        if self.deterministic_sampling:
            sample = self.sample_deterministically(eps, sigmas, cond_sample, cond)
        else:
            sample = self.sample_stochastically(eps, sigmas, cond_sample, cond)

        sample = sample.to(th.float32)
        return sample

    def sample_deterministically(
        self,
        eps,
        sigmas,
        cond_sample=None,
        cond=None,
        guidance_scale: float | None = None,
    ):
        sample_next = eps
        for i, (sigma, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):
            sample_curr = sample_next
            pred_curr = self._guided_pred(
                sample_curr.to(self.dtype),
                sigma.to(self.dtype).repeat(len(sample_curr)),
                cond_sample,
                cond,
                guidance_scale,
            ).to(th.float64)
            d_cur = (sample_curr - pred_curr) / sigma
            sample_next = sample_curr + d_cur * (sigma_next - sigma)

            # second order correction
            if i < self.num_sampling_steps - 1:
                pred_next = self._guided_pred(
                    sample_next.to(self.dtype),
                    sigma_next.to(self.dtype).repeat(len(sample_curr)),
                    cond_sample,
                    cond,
                    guidance_scale,
                ).to(th.float64)
                d_prime = (sample_next - pred_next) / sigma_next
                sample_next = sample_curr + (sigma_next - sigma) * (
                    0.5 * d_cur + 0.5 * d_prime
                )

        return sample_next

    def sample_stochastically(
        self,
        eps,
        sigmas,
        cond_sample=None,
        cond=None,
        guidance_scale: float | None = None,
    ):
        sample_next = eps
        for i, (sigma, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):
            sample_curr = sample_next

            # increase noise temporarily
            sigma_hat = self.edm.sigma_hat(sigma, self.num_sampling_steps)
            noise = th.randn_like(sample_curr) * self.edm.S_noise
            sample_hat = sample_curr + noise * (sigma_hat**2 - sigma**2) ** 0.5

            # euler step
            pred_hat = self._guided_pred(
                sample_hat.to(self.dtype),
                sigma_hat.to(self.dtype).repeat(len(sample_hat)),
                cond_sample,
                cond,
                guidance_scale,
            ).to(th.float64)
            d_cur = (sample_hat - pred_hat) / sigma_hat
            sample_next = sample_hat + d_cur * (sigma_next - sigma_hat)

            # second order correction
            if i < self.num_sampling_steps - 1:
                pred_next = self._guided_pred(
                    sample_next.to(self.dtype),
                    sigma_next.to(self.dtype).repeat(len(sample_hat)),
                    cond_sample,
                    cond,
                    guidance_scale,
                ).to(th.float64)
                d_prime = (sample_next - pred_next) / sigma_next
                sample_next = sample_hat + (sigma_next - sigma_hat) * (
                    0.5 * d_cur + 0.5 * d_prime
                )

        return sample_next

    def _guided_pred(
        self,
        sample_in: th.Tensor,
        sigma: th.Tensor,
        cond_sample=None,
        cond=None,
        guidance_scale: float | None = None,
    ) -> th.Tensor:
        """
        Compute guided prediction using Classifier-Free Guidance (CFG).

        If guidance_scale <= 1 or None, fall back to single-branch conditional prediction.
        Otherwise, run unconditional and conditional branches and mix:
            pred = pred_uncond + s * (pred_cond - pred_uncond)

        Unconditional branch drops both cond and cond_sample (strongest guidance).
        """
        gs = self.guidance_scale if guidance_scale is None else guidance_scale
        if gs is None or gs <= 1.0:
            return self(sample_in, sigma, cond_sample, cond)

        # unconditional prediction (drop both cond and cond_sample)
        pred_uncond = self(sample_in, sigma, None, None)
        # conditional prediction
        pred_cond = self(sample_in, sigma, cond_sample, cond)
        return pred_uncond + (gs * (pred_cond - pred_uncond))

    @th.no_grad()
    def evaluate(self, batch):
        """Evaluate the model on a batch of data."""
        sample = batch["signal"]
        cond_sample = batch["cond_signal"] if "cond_signal" in batch else None
        cond = batch["cond"] if "cond" in batch else None
        return self.sample(sample.shape, cond_sample, cond)
    
    def predict_step(self, batch, batch_idx=None):
        """Lightning predict step for multi-GPU inference with sample indices and target data."""
        predictions = self.evaluate(batch)
        # Return predictions + targets + indices for 1:1 alignment.
        result = {
            "predictions": predictions,
            "target_signal": batch["signal"],
            "target_waveform": batch["wfs"],
            "conditions": batch["cond"],
            "indices": batch["original_index"]
        }
        if "wavelet_meta" in batch:
            result["wavelet_meta"] = batch["wavelet_meta"]
        return result

    def on_train_start(self):
        """Record the training start timestamp."""
        self.train_start_time = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Training started")
    
    def on_train_epoch_start(self):
        """Record epoch start time."""
        epoch_start_time = time.time()
        self.epoch_start_times.append(epoch_start_time)
    
    def on_train_epoch_end(self):
        """Record epoch end time and timing statistics."""
        if self.epoch_start_times:
            epoch_duration = (time.time() - self.epoch_start_times[-1]) / 60.0
            self.epoch_training_times.append(epoch_duration)
            
            # epoch/duration_min: 3 significant digits
            duration_formatted = float(f"{epoch_duration:.3g}")
            self.log("epoch/duration_min", duration_formatted, sync_dist=True, on_step=False, on_epoch=True)
            
            if self.train_start_time is not None:
                total_elapsed = (time.time() - self.train_start_time) / 60.0
                # epoch/total_elapsed_min: 2 decimals
                elapsed_formatted = round(total_elapsed, 2)
                self.log("epoch/total_elapsed_min", elapsed_formatted, sync_dist=True, on_step=False, on_epoch=True)
    
    def _get_gpu_info(self):
        """Return (gpu_name, gpu_count)."""
        try:
            if not th.cuda.is_available():
                return None, 0
            gpu_count = th.cuda.device_count()
            gpu_name = th.cuda.get_device_name(0) if gpu_count > 0 else None
            return gpu_name, gpu_count
        except Exception:
            return None, 0

    def on_fit_end(self):
        """Write a timing summary file at the end of training."""
        if self.train_start_time is None:
            return

        train_end_time = time.time()
        total_duration = (train_end_time - self.train_start_time) / 60.0

        # Output directory
        if hasattr(self.trainer, 'default_root_dir') and self.trainer.default_root_dir:
            output_dir = self.trainer.default_root_dir
        else:
            output_dir = './outputs'

        os.makedirs(output_dir, exist_ok=True)

        # Timing file
        timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
        timing_file = os.path.join(output_dir, f"training_timing_{timestamp_file}.txt")

        # Summary statistics
        avg_training_step_time = self.total_training_time / max(self.total_training_steps, 1)
        avg_validation_step_time = self.total_validation_time / max(self.total_validation_steps, 1)
        avg_epoch_time = sum(self.epoch_training_times) / max(len(self.epoch_training_times), 1)

        # GPU info
        gpu_name, gpu_count = self._get_gpu_info()

        timestamp_start = datetime.fromtimestamp(self.train_start_time).strftime("%Y-%m-%d %H:%M:%S")
        timestamp_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(timing_file, 'w', encoding='utf-8') as f:
            f.write("Training timing report\n")
            f.write(f"{'='*50}\n")
            f.write(f"start: {timestamp_start}\n")
            f.write(f"end:   {timestamp_end}\n")
            f.write(f"total_min: {total_duration:.2e} (hours: {total_duration/60:.2f})\n")
            f.write(f"epochs: {len(self.epoch_training_times)}\n")
            f.write(f"avg_epoch_min: {avg_epoch_time:.2e}\n")

            if gpu_count > 0:
                f.write(f"gpus: {gpu_count}\n")
                f.write(f"gpu_name: {gpu_name}\n")

            f.write("\nTraining steps:\n")
            f.write(f"  total_steps: {self.total_training_steps}\n")
            f.write(f"  total_time_min: {self.total_training_time:.2e}\n")
            f.write(f"  avg_step_min: {avg_training_step_time:.2e}\n")
            f.write("\nValidation steps:\n")
            f.write(f"  total_steps: {self.total_validation_steps}\n")
            f.write(f"  total_time_min: {self.total_validation_time:.2e}\n")
            f.write(f"  avg_step_min: {avg_validation_step_time:.2e}\n")

            f.write("\nThroughput:\n")
            if self.total_training_steps > 0:
                steps_per_minute = self.total_training_steps / max(self.total_training_time, 0.001)
                f.write(f"  train_steps_per_min: {steps_per_minute:.2f}\n")
            if self.total_validation_steps > 0:
                val_steps_per_minute = self.total_validation_steps / max(self.total_validation_time, 0.001)
                f.write(f"  val_steps_per_min: {val_steps_per_minute:.2f}\n")

        timestamp_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp_end}] Training finished")
        print(f"  - total: {total_duration:.2e} min (hours: {total_duration/60:.2f})")
        print(f"  - epochs: {len(self.epoch_training_times)}")
        print(f"  - avg_epoch: {avg_epoch_time:.2e} min")
        if gpu_count > 0:
            print(f"  - GPU: {gpu_count}× {gpu_name}")
        print(f"  - timing_file: {timing_file}")

    def configure_optimizers(self):
        optimizer = th.optim.Adam(
            self.parameters(), lr=self.optimizer_params["learning_rate"]
        )
        lr_scheduler = th.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.optimizer_params["max_steps"]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step"},
        }
