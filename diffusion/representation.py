"""
Waveform representation utilities (db6 wavelet packet).

- Use PyWavelets WaveletPacket at a fixed level (default: 7) to produce
  coefficients with shape [channels, 2**level, width].
- Zero-pad to a suitable length so the transform is invertible.
- Provide a paired forward/inverse transform with per-node normalization.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Sequence

import numpy as np

try:  # PyWavelets is the only dependency of this module.
    import pywt

    _PYWT_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - import guard
    pywt = None  # type: ignore[assignment]
    _PYWT_IMPORT_ERROR = str(exc)

DEFAULT_WAVELET_NAME = "db6"
DEFAULT_WAVELET_LEVEL = 7


def _ensure_pywt() -> None:
    if pywt is None:
        raise ImportError(
            "PyWavelets is required to compute db6 wavelet packet coefficients."
            f" Original error: {_PYWT_IMPORT_ERROR}"
        )


def _as_2d_waveform(waveform: np.ndarray) -> np.ndarray:
    arr = np.asarray(waveform, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"Expected waveform shape [C, T], got {arr.shape}")
    return arr


def _pad_signal(signal: np.ndarray, level: int) -> tuple[np.ndarray, int, int, int]:
    """Pad the signal tail to a power-of-two length (>= 2**level)."""
    orig_len = int(signal.shape[-1])
    min_len = 2 ** level
    if orig_len <= 0:
        target_len = min_len
    else:
        pow_val = int(np.ceil(np.log2(max(orig_len, min_len))))
        target_len = 2**pow_val
    pad = max(0, target_len - orig_len)
    if pad > 0:
        padded = np.pad(signal, (0, pad), mode="constant")
    else:
        padded = signal
    return padded.astype(np.float32), pad, orig_len, target_len


@lru_cache(maxsize=None)
def _wavelet_nodes(level: int, wavelet: str) -> tuple[str, ...]:
    """Return WaveletPacket node paths at `level` (frequency order)."""
    _ensure_pywt()
    dummy = np.zeros(2**level, dtype=np.float32)
    packet = pywt.WaveletPacket(
        data=dummy, wavelet=wavelet, mode="symmetric", maxlevel=level
    )
    nodes = packet.get_level(level, order="freq")
    return tuple(node.path for node in nodes)


def _channel_wavelet_coeffs(
    channel: np.ndarray,
    level: int,
    wavelet: str,
) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
    padded, pad, orig_len, padded_len = _pad_signal(channel, level)
    _ensure_pywt()
    packet = pywt.WaveletPacket(
        data=padded, wavelet=wavelet, mode="symmetric", maxlevel=level
    )
    nodes = _wavelet_nodes(level, wavelet)
    coeffs = np.stack([packet[path].data for path in nodes], axis=0).astype(np.float32)

    # Per-node normalization (store mean/std for inverse transform).
    DB6_EPS = 1e-8
    mean = coeffs.mean(axis=1, keepdims=True)  # [nodes, 1]
    coeffs = coeffs - mean
    std = np.sqrt(np.mean(coeffs**2, axis=1, keepdims=True)) + DB6_EPS  # [nodes, 1]
    coeffs = coeffs / std

    meta = {
        "mean": mean.squeeze(-1).astype(np.float32),  # [nodes,]
        "std": std.squeeze(-1).astype(np.float32),    # [nodes,]
        "pad": np.array(pad, dtype=np.int32),
        "orig_len": np.array(orig_len, dtype=np.int32),
        "padded_len": np.array(padded_len, dtype=np.int32),
        "coeff_len": np.array(coeffs.shape[-1], dtype=np.int32),
        "level": np.array(level, dtype=np.int32),
    }
    return coeffs, meta


def get_db6_wavelet_representation(
    waveform: np.ndarray,
    level: int = DEFAULT_WAVELET_LEVEL,
    wavelet: str = DEFAULT_WAVELET_NAME,
) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Compute db6 wavelet packet coefficients and return (coeffs, meta).

    Coefficients are normalized per node: coeffs = (coeffs - mean) / std.
    """
    arr = _as_2d_waveform(waveform)
    coeffs_list = []
    meta_list = []
    for channel in arr:
        coeffs, meta = _channel_wavelet_coeffs(channel, level=level, wavelet=wavelet)
        coeffs_list.append(coeffs)
        meta_list.append(meta)

    stacked = np.stack(coeffs_list, axis=0)

    # Keep per-channel mean/std for inverse normalization.
    meta = {
        "mean": np.stack([m["mean"] for m in meta_list], axis=0),  # [channels, nodes]
        "std": np.stack([m["std"] for m in meta_list], axis=0),    # [channels, nodes]
        "pad": np.stack([m["pad"] for m in meta_list], axis=0),
        "orig_len": np.stack([m["orig_len"] for m in meta_list], axis=0),
        "padded_len": np.stack([m["padded_len"] for m in meta_list], axis=0),
        "coeff_len": np.stack([m["coeff_len"] for m in meta_list], axis=0),
        "level": np.stack([m["level"] for m in meta_list], axis=0),
    }
    return stacked, meta


def get_representation(
    waveform: np.ndarray,
    level: int = DEFAULT_WAVELET_LEVEL,
    wavelet: str = DEFAULT_WAVELET_NAME,
) -> np.ndarray:
    """Backward-compatible alias returning only the coefficient tensor."""
    coeffs, _ = get_db6_wavelet_representation(waveform, level=level, wavelet=wavelet)
    return coeffs


def _resolve_lengths(
    length: int | Sequence[int] | None,
    num_channels: int,
) -> list[int | None]:
    if length is None:
        return [None] * num_channels
    if isinstance(length, Sequence) and not isinstance(length, (str, bytes)):
        values = list(length)
        if len(values) == num_channels:
            return [int(v) for v in values]
    return [int(length)] * num_channels


def invert_representation(
    representation,
    level: int = DEFAULT_WAVELET_LEVEL,
    wavelet: str = DEFAULT_WAVELET_NAME,
    original_length: int | Sequence[int] | None = None,
    wavelet_meta: Dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Invert wavelet packet coefficients back to waveform."""

    coeffs = representation
    if hasattr(coeffs, "detach"):  # support torch.Tensor
        coeffs = coeffs.detach().cpu().numpy()
    coeffs = np.asarray(coeffs, dtype=np.float32)

    squeeze_batch = False
    if coeffs.ndim == 3:
        squeeze_batch = True
        coeffs = coeffs[None, ...]
    if coeffs.ndim != 4:
        raise ValueError(f"Expected representation shape [B, C, N, W], got {representation!r}")

    # Inverse normalization: coeffs_denorm = coeffs * std + mean
    if wavelet_meta is not None and "mean" in wavelet_meta and "std" in wavelet_meta:
        mean = np.asarray(wavelet_meta["mean"], dtype=np.float32)
        std = np.asarray(wavelet_meta["std"], dtype=np.float32)

        # Adjust shapes to match coeffs: [B, C, N, W]
        if mean.ndim == 3:  # [B, C, N]
            mean = mean[..., None]  # [B, C, N, 1]
            std = std[..., None]    # [B, C, N, 1]
        elif mean.ndim == 2:  # [C, N]
            mean = mean[None, :, :, None]  # [1, C, N, 1]
            std = std[None, :, :, None]    # [1, C, N, 1]

        # Denormalize
        coeffs = coeffs * std + mean

    nodes = _wavelet_nodes(level, wavelet)
    batch_waveforms = []

    for sample in coeffs:
        channel_waveforms = []
        channel_lengths = _resolve_lengths(original_length, num_channels=sample.shape[0])
        for ch_idx, ch_coeffs in enumerate(sample):
            _ensure_pywt()
            packet = pywt.WaveletPacket(
                data=None, wavelet=wavelet, mode="symmetric", maxlevel=level
            )
            for path, node_coeff in zip(nodes, ch_coeffs):
                packet[path] = np.asarray(node_coeff, dtype=np.float32)
            reconstructed = packet.reconstruct(update=False)
            target_len = channel_lengths[ch_idx]
            if target_len is not None:
                reconstructed = reconstructed[:target_len]
            channel_waveforms.append(reconstructed.astype(np.float32))
        batch_waveforms.append(np.stack(channel_waveforms, axis=0))

    result = np.stack(batch_waveforms, axis=0)
    if squeeze_batch:
        result = result[0]
    return result
