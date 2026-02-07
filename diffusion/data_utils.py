"""
Data loading and (de-)normalization utilities.

This repository uses a single public dataset split: `test`.
"""

import multiprocessing as mp
from typing import Dict, Optional
import numpy as np
from h5py import File
from torch.utils.data import DataLoader

from diffusion.dataset import Dataset


# ---------------------------
# Normalization helpers
# ---------------------------


def load_norm_stats(h5_path: str) -> Dict[str, np.ndarray]:
    """Load normalization statistics from an HDF5 dataset."""
    stats: Dict[str, np.ndarray] = {}
    try:
        with File(h5_path, "r") as f:
            if "_norm_stats" not in f:
                return stats
            group = f["_norm_stats"]
            for key in group.keys():
                stats[key] = np.array(group[key], dtype=np.float32)
    except OSError as err:
        print(f"Failed to read normalization stats: {err}")
    return stats


def _invert_log_cdf(
    cdf_values: np.ndarray,
    log_sorted: np.ndarray,
    cdf_sorted: np.ndarray,
) -> np.ndarray:
    """Invert a log1p+empirical-CDF normalization back to raw non-negative values."""
    if cdf_values.size == 0:
        return cdf_values

    flat = cdf_values.reshape(-1)
    clipped = np.clip(flat, cdf_sorted[0], cdf_sorted[-1])
    log_vals = np.interp(clipped, cdf_sorted, log_sorted)
    restored = np.expm1(log_vals)
    return restored.reshape(cdf_values.shape)


def denormalize_conditions(
    normalized: Dict[str, np.ndarray],
    original_indices: np.ndarray,
    stats: Dict[str, np.ndarray],
    datapath: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Denormalize condition variables using stored statistics."""
    if normalized is None or len(normalized) == 0:
        return {}

    denorm: Dict[str, np.ndarray] = {}
    dataset_pga_cache: Optional[np.ndarray] = None

    def _ensure_dataset_pga() -> Optional[np.ndarray]:
        nonlocal dataset_pga_cache
        if dataset_pga_cache is not None:
            return dataset_pga_cache
        if datapath is None:
            return None
        try:
            with File(datapath, "r") as f:
                if "pga_NoNorm" not in f:
                    raise KeyError("pga_NoNorm not found for SA denorm")
                sort_order = np.argsort(original_indices)
                sorted_indices = original_indices[sort_order]
                pga_raw = f["pga_NoNorm"][sorted_indices]
                restore_order = np.argsort(sort_order)
                pga_raw = pga_raw[restore_order]
                if pga_raw.ndim == 2 and pga_raw.shape[1] == 1:
                    pga_raw = pga_raw[:, 0]
                dataset_pga_cache = pga_raw.astype(np.float32)
        except Exception as err:
            print(f"  Warning: failed to read raw PGA from {datapath}: {err}")
            dataset_pga_cache = None
        return dataset_pga_cache

    def _require_pga() -> Optional[np.ndarray]:
        """Return raw PGA values with shape [N] if available."""
        if "pga" in denorm:
            pga_vals = denorm["pga"]
            return pga_vals.reshape(-1)
        pga_vals = _ensure_dataset_pga()
        if pga_vals is not None:
            denorm["pga"] = pga_vals
        return pga_vals

    if "pga" in normalized and "pga_log_sorted" in stats and "pga_cdf_sorted" in stats:
        denorm["pga"] = _invert_log_cdf(
            normalized["pga"], stats["pga_log_sorted"], stats["pga_cdf_sorted"]
        )

    if "arias" in normalized and "arias_log_sorted" in stats and "arias_cdf_sorted" in stats:
        denorm["arias"] = _invert_log_cdf(
            normalized["arias"], stats["arias_log_sorted"], stats["arias_cdf_sorted"]
        )

    # T5 / D5-95 / tc use the same log1p+CDF normalization as Arias intensity.
    if "t5_norm" in normalized and "t5_log_sorted" in stats and "t5_cdf_sorted" in stats:
        denorm["T5"] = _invert_log_cdf(
            normalized["t5_norm"], stats["t5_log_sorted"], stats["t5_cdf_sorted"]
        )

    if "d595_norm" in normalized and "d595_log_sorted" in stats and "d595_cdf_sorted" in stats:
        denorm["D5_95"] = _invert_log_cdf(
            normalized["d595_norm"], stats["d595_log_sorted"], stats["d595_cdf_sorted"]
        )

    # Time centroid (tc)
    if "tc_norm" in normalized and "tc_log_sorted" in stats and "tc_cdf_sorted" in stats:
        denorm["tc"] = _invert_log_cdf(
            normalized["tc_norm"], stats["tc_log_sorted"], stats["tc_cdf_sorted"]
        )

    if "sa" in normalized:
        sa_values = normalized["sa"]
        if original_indices.size == sa_values.shape[0]:
            pga_vals = _require_pga()
            if pga_vals is not None:
                scale = pga_vals[:, np.newaxis]
                denorm["sa"] = sa_values * scale
            elif "sa_max_values" in stats:
                max_vals = stats["sa_max_values"][original_indices]
                if sa_values.ndim == 1:
                    denorm["sa"] = sa_values * max_vals
                else:
                    denorm["sa"] = sa_values * max_vals[:, None]

    return denorm


# ---------------------------
# Data loading helpers
# ---------------------------


def create_dataset(
    datapath: str,
    cut: Optional[int] = None,
    cond_configs: Optional[Dict] = None,
    wavelet_level: int = 7,
    wavelet_name: str = "db6",
    include_wavelet_meta: bool = True,
) -> Dataset:
    """Create a Dataset instance (public split: `test`)."""
    return Dataset(
        datapath=datapath,
        cut=cut,
        cond_configs=cond_configs or {},
        split="test",
        wavelet_level=wavelet_level,
        wavelet_name=wavelet_name,
        include_wavelet_meta=include_wavelet_meta,
    )


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 0,
    mode: str = "train",
    device: str = "cuda",
    pin_memory: bool = True,
) -> DataLoader:
    """Create a PyTorch DataLoader."""
    if mode == "train":
        shuffle = True
        persistent_workers = num_workers > 0
        drop_last = True
    elif mode == "eval":
        shuffle = False
        persistent_workers = False
        drop_last = False
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose from 'train', 'eval'")

    # pin_memory only helps on CUDA.
    use_pin_memory = pin_memory and (device == "cuda")

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        pin_memory=use_pin_memory,
    )

    # Multiprocessing settings
    if num_workers > 0:
        loader_kwargs.update(
            multiprocessing_context=mp.get_context("spawn"),
            prefetch_factor=1,
        )

    return DataLoader(dataset, **loader_kwargs)


def create_train_eval_dataloaders(
    config: Dict,
    include_meta_train: bool = False,
    include_meta_eval: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Create (train_loader, eval_loader).

    Note: the public release only provides the `test` split. For convenience,
    we reuse the same dataset for both loaders and differ only in DataLoader
    shuffling and meta inclusion.
    """
    train_dataset = create_dataset(
        datapath=config["datapath"],
        cut=config["cut_t"],
        cond_configs=config.get("cond_configs", {}),
        wavelet_level=config.get("wavelet_level", 7),
        wavelet_name=config.get("wavelet_name", "db6"),
        include_wavelet_meta=include_meta_train,
    )

    eval_dataset = create_dataset(
        datapath=config["datapath"],
        cut=config["cut_t"],
        cond_configs=config.get("cond_configs", {}),
        wavelet_level=config.get("wavelet_level", 7),
        wavelet_name=config.get("wavelet_name", "db6"),
        include_wavelet_meta=include_meta_eval,
    )

    used_batch_size = config.get("batch_size", 64)

    train_num_workers = config.get("train_num_workers", 0)
    eval_num_workers = config.get("eval_num_workers", 0)
    device = config.get("device", "cuda")

    # Create DataLoaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=used_batch_size,
        num_workers=train_num_workers,
        mode="train",
        device=device,
    )

    eval_loader = create_dataloader(
        eval_dataset,
        batch_size=used_batch_size,
        num_workers=eval_num_workers,
        mode="eval",
        device=device,
    )

    return train_loader, eval_loader


def create_eval_dataloader(
    config: Dict,
    include_wavelet_meta: bool = False,
) -> DataLoader:
    """Create a single evaluation DataLoader (split: `test`)."""
    dataset = create_dataset(
        datapath=config["datapath"],
        cut=config["cut_t"],
        cond_configs=config.get("cond_configs", {}),
        wavelet_level=config.get("wavelet_level", 7),
        wavelet_name=config.get("wavelet_name", "db6"),
        include_wavelet_meta=include_wavelet_meta,
    )

    eval_num_workers = config.get("eval_num_workers", 0)
    eval_batch_size = config.get("batch_size", 64)
    device = config.get("device", "cuda")

    return create_dataloader(
        dataset,
        batch_size=eval_batch_size,
        num_workers=eval_num_workers,
        mode="eval",
        device=device,
    )
