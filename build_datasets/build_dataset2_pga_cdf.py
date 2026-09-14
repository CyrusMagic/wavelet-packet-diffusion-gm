#!/usr/bin/env python3
"""
Prepare NGA dataset with SA/PGA normalization and PGA CDF normalization.

Optionally exports IM features (Arias intensity, T5, D5-95, tc, Husid curve).
You can disable IM export via --no-im. The IM computation uses `--dt` when
provided; otherwise it reads `_meta.target_dt` from the source file, falling
back to 0.01s.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np

EPS = 1e-8
G = 9.80665  # gravitational acceleration (m/s^2)
HUSID_RESAMPLE_POINTS = 256


def compute_cdf(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply log1p and map empirical CDF to [-1, 1] for non-negative values."""
    log_vals = np.log1p(np.clip(values, a_min=0.0, a_max=None))
    order = np.argsort(log_vals)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = (np.arange(log_vals.size, dtype=np.float64) + 0.5) / log_vals.size
    cdf = 2.0 * ranks - 1.0
    log_sorted = log_vals[order]
    cdf_sorted = cdf[order]
    return cdf, log_sorted, cdf_sorted


def _safe_percent_time(J: np.ndarray, t: np.ndarray, q: float) -> float:
    """Return the first time reaching percentile q; fallback to the last time."""
    # q: 0..1
    idx = np.searchsorted(J, q, side="left")
    idx = int(np.clip(idx, 0, len(t) - 1))
    return float(t[idx])


def compute_im_features(acc: np.ndarray, dt: float) -> Tuple[np.ndarray, dict]:
    """Compute IM features from acceleration waveforms."""
    N, T = acc.shape
    t = np.arange(T, dtype=np.float64) * float(dt)

    # Arias intensity (raw)
    # IA = pi/(2g) * \int a^2 dt
    e = acc.astype(np.float64) ** 2
    ia_raw = (np.pi / (2.0 * G)) * (e.sum(axis=1) * dt)
    ia_cdf, ia_log_sorted, ia_cdf_sorted = compute_cdf(ia_raw)

    # Husid / timing features
    # J_i = (\sum_{k<=i} e_k dt) / E
    E = e.sum(axis=1) * dt  # [N]
    # Guard against divide-by-zero (these samples fall back to 0 for times/durations).
    E_safe = np.where(E > EPS, E, 1.0)
    J = np.cumsum(e, axis=1) * dt
    J = (J.T / E_safe).T  # normalize to [0, 1]

    # Per-sample percent times
    T5 = np.zeros(N, dtype=np.float64)
    T25 = np.zeros(N, dtype=np.float64)
    T50 = np.zeros(N, dtype=np.float64)
    T75 = np.zeros(N, dtype=np.float64)
    T95 = np.zeros(N, dtype=np.float64)
    for i in range(N):
        Ji = J[i]
        # Very small energy -> return zeros.
        if E[i] <= EPS:
            T5[i] = T25[i] = T50[i] = T75[i] = T95[i] = 0.0
            continue
        T5[i] = _safe_percent_time(Ji, t, 0.05)
        T25[i] = _safe_percent_time(Ji, t, 0.25)
        T50[i] = _safe_percent_time(Ji, t, 0.50)
        T75[i] = _safe_percent_time(Ji, t, 0.75)
        T95[i] = _safe_percent_time(Ji, t, 0.95)

    # Husid curve resampling
    time_axis = np.arange(T, dtype=np.float64) * dt
    resample_times = np.linspace(
        0.0, time_axis[-1] if T > 0 else 0.0, HUSID_RESAMPLE_POINTS, endpoint=True
    )
    husid_resampled = np.zeros((N, HUSID_RESAMPLE_POINTS), dtype=np.float64)
    for i in range(N):
        if E[i] <= EPS:
            continue
        husid_resampled[i] = np.interp(
            resample_times,
            time_axis,
            J[i],
            left=0.0,
            right=1.0,
        )

    D595 = T95 - T5
    D2575 = np.maximum(0.0, T75 - T25)

    # Energy centroid within [T5, T95]
    tc = np.zeros(N, dtype=np.float64)
    for i in range(N):
        if E[i] <= EPS or D595[i] <= 0:
            tc[i] = 0.0
            continue
        # Weights and time within the window
        i5 = int(np.searchsorted(t, T5[i], side="left"))
        i95 = int(np.searchsorted(t, T95[i], side="left"))
        i95 = max(i95, i5 + 1)
        tw = t[i5 : i95 + 1]
        rw = e[i, i5 : i95 + 1] / (E[i] / dt)  # r = e / (E/dt)
        denom = np.sum(rw)
        if denom <= EPS:
            tc[i] = T5[i]
        else:
            tc[i] = float(np.sum(tw * rw) / denom)

    # Normalize features via CDF normalization (same idea as Arias intensity).

    # T5 CDF normalization
    t5_cdf, t5_log_sorted, t5_cdf_sorted = compute_cdf(T5)

    # D5-95 CDF normalization
    d595_cdf, d595_log_sorted, d595_cdf_sorted = compute_cdf(D595)

    # tc (time centroid) CDF normalization
    tc_cdf, tc_log_sorted, tc_cdf_sorted = compute_cdf(tc)

    extras = {
        "arias_NoNorm": ia_raw.astype(np.float32),
        "t5_raw": T5.astype(np.float32),
        "d595_raw": D595.astype(np.float32),
        "d2575_raw": D2575.astype(np.float32),
        "tc_raw": tc.astype(np.float32),
        "t5_norm": t5_cdf.astype(np.float32),
        "d595_norm": d595_cdf.astype(np.float32),
        "tc_norm": tc_cdf.astype(np.float32),
        "husid_resampled": husid_resampled.astype(np.float32),
        "husid_time_axis": resample_times.astype(np.float32),
        # Lookup tables for denormalization
        "ia_log_sorted": ia_log_sorted.astype(np.float32),
        "ia_cdf_sorted": ia_cdf_sorted.astype(np.float32),
        "t5_log_sorted": t5_log_sorted.astype(np.float32),
        "t5_cdf_sorted": t5_cdf_sorted.astype(np.float32),
        "d595_log_sorted": d595_log_sorted.astype(np.float32),
        "d595_cdf_sorted": d595_cdf_sorted.astype(np.float32),
        "tc_log_sorted": tc_log_sorted.astype(np.float32),
        "tc_cdf_sorted": tc_cdf_sorted.astype(np.float32),
    }
    return ia_cdf.astype(np.float32), extras


def _read_attr_float(group: h5py.Group | None, name: str) -> float | None:
    if group is None:
        return None
    if name not in group.attrs:
        return None
    try:
        value = group.attrs[name]
    except Exception:
        return None
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        value = value.astype(np.float64).flat[0]
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def copy_alignment_info(src: h5py.File, dst: h5py.File) -> None:
    if "alignment_info" in src:
        src.copy("alignment_info", dst)


def process_dataset(
    source: Path, target: Path, with_im: bool = True, dt: float | None = None
) -> float:
    with h5py.File(source, "r") as src, h5py.File(target, "w") as dst:
        print(f"[load] reading wfs from {source} ...", flush=True)
        wfs = np.array(src["wfs"], dtype=np.float32)
        print(f"[load] wfs loaded with shape {wfs.shape}", flush=True)
        if wfs.ndim == 2:
            wfs = wfs[:, np.newaxis, :]
        elif wfs.ndim != 3:
            raise ValueError(
                f"Unexpected wfs shape {wfs.shape}; expect [N, C, T] or [N, T]"
            )
        if wfs.shape[1] == 0:
            raise ValueError("wfs channel dimension cannot be zero")
        dst.create_dataset("wfs", data=wfs)

        if "wavelet_db6" in src and "wavelet_db6" not in dst:
            print("[wavelet] loading compressed wavelet_db6 data ...", flush=True)
            src_wavelet = src["wavelet_db6"]
            coeffs = np.array(src_wavelet["coeffs"], dtype=np.float32)

            wavelet_group = dst.create_group("wavelet_db6")
            wavelet_group.create_dataset("coeffs", data=coeffs)

            if "meta" in src_wavelet:
                src_meta = src_wavelet["meta"]
                meta_group = wavelet_group.create_group("meta")
                for key in src_meta.keys():
                    meta_data = np.array(src_meta[key])
                    meta_group.create_dataset(key, data=meta_data)
                for attr, value in src_meta.attrs.items():
                    meta_group.attrs[attr] = value

            for attr, value in src_wavelet.attrs.items():
                wavelet_group.attrs[attr] = value

            print(
                "[wavelet] wavelet_db6 coefficients written without compression",
                flush=True,
            )

        print("[load] reading SA array ...", flush=True)
        sa_raw = np.array(src["sa"], dtype=np.float32)
        print(f"[load] sa loaded with shape {sa_raw.shape}", flush=True)
        if sa_raw.ndim == 3 and sa_raw.shape[1] in (1,):
            sa_flat = sa_raw[:, 0, :]
        elif sa_raw.ndim == 2:
            sa_flat = sa_raw
            sa_raw = sa_raw[:, np.newaxis, :]
        else:
            raise ValueError(
                f"Unexpected sa shape {sa_raw.shape}; expect [N, 1, K] or [N, K]"
            )

        print("[load] reading PGA array ...", flush=True)
        pga_raw = np.array(src["pga"], dtype=np.float64)
        print(f"[load] pga loaded with shape {pga_raw.shape}", flush=True)
        pga_shape = pga_raw.shape
        pga_flat = pga_raw.reshape(pga_shape[0], -1)
        if pga_flat.shape[1] != 1:
            raise ValueError(f"Unexpected pga shape {pga_shape}; expect [N] or [N, 1]")
        pga_flat = pga_flat[:, 0]
        print("[norm] computing PGA CDF ...", flush=True)
        pga_cdf, log_sorted, cdf_sorted = compute_cdf(pga_flat)
        print("[norm] PGA CDF done", flush=True)
        dst.create_dataset("pga", data=pga_cdf.reshape(pga_shape).astype(np.float32))
        dst.create_dataset("pga_NoNorm", data=pga_raw.astype(np.float32))

        safe_pga_vec = np.where(pga_flat > EPS, pga_flat, 1.0)

        print("[norm] computing SA PGA normalization ...", flush=True)
        sa_pga_norm = sa_flat / safe_pga_vec[:, np.newaxis]
        print(
            f"[norm] SA PGA normalization done. SA[0] mean after norm: {sa_pga_norm[:, 0].mean():.4f}",
            flush=True,
        )
        dst.create_dataset(
            "sa",
            data=sa_pga_norm[:, np.newaxis, :].astype(np.float32),
        )
        dst.create_dataset(
            "sa_NoNorm",
            data=sa_raw.astype(np.float32),
        )

        stats = dst.require_group("_norm_stats")
        stats.create_dataset("pga_log_sorted", data=log_sorted.astype(np.float32))
        stats.create_dataset("pga_cdf_sorted", data=cdf_sorted.astype(np.float32))
        stats.create_dataset(
            "pga_cdf_eps", data=np.array(1.0 / pga_flat.size, dtype=np.float32)
        )

        meta_group = src.get("_meta") if "_meta" in src else None
        effective_dt = (
            dt if dt is not None else _read_attr_float(meta_group, "target_dt")
        )
        if effective_dt is None:
            effective_dt = 0.01

        if with_im:
            print(
                f"[im] computing IM features for {wfs.shape[0]} waveforms (dt={effective_dt:.6f}s) ...",
                flush=True,
            )
            acc = wfs[:, 0, :].astype(np.float64)
            ia_cdf, extras = compute_im_features(acc, effective_dt)
            print("[im] IM features computed", flush=True)
            dst.create_dataset("arias", data=ia_cdf)
            dst.create_dataset("t5_norm", data=extras["t5_norm"])  # type: ignore[index]
            dst.create_dataset("d595_norm", data=extras["d595_norm"])  # type: ignore[index]
            dst.create_dataset("tc_norm", data=extras["tc_norm"])  # type: ignore[index]
            dst.create_dataset("husid", data=extras["husid_resampled"])  # type: ignore[index]
            dst.create_dataset("arias_NoNorm", data=extras["arias_NoNorm"])  # type: ignore[index]
            dst.create_dataset("T5", data=extras["t5_raw"])  # type: ignore[index]
            dst.create_dataset("D5_95", data=extras["d595_raw"])  # type: ignore[index]
            dst.create_dataset("D25_75", data=extras["d2575_raw"])  # type: ignore[index]
            dst.create_dataset("t_c", data=extras["tc_raw"])  # type: ignore[index]
            stats.create_dataset("arias_log_sorted", data=extras["ia_log_sorted"])  # type: ignore[index]
            stats.create_dataset("arias_cdf_sorted", data=extras["ia_cdf_sorted"])  # type: ignore[index]
            # Save lookup tables for denormalization (T5, D5-95, tc).
            stats.create_dataset("t5_log_sorted", data=extras["t5_log_sorted"])  # type: ignore[index]
            stats.create_dataset("t5_cdf_sorted", data=extras["t5_cdf_sorted"])  # type: ignore[index]
            stats.create_dataset("d595_log_sorted", data=extras["d595_log_sorted"])  # type: ignore[index]
            stats.create_dataset("d595_cdf_sorted", data=extras["d595_cdf_sorted"])  # type: ignore[index]
            stats.create_dataset("tc_log_sorted", data=extras["tc_log_sorted"])  # type: ignore[index]
            stats.create_dataset("tc_cdf_sorted", data=extras["tc_cdf_sorted"])  # type: ignore[index]
            stats.create_dataset("dt", data=np.array(effective_dt, dtype=np.float32))
            stats.create_dataset(
                "husid_time_axis",
                data=extras["husid_time_axis"],
            )

        # Copy metadata
        for name in ("period", "rsn", "event", "station", "component", "source_dt"):
            if name in src and name not in dst:
                src.copy(name, dst)

        if meta_group is not None and "_meta" not in dst:
            src.copy("_meta", dst)

        copy_alignment_info(src, dst)

    return effective_dt


DATA_ROOT = Path("./datasets")
DEFAULT_SOURCE = DATA_ROOT / "step1_NGAH1_len16k_symmetric_freq.h5"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=("Source NGA HDF5 path"),
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=None,
        help=("Output HDF5 path (default: infer from source name: step1 -> step2)"),
    )
    parser.add_argument(
        "--no-im",
        action="store_true",
        help="Skip IM feature computation (default: compute IA/T5/D5-95/tc/Husid)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Sampling interval (seconds) for IM computation. Default: read _meta.target_dt; fallback to 0.01.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    source: Path = args.source.expanduser().resolve()

    # Infer target filename: replace step1_ with step2_
    if args.target:
        target = args.target.expanduser().resolve()
    else:
        stem = source.stem
        new_stem = stem.replace("step1_", "step2_", 1)
        target = source.with_name(f"{new_stem}{source.suffix}")

    target.parent.mkdir(parents=True, exist_ok=True)
    with_im = not args.no_im
    effective_dt = process_dataset(source, target, with_im=with_im, dt=args.dt)
    dt_str = f"{effective_dt:.6f}" if effective_dt is not None else "unknown"
    print(f"Wrote dataset to {target} (with_im={with_im}, dt={dt_str})")


if __name__ == "__main__":
    main()
