#!/usr/bin/env python3
"""Prepare NGA waveforms at 100 Hz (optional db6 wavelet export)."""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
from tqdm import tqdm

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from diffusion.representation import get_db6_wavelet_representation

    _WAVELET_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - optional dependency at runtime
    get_db6_wavelet_representation = None  # type: ignore[assignment]
    _WAVELET_IMPORT_ERROR = str(exc)

G_TO_CMS2 = 980.0  # Convert acceleration from g to cm/s^2


def nigam_jennings(
    accel: np.ndarray,
    period: Optional[np.ndarray] = None,
    damp: float | np.ndarray = 0.05,
    dt: float = 0.01,
) -> np.ndarray:
    """Compute pseudo spectral acceleration with the Nigam-Jennings method.
    """
    if period is None:
        period = np.logspace(np.log10(0.01), np.log10(10.0), 130)
    accel = np.asarray(accel, dtype=np.float64)

    damp_array = np.atleast_1d(damp).astype(np.float64)
    is_single_damp = damp_array.size == 1
    rows = accel.shape[0]
    n_periods = period.shape[0]

    sa_results = []
    for h in damp_array:
        new_disp = np.zeros((rows, n_periods), dtype=np.float64)
        new_vel = np.zeros((rows, n_periods), dtype=np.float64)
        new_accel = np.zeros((rows, n_periods), dtype=np.float64)
        omega = np.zeros(n_periods, dtype=np.float64)
        valid = ~np.isclose(period, 0.0)
        omega[valid] = 2.0 * np.pi / period[valid]
        d1 = np.exp(-h * omega * dt)
        d2 = np.sqrt(np.maximum(1.0 - h**2.0, 0.0))
        d3 = np.sin(d2 * omega * dt)
        d4 = np.cos(d2 * omega * dt)
        inv_omega = np.zeros_like(omega)
        nz = omega != 0.0
        inv_omega[nz] = 1.0 / omega[nz]
        inv_omega2 = inv_omega**2
        inv_omega3 = inv_omega2 * inv_omega
        d5 = (2.0 * h**2 - 1.0) * inv_omega2 / dt
        d6 = (2.0 * h) * inv_omega3 / dt
        d7 = inv_omega2
        factor = (d3 * inv_omega) / d2
        a11 = d1 * (h * d3 / d2 + d4)
        a12 = d1 * factor
        a21 = -omega * d1 * d3 / d2
        a22 = d1 * (d4 - h * d3 / d2)
        b11 = d1 * ((d5 + h * inv_omega) * factor + (d6 + d7) * d4) - d6
        b12 = -d1 * (d5 * factor + d6 * d4) - d7 + d6
        term = d2 * omega * d3 + h * omega * d4
        b21 = (
            d1 * ((d5 + h * inv_omega) * (d4 - h * d3 / d2) - (d6 + d7) * term)
            + d7 / dt
        )
        b22 = -d1 * (d5 * (d4 - h * d3 / d2) - d6 * term) - d7 / dt
        if rows > 0:
            new_vel[0, :] = -accel[0] * dt
            new_accel[0, :] = 2.0 * h * omega * accel[0] * dt
        for i in range(1, rows - 1):
            new_disp[i + 1, :] = (
                a11 * new_disp[i, :]
                + a12 * new_vel[i, :]
                + b11 * accel[i]
                + b12 * accel[i + 1]
            )
            new_vel[i + 1, :] = (
                a21 * new_disp[i, :]
                + a22 * new_vel[i, :]
                + b21 * accel[i]
                + b22 * accel[i + 1]
            )
            new_accel[i + 1, :] = -(
                2.0 * h * omega * new_vel[i + 1, :] + omega**2 * new_disp[i + 1, :]
            )
        sa_results.append(np.max(np.abs(new_accel), axis=0))

    if is_single_damp:
        return sa_results[0]
    return np.stack(sa_results, axis=0)


@dataclass(frozen=True)
class RecordMeta:
    rsn: str
    event: str
    station: str
    component: str
    component_upper: str
    dt: float
    npts: int

    @property
    def key(self) -> Tuple[str, str, str]:
        return self.rsn, self.event, self.station


def decode_ascii(raw: np.bytes_) -> str:
    return raw.decode("utf-8", errors="ignore").strip()


def load_records_metadata(h5_file: h5py.File) -> List[RecordMeta]:
    records = h5_file["metadata"]["records_index"]
    dtype_id = records.id.get_type()
    count = records.shape[0]
    item_size = dtype_id.get_size()

    raw = np.empty((count,), dtype=np.dtype(("V", item_size)))
    records.id.read(
        h5py.h5s.create_simple((count,)), records.id.get_space(), raw, mtype=dtype_id
    )

    view_dtype = np.dtype(
        {
            "names": [
                "rsn",
                "event",
                "station",
                "component",
                "dt",
                "npts",
                "magnitude",
                "vs30",
                "hypd",
            ],
            "formats": [
                "S10",
                "S50",
                "S50",
                "S10",
                "<f4",
                "<i4",
                "<f4",
                "<f4",
                "<f4",
            ],
        }
    )
    data = raw.view(view_dtype)

    metas: List[RecordMeta] = []
    for row in data:
        comp_raw = decode_ascii(row["component"])
        comp_upper = comp_raw.upper()
        if not comp_upper:
            continue
        metas.append(
            RecordMeta(
                rsn=decode_ascii(row["rsn"]),
                event=decode_ascii(row["event"]),
                station=decode_ascii(row["station"]),
                component=comp_raw,
                component_upper=comp_upper,
                dt=float(row["dt"]),
                npts=int(row["npts"]),
            )
        )
    return metas


def filter_records_by_mode(records: List[RecordMeta], mode: str) -> List[RecordMeta]:
    """Filter records based on export mode (h1 or 3c)."""
    if mode == "h1":
        return [r for r in records if r.component_upper == "H1"]
    elif mode == "3c":
        return group_3c_records(records)
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'h1' or '3c'.")


def group_3c_records(records: List[RecordMeta]) -> List[RecordMeta]:
    """Group records by (rsn, event, station) and keep only complete 3C sets (H1, H2, V)."""
    grouped: Dict[Tuple[str, str, str], Dict[str, RecordMeta]] = {}
    for meta in records:
        key = meta.key
        comp = meta.component_upper
        if comp in {"H1", "H2", "V"}:
            grouped.setdefault(key, {})[comp] = meta

    # Keep only complete sets and maintain H1→H2→V order
    result: List[RecordMeta] = []
    for key, comps in grouped.items():
        if {"H1", "H2", "V"}.issubset(comps.keys()):
            result.extend([comps["H1"], comps["H2"], comps["V"]])

    return result


def resample_to_target(
    wave: np.ndarray, source_dt: float, target_dt: float, target_len: int
) -> np.ndarray:
    if wave.size == 0 or source_dt <= 0:
        return np.zeros(target_len, dtype=np.float32)
    source_t = np.arange(wave.size, dtype=np.float64) * source_dt
    target_t = np.arange(target_len, dtype=np.float64) * target_dt
    resampled = np.interp(target_t, source_t, wave, left=0.0, right=0.0)
    return resampled.astype(np.float32)


def compute_sa_pga(
    wave: np.ndarray,
    period: np.ndarray,
    dt: float,
) -> Tuple[np.ndarray, float]:
    """Compute 5% damped SA and PGA."""
    wave = wave.astype(np.float64)
    sa_single = nigam_jennings(wave, period=period, damp=0.05, dt=dt)

    pga = float(np.max(np.abs(wave)))
    return sa_single.astype(np.float32), pga


@dataclass
class WaveletExportConfig:
    level: int = 7
    crop_to: Optional[int] = 128


def _stack_wavelet_meta(
    meta_list: List[Dict[str, np.ndarray]],
) -> Dict[str, np.ndarray]:
    if not meta_list:
        raise ValueError("wavelet meta list is empty")

    stacked: Dict[str, np.ndarray] = {}
    keys = ["mean", "std", "coeff_len", "pad"]
    for key in keys:
        stacked[key] = np.stack([np.asarray(meta[key]) for meta in meta_list], axis=0)

    stacked["orig_len"] = np.asarray(
        [int(np.asarray(meta.get("orig_len", 0)).item()) for meta in meta_list],
        dtype=np.int32,
    )
    stacked["level"] = np.asarray(
        [int(np.asarray(meta.get("level", 0)).item()) for meta in meta_list],
        dtype=np.int32,
    )
    return stacked


def _compute_wavelet_features(
    waveforms: np.ndarray,
    config: WaveletExportConfig,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, int]]:
    if get_db6_wavelet_representation is None:
        raise ImportError(
            "diffusion.representation.get_db6_wavelet_representation not found. "
            "Please install project dependencies."
        )

    coeffs_list: List[np.ndarray] = []
    meta_list: List[Dict[str, np.ndarray]] = []
    crop_to = config.crop_to
    detected_width: Optional[int] = None
    for sample in tqdm(waveforms, desc="Computing db6 wavelet", leave=False):
        coeffs, meta = get_db6_wavelet_representation(sample, level=config.level)
        if detected_width is None:
            detected_width = coeffs.shape[-1]
            logging.info("Wavelet coeff shape (before crop): %s", coeffs.shape)
            logging.info("Wavelet coeff width detected: %d", detected_width)
            if crop_to is not None and detected_width <= crop_to:
                logging.info(
                    "Coeff width <= crop target (%d); skipping crop.",
                    crop_to,
                )
                crop_to = None
        if crop_to is not None and coeffs.shape[-1] > crop_to:
            logging.info(
                "Cropping wavelet coeff from width %d to %d", coeffs.shape[-1], crop_to
            )
            coeffs = coeffs[..., :crop_to]
        coeffs_list.append(coeffs.astype(np.float32))
        meta_list.append(meta)

    coeff_tensor = np.stack(coeffs_list, axis=0)
    meta_tensor = _stack_wavelet_meta(meta_list)
    info = {
        "detected_width": int(detected_width or coeff_tensor.shape[-1]),
        "crop_used": -1 if crop_to is None else int(crop_to),
    }
    return coeff_tensor, meta_tensor, info


def process_dataset(
    source_path: Path,
    target_path: Path,
    target_dt: float = 0.01,
    target_duration: float = 120.0,
    target_len: Optional[int] = None,
    period: Optional[np.ndarray] = None,
    limit: Optional[int] = None,
    mode: str = "h1",
    wavelet_config: Optional[WaveletExportConfig] = None,
) -> None:
    if target_len is None:
        target_len = int(round(target_duration / target_dt))
    else:
        target_duration = float(target_len) * target_dt

    if period is None:
        period = np.logspace(np.log10(0.01), np.log10(10.0), 130)

    logging.info("Loading source file: %s", source_path)
    with h5py.File(source_path, "r") as src:
        records = load_records_metadata(src)
        logging.info("Loaded %d total records from metadata", len(records))

        # Filter records by mode
        records = filter_records_by_mode(records, mode)
        logging.info("After %s filtering: %d records", mode.upper(), len(records))

        if limit is not None:
            records = records[:limit]
            logging.info("Limited to %d records for testing", len(records))

        wfs_store: List[np.ndarray] = []
        sa_store: List[np.ndarray] = []
        # Public release: single-damping response spectrum only.
        pga_store: List[float] = []
        rsn_list: List[str] = []
        event_list: List[str] = []
        station_list: List[str] = []
        component_labels: List[str] = []
        source_dt_list: List[float] = []

        skipped_dt = 0
        missing_records = 0

        for meta in tqdm(records, desc="Processing records"):
            dt_val = float(meta.dt)
            if dt_val <= 0:
                skipped_dt += 1
                continue

            h5_path = f"events/{meta.event}/stations/{meta.station}/acceleration/{meta.component}"
            if h5_path not in src:
                missing_records += 1
                logging.debug(
                    "Dataset not found for %s/%s/%s/%s",
                    meta.rsn,
                    meta.event,
                    meta.station,
                    meta.component,
                )
                continue

            dataset = src[h5_path]
            data = dataset.astype("<f4")[...].astype(np.float64) * G_TO_CMS2
            resampled = resample_to_target(data, dt_val, target_dt, target_len)
            sa_vals, pga_val = compute_sa_pga(
                resampled,
                period,
                target_dt,
            )

            wfs_store.append(resampled)
            sa_store.append(sa_vals)
            pga_store.append(pga_val)
            rsn_list.append(meta.rsn)
            event_list.append(meta.event)
            station_list.append(meta.station)
            component_labels.append(meta.component)
            source_dt_list.append(dt_val)

    if not wfs_store:
        raise RuntimeError(
            "No waveform data were processed. Check source file and filters."
        )

    wfs = np.stack(wfs_store, axis=0).astype(np.float32)
    sa = np.stack(sa_store, axis=0).astype(np.float32)
    pga = np.array(pga_store, dtype=np.float32)
    dt_arr = np.array(source_dt_list, dtype=np.float32)

    # Reshape based on mode
    if mode == "h1":
        # H1 mode: (N, 1, T) - single channel
        wfs = np.expand_dims(wfs, axis=1)
        sa = np.expand_dims(sa, axis=1)
        pga = np.expand_dims(pga, axis=1)
    elif mode == "3c":
        # 3C mode: (N/3, 3, T) - three channels (H1, H2, V)
        num_samples = wfs.shape[0] // 3
        wfs = wfs[: num_samples * 3].reshape(num_samples, 3, -1)
        sa = sa[: num_samples * 3].reshape(num_samples, 3, -1)
        pga = pga[: num_samples * 3].reshape(num_samples, 3)

        # Group metadata for 3C mode (keep only first of each triplet)
        rsn_list = rsn_list[::3]
        event_list = event_list[::3]
        station_list = station_list[::3]
        component_labels = ["H1+H2+V"] * num_samples
        dt_arr = dt_arr[::3]

    rsn_arr = np.array(rsn_list, dtype=object)
    event_arr = np.array(event_list, dtype=object)
    station_arr = np.array(station_list, dtype=object)
    comp_arr = np.array(component_labels, dtype=object)

    wavelet_result: Optional[
        Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, int]]
    ] = None
    if wavelet_config is not None:
        logging.info(
            "Computing db6 wavelet features (level=%d, crop_to=%s)...",
            wavelet_config.level,
            "None" if wavelet_config.crop_to is None else str(wavelet_config.crop_to),
        )
        wavelet_result = _compute_wavelet_features(wfs, wavelet_config)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Writing dataset to %s", target_path)
    utf_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(target_path, "w") as dst:
        dst.create_dataset("wfs", data=wfs, compression="gzip", compression_opts=4)
        dst.create_dataset("sa", data=sa, compression="gzip", compression_opts=4)
        dst.create_dataset("pga", data=pga)
        dst.create_dataset("period", data=period.astype(np.float32))
        dst.create_dataset("rsn", data=rsn_arr, dtype=utf_dtype)
        dst.create_dataset("event", data=event_arr, dtype=utf_dtype)
        dst.create_dataset("station", data=station_arr, dtype=utf_dtype)
        dst.create_dataset("component", data=comp_arr, dtype=utf_dtype)
        dst.create_dataset("source_dt", data=dt_arr)
        meta = dst.create_group("_meta")
        meta.attrs["target_dt"] = target_dt
        meta.attrs["target_duration"] = target_duration
        meta.attrs["target_length"] = float(target_len)
        meta.attrs["unit"] = "cm/s^2"
        meta.attrs["skipped_nonpositive_dt"] = float(skipped_dt)
        meta.attrs["missing_records"] = float(missing_records)

        if wavelet_result is not None:
            coeff_tensor, meta_tensor, wavelet_info = wavelet_result
            wavelet_group = dst.create_group("wavelet_db6")
            wavelet_group.create_dataset(
                "coeffs",
                data=coeff_tensor,
                compression="gzip",
                compression_opts=4,
            )
            meta_group = wavelet_group.create_group("meta")
            for key, value in meta_tensor.items():
                meta_group.create_dataset(
                    key,
                    data=value,
                    compression="gzip" if value.ndim >= 2 else None,
                    compression_opts=4 if value.ndim >= 2 else None,
                )
            meta_group.attrs["stored_length"] = coeff_tensor.shape[-1]
            meta_group.attrs["level"] = wavelet_config.level
            meta_group.attrs["detected_width"] = wavelet_info.get("detected_width", -1)
            meta_group.attrs["crop_to"] = wavelet_info.get("crop_used", -1)

    logging.info(
        "Finished writing %d records (wfs %s, sa %s, pga %s)",
        wfs.shape[0],
        wfs.shape,
        sa.shape,
        pga.shape,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert NGAW2 meta accelerograms to H1 or 3-component 100 Hz dataset"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("datasets/NGAW2_acc_meta.h5"),
        help="Input NGA meta HDF5 file",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=None,
        help="Output dataset path (auto-generated if not specified)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["h1", "3c"],
        default="h1",
        help="Export mode: 'h1' (H1 component only) or '3c' (3-component H1+H2+V)",
    )
    parser.add_argument(
        "--target-dt",
        type=float,
        default=0.01,
        help="Target sampling interval in seconds (default 0.01 -> 100 Hz)",
    )
    parser.add_argument(
        "--target-length",
        type=int,
        default=None,
        help="Target waveform length (samples). Overrides --duration if provided.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=163.84,
        help="Target duration in seconds (default 163.84 s -> 16384 points @ 100Hz)",
    )
    parser.add_argument(
        "--period-min",
        type=float,
        default=0.01,
        help="Minimum period for SA computation",
    )
    parser.add_argument(
        "--period-max",
        type=float,
        default=10.0,
        help="Maximum period for SA computation",
    )
    parser.add_argument(
        "--period-count",
        type=int,
        default=130,
        help="Number of logarithmically spaced period samples",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default INFO)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of record pairs processed (for quick tests)",
    )
    parser.add_argument(
        "--no-wavelet-db6",
        action="store_true",
        help="Disable exporting db6 wavelet packet coefficients (enabled by default)",
    )
    parser.add_argument(
        "--wavelet-level",
        type=int,
        default=7,
        help="Wavelet packet level (default: 7)",
    )
    parser.add_argument(
        "--wavelet-crop",
        type=int,
        default=-1,
        help=(
            "Optional crop length for wavelet packet coefficients. "
            "Use -1 to disable cropping (default)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="[%(levelname)s] %(message)s",
    )

    # Auto-generate target filename if not specified
    base_length = int(round(args.duration / args.target_dt))
    target_len = args.target_length or base_length

    export_wavelet = not args.no_wavelet_db6
    if export_wavelet and get_db6_wavelet_representation is None:
        details = f" Original error: {_WAVELET_IMPORT_ERROR}" if _WAVELET_IMPORT_ERROR else ""
        raise ImportError(
            "db6 wavelet export requested; please install project dependencies (e.g., PyWavelets)."
            + details
        )

    if args.target is None:
        prefix = "step1_NGAH1" if args.mode == "h1" else "step1_NGA3C"
        parts = [f"datasets/{prefix}"]

        # Add length suffix
        if target_len == 16384:
            parts.append("len16k")
        elif target_len != base_length:
            # Non-default length
            parts.append(f"len{target_len}")

        # Add wavelet mode/order tags (fixed in this repo)
        if export_wavelet:
            parts.append("symmetric")  # mode
            parts.append("freq")  # order

        args.target = Path("_".join(parts) + ".h5")
        logging.info("Auto-generated target path: %s", args.target)

    crop_value = args.wavelet_crop if args.wavelet_crop > 0 else None
    wavelet_config = (
        WaveletExportConfig(level=args.wavelet_level, crop_to=crop_value)
        if export_wavelet
        else None
    )

    period = np.logspace(
        np.log10(args.period_min), np.log10(args.period_max), args.period_count
    )
    process_dataset(
        source_path=args.source,
        target_path=args.target,
        target_dt=args.target_dt,
        target_duration=args.duration,
        target_len=target_len,
        period=period,
        limit=args.limit,
        mode=args.mode,
        wavelet_config=wavelet_config,
    )


if __name__ == "__main__":
    main()
