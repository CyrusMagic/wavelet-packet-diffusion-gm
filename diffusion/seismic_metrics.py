"""
Intensity measure (IM) utilities for ground motion waveforms.

Supported metrics:
- PGA
- Arias intensity
- T5 and D5-95 (based on Arias cumulative energy)
- Husid curve (normalized cumulative energy)
"""

import numpy as np
from typing import Dict, Tuple


class SeismicMetrics:
    """Compute common intensity measures for acceleration time histories."""

    def __init__(self, dt: float = 0.01, g: float = 9.81, husid_points: int = 256):
        """Args:
        dt: sampling interval in seconds.
        g: gravitational acceleration (m/s^2).
        husid_points: number of samples for the Husid curve.
        """
        self.dt = dt
        self.g = g
        self.husid_points = husid_points
        self.husid_time_axis = None

    def peak_ground_acceleration(self, acc: np.ndarray) -> float:
        """Peak ground acceleration (PGA)."""
        if acc.ndim > 1:
            acc = acc.squeeze()
        return float(np.max(np.abs(acc)))

    def arias_intensity(self, acc: np.ndarray) -> float:
        """Arias intensity."""
        if acc.ndim > 1:
            acc = acc.squeeze()
        return float(np.pi / (2 * self.g) * np.sum(acc**2) * self.dt)

    def significant_duration(self, acc: np.ndarray) -> float:
        """Significant duration D5-95 (seconds)."""
        if acc.ndim > 1:
            acc = acc.squeeze()

        ai_cumulative = np.cumsum(acc**2) * self.dt
        ai_total = ai_cumulative[-1]

        if ai_total <= 0:
            return 0.0

        idx_5 = np.where(ai_cumulative >= 0.05 * ai_total)[0]
        idx_95 = np.where(ai_cumulative >= 0.95 * ai_total)[0]

        if len(idx_5) == 0 or len(idx_95) == 0:
            return 0.0

        t5 = float(idx_5[0] * self.dt)
        t95 = float(idx_95[0] * self.dt)
        return max(0.0, t95 - t5)

    def t5_time(self, acc: np.ndarray) -> float:
        """Time when cumulative Arias intensity reaches 5% (seconds)."""
        if acc.ndim > 1:
            acc = acc.squeeze()

        ai_cumulative = np.cumsum(acc**2) * self.dt
        ai_total = ai_cumulative[-1]

        if ai_total <= 0:
            return 0.0

        idx_5 = np.where(ai_cumulative >= 0.05 * ai_total)[0]
        if len(idx_5) == 0:
            return 0.0

        return float(idx_5[0] * self.dt)

    def get_arias_percentile_time(self, acc: np.ndarray, percentile: float = 0.99) -> float:
        """Time when cumulative Arias intensity reaches `percentile` (seconds)."""
        if acc.ndim > 1:
            acc = acc.squeeze()

        ai_cumulative = np.cumsum(acc**2) * self.dt
        ai_total = ai_cumulative[-1]

        if ai_total <= 0:
            return float(len(acc) * self.dt)

        threshold = percentile * ai_total
        idx = np.where(ai_cumulative >= threshold)[0]

        if len(idx) == 0:
            return float(len(acc) * self.dt)

        return float(idx[0] * self.dt)

    def husid_curve(self, acc: np.ndarray) -> np.ndarray:
        """Husid curve J(t) resampled to `husid_points` in [0, 1]."""
        if acc.ndim > 1:
            acc = acc.squeeze()

        acc_flat = acc.astype(np.float64)
        energy = acc_flat**2
        cumulative = np.cumsum(energy) * self.dt
        total = cumulative[-1] if cumulative.size > 0 else 0.0

        if total <= 0 or cumulative.size == 0:
            return np.zeros(self.husid_points, dtype=np.float32)

        j_t = cumulative / total
        time_axis = np.arange(cumulative.size, dtype=np.float64) * self.dt
        resample_times = np.linspace(
            0.0, time_axis[-1], self.husid_points, endpoint=True
        )

        # Cache the time axis for potential downstream use.
        if (
            self.husid_time_axis is None
            or len(self.husid_time_axis) != self.husid_points
            or not np.allclose(self.husid_time_axis[-1], resample_times[-1])
        ):
            self.husid_time_axis = resample_times

        return np.interp(
            resample_times,
            time_axis,
            j_t,
            left=0.0,
            right=1.0,
        ).astype(np.float32)

    def calculate_all_metrics(self, acc: np.ndarray) -> Dict[str, float | np.ndarray]:
        """Compute a bundle of metrics in one pass."""
        if acc.ndim > 1:
            acc = acc.squeeze()

        metrics = {}

        metrics['pga'] = self.peak_ground_acceleration(acc)

        metrics['arias_intensity'] = self.arias_intensity(acc)

        metrics['t5'] = self.t5_time(acc)
        metrics['d595'] = self.significant_duration(acc)

        metrics['husid_curve'] = self.husid_curve(acc)

        return metrics


def compare_waveforms_metrics(
    acc_target: np.ndarray,
    acc_pred: np.ndarray,
    calculator: SeismicMetrics = None
) -> Dict[str, float]:
    """Compare IMs between a target and predicted waveform."""
    if calculator is None:
        calculator = SeismicMetrics()

    metrics_target = calculator.calculate_all_metrics(acc_target)
    metrics_pred = calculator.calculate_all_metrics(acc_pred)

    return {
        'pga_target': metrics_target['pga'],
        'pga_pred': metrics_pred['pga'],
        't5_target': metrics_target['t5'],
        't5_pred': metrics_pred['t5'],
        'd595_target': metrics_target['d595'],
        'd595_pred': metrics_pred['d595'],
        'ia_target': metrics_target['arias_intensity'],
        'ia_pred': metrics_pred['arias_intensity'],
        'husid_target': metrics_target['husid_curve'],
        'husid_pred': metrics_pred['husid_curve'],
    }
