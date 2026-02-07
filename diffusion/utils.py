from collections.abc import Mapping, Sequence
from pathlib import Path

import PIL
import pytorch_lightning as pl
import torch
import numpy as np


def min_max_norm(x, x_min, x_max, range="[0,1]", mode="add"):

    if range == "[0,1]":
        if mode == "sub":
            return (x - x_min) / (x_max - x_min)
        elif mode == "add":
            return x * (x_max - x_min) + x_min
        else:
            raise NotImplementedError

    elif range == "[-1,1]":
        if mode == "sub":
            return 2.0 * ((x - x_min) / (x_max - x_min)) - 1.0
        elif mode == "add":
            x = (x + 1.0) / 2.0
            return x * (x_max - x_min) + x_min
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError


def to_numpy(x):
    if isinstance(x, Sequence):
        return x.__class__(to_numpy(v) for v in x)
    elif isinstance(x, Mapping):
        return x.__class__((k, to_numpy(v)) for k, v in x.items())
    else:
        return x.numpy(force=True) if isinstance(x, torch.Tensor) else x


class NumpyArgMixin:
    """Mixin for automatic conversion of method arguments to numpy arrays."""

    def __getattribute__(self, name):
        """Return a function wrapper that converts method arguments to numpy arrays."""
        attr = super().__getattribute__(name)
        if not callable(attr):
            return attr

        def wrapper(*args, **kwargs):
            args = to_numpy(args)
            kwargs = to_numpy(kwargs)
            return attr(*args, **kwargs)

        return wrapper


def load_model(type: type[pl.LightningModule], path: Path, **kwargs):
    """Load a PyTorch Lightning model checkpoint.

    Parameters
    ----------
    type : Type[pl.LightningModule]
        The type of the model to load.
    path : Path
        The checkpoint path.
    **kwargs : dict
        The keyword arguments to pass to the load_from_checkpoint method.
        Instead one can add `self.save_hyperparameters()` to the init method
        of the model.

    Returns
    -------
    model : pl.LightningModule
        The trained model.

    """
    if not path.exists():
        logging.info("Model not found. Returning None.")
        return None
    model = type.load_from_checkpoint(path, **kwargs)
    return model


def fig2PIL(fig):
    """Convert a matplotlib figure to a PIL Image.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The matplotlib figure.

    Returns
    -------
    PIL.Image
        The PIL Image.

    """
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = fig.canvas.tostring_rgb()
    return PIL.Image.frombytes(mode="RGB", size=(w, h), data=buf)


def get_last_checkpoint(dirpath):
    checkpoints = sorted(list(Path(dirpath).glob("*.ckpt")))
    if len(checkpoints) == 0:
        print("No checkpoint found. Returning None.")
        return None
    # Load the checkpoint with the latest epoch
    checkpoint = checkpoints[-1]
    print(f"Last checkpoint is : {checkpoint}")
    return checkpoint


def nigam_jennings(accel, period=None, damp=0.05, dt=0.01):
    if period is None:
        period = np.logspace(np.log10(0.01), np.log10(10), 130)
    # Nigam-Jennings method.
    # t0 is period, kcc is damping
    pi, cos, sin = np.pi, np.cos, np.sin
    exp = np.exp
    # initial data
    rows = len(accel)
    columns = len(period)
    new_disp = np.zeros((rows, columns))
    new_vel = np.zeros((rows, columns))
    new_accel = np.zeros((rows, columns))
    # omega
    omega = np.zeros(columns)
    maks = ~np.isclose(period, 0)
    omega[maks] = 2 * pi / period[maks]
    h = damp
    # parameters
    d1 = exp(-h * omega * dt)
    d2 = (1 - h**2.0) ** 0.5
    d3 = sin(d2 * omega * dt)
    d4 = cos(d2 * omega * dt)
    d5 = (2.0 * h**2 - 1) / (omega**2 * dt)
    d6 = (2.0 * h) / (omega**3 * dt)
    d7 = 1.0 / omega**2
    # coefficient
    a11 = d1 * (h * d3 / d2 + d4)
    a12 = d1 * d3 / (d2 * omega)
    a21 = -omega * d1 * d3 / d2
    a22 = d1 * (d4 - h * d3 / d2)
    b11 = d1 * ((d5 + h / omega) * d3 / (d2 * omega) + (d6 + d7) * d4) - d6
    b12 = -d1 * ((d5 * d3 / (d2 * omega) + d6 * d4)) - d7 + d6
    b21 = (
        d1
        * (
            (d5 + h / omega) * (d4 - h * d3 / d2)
            - (d6 + d7) * (omega * d2 * d3 + h * omega * d4)
        )
        + d7 / dt
    )
    b22 = (
        -d1 * (d5 * (d4 - h * d3 / d2) - d6 * (d2 * omega * d3 + h * omega * d4))
        - d7 / dt
    )
    # initial velocity is zero
    new_vel[0, :] = -accel[0] * dt
    new_accel[0, :] = 2.0 * damp * omega * accel[0] * dt
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
            2 * damp * omega * new_vel[i + 1, :] + omega**2 * new_disp[i + 1, :]
        )
    Sa = np.max(np.abs(new_accel), axis=0)
    return Sa
