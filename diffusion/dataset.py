"""
Dataset utilities for earthquake ground motion generation.

This public version only exposes a single split: `test`.
"""

from functools import partial
from typing import Callable, Optional

from h5py import File
import numpy as np
import torch as th

from diffusion.representation import get_db6_wavelet_representation


class Dataset(th.utils.data.Dataset):
    """HDF5 dataset wrapper.

    - Uses db6 wavelet packet coefficients as model inputs.
    - If `wavelet_db6/coeffs` exists in the HDF5 file, it is loaded directly;
      otherwise coefficients are computed on the fly.
    """

    def __init__(
        self,
        datapath,
        cut=None,
        cond_configs=None,
        split="test",
        wavelet_level: int = 7,
        wavelet_name: str = "db6",
        representation_fn: Optional[Callable] = None,
        include_wavelet_meta: bool = True,
    ):
        super().__init__()
        self.cut = cut
        self.cond_configs = cond_configs or {}
        self.wavelet_level = wavelet_level
        self.wavelet_name = wavelet_name
        self.include_wavelet_meta = include_wavelet_meta
        self._index_lookup: dict[int, int] = {}

        # Pre-build wavelet transform function.
        if representation_fn is not None:
            self.representation_fn = representation_fn
        else:
            self.representation_fn = partial(
                get_db6_wavelet_representation,
                level=wavelet_level,
                wavelet=wavelet_name,
            )

        self.datapath = datapath
        # Open HDF5 lazily per worker to avoid sharing a handle across processes.
        self._file: Optional[File] = None
        self._cond_names = []
        self._has_wavelet_store = False
        self._wavelet_meta_keys: list[str] = []

        with File(datapath, "r") as f:
            self.total_waveforms = len(f["wfs"])
            self.waveform_shape = f["wfs"].shape
            print("Available datasets in", datapath)
            for key in f.keys():
                print(f"- {key}")

            if "sa" in f:
                try:
                    _ = int(f["sa"].shape[-1])
                except Exception:
                    pass

            if "wavelet_db6" in f and "coeffs" in f["wavelet_db6"]:
                self._has_wavelet_store = True
                print(
                    f"Detected precomputed wavelet_db6 coeffs: "
                    f"{f['wavelet_db6']['coeffs'].shape}"
                )
                meta_group = f["wavelet_db6"].get("meta")
                if meta_group is not None:
                    self._wavelet_meta_keys = sorted(meta_group.keys())

            for cond_name, config in (self.cond_configs or {}).items():
                if cond_name in f:
                    self._cond_names.append(cond_name)
                    print(f"Loaded {cond_name}: {f[cond_name].shape}")
                else:
                    print(f"Warning: {cond_name} not found in dataset")

        if split != "test":
            raise ValueError(
                f"Only split='test' is supported in the public release; got {split!r}."
            )
        self.indices = np.arange(self.total_waveforms)
        self._index_lookup = {
            int(original_idx): pos for pos, original_idx in enumerate(self.indices)
        }

    def get_feature(self, key):
        """Return a feature array for the current split."""
        file = self._ensure_file()
        return file[key][:][self.indices]

    def __del__(self):
        self._close_file()

    def __getstate__(self):
        """Drop the HDF5 handle during pickling."""
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def _ensure_file(self) -> File:
        """Open HDF5 lazily (per worker)."""
        if self._file is None:
            self._file = File(self.datapath, "r")
        return self._file

    def _close_file(self) -> None:
        """Close the cached HDF5 handle."""
        if self._file is not None:
            try:
                self._file.close()
            finally:
                self._file = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        """Return one sample."""
        original_index = self.indices[index]
        file = self._ensure_file()
        waveform = file["wfs"][original_index]

        if self.cut:
            waveform = waveform[:, : self.cut]

        wavelet_meta = None
        if self._has_wavelet_store:
            signal = np.array(
                file["wavelet_db6"]["coeffs"][original_index], dtype=np.float32
            )
            if self._wavelet_meta_keys:
                wavelet_meta = {}
                meta_group = file["wavelet_db6"]["meta"]
                for key in self._wavelet_meta_keys:
                    wavelet_meta[key] = th.tensor(
                        meta_group[key][original_index], dtype=th.float32
                    )
        else:
            signal, meta = self.representation_fn(waveform)
            signal = signal.astype(np.float32)
            wavelet_meta = {
                key: th.tensor(value, dtype=th.float32)
                for key, value in meta.items()
            }

        # Pad to a multiple of 16 for UNet downsampling.
        pad_frames = (-signal.shape[-1]) % 16
        if pad_frames:
            signal = np.pad(signal, ((0, 0), (0, 0), (0, pad_frames)), mode="constant")

        out = {
            "wfs": th.tensor(waveform, dtype=th.float32),
            "signal": th.tensor(signal, dtype=th.float32),
            "original_index": original_index,
        }

        if self.include_wavelet_meta and wavelet_meta is not None:
            out["wavelet_meta"] = wavelet_meta

        # Conditions
        cond_dict = {}

        for cond_name in self._cond_names:
            raw_value = file[cond_name][original_index]
            value = np.asarray(raw_value, dtype=np.float32)
            config = self.cond_configs.get(cond_name, {})
            cond_type = config.get("type", "vector")

            if cond_type == "vector":
                value = value.reshape(-1)
                expected_len = int(config.get("length", value.shape[-1]))
                if value.size != expected_len:
                    value = value[:expected_len]
            else:  # scalar
                value = value.reshape(-1)
                if value.size > 1:
                    value = value[:1]

            cond_dict[cond_name] = th.tensor(value, dtype=th.float32)
        out["cond"] = cond_dict

        # RSN metadata (optional)
        if "rsn" in file:
            try:
                rsn_value = file["rsn"][original_index]
                # Handle bytes.
                if isinstance(rsn_value, bytes):
                    rsn_value = rsn_value.decode("utf-8")
                out["rsn"] = str(rsn_value)
            except Exception:
                pass  # Ignore missing/invalid RSN.

        return out

    def has_original_index(self, original_index: int) -> bool:
        return int(original_index) in self._index_lookup

    def get_sample_by_original_index(self, original_index: int):
        pos = self._index_lookup.get(int(original_index))
        if pos is None:
            raise KeyError(f"Original index {original_index} not found in dataset")
        return self[pos]
