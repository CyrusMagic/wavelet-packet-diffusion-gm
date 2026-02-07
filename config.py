"""
Project configuration.
"""

import os


class DatasetConfig:
    """Dataset-related configuration."""

    # Dataset name (HDF5 file stem under DATASETS_DIR).
    DATASET_NAME = "step2_NGAH1_len16k_symmetric_freq"

    @staticmethod
    def get_dataset_name(*_args, **_kwargs) -> str:
        """Return the default dataset name."""
        return DatasetConfig.DATASET_NAME

    # Wavelet packet settings
    WAVELET_NAME = "db6"
    WAVELET_LEVEL = 7

    # Sampling
    DEFAULT_DT = 0.01  # seconds (100 Hz)

    # Conditioning variables
    # Public release: single damping ratio response spectrum (5%).
    DAMPING_RATIO = 0.05
    NUM_PERIODS = 130
    COND_CONFIGS = {
        # cond0: SA only (PGA is recovered from SA[0] when needed)
        0: {
            "sa": {"length": 130, "type": "vector"},
        },
        # cond1: SA + IM features (PGA is recovered from SA[0] when needed)
        1: {
            "sa": {"length": 130, "type": "vector"},
            "arias": {"length": 1, "type": "scalar"},  # Arias intensity (CDF-normalized)
            "t5_norm": {"length": 1, "type": "scalar"},  # T5 time (CDF-normalized)
            "d595_norm": {"length": 1, "type": "scalar"},  # D5-95 duration (CDF-normalized)
            "tc_norm": {
                "length": 1,
                "type": "scalar",
            },  # time centroid (CDF-normalized)
            "husid": {"length": 256, "type": "vector"},
        },
    }


class TrainingConfig:
    """Training-related configuration."""

    # Core hyperparameters
    MAX_EPOCHS = 300
    LEARNING_RATE = 1e-4
    EMA_DECAY = 0.999

    # Batch sizes (single default profile)
    BATCH_SIZES = {
        "default": {
            "train": 110,
            "eval": 512,
        },
    }

    # DataLoader
    MAX_TRAIN_WORKERS = 4
    EVAL_WORKERS = 0

    # PyTorch Lightning Trainer
    PRECISION = "bf16-mixed"
    LOG_EVERY_N_STEPS = 10
    NUM_SANITY_VAL_STEPS = 0
    CHECK_VAL_EVERY_N_EPOCH = 5


class ModelConfig:
    """Model architecture configuration."""

    # UNet
    MODEL_CHANNELS = 64
    CHANNEL_MULT = (1, 2, 4, 4)
    ATTENTION_RESOLUTIONS = (8,)
    NUM_RES_BLOCKS = 2
    NUM_HEADS = 4
    DROPOUT = 0.1
    FLASH_ATTENTION = False


class SamplingConfig:
    """Sampling / inference configuration."""

    # CFG (Classifier-Free Guidance)
    CFG_DROPOUT_PROB = 0.2
    GUIDANCE_SCALE = 2.0

    # Diffusion sampling
    NUM_SAMPLING_STEPS = 25
    DETERMINISTIC_SAMPLING = True

    # Random conditional sampling mode
    NUM_CONDITIONS = 30
    SAMPLES_PER_CONDITION = 100


class PathConfig:
    """Path configuration."""

    # Base paths
    DATASETS_DIR = "./datasets"
    OUTPUTS_DIR = "./outputs/results"

    @staticmethod
    def get_dataset_path(dataset_name: str) -> str:
        """Return HDF5 path for a dataset name."""
        return f"{PathConfig.DATASETS_DIR}/{dataset_name}.h5"

    @staticmethod
    def get_output_dir(model_name: str) -> str:
        """Return output directory for a model name."""
        return f"{PathConfig.OUTPUTS_DIR}/{model_name}"

    @staticmethod
    def get_edm_checkpoint(model_name: str) -> str:
        """Return checkpoint path for EDM."""
        return f"{PathConfig.OUTPUTS_DIR}/{model_name}/last.ckpt"


class Config:
    """Convenience wrapper for accessing configuration groups."""

    Dataset = DatasetConfig
    Training = TrainingConfig
    Model = ModelConfig
    Sampling = SamplingConfig
    Path = PathConfig

    # Common shortcuts
    WAVELET_NAME = DatasetConfig.WAVELET_NAME
    WAVELET_LEVEL = DatasetConfig.WAVELET_LEVEL
    COND_CONFIGS = DatasetConfig.COND_CONFIGS
    MAX_EPOCHS = TrainingConfig.MAX_EPOCHS
    LEARNING_RATE = TrainingConfig.LEARNING_RATE
