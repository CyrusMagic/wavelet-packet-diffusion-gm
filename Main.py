import multiprocessing as mp
from pathlib import Path
import h5py
import os
import torch

from config import Config
from train_eval import train, eval


# Reduce per-worker file descriptor usage to avoid "Too many open files"
try:
    import torch.multiprocessing as torch_mp

    torch_mp.set_sharing_strategy("file_system")
except Exception as _e:
    print(f"Warning: failed to set torch multiprocessing sharing strategy: {_e}")

# ---------------------------
# Global configuration
# ---------------------------
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
pl_accelerator = "gpu" if device == "cuda" else device

# User-configurable switches
STATE = "train"  # "train" or "test"
RANDOM_INFERENCE_MODE = False  # random conditional sampling
COND_CONFIG_ID = 1  # 0: SA, 1: SA+IM features
RANDOM_SEED = 42

# Auto-generated model name
cond_tag = f"cond{COND_CONFIG_ID}"
model_name = f"NGAH1-2d_{cond_tag}_CFG0d2"


def gen_config():
    """Build runtime configuration dictionaries."""
    # 1) Read dataset metadata
    dataset_name = Config.Dataset.get_dataset_name()
    dataset_path = Path(Config.Path.get_dataset_path(dataset_name))
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with h5py.File(dataset_path, "r") as f:
        wfs_shape = f["wfs"].shape
        dataset_channels = wfs_shape[1] if len(wfs_shape) >= 2 else 1
        dataset_length = wfs_shape[-1]
        # Sampling interval
        dataset_dt = Config.Dataset.DEFAULT_DT
        meta_group = f.get("_meta")
        if meta_group is not None and "target_dt" in meta_group.attrs:
            try:
                dataset_dt = float(meta_group.attrs["target_dt"])
            except Exception:
                pass

    # 2) Batch sizes
    batch_cfg = Config.Training.BATCH_SIZES["default"]

    # 3) DataLoader workers
    cpu_count = max(1, mp.cpu_count())
    active_gpus = max(1, available_gpus) if device == "cuda" else 1
    train_num_workers = min(Config.Training.MAX_TRAIN_WORKERS, max(0, cpu_count - 1))
    eval_num_workers = Config.Training.EVAL_WORKERS

    # 4) Condition configuration
    selected_cond_configs = Config.Dataset.COND_CONFIGS.get(
        COND_CONFIG_ID, Config.Dataset.COND_CONFIGS[0]
    )

    modelConfig = {
        # Runtime switches
        "state": STATE,
        "random_inference_mode": RANDOM_INFERENCE_MODE,
        # Base settings
        "name": model_name,
        "datapath": str(dataset_path),
        "outputdir": Config.Path.get_output_dir(model_name),
        "device": device,
        # Dataset settings
        "channels": dataset_channels,
        "cut_t": dataset_length,
        "dt": dataset_dt,
        "cond_configs": selected_cond_configs,
        "wavelet_level": Config.WAVELET_LEVEL,
        "wavelet_name": Config.WAVELET_NAME,
        "input_representation": "wavelet",
        # Training
        "max_epochs": Config.MAX_EPOCHS,
        "batch_size": batch_cfg["train"],
        "lr": Config.LEARNING_RATE,
        "ema_decay": Config.Training.EMA_DECAY,
        "resume": False,
        "train_num_workers": train_num_workers,
        "eval_num_workers": eval_num_workers,
        # Sampling
        "cfg_dropout_prob": Config.Sampling.CFG_DROPOUT_PROB,
        "guidance_scale": Config.Sampling.GUIDANCE_SCALE,
        "num_sampling_steps": Config.Sampling.NUM_SAMPLING_STEPS,
        "deterministic_sampling": Config.Sampling.DETERMINISTIC_SAMPLING,
        "num_conditions": Config.Sampling.NUM_CONDITIONS,
        "samples_per_condition": Config.Sampling.SAMPLES_PER_CONDITION,
    }

    # 5) UNet configuration (2D wavelet map + cross-attention conditioning)
    in_out_channels = modelConfig["channels"]
    unet_config = {
        "in_channels": in_out_channels,
        "out_channels": in_out_channels,
        "cond_configs": selected_cond_configs,
        "dims": 2,
        "conv_kernel_size": 3,
        "model_channels": Config.Model.MODEL_CHANNELS,
        "channel_mult": Config.Model.CHANNEL_MULT,
        "attention_resolutions": Config.Model.ATTENTION_RESOLUTIONS,
        "num_res_blocks": Config.Model.NUM_RES_BLOCKS,
        "num_heads": Config.Model.NUM_HEADS,
        "dropout": Config.Model.DROPOUT,
        "flash_attention": Config.Model.FLASH_ATTENTION,
    }

    # 6) Lightning Trainer
    device_count = active_gpus if device == "cuda" else 1
    trainer_params = {
        "precision": Config.Training.PRECISION,
        "accelerator": pl_accelerator,
        "devices": device_count,
        "log_every_n_steps": Config.Training.LOG_EVERY_N_STEPS,
        "max_epochs": modelConfig["max_epochs"],
        "num_sanity_val_steps": Config.Training.NUM_SANITY_VAL_STEPS,
        "check_val_every_n_epoch": Config.Training.CHECK_VAL_EVERY_N_EPOCH,
    }

    # DDP strategy (multi-GPU)
    try:
        world_size_env = os.environ.get("WORLD_SIZE") or os.environ.get("SLURM_NTASKS")
        world_size = int(world_size_env) if world_size_env else 1
    except Exception:
        world_size = 1

    devices_count = device_count if isinstance(device_count, int) else len(device_count)
    use_ddp = (world_size > 1 or devices_count > 1) and device in {"cuda", "gpu"}
    if use_ddp:
        trainer_params["strategy"] = "ddp_find_unused_parameters_true"

    # 7) Evaluation
    eval_config = {
        "split": "test",
        "device": device,
        "batch_size": batch_cfg["eval"],
        "edm_checkpoint": Config.Path.get_edm_checkpoint(model_name),
        "guidance_scale": modelConfig["guidance_scale"],
        "num_workers": eval_num_workers,
        "random_seed": RANDOM_SEED,
    }

    return modelConfig, unet_config, trainer_params, eval_config


def run_training_mode(modelConfig, unet_config, trainer_params):
    """Train the EDM model."""
    train(modelConfig, unet_config, trainer_params)


def run_inference_mode(modelConfig, eval_config):
    """Run inference/evaluation."""
    if modelConfig.get("random_inference_mode", False):
        from train_eval import eval_random_diversity

        eval_random_diversity(modelConfig, eval_config)
    else:
        eval(modelConfig, eval_config)


def main():
    """Entry point."""
    modelConfig, unet_config, trainer_params, eval_config = gen_config()
    print(f"Run: {modelConfig['name']}")

    if STATE == "train":
        run_training_mode(modelConfig, unet_config, trainer_params)

    else:
        run_inference_mode(modelConfig, eval_config)


if __name__ == "__main__":
    main()
