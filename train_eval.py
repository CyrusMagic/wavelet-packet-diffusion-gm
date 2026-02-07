from typing import Dict, List
import random

import numpy as np
import torch
import pytorch_lightning as pl
from h5py import File, string_dtype
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from diffusion.edm import LightningEDM
from diffusion.representation import invert_representation
from diffusion.utils import get_last_checkpoint
from diffusion.data_utils import (
    load_norm_stats, denormalize_conditions,
    create_dataset, create_dataloader,
    create_train_eval_dataloaders
)

import os
import time
from datetime import datetime

# Enable TF32 matmul on supported GPUs (faster with minimal impact on results).
torch.set_float32_matmul_precision("high")



def train(config: Dict, unet_config: Dict, trainer_params: Dict):
    # Use the unified data-loading helpers (db6 wavelet packet coefficients).
    train_loader, test_loader = create_train_eval_dataloaders(config)
    max_steps = config["max_epochs"] * len(train_loader)

    optimizer_params = {"learning_rate": config["lr"], "max_steps": max_steps}

    print("Build lightning module...")
    model = LightningEDM(
        modelConfig=config,
        unet_config=unet_config,
        trainer_params=trainer_params,
        optimizer_params=optimizer_params,
    )
    print("Build Pytorch Lightning Trainer...")
    # learning rate logger
    callbacks = [LearningRateMonitor()]

    # save checkpoints to 'model_path' whenever 'val_loss' has a new min
    if (
        "enable_checkpointing" not in trainer_params
        or trainer_params["enable_checkpointing"]
    ):
        callbacks.append(
            ModelCheckpoint(
                dirpath=config["outputdir"],
                filename="{name}_{epoch}-val_loss={validation/loss:.2e}",
                monitor="validation/loss",
                auto_insert_metric_name=False,
                mode="min",
                save_top_k=3,
                save_last=True,
            )
        )

    # CSV logger (easy-to-parse training curves)
    csv_logger = CSVLogger(
        save_dir=config["outputdir"],
        name="training_logs",
        flush_logs_every_n_steps=50,
    )

    # Define Trainer
    trainer = pl.Trainer(
        **trainer_params,
        callbacks=callbacks,
        default_root_dir=config["outputdir"],
        logger=csv_logger,
    )

    print("Start training...")
    checkpoint = (
        get_last_checkpoint(trainer.default_root_dir) if config["resume"] else None
    )
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=test_loader,
        ckpt_path=checkpoint,
    )

    print("Done!")


def eval(config: Dict, eval_config: Dict):
    # Evaluation timing (single device).
    eval_start_time = time.time()
    timestamp_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp_start}] Start evaluation: {config['name']}")

    eval_dir = os.path.join(config["outputdir"], "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    split = eval_config["split"]
    name = config["name"]

    # Important: include wavelet_meta for correct inverse transform.
    dataset = create_dataset(
        datapath=config["datapath"],
        cut=config["cut_t"],
        cond_configs=config["cond_configs"],
        wavelet_level=config.get("wavelet_level", 7),
        wavelet_name=config.get("wavelet_name", "db6"),
        include_wavelet_meta=True,
    )

    eval_batch_size = eval_config["batch_size"]
    eval_num_workers = eval_config.get("num_workers", 0)
    print(
        f"[DEBUG] dataset_size={len(dataset)}, batch_size={eval_batch_size}, "
        f"num_workers={eval_num_workers}"
    )

    loader = create_dataloader(
        dataset,
        batch_size=eval_batch_size,
        num_workers=eval_num_workers,
        mode="eval",
        device=config.get("device", "cuda"),
    )

    data_load_time = time.time()
    data_duration = (data_load_time - eval_start_time) / 60.0
    timestamp_data = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp_data}] Dataset ready - {data_duration:.1f} min")

    print("Loading model...")
    edm = LightningEDM.load_from_checkpoint(eval_config["edm_checkpoint"])
    # allow overriding guidance scale at evaluation time (CFG sampling)
    if "guidance_scale" in eval_config:
        edm.guidance_scale = eval_config["guidance_scale"]

    model_load_time = time.time()
    model_duration = (model_load_time - data_load_time) / 60.0
    timestamp_model = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp_model}] Model loaded - {model_duration:.1f} min")

    # Run inference + export.
    eval_with_improved_inference(
        config,
        eval_config,
        edm,
        dataset,
        loader,
        eval_dir,
        name,
        split,
        eval_start_time,
        data_duration,
        model_duration,
    )


def eval_with_improved_inference(
    config,
    eval_config,
    edm,
    dataset,
    loader,
    eval_dir,
    name,
    split,
    eval_start_time,
    data_duration,
    model_duration,
):
    """Single-device inference (public release)."""
    norm_stats = load_norm_stats(config["datapath"])

    # Force single device.
    accelerator = "gpu" if eval_config["device"] == "cuda" else eval_config["device"]
    print(f"Single-device inference (accelerator={accelerator}, devices=1)")

    trainer_kwargs = {
        "accelerator": accelerator,
        "devices": 1,
        "logger": False,
        "enable_checkpointing": False,
        "enable_progress_bar": True,
        "enable_model_summary": False,
    }

    trainer = pl.Trainer(**trainer_kwargs)

    inference_start_time = time.time()
    timestamp_inference = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp_inference}] Start sampling...")

    print(f"[DEBUG] dataloader_batch_size={loader.batch_size}, num_workers={loader.num_workers}")
    print(
        f"[DEBUG] num_samples={len(dataset)}, expected_num_batches="
        f"{(len(dataset) + loader.batch_size - 1) // loader.batch_size}"
    )

    # Lightning prediction loop
    predictions = trainer.predict(edm, loader)

    # Merge outputs
    if predictions:
        all_pred_signals = torch.cat(
            [pred["predictions"] for pred in predictions], dim=0
        )
        all_target_signals = torch.cat(
            [pred["target_signal"] for pred in predictions], dim=0
        )
        all_target_waveforms = torch.cat(
            [pred["target_waveform"] for pred in predictions], dim=0
        )
        all_indices = torch.cat([pred["indices"] for pred in predictions], dim=0)

        # Collect condition tensors
        all_conditions = {}
        for cond_name in predictions[0]["conditions"].keys():
            all_conditions[cond_name] = torch.cat(
                [pred["conditions"][cond_name] for pred in predictions], dim=0
            )

        # Collect wavelet_meta for inverse transform.
        all_wavelet_meta = None
        if "wavelet_meta" in predictions[0]:
            all_wavelet_meta = {}
            for meta_key in predictions[0]["wavelet_meta"].keys():
                all_wavelet_meta[meta_key] = torch.cat(
                    [pred["wavelet_meta"][meta_key] for pred in predictions], dim=0
                )

        num_samples = all_pred_signals.shape[0]
        print(f"Processed {num_samples} samples")
    else:
        all_pred_signals = torch.empty(0, dtype=torch.float32)
        all_target_signals = torch.empty(0, dtype=torch.float32)
        all_target_waveforms = torch.empty(0, dtype=torch.float32)
        all_indices = torch.empty(0, dtype=torch.long)
        all_conditions = {}
        all_wavelet_meta = None
        num_samples = 0

    # Sort everything by original indices to keep a stable order.
    if all_pred_signals.numel() > 0 and all_indices.numel() > 0:
        sorted_indices = torch.argsort(all_indices)
        all_pred_signals = all_pred_signals[sorted_indices].cpu().numpy()
        all_target_signals = all_target_signals[sorted_indices].cpu().numpy()
        all_target_waveforms = all_target_waveforms[sorted_indices].cpu().numpy()
        all_indices_sorted = all_indices[sorted_indices].numpy()

        # Sort conditions
        sorted_conditions = {}
        for cond_name, cond_data in all_conditions.items():
            sorted_conditions[cond_name] = cond_data[sorted_indices].cpu().numpy()

        # Sort wavelet_meta
        sorted_wavelet_meta = None
        if all_wavelet_meta is not None:
            sorted_wavelet_meta = {}
            for meta_key, meta_data in all_wavelet_meta.items():
                sorted_wavelet_meta[meta_key] = meta_data[sorted_indices].cpu().numpy()

        print(f"[DEBUG] predictions_shape={all_pred_signals.shape}")
        print(f"[DEBUG] target_signal_shape={all_target_signals.shape}")
        print(f"[DEBUG] target_waveform_shape={all_target_waveforms.shape}")
        print(
            f"[DEBUG] index_range=[{all_indices_sorted.min()}, {all_indices_sorted.max()}]"
        )

        # Sanity check
        expected_samples = len(dataset)
        actual_samples = all_pred_signals.shape[0]
        print(f"[DEBUG] num_predictions={actual_samples}, expected={expected_samples}")

        if actual_samples != expected_samples:
            print(
                f"Warning: num_predictions({actual_samples}) != dataset_size({expected_samples})"
            )
    else:
        all_pred_signals = np.empty((0,), dtype=np.float32)
        all_target_signals = np.empty((0,), dtype=np.float32)
        all_target_waveforms = np.empty((0,), dtype=np.float32)
        sorted_conditions = {}
        sorted_wavelet_meta = None
        all_indices_sorted = np.empty((0,), dtype=np.int64)

    device_info = f"Lightning-Single({accelerator})"

    print(f"[DEBUG] target_signal_shape={all_target_signals.shape}")
    print(f"[DEBUG] target_waveform_shape={all_target_waveforms.shape}")

    # Invert db6 wavelet packet representation back to waveforms.
    all_pred_waveforms = invert_representation(
        all_pred_signals,
        level=config.get("wavelet_level", 7),
        wavelet=config.get("wavelet_name", "db6"),
        original_length=config.get("cut_t"),
        wavelet_meta=sorted_wavelet_meta,
    )
    all_pred_waveforms = all_pred_waveforms[..., : config.get("cut_t", all_pred_waveforms.shape[-1])]
    print(f"[DEBUG] predicted_waveform_shape={all_pred_waveforms.shape}")

    # Quick correspondence checks
    if len(all_pred_signals) > 0:
        print("\n[DEBUG] correspondence_check")
        print(f"  num_predictions: {len(all_pred_signals)}")
        print(f"  num_targets: {len(all_target_signals)}")
        print(f"  num_indices: {len(all_indices_sorted)}")

        # Check PGA of the first few samples
        pred_pga = np.max(np.abs(all_pred_waveforms[:3]), axis=(1, 2))
        target_pga = np.max(np.abs(all_target_waveforms[:3]), axis=(1, 2))
        print(f"  predicted_pga[:3]: {pred_pga}")
        print(f"  target_pga[:3]: {target_pga}")

        if "sa" in sorted_conditions:
            print(f"  sa_shape: {sorted_conditions['sa'].shape}")
            print(f"  sa_T0[:3]: {sorted_conditions['sa'][:3, 0]}")

    denorm_conditions = denormalize_conditions(
        sorted_conditions,
        all_indices_sorted,
        norm_stats,
        datapath=config.get("datapath"),
    )

    # If PGA is not explicitly provided as a condition (cond0/cond1),
    # recover target PGA from denormalized conditions and rescale waveforms.
    needs_pga_scaling = "pga" not in config.get("cond_configs", {})
    if needs_pga_scaling and all_pred_waveforms.size > 0:
        pga_target = denorm_conditions.get("pga")
        if pga_target is not None and pga_target.size > 0:
            pga_target = pga_target.reshape(-1)
            pga_gen = np.max(np.abs(all_pred_waveforms), axis=(1, 2))
            safe_pga_gen = np.where(pga_gen > 1e-8, pga_gen, 1.0)
            scale_factors = pga_target / safe_pga_gen
            all_pred_waveforms = all_pred_waveforms * scale_factors[:, None, None]
            denorm_conditions["pga"] = pga_target

            print("Applied PGA scaling (missing PGA in conditions):")
            print(
                f"  mean_generated_pga (before): {pga_gen.mean():.4f}, after: "
                f"{np.max(np.abs(all_pred_waveforms), axis=(1, 2)).mean():.4f}"
            )
            print(
                f"  scale_factor_range: [{scale_factors.min():.4f}, {scale_factors.max():.4f}]"
            )
        else:
            sa_denorm = denorm_conditions.get("sa")
            if sa_denorm is not None and sa_denorm.size > 0:
                pga_target = sa_denorm[:, 0]
                pga_gen = np.max(np.abs(all_pred_waveforms), axis=(1, 2))
                safe_pga_gen = np.where(pga_gen > 1e-8, pga_gen, 1.0)
                scale_factors = pga_target / safe_pga_gen
                all_pred_waveforms = all_pred_waveforms * scale_factors[:, None, None]
                denorm_conditions["pga"] = pga_target

                print("Applied PGA scaling (recovered from SA[0]):")
                print(
                    f"  mean_generated_pga (before): {pga_gen.mean():.4f}, after: "
                    f"{np.max(np.abs(all_pred_waveforms), axis=(1, 2)).mean():.4f}"
                )
                print(
                    f"  scale_factor_range: [{scale_factors.min():.4f}, {scale_factors.max():.4f}]"
                )
            else:
                print(
                    "Warning: PGA is missing and cannot be recovered from SA; "
                    "generated amplitudes may be mismatched."
                )

    # Save results
    mode_suffix = ""
    if config.get("random_inference_mode"):
        mode_suffix = "_random"

    eval_filename = f"{name}_{split}{mode_suffix}.h5"
    eval_path = os.path.join(eval_dir, eval_filename)

    with File(eval_path, "w") as f:
        for cond_name, cond_data in sorted_conditions.items():
            f.create_dataset(cond_name, data=cond_data)

        for cond_name, cond_data in denorm_conditions.items():
            f.create_dataset(f"{cond_name}_denorm", data=cond_data)

        f.create_dataset("target_waveform", data=all_target_waveforms)
        f.create_dataset("predicted_waveform", data=all_pred_waveforms)
        f.create_dataset("target_signal", data=all_target_signals)
        f.create_dataset("predicted_signal", data=all_pred_signals)
        f.create_dataset("sample_indices", data=all_indices_sorted)

    print(f"Saved evaluation results to: {eval_path}")

    _save_timing_info(
        eval_dir,
        config["name"],
        eval_start_time,
        inference_start_time,
        data_duration,
        model_duration,
        len(dataset),
        device_info,
        eval_config["batch_size"],
    )


def _save_timing_info(
    eval_dir,
    name,
    eval_start_time,
    inference_start_time,
    data_duration,
    model_duration,
    dataset_size,
    device_info,
    batch_size,
):
    """Write timing information to a text file."""
    eval_end_time = time.time()
    total_duration = (eval_end_time - eval_start_time) / 60.0
    inference_duration = (eval_end_time - inference_start_time) / 60.0

    timestamp_start = datetime.fromtimestamp(eval_start_time).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    timestamp_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"[{timestamp_end}] Evaluation finished")
    print(f"  - total: {total_duration:.1f} min")
    print(f"  - data:  {data_duration:.1f} min")
    print(f"  - model: {model_duration:.1f} min")
    print(f"  - infer: {inference_duration:.1f} min")

    timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    timing_file = os.path.join(
        eval_dir, f"{name}_inference_timing_{timestamp_file}.txt"
    )
    with open(timing_file, "w", encoding="utf-8") as f:
        f.write(f"Inference timing - {name}\n")
        f.write(f"start: {timestamp_start}\n")
        f.write(f"end:   {timestamp_end}\n")
        f.write(f"total_min: {total_duration:.1f}\n")
        f.write(f"data_min:  {data_duration:.1f}\n")
        f.write(f"model_min: {model_duration:.1f}\n")
        f.write(f"infer_min: {inference_duration:.1f}\n")
        f.write(f"num_samples: {dataset_size}\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"device: {device_info}\n")

    print("Done!")


def eval_random_diversity(config: Dict, eval_config: Dict):
    """Random conditional sampling: pick random conditions and generate multiple samples per condition."""
    import numpy as np
    import random
    from datetime import datetime
    import time

    eval_start_time = time.time()
    timestamp_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp_start}] Start random-conditional sampling: {config['name']}")

    eval_dir = os.path.join(config["outputdir"], "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    num_conditions = config.get("num_conditions", 5)
    samples_per_condition = config.get("samples_per_condition", 100)
    name = config["name"]

    print("Loading dataset...")
    dataset = create_dataset(
        datapath=config["datapath"],
        cut=config["cut_t"],
        cond_configs=config.get("cond_configs", {}),
        wavelet_level=config.get("wavelet_level", 7),
        wavelet_name=config.get("wavelet_name", "db6"),
    )
    norm_stats = load_norm_stats(config["datapath"])

    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty; cannot run random sampling.")

    sample_shape = tuple(dataset[0]["signal"].shape)

    # Use a fixed seed so cond0/cond1 pick identical conditions if needed.
    random_seed = eval_config.get("random_seed", 42)
    random.seed(random_seed)
    np.random.seed(random_seed)
    print(
        f"Sampling {num_conditions} conditions from {len(dataset)} records (seed={random_seed})..."
    )
    selected_indices = random.sample(range(len(dataset)), num_conditions)

    # Gather selected conditions
    selected_conditions = {}
    selected_waveforms = []
    selected_original_indices = []
    selected_rsn = []
    selected_wavelet_meta_lists = None

    for i, idx in enumerate(selected_indices):
        sample = dataset[idx]
        selected_waveforms.append(sample["wfs"].numpy())
        selected_original_indices.append(int(sample["original_index"]))

        # RSN (optional)
        if "rsn" in sample:
            selected_rsn.append(str(sample["rsn"]))
        else:
            selected_rsn.append(None)

        for cond_name, cond_value in sample["cond"].items():
            if cond_name not in selected_conditions:
                selected_conditions[cond_name] = []
            selected_conditions[cond_name].append(cond_value.numpy())

        if "wavelet_meta" in sample:
            if selected_wavelet_meta_lists is None:
                selected_wavelet_meta_lists = {
                    k: [] for k in sample["wavelet_meta"].keys()
                }
            for meta_key, meta_value in sample["wavelet_meta"].items():
                selected_wavelet_meta_lists[meta_key].append(
                    meta_value.detach().cpu().numpy()
                )

    # Convert to numpy
    for cond_name in selected_conditions:
        selected_conditions[cond_name] = np.array(selected_conditions[cond_name])
    selected_waveforms = np.array(selected_waveforms)
    selected_original_indices = np.array(selected_original_indices, dtype=np.int64)

    selected_wavelet_meta = None
    if selected_wavelet_meta_lists is not None:
        selected_wavelet_meta = {}
        for meta_key, value_list in selected_wavelet_meta_lists.items():
            try:
                selected_wavelet_meta[meta_key] = np.stack(value_list, axis=0)
            except ValueError:
                selected_wavelet_meta[meta_key] = np.array(value_list)

    selected_conditions_denorm = denormalize_conditions(
        selected_conditions,
        selected_original_indices,
        norm_stats,
        datapath=config.get("datapath"),
    )

    print("Selected condition shapes:")
    for cond_name, cond_data in selected_conditions.items():
        print(f"  {cond_name}: {cond_data.shape}")
    print(f"  original_waveforms: {selected_waveforms.shape}")
    if selected_wavelet_meta is not None:
        print("  wavelet_meta keys:", ", ".join(selected_wavelet_meta.keys()))

    print("Loading model...")
    edm = LightningEDM.load_from_checkpoint(eval_config["edm_checkpoint"])
    if "guidance_scale" in eval_config:
        edm.guidance_scale = eval_config["guidance_scale"]

    device = eval_config["device"]
    edm.to(device)
    edm.eval()

    print(f"Generating: {num_conditions} conditions x {samples_per_condition} samples...")

    all_generated_waveforms = []
    all_generated_signals = []
    all_condition_indices = []
    generated_wavelet_meta_lists = (
        {k: [] for k in selected_wavelet_meta.keys()} if selected_wavelet_meta is not None else None
    )

    for cond_idx in range(num_conditions):
        print(f"Condition {cond_idx + 1}/{num_conditions}...")

        # Prepare batch conditions
        current_conditions = {}
        for cond_name, cond_data in selected_conditions.items():
            current_conditions[cond_name] = torch.tensor(
                np.repeat(
                    cond_data[cond_idx : cond_idx + 1], samples_per_condition, axis=0
                ),
                dtype=torch.float32,
                device=device,
            )

        generation_shape = (samples_per_condition,) + sample_shape

        # Generate samples
        with torch.no_grad():
            edm.deterministic_sampling = False
            cond_payload = current_conditions if current_conditions else None
            generated_signals = edm.sample(
                generation_shape,
                cond=cond_payload,
            )

            # Convert wavelet coefficients back to waveforms.
            generated_np = generated_signals.cpu().numpy()
            meta_for_condition = None
            if selected_wavelet_meta is not None:
                meta_for_condition = {}
                for meta_key, meta_data in selected_wavelet_meta.items():
                    cond_meta = meta_data[cond_idx : cond_idx + 1]
                    repeated = np.repeat(cond_meta, samples_per_condition, axis=0)
                    meta_for_condition[meta_key] = repeated.astype(
                        np.float32, copy=False
                    )
                if generated_wavelet_meta_lists is not None:
                    for meta_key, meta_values in meta_for_condition.items():
                        generated_wavelet_meta_lists[meta_key].append(meta_values)
            generated_waveforms = invert_representation(
                generated_np,
                level=config.get("wavelet_level", 7),
                wavelet=config.get("wavelet_name", "db6"),
                original_length=config.get("cut_t"),
                wavelet_meta=meta_for_condition,
            )

        all_generated_signals.append(generated_np)
        all_generated_waveforms.append(generated_waveforms)
        all_condition_indices.extend([cond_idx] * samples_per_condition)

        print(f"  Done: {samples_per_condition} samples")

    # Merge results
    all_generated_signals = np.concatenate(all_generated_signals, axis=0)
    all_generated_waveforms = np.concatenate(all_generated_waveforms, axis=0)
    all_condition_indices = np.array(all_condition_indices)
    generated_wavelet_meta = None
    if generated_wavelet_meta_lists is not None:
        generated_wavelet_meta = {}
        for meta_key, parts in generated_wavelet_meta_lists.items():
            generated_wavelet_meta[meta_key] = np.concatenate(parts, axis=0)

    print("Generated output summary:")
    print(f"  generated_signals: {all_generated_signals.shape}")
    print(f"  generated_waveforms: {all_generated_waveforms.shape}")
    print(f"  condition_indices: {all_condition_indices.shape}")

    # Optionally rescale to match target PGA if PGA is not explicitly conditioned.
    needs_pga_scaling = "pga" not in config.get("cond_configs", {})
    if needs_pga_scaling and all_generated_waveforms.size > 0:
        pga_denorm = selected_conditions_denorm.get("pga")
        sa_denorm = selected_conditions_denorm.get("sa")
        if pga_denorm is not None and pga_denorm.size >= num_conditions:
            pga_values = pga_denorm.reshape(-1)
        elif sa_denorm is not None and sa_denorm.size > 0:
            pga_values = sa_denorm[:, 0]
            selected_conditions_denorm["pga"] = pga_values
        else:
            pga_values = None

        if pga_values is not None:
            for cond_idx in range(num_conditions):
                mask = all_condition_indices == cond_idx
                if not np.any(mask):
                    continue

                pga_target = pga_values[cond_idx]
                waveforms_for_cond = all_generated_waveforms[mask]
                pga_gen = np.max(np.abs(waveforms_for_cond), axis=(1, 2))
                safe_pga_gen = np.where(pga_gen > 1e-8, pga_gen, 1.0)
                scale_factors = pga_target / safe_pga_gen
                all_generated_waveforms[mask] = waveforms_for_cond * scale_factors[:, None, None]

            print("Applied PGA scaling:")
            print(
                f"  mean_PGA_after: "
                f"{np.max(np.abs(all_generated_waveforms), axis=(1, 2)).mean():.4f}"
            )
        else:
            print(
                "Warning: PGA is missing and cannot be recovered from SA; "
                "generated amplitudes may be mismatched."
            )

    # Save results
    filename = (
        f"random_diversity_{num_conditions}cond_{samples_per_condition}samples.h5"
    )
    filepath = os.path.join(eval_dir, filename)

    print(f"Saving to: {filepath}")
    with File(filepath, "w") as f:
        cond_group = f.create_group("selected_conditions")
        for cond_name, cond_data in selected_conditions.items():
            cond_group.create_dataset(cond_name, data=cond_data)

        if selected_wavelet_meta is not None:
            meta_group = f.create_group("selected_wavelet_meta")
            for meta_key, meta_data in selected_wavelet_meta.items():
                meta_group.create_dataset(meta_key, data=meta_data, compression="gzip")

        if selected_conditions_denorm:
            denorm_group = f.create_group("selected_conditions_denorm")
            for cond_name, cond_data in selected_conditions_denorm.items():
                denorm_group.create_dataset(cond_name, data=cond_data)

        f.create_dataset("original_waveforms", data=selected_waveforms)
        f.create_dataset("selected_original_indices", data=selected_original_indices)

        # RSN (optional)
        if selected_rsn and any(rsn is not None for rsn in selected_rsn):
            import h5py
            rsn_array = np.array(
                [str(r) if r is not None else "" for r in selected_rsn],
                dtype=h5py.string_dtype(),
            )
            f.create_dataset("rsn", data=rsn_array)
            print(f"  saved_rsn_values: {len([r for r in selected_rsn if r])}")

        f.create_dataset("generated_signals", data=all_generated_signals)
        f.create_dataset("generated_waveforms", data=all_generated_waveforms)
        f.create_dataset("condition_indices", data=all_condition_indices)
        if generated_wavelet_meta is not None:
            gen_meta_group = f.create_group("generated_wavelet_meta")
            for meta_key, meta_data in generated_wavelet_meta.items():
                gen_meta_group.create_dataset(
                    meta_key, data=meta_data, compression="gzip"
                )

        f.attrs["num_conditions"] = num_conditions
        f.attrs["samples_per_condition"] = samples_per_condition
        f.attrs["total_samples"] = len(all_generated_waveforms)
        f.attrs["model_name"] = config["name"]
        f.attrs["timestamp"] = timestamp_start
        f.attrs["selected_indices"] = selected_indices

    eval_end_time = time.time()
    total_duration = (eval_end_time - eval_start_time) / 60.0
    timestamp_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"[{timestamp_end}] Random-conditional sampling finished")
    print(f"  - total: {total_duration:.1f} min")
    print(f"  - num_conditions: {num_conditions}")
    print(f"  - samples_per_condition: {samples_per_condition}")
    print(f"  - total_generated: {len(all_generated_waveforms)}")

    print("Done!")
