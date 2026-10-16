#!/usr/bin/env python3
"""Evaluate full RQ-VQA and three zero-masked branches without retraining.

Reuse configs/multi_dataset_test.yaml. Results default to its output_root/ablation.
Positive delta_<correlation> means degradation relative to the full model;
positive increase_<RMSE> also means degradation. This measures checkpoint
dependence, not performance after retraining or inference speed after pruning.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import test_multi_dataset as base


VARIANTS = {
    "full": None,
    "without_slowfast": (1024, 1280),
    "without_fastvqa": (1280, 2048),
    "without_swin": (0, 1024),
}
CORRELATIONS = ("SRCC", "KRCC", "PLCC_raw", "PLCC_fitted")
ERRORS = ("RMSE_raw", "RMSE_fitted")


@contextmanager
def mask_branch(model, variant: str):
    """Mask the fused input after BoT, preserving biases and optional LIQE.

    Hooks registered on the underlying quality head also run on DataParallel
    replicas. Clone the tensor so the original features are never modified.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown ablation variant: {variant}")
    expected_dim = 2048 + (495 if model.use_liqe else 0)
    if model.quality[0].in_features != expected_dim:
        raise ValueError("Unexpected RQ-VQA fusion dimensions; check branch layout")
    interval = VARIANTS[variant]
    if interval is None:
        yield
        return

    def mask_input(module, inputs):
        features, = inputs
        if features.ndim != 2 or features.shape[1] != expected_dim:
            raise ValueError(f"Expected fused features [B*T, {expected_dim}]")
        masked = features.clone()
        masked[:, interval[0]:interval[1]] = 0
        return (masked,)

    handle = model.quality.register_forward_pre_hook(mask_input)
    try:
        yield
    finally:
        handle.remove()


def metric_changes(full, current):
    changes = {}
    for key in CORRELATIONS + ERRORS:
        reference, value = full.get(key), current.get(key)
        prefix = "delta_" if key in CORRELATIONS else "increase_"
        changes[prefix + key] = (
            None if reference is None or value is None
            else reference - value if key in CORRELATIONS else value - reference
        )
    return changes


def write_reports(output_dir: Path, results):
    import pandas as pd

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {key: value for key, value in item.items()
         if key not in {"checkpoints", "fit_error", "note"}}
        for item in results
    ]).to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def evaluate_ablation(name, args, rows, output_dir, model, inference_model,
                      checkpoints, device, transform, config):
    import pandas as pd

    results = []
    combined = None
    for variant in VARIANTS:
        variant_dir = output_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{name}] Ablation: {variant}", flush=True)
        with mask_branch(model, variant):
            metrics = base.evaluate_dataset(
                name, args, rows, variant_dir, model, inference_model,
                checkpoints, device, transform, config,
            )
        metrics.update(variant=variant, use_liqe=model.use_liqe)
        metrics.update(metric_changes(results[0] if results else metrics, metrics))
        results.append(metrics)
        table = pd.read_csv(variant_dir / "predictions.csv", dtype={"filename": str})
        if combined is None:
            combined = table[["filename", "mos"]].copy()
        elif not table[["filename", "mos"]].equals(combined[["filename", "mos"]]):
            raise ValueError(f"[{name}] Video order or labels changed between variants")
        combined["prediction_" + variant] = table["prediction"]
        print(f"[{name}/{variant}] SRCC={metrics['SRCC']} "
              f"delta_SRCC={metrics['delta_SRCC']}", flush=True)
    combined.to_csv(output_dir / "predictions.csv", index=False)
    write_reports(output_dir, results)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Existing multi-dataset YAML configuration")
    parser.add_argument("--dataset", action="append", help="Only run the named data entry; repeatable")
    parser.add_argument("--validate-only", action="store_true", help="Check labels and extracted inputs only")
    parser.add_argument("--output-root", help="Override ablation output directory (relative to repository root)")
    cli = parser.parse_args(argv)
    config = base.load_config(cli.config)
    if config.get("visible_devices") is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config["visible_devices"])
    selected = set(cli.dataset or [])
    unknown = selected - set(config["data"])
    if unknown:
        parser.error("unknown dataset(s): " + ", ".join(sorted(unknown)))
    output_root = Path(cli.output_root).expanduser() if cli.output_root else Path(config["output_root"]) / "ablation"
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    use_liqe = config["model"]["args"]["use_liqe"]
    prepared = {}
    for name, dataset in config["data"].items():
        if selected and name not in selected:
            continue
        args = dataset["args"]
        rows = base.parse_annotation(
            args["anno_file"], filename_mode=args["filename_mode"],
            data_prefix=args.get("data_prefix"),
        )
        base.validate_features(rows, args, use_liqe)
        prepared[name] = (args, rows, output_root / name)
        print(f"[{name}] checked {len(rows)} videos", flush=True)
    if cli.validate_only:
        print("All selected datasets passed annotation and feature validation.", flush=True)
        return 0

    import numpy as np
    import torch
    from torchvision import transforms
    import models

    checkpoints = base._checkpoint_files(config["test_load_path"], use_liqe, config.get("n_exp"))
    seed = config["random_seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for RQ-VQA inference")
    device = torch.device("cuda:0")
    model = models.RQ_VQA(use_liqe=use_liqe, pretrained=False).to(device)
    inference_model = torch.nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
    transform = transforms.Compose([
        transforms.Resize(384), transforms.CenterCrop(384), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    if use_liqe:
        print("LIQE remains enabled in all four variants.", flush=True)
    print("Zero masking measures checkpoint dependence, not pruned-model speed.", flush=True)
    summary = []
    for name, (args, rows, dataset_output) in prepared.items():
        summary.extend(evaluate_ablation(
            name, args, rows, dataset_output, model, inference_model,
            checkpoints, device, transform, config,
        ))
    write_reports(output_root, summary)
    print(f"Completed {len(prepared)} dataset(s), four variants each. "
          f"Summary: {output_root / 'summary.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
