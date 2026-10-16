#!/usr/bin/env python3
"""Evaluate one RQ-VQA checkpoint on several precomputed-feature datasets."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import re
import shlex
import sys
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]


def _split_annotation_line(line: str) -> list[str]:
    """Split comma, semicolon, tab, or ordinary whitespace annotations."""
    if "," in line:
        return [part.strip() for part in next(csv.reader([line]))]
    if ";" in line:
        return [part.strip() for part in next(csv.reader([line], delimiter=";"))]
    return shlex.split(line, comments=False, posix=True)


def _looks_like_header(fields: list[str]) -> bool:
    if not fields:
        return False
    first = fields[0].strip().lower()
    last = fields[-1].strip().lower()
    return first in {"filename", "video", "video_name", "path"} and last in {
        "score", "mos", "label", "quality"
    }


def _strip_data_prefix(name: str, data_prefix: str | None) -> str:
    normalized = name.replace("\\", "/")
    if data_prefix:
        prefix = str(data_prefix).replace("\\", "/").rstrip("/")
        if normalized == prefix:
            raise ValueError("annotation path points to data_prefix, not a video")
        if normalized.startswith(prefix + "/"):
            normalized = normalized[len(prefix) + 1 :]
    return normalized


def normalize_filename(name: str, mode: str = "basename", data_prefix: str | None = None) -> str:
    """Normalize a label filename to the path layout used by extracted features."""
    name = _strip_data_prefix(name.strip(), data_prefix)
    path = PurePosixPath(name)
    if mode == "basename":
        result = path.name
    elif mode == "relative":
        if path.is_absolute():
            raise ValueError(
                f"absolute annotation path {name!r} is outside data_prefix; "
                "set args.data_prefix or use filename_mode: basename"
            )
        if ".." in path.parts:
            raise ValueError(f"unsafe relative video path: {name}")
        result = path.as_posix()
    else:
        raise ValueError("filename_mode must be 'basename' or 'relative'")
    if not result or result in {".", "/"}:
        raise ValueError(f"invalid video filename: {name!r}")
    return result


def parse_annotation(
    path: str | Path,
    *,
    filename_mode: str = "basename",
    data_prefix: str | None = None,
) -> list[tuple[str, float]]:
    """Read either ``video MOS`` or ``video -1 -1 MOS`` label files."""
    path = Path(path)
    rows: list[tuple[str, float]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                fields = _split_annotation_line(line)
            except (csv.Error, ValueError) as error:
                raise ValueError(f"{path}:{line_number}: cannot parse annotation") from error
            if not rows and _looks_like_header(fields):
                continue
            if len(fields) not in (2, 4):
                raise ValueError(
                    f"{path}:{line_number}: expected 'video MOS' or "
                    f"'video -1 -1 MOS', got {len(fields)} fields"
                )
            try:
                score = float(fields[-1])
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: MOS must be numeric") from error
            if not math.isfinite(score):
                raise ValueError(f"{path}:{line_number}: MOS must be finite")
            try:
                filename = normalize_filename(fields[0], filename_mode, data_prefix)
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            key = filename.casefold()
            if key in seen:
                raise ValueError(f"{path}:{line_number}: duplicate normalized filename {filename!r}")
            seen.add(key)
            rows.append((filename, score))
    if len(rows) < 2:
        raise ValueError(f"{path}: at least two labeled videos are required")
    return rows


def _expand_string(value: str, variables: dict[str, str]) -> str:
    result = os.path.expanduser(os.path.expandvars(value))
    pattern = re.compile(r"\$\{([^}]+)\}")
    for _ in range(10):
        changed = False

        def replace(match: re.Match[str]) -> str:
            nonlocal changed
            key = match.group(1)
            if key in variables:
                changed = True
                return str(variables[key])
            return match.group(0)

        updated = pattern.sub(replace, result)
        result = updated
        if not changed:
            break
    unresolved = pattern.findall(result)
    if unresolved:
        raise ValueError(f"unresolved variable(s) in {value!r}: {', '.join(unresolved)}")
    return result


def _absolute_like(value: str) -> bool:
    return value.startswith("/") or bool(re.match(r"^[A-Za-z]:[\\/]", value))


def _resolve_path(value: str | Path | None, variables: dict[str, str]) -> str | None:
    if value is None:
        return None
    expanded = _expand_string(str(value), variables)
    if _absolute_like(expanded):
        return expanded
    return str((ROOT / expanded).resolve())


def load_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is required: pip install pyyaml") from error
    config_path = Path(path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    allowed = {
        "name", "visible_devices", "num_workers", "batch_size", "n_exp",
        "random_seed", "output_root", "test_load_path", "model", "data",
    }
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise ValueError(f"unknown top-level config key(s): {', '.join(unknown)}")
    if not isinstance(config.get("data"), dict) or not config["data"]:
        raise ValueError("config.data must contain at least one dataset")
    variables = {
        "project_root": str(ROOT),
        "config_dir": str(config_path.parent),
    }
    config["test_load_path"] = _resolve_path(config.get("test_load_path"), variables)
    config["output_root"] = _resolve_path(
        config.get("output_root", "results/multi_dataset"), variables
    )
    if not config["test_load_path"]:
        raise ValueError("test_load_path is required")
    model = config.setdefault("model", {})
    if not isinstance(model, dict):
        raise ValueError("model must be a mapping")
    if model.get("type", "RQ_VQA") != "RQ_VQA":
        raise ValueError("only model.type: RQ_VQA is supported")
    model_args = model.setdefault("args", {})
    if not isinstance(model_args, dict):
        raise ValueError("model.args must be a mapping")
    if set(model_args) - {"use_liqe"}:
        raise ValueError("model.args only supports use_liqe")
    if not isinstance(model_args.get("use_liqe", False), bool):
        raise ValueError("model.args.use_liqe must be true or false")
    model_args["use_liqe"] = model_args.get("use_liqe", False)

    for key, minimum in (("num_workers", 0), ("batch_size", 1), ("random_seed", 0)):
        value = config.get(key, {"num_workers": 4, "batch_size": 1, "random_seed": 8}[key])
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
        config[key] = value
    if config.get("n_exp") is not None and (
        isinstance(config["n_exp"], bool) or not isinstance(config["n_exp"], int) or config["n_exp"] < 1
    ):
        raise ValueError("n_exp must be null or a positive integer")

    resolved_data: dict[str, Any] = {}
    for dataset_name, dataset in config["data"].items():
        if not isinstance(dataset, dict):
            raise ValueError(f"data.{dataset_name} must be a mapping")
        if not isinstance(dataset.get("enabled", True), bool):
            raise ValueError(f"data.{dataset_name}.enabled must be true or false")
        if not dataset.get("enabled", True):
            continue
        if dataset.get("type", "RQVQAPrecomputedDataset") != "RQVQAPrecomputedDataset":
            raise ValueError(f"data.{dataset_name}: unsupported dataset type")
        args = dataset.get("args")
        if not isinstance(args, dict):
            raise ValueError(f"data.{dataset_name}.args must be a mapping")
        feature_root = args.get("feature_root")
        dataset_vars = dict(variables)
        if feature_root is not None:
            dataset_vars["feature_root"] = _resolve_path(feature_root, variables) or ""
        item = {"type": "RQVQAPrecomputedDataset", "args": dict(args)}
        item_args = item["args"]
        for key in ("anno_file", "feature_root", "frames_dir", "slowfast_dir", "fastvqa_dir", "liqe_dir"):
            if key in item_args:
                item_args[key] = _resolve_path(item_args[key], dataset_vars)
        if item_args.get("data_prefix") is not None:
            item_args["data_prefix"] = _expand_string(str(item_args["data_prefix"]), dataset_vars)
        for required in ("anno_file", "frames_dir", "slowfast_dir", "fastvqa_dir"):
            if not item_args.get(required):
                raise ValueError(f"data.{dataset_name}.args.{required} is required")
        mode = item_args.setdefault("filename_mode", "basename")
        if mode not in {"basename", "relative"}:
            raise ValueError(f"data.{dataset_name}.args.filename_mode is invalid")
        resolved_data[str(dataset_name)] = item
    if not resolved_data:
        raise ValueError("no enabled datasets")
    config["data"] = resolved_data
    return config


def write_normalized_csv(path: Path, rows: Iterable[tuple[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "score"])
        writer.writerows(rows)


def _video_stem(filename: str) -> str:
    path = PurePosixPath(filename)
    return str(path.with_suffix(""))


def validate_features(rows: list[tuple[str, float]], args: dict[str, Any], use_liqe: bool) -> None:
    roots = {
        "frames": Path(args["frames_dir"]),
        "slowfast": Path(args["slowfast_dir"]),
        "fastvqa": Path(args["fastvqa_dir"]),
    }
    if use_liqe:
        if not args.get("liqe_dir"):
            raise ValueError("liqe_dir is required because model.args.use_liqe is true")
        roots["liqe"] = Path(args["liqe_dir"])
    missing: list[Path] = []
    for filename, _ in rows:
        stem = _video_stem(filename)
        for index in range(8):
            candidates = [roots["frames"] / stem / f"{index:03d}.png"]
            candidates += [roots["slowfast"] / stem / f"feature_{index}_fast_feature.npy"]
            if use_liqe:
                candidates += [roots["liqe"] / stem / f"feature_{index}_LIQE_feature.npy"]
            missing.extend(path for path in candidates if not path.is_file())
        fastvqa = roots["fastvqa"] / f"{stem}.npy"
        flat_fastvqa = roots["fastvqa"] / f"{PurePosixPath(stem).name}.npy"
        if not fastvqa.is_file() and not flat_fastvqa.is_file():
            missing.append(fastvqa)
        if len(missing) >= 20:
            break
    if missing:
        examples = "\n".join(f"  - {path}" for path in missing[:20])
        raise FileNotFoundError(f"missing extracted inputs (showing up to 20):\n{examples}")


def _safe_float(value: Any) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def calculate_metrics(labels, predictions) -> dict[str, Any]:
    import numpy as np
    from scipy.stats import kendalltau, pearsonr, spearmanr
    from utils import fit_function

    labels = np.asarray(labels, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    if len(labels) < 2 or not np.isfinite(labels).all() or not np.isfinite(predictions).all():
        raise ValueError("metrics require at least two finite labels and predictions")
    result: dict[str, Any] = {
        "num_videos": int(len(labels)),
        "SRCC": None,
        "KRCC": None,
        "PLCC_raw": None,
        "RMSE_raw": _safe_float(np.sqrt(np.mean((predictions - labels) ** 2))),
        "PLCC_fitted": None,
        "RMSE_fitted": None,
    }
    if np.std(labels) > 0 and np.std(predictions) > 0:
        result.update(
            SRCC=_safe_float(spearmanr(labels, predictions)[0]),
            KRCC=_safe_float(kendalltau(labels, predictions)[0]),
            PLCC_raw=_safe_float(pearsonr(labels, predictions)[0]),
        )
        try:
            fitted = fit_function(labels, predictions)
            result["PLCC_fitted"] = _safe_float(pearsonr(labels, fitted)[0])
            result["RMSE_fitted"] = _safe_float(np.sqrt(np.mean((fitted - labels) ** 2)))
        except Exception as error:  # scipy fitting can fail for degenerate predictions.
            result["fit_error"] = str(error)
    else:
        result["note"] = "Correlation is undefined because MOS or predictions are constant."
    return result


def _checkpoint_files(path: str, use_liqe: bool, n_exp: int | None) -> list[Path]:
    checkpoint = Path(path)
    if checkpoint.is_file():
        files = [checkpoint]
    elif checkpoint.is_dir():
        tag = "LIQE" if use_liqe else "noLIQE"
        files = sorted(checkpoint.glob(f"RQ_VQA_{tag}_*.pth"))
    else:
        files = []
    if not files:
        raise FileNotFoundError(f"no matching checkpoint found at {checkpoint}")
    if n_exp is not None:
        if not isinstance(n_exp, int) or not 1 <= n_exp <= len(files):
            raise ValueError("n_exp must be between 1 and the number of matching checkpoints")
        files = files[:n_exp]
    return files


def evaluate_dataset(
    name: str,
    args: dict[str, Any],
    rows: list[tuple[str, float]],
    output_dir: Path,
    model,
    inference_model,
    checkpoints: list[Path],
    device,
    transform,
    config: dict[str, Any],
) -> dict[str, Any]:
    import numpy as np
    import pandas as pd
    import torch
    from data_loader import BVQA_VideoDataset
    from test import load_checkpoint

    annotation = output_dir / "normalized_annotations.csv"
    write_normalized_csv(annotation, rows)
    use_liqe = config["model"]["args"]["use_liqe"]
    dataset = BVQA_VideoDataset(
        args["frames_dir"], args["slowfast_dir"], args["fastvqa_dir"],
        str(annotation), transform, 384, use_liqe=use_liqe,
        data_dir_LIQE=args.get("liqe_dir"),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 1)),
        shuffle=False,
        num_workers=int(config.get("num_workers", 4)),
        pin_memory=device.type == "cuda",
        persistent_workers=int(config.get("num_workers", 4)) > 0,
    )
    all_predictions = []
    for checkpoint in checkpoints:
        print(f"[{name}] Loading {checkpoint}", flush=True)
        load_checkpoint(model, checkpoint)
        inference_model.eval()
        scores = []
        with torch.no_grad():
            for *inputs, _ in loader:
                batch = inference_model(*[value.to(device, non_blocking=True) for value in inputs])
                scores.extend(batch.detach().float().cpu().numpy().tolist())
        all_predictions.append(scores)
    predictions = np.mean(np.asarray(all_predictions, dtype=float), axis=0)
    labels = np.asarray([score for _, score in rows], dtype=float)
    table = pd.DataFrame({
        "filename": dataset.video_names,
        "mos": labels,
        "prediction": predictions,
    })
    table.to_csv(output_dir / "predictions.csv", index=False)
    metrics = calculate_metrics(labels, predictions)
    metrics.update(dataset=name, checkpoints=[str(path) for path in checkpoints])
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"[{name}] " + json.dumps(metrics, ensure_ascii=False, allow_nan=False), flush=True)
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML evaluation configuration")
    parser.add_argument("--dataset", action="append", help="Only run the named data entry; repeatable")
    parser.add_argument("--validate-only", action="store_true", help="Parse labels and check feature files")
    cli = parser.parse_args(argv)
    config = load_config(cli.config)
    if config.get("visible_devices") is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config["visible_devices"])
    selected = set(cli.dataset or [])
    unknown = selected - set(config["data"])
    if unknown:
        parser.error("unknown dataset(s): " + ", ".join(sorted(unknown)))

    prepared: dict[str, tuple[dict[str, Any], list[tuple[str, float]], Path]] = {}
    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    use_liqe = config["model"]["args"]["use_liqe"]
    for name, dataset in config["data"].items():
        if selected and name not in selected:
            continue
        args = dataset["args"]
        rows = parse_annotation(
            args["anno_file"], filename_mode=args["filename_mode"],
            data_prefix=args.get("data_prefix"),
        )
        validate_features(rows, args, use_liqe)
        dataset_output = output_root / name
        dataset_output.mkdir(parents=True, exist_ok=True)
        write_normalized_csv(dataset_output / "normalized_annotations.csv", rows)
        prepared[name] = (args, rows, dataset_output)
        print(f"[{name}] checked {len(rows)} videos", flush=True)
    if cli.validate_only:
        print("All selected datasets passed annotation and feature validation.", flush=True)
        return 0

    sys.path.insert(0, str(ROOT))
    import numpy as np
    import pandas as pd
    import torch
    from torchvision import transforms
    import models

    seed = int(config.get("random_seed", 8))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for RQ-VQA inference")
    device = torch.device("cuda:0")
    model = models.RQ_VQA(use_liqe=use_liqe, pretrained=False).to(device)
    inference_model = model
    if torch.cuda.device_count() > 1:
        inference_model = torch.nn.DataParallel(model)
    checkpoints = _checkpoint_files(config["test_load_path"], use_liqe, config.get("n_exp"))
    transform = transforms.Compose([
        transforms.Resize(384), transforms.CenterCrop(384), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    summary = []
    for name, (args, rows, dataset_output) in prepared.items():
        summary.append(evaluate_dataset(
            name, args, rows, dataset_output, model, inference_model,
            checkpoints, device, transform, config,
        ))
    pd.DataFrame([
        {key: value for key, value in item.items() if key not in {"checkpoints", "fit_error", "note"}}
        for item in summary
    ]).to_csv(output_root / "summary.csv", index=False)
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Completed {len(summary)} dataset(s). Summary: {output_root / 'summary.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
