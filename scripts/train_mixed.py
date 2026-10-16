#!/usr/bin/env python3
"""Prepare mixed-dataset feature links and launch the existing RQ-VQA trainer."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")
FEATURE_DIRS = ("image_384p", "SlowFast", "FASTVQA")


@dataclass(frozen=True)
class FeatureMatch:
    source_name: str
    root: Path
    stem: Path
    fastvqa: Path


def split_fields(line: str) -> list[str]:
    if "," in line:
        return [item.strip() for item in next(csv.reader([line]))]
    if ";" in line:
        return [item.strip() for item in next(csv.reader([line], delimiter=";"))]
    return line.split()


def read_annotations(path: str | Path) -> list[tuple[Path, float]]:
    """Read absolute-video annotations in either two- or four-column form."""
    path = Path(path)
    rows: list[tuple[Path, float]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            fields = split_fields(line)
            if not rows and fields and fields[0].lower() in {"filename", "video", "path"}:
                continue
            if len(fields) not in (2, 4):
                raise ValueError(
                    f"{path}:{line_number}: expected 'absolute_video MOS' or "
                    f"'absolute_video -1 -1 MOS', got {len(fields)} fields"
                )
            video = Path(fields[0]).expanduser()
            if not video.is_absolute():
                raise ValueError(f"{path}:{line_number}: video path must be absolute: {video}")
            if not video.is_file():
                raise FileNotFoundError(f"{path}:{line_number}: video does not exist: {video}")
            try:
                score = float(fields[-1])
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: MOS must be numeric") from error
            if not math.isfinite(score):
                raise ValueError(f"{path}:{line_number}: MOS must be finite")
            key = os.path.normcase(str(video.resolve()))
            if key in seen:
                raise ValueError(f"{path}:{line_number}: duplicate video: {video}")
            seen.add(key)
            rows.append((video.resolve(), score))
    if len(rows) < 2:
        raise ValueError(f"{path}: at least two videos are required")
    return rows


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.-")
    return value or "dataset"


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def feature_roots(search_root: Path) -> list[Path]:
    roots = []
    for frames in search_root.rglob("image_384p"):
        root = frames.parent
        if all((root / folder).is_dir() for folder in FEATURE_DIRS):
            roots.append(root.resolve())
    unique = sorted(set(roots), key=str)
    if not unique:
        raise FileNotFoundError(
            f"No directory containing image_384p, SlowFast and FASTVQA was found under {search_root}"
        )
    return unique


def parse_rules(config: dict[str, Any], search_root: Path) -> list[dict[str, Any]]:
    rules = config.get("source_rules", [])
    if not isinstance(rules, list):
        raise ValueError("source_rules must be a list")
    parsed = []
    for index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            raise ValueError(f"source_rules[{index}] must be a mapping")
        unknown = set(rule) - {"name", "video_prefix", "feature_root"}
        if unknown:
            raise ValueError(f"source_rules[{index}] has unknown keys: {sorted(unknown)}")
        if not all(rule.get(key) for key in ("name", "video_prefix", "feature_root")):
            raise ValueError(
                f"source_rules[{index}] requires name, video_prefix and feature_root"
            )
        prefix = Path(rule["video_prefix"]).expanduser().resolve()
        root = Path(rule["feature_root"]).expanduser().resolve()
        if not is_relative_to(root, search_root) and root != search_root:
            raise ValueError(f"source rule feature_root must be under {search_root}: {root}")
        if not root.is_dir():
            raise FileNotFoundError(root)
        parsed.append({"name": safe_name(str(rule["name"])), "video_prefix": prefix,
                       "feature_root": root})
    return parsed


def matching_rule(video: Path, rules: list[dict[str, Any]]) -> dict[str, Any] | None:
    matches = [rule for rule in rules if is_relative_to(video, rule["video_prefix"])]
    if not matches:
        return None
    matches.sort(key=lambda rule: len(rule["video_prefix"].parts), reverse=True)
    if len(matches) > 1 and len(matches[0]["video_prefix"].parts) == len(matches[1]["video_prefix"].parts):
        raise ValueError(f"Multiple equally specific source_rules match {video}")
    return matches[0]


def candidate_stems(video: Path, rule: dict[str, Any] | None) -> list[Path]:
    stems = []
    if rule is not None:
        relative = video.relative_to(rule["video_prefix"])
        stems.append(Path(str(relative.with_suffix(""))))
    stems.append(Path(video.stem))
    result = []
    for stem in stems:
        if stem not in result:
            result.append(stem)
    return result


def complete_match(root: Path, stem: Path, use_liqe: bool) -> FeatureMatch | None:
    frames = [root / "image_384p" / stem / f"{index:03d}.png" for index in range(8)]
    slowfast = [
        root / "SlowFast" / stem / f"feature_{index}_fast_feature.npy"
        for index in range(8)
    ]
    fastvqa = root / "FASTVQA" / (str(stem) + ".npy")
    if not fastvqa.is_file():
        fastvqa = root / "FASTVQA" / f"{stem.name}.npy"
    required = frames + slowfast + [fastvqa]
    if use_liqe:
        required += [
            root / "LIQE" / stem / f"feature_{index}_LIQE_feature.npy"
            for index in range(8)
        ]
    if not all(path.is_file() for path in required):
        return None
    return FeatureMatch("", root, stem, fastvqa)


def locate_features(
    video: Path,
    roots: list[Path],
    rules: list[dict[str, Any]],
    search_root: Path,
    use_liqe: bool,
) -> FeatureMatch:
    rule = matching_rule(video, rules)
    allowed_roots = roots
    if rule is not None:
        allowed_roots = [root for root in roots if is_relative_to(root, rule["feature_root"])]
    matches = []
    for root in allowed_roots:
        for stem in candidate_stems(video, rule):
            match = complete_match(root, stem, use_liqe)
            if match is not None:
                if rule is not None:
                    source_name = rule["name"]
                else:
                    relative_root = root.relative_to(search_root)
                    source_name = safe_name(relative_root.parts[0] if relative_root.parts else root.name)
                matches.append(FeatureMatch(source_name, root, stem, match.fastvqa))
    # The same root may match both a relative stem and its basename.
    matches = list(dict.fromkeys(matches))
    if not matches:
        hint = ""
        if rule is None:
            hint = " Add a source_rules entry if its feature layout uses a relative subpath."
        raise FileNotFoundError(f"No complete feature set found for {video}.{hint}")
    if len(matches) > 1:
        choices = "\n".join(f"  - {item.root} (stem={item.stem})" for item in matches)
        raise ValueError(
            f"Ambiguous features for {video}:\n{choices}\n"
            "Add a source_rules entry mapping this video's directory to one feature_root."
        )
    return matches[0]


def ensure_symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() == source.resolve():
            return
        raise FileExistsError(f"Existing link points elsewhere: {destination}")
    if destination.exists():
        raise FileExistsError(f"Destination already exists and is not the expected link: {destination}")
    destination.symlink_to(source, target_is_directory=source.is_dir())


def link_sample(
    workspace: Path,
    split: str,
    relative_name: PurePosixPath,
    match: FeatureMatch,
    use_liqe: bool,
) -> None:
    relative_stem = Path(str(relative_name.with_suffix("")))
    ensure_symlink(
        match.root / "image_384p" / match.stem,
        workspace / split / "image_384p" / relative_stem,
    )
    ensure_symlink(
        match.root / "SlowFast" / match.stem,
        workspace / split / "SlowFast" / relative_stem,
    )
    ensure_symlink(
        match.fastvqa,
        workspace / split / "FASTVQA" / (str(relative_stem) + ".npy"),
    )
    if use_liqe:
        ensure_symlink(
            match.root / "LIQE" / match.stem,
            workspace / split / "LIQE" / relative_stem,
        )


def load_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is required: pip install pyyaml") from error
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8-sig") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping")
    allowed = {"name", "splits", "feature_search_root", "prepared_root",
               "source_rules", "nproc_per_node", "training"}
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"Unknown mixed config keys: {sorted(unknown)}")
    if not isinstance(config.get("splits"), dict):
        raise ValueError("splits must be a mapping containing train, val and test")
    for split in SPLITS:
        if not config["splits"].get(split):
            raise ValueError(f"splits.{split} is required")
        config["splits"][split] = str(Path(config["splits"][split]).expanduser().resolve())
    for key in ("feature_search_root", "prepared_root"):
        if not config.get(key):
            raise ValueError(f"{key} is required")
        config[key] = str(Path(config[key]).expanduser().resolve())
    if not isinstance(config.get("training"), dict):
        raise ValueError("training must be a mapping of train.py configuration values")
    nproc = config.get("nproc_per_node", 4)
    if isinstance(nproc, bool) or not isinstance(nproc, int) or nproc < 1:
        raise ValueError("nproc_per_node must be a positive integer")
    config["nproc_per_node"] = nproc
    return config


def prepare(config: dict[str, Any]) -> Path:
    search_root = Path(config["feature_search_root"])
    workspace = Path(config["prepared_root"])
    if not search_root.is_dir():
        raise FileNotFoundError(search_root)
    roots = [
        root for root in feature_roots(search_root)
        if root != workspace and not is_relative_to(root, workspace)
    ]
    if not roots:
        raise FileNotFoundError(
            f"No source feature directories remain after excluding prepared_root {workspace}"
        )
    rules = parse_rules(config, search_root)
    use_liqe = bool(config["training"].get("use_liqe", False))
    rows_by_split = {split: read_annotations(config["splits"][split]) for split in SPLITS}

    owners: dict[str, str] = {}
    prepared: dict[str, list[tuple[str, float]]] = {split: [] for split in SPLITS}
    for split, rows in rows_by_split.items():
        source_counts: dict[str, int] = {}
        source_scores: dict[str, list[float]] = {}
        for video, score in rows:
            key = os.path.normcase(str(video))
            if key in owners:
                raise ValueError(f"Video appears in both {owners[key]} and {split}: {video}")
            owners[key] = split
            match = locate_features(video, roots, rules, search_root, use_liqe)
            source_counts[match.source_name] = source_counts.get(match.source_name, 0) + 1
            source_scores.setdefault(match.source_name, []).append(score)
            # Preserve the feature-relative stem so two videos with the same
            # basename in different source folders cannot overwrite each other.
            relative_name = PurePosixPath(match.source_name) / PurePosixPath(
                str(match.stem).replace("\\", "/") + video.suffix
            )
            link_sample(workspace, split, relative_name, match, use_liqe)
            prepared[split].append((relative_name.as_posix(), score))
        detail = ", ".join(
            f"{name}={count} (MOS {min(source_scores[name]):.4g}..{max(source_scores[name]):.4g})"
            for name, count in sorted(source_counts.items())
        )
        print(f"{split}: prepared {len(rows)} videos ({detail})", flush=True)

    annotation_dir = workspace / "annotations"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in prepared.items():
        path = annotation_dir / f"{split}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["filename", "score"])
            writer.writerows(rows)
    return write_training_config(config, workspace)


def write_training_config(config: dict[str, Any], workspace: Path) -> Path:
    import yaml

    training = dict(config["training"])
    reserved = {
        "feature_root", "datainfo", "val_datainfo", "videos_dir", "slowfast_dir",
        "fastvqa_dir", "liqe_dir", "val_videos_dir", "val_slowfast_dir",
        "val_fastvqa_dir", "val_liqe_dir",
    }
    overridden = sorted(reserved & set(training))
    if overridden:
        raise ValueError(
            "training must not override generated data paths: " + ", ".join(overridden)
        )
    generated = dict(training)
    generated.update({
        "feature_root": str(workspace),
        "datainfo": "${feature_root}/annotations/train.csv",
        "val_datainfo": "${feature_root}/annotations/val.csv",
        "videos_dir": "${feature_root}/train/image_384p",
        "slowfast_dir": "${feature_root}/train/SlowFast",
        "fastvqa_dir": "${feature_root}/train/FASTVQA",
        "val_videos_dir": "${feature_root}/val/image_384p",
        "val_slowfast_dir": "${feature_root}/val/SlowFast",
        "val_fastvqa_dir": "${feature_root}/val/FASTVQA",
    })
    if training.get("use_liqe", False):
        generated.update(
            liqe_dir="${feature_root}/train/LIQE",
            val_liqe_dir="${feature_root}/val/LIQE",
        )
    nproc = config["nproc_per_node"]
    if int(generated.get("expected_world_size", nproc)) != nproc:
        raise ValueError("training.expected_world_size must equal nproc_per_node")
    generated["expected_world_size"] = nproc
    path = workspace / "resolved_train_config.yaml"
    path.write_text(yaml.safe_dump(generated, sort_keys=False), encoding="utf-8")
    print(f"Generated training config: {path}", flush=True)
    return path


def validate_prepared(training_config: Path) -> None:
    expression = (
        "from train_options import parse_config, validate_explicit_data; "
        f"c=parse_config(['--config', {str(training_config)!r}]); "
        "validate_explicit_data(c)"
    )
    command = [
        sys.executable,
        "-c",
        expression,
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    args = parser.parse_args(argv)
    if args.prepare_only and args.skip_prepare:
        parser.error("--prepare-only and --skip-prepare cannot be used together")
    config = load_config(args.config)
    workspace = Path(config["prepared_root"])
    training_config = workspace / "resolved_train_config.yaml"
    if not args.skip_prepare:
        training_config = prepare(config)
    elif not training_config.is_file():
        raise FileNotFoundError(training_config)
    validate_prepared(training_config)
    if args.prepare_only:
        print("Mixed dataset preparation and validation completed.", flush=True)
        return 0
    command = [
        sys.executable, "-m", "torch.distributed.run", "--standalone",
        f"--nproc_per_node={config['nproc_per_node']}",
        str(ROOT / "train.py"), "--config", str(training_config),
    ]
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
