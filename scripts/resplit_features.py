"""Reorganize precomputed RQ-VQA features using existing relative-path splits."""
import argparse
import csv
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil


SPLITS = ('train', 'val', 'test')


def read_split(path):
    rows = []
    seen = set()
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            delimiter = ',' if ',' in line else ';' if ';' in line else None
            fields = [v.strip() for v in next(csv.reader([line], delimiter=delimiter))] if delimiter else line.split()
            if (not rows and fields[0].lower() in ('filename', 'video', 'path')
                    and fields[-1].lower() in ('score', 'mos', 'label')):
                continue
            if len(fields) not in (2, 4):
                raise ValueError(f'{path}:{number}: expected video,MOS or video,-1,-1,MOS')
            raw_name = fields[0].replace('\\', '/')
            video = PurePosixPath(raw_name)
            if not raw_name or not video.suffix:
                raise ValueError(f'{path}:{number}: invalid video path: {raw_name}')
            if video.is_absolute():
                # Precomputed features are stored by video basename.
                name = video.name
            else:
                if '..' in video.parts or ':' in raw_name:
                    raise ValueError(f'{path}:{number}: unsafe relative video path: {raw_name}')
                name = video.as_posix()
            if name in seen:
                raise ValueError(f'{path}:{number}: duplicate video: {name}')
            try:
                score = float(fields[-1])
            except ValueError as error:
                raise ValueError(f'{path}:{number}: MOS must be numeric') from error
            if not math.isfinite(score):
                raise ValueError(f'{path}:{number}: MOS must be finite')
            seen.add(name)
            rows.append((name, score))
    if not rows:
        raise ValueError(f'Empty annotation: {path}')
    return rows


def locate_group(source, feature_type, stem, leaves):
    """Prefer an exact mirrored path; only fall back when that path is absent."""
    exact = source / feature_type / stem
    flat = source / feature_type / stem.name
    folder = exact if exact.exists() else flat
    paths = [folder / leaf for leaf in leaves]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError('Missing source features:\n  ' + '\n  '.join(missing))
    return paths


def sample_files(source, name, use_liqe=False):
    stem = Path(PurePosixPath(name).with_suffix('').as_posix())
    files = []
    groups = {
        'image_384p': [f'{i:03d}.png' for i in range(8)],
        'SlowFast': [f'feature_{i}_fast_feature.npy' for i in range(8)],
    }
    if use_liqe:
        groups['LIQE'] = [f'feature_{i}_LIQE_feature.npy' for i in range(8)]
    for feature_type, leaves in groups.items():
        paths = locate_group(source, feature_type, stem, leaves)
        files.extend((path, Path(feature_type) / stem / leaf)
                     for path, leaf in zip(paths, leaves))
    fast = source / 'FASTVQA' / (str(stem) + '.npy')
    if not fast.is_file():
        fast = source / 'FASTVQA' / (stem.name + '.npy')
    if not fast.is_file():
        raise FileNotFoundError(fast)
    files.append((fast, Path('FASTVQA') / (str(stem) + '.npy')))
    return files


def make_plan(source, output, split_rows, use_liqe=False):
    """Validate every split and destination before creating any output."""
    seen_names, seen_stems, source_owners = {}, {}, {}
    plan = []
    for split, rows in split_rows.items():
        for name, score in rows:
            stem = str(PurePosixPath(name).with_suffix(''))
            if name in seen_names or stem in seen_stems:
                previous = seen_names.get(name, seen_stems.get(stem))
                raise ValueError(f'Overlapping video/feature stem in {previous} and {split}: {name}')
            seen_names[name] = split
            seen_stems[stem] = split
            files = sample_files(source, name, use_liqe)
            for path, relative in files:
                key = str(path.resolve())
                owner = source_owners.get(key)
                if owner is not None and owner != name:
                    raise ValueError(f'Different video names resolve to the same feature: {owner}, {name}: {path}')
                source_owners[key] = name
                destination = output / split / relative
                if destination.is_symlink():
                    if destination.resolve() != path.resolve():
                        raise FileExistsError(f'Existing link points elsewhere: {destination}')
                elif destination.exists():
                    if not destination.is_file() or not os.path.samefile(path, destination):
                        raise FileExistsError(f'Refusing to overwrite existing file: {destination}')
            plan.append((split, name, score, files))
    return plan


def verify_content(plan):
    import numpy as np
    from PIL import Image

    checked = set()
    for _, _, _, files in plan:
        for source, relative in files:
            key = str(source.resolve())
            if key in checked:
                continue
            checked.add(key)
            feature_type = relative.parts[0]
            if feature_type == 'image_384p':
                with Image.open(source) as frame:
                    if min(frame.size) < 384:
                        raise ValueError(f'Frame smaller than 384 pixels: {source}')
                    frame.verify()
            else:
                dimension = {'SlowFast': 256, 'FASTVQA': 768, 'LIQE': 495}[feature_type]
                array = np.load(source, allow_pickle=False).squeeze()
                if array.shape != (dimension,) or not np.isfinite(array).all():
                    raise ValueError(f'Invalid {feature_type} feature: {source}, shape={array.shape}')


def create_file(source, destination, mode):
    if destination.exists() or destination.is_symlink():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == 'symlink':
        destination.symlink_to(source.resolve())
    elif mode == 'hardlink':
        os.link(source, destination)
    else:
        shutil.copy2(source, destination)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['filename', 'score'])
        writer.writerows(rows)
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Old feature split, e.g. SDR/test')
    parser.add_argument('--output', type=Path, required=True)
    for split in SPLITS:
        parser.add_argument(f'--{split}-txt', type=Path, required=True)
    parser.add_argument('--mode', choices=('symlink', 'hardlink', 'copy'), default='symlink')
    parser.add_argument('--use-liqe', action='store_true')
    parser.add_argument('--check-only', action='store_true', help='Validate paths and feature contents; create nothing')
    args = parser.parse_args(argv)
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('Source and output must be separate directories')
    rows = {split: read_split(getattr(args, f'{split}_txt')) for split in SPLITS}
    plan = make_plan(source, output, rows, args.use_liqe)
    verify_content(plan)
    for split, split_rows in rows.items():
        count = sum(len(files) for item_split, _, _, files in plan if item_split == split)
        print(f'{split}: checked {len(split_rows)} videos, {count} feature files', flush=True)
    if args.check_only:
        print('All splits and source features are valid; no output was created.', flush=True)
        return
    for split, _, _, files in plan:
        for path, relative in files:
            create_file(path, output / split / relative, args.mode)
    for split, split_rows in rows.items():
        write_csv(output / 'annotations' / f'{split}.csv', split_rows)
    manifest = [{'split': split, 'filename': name, 'score': score,
                 'files': {relative.as_posix(): str(path) for path, relative in files}}
                for split, name, score, files in plan]
    (output / 'annotations' / 'feature_sources.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    print(f'Completed ({args.mode}): {output}', flush=True)


if __name__ == '__main__':
    main()
