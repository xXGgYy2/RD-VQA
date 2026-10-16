"""Prepare VDPVE annotations, frames and frozen features for RQ-VQA."""
import argparse
import csv
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANN = '/mnt/data/local-disk2/xyy/datasets/FineVD/FineVD'
STAGES = ('frames', 'slowfast', 'fastvqa', 'liqe')


def read_annotations(path):
    rows = []
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        for line, row in enumerate(csv.reader(handle), 1):
            if not row:
                continue
            if len(row) != 4:
                raise ValueError(f'{path}:{line}: expected filename,source,0,MOS')
            name, source, _, score = (item.strip() for item in row)
            if (not name or Path(name).is_absolute() or '..' in Path(name).parts
                    or '\\' in name or Path(name).suffix.lower() != '.mp4'):
                raise ValueError(f'{path}:{line}: invalid relative MP4 filename: {name}')
            if not source or not math.isfinite(float(score)):
                raise ValueError(f'{path}:{line}: invalid source or MOS')
            rows.append((name, source, score))
    if not rows:
        raise ValueError(f'Empty annotation: {path}')
    if len({row[0] for row in rows}) != len(rows):
        raise ValueError(f'Duplicate video in {path}')
    # The original FAST-VQA extractor saves flat basenames.
    if len({Path(row[0]).stem for row in rows}) != len(rows):
        raise ValueError(f'Duplicate video basename in {path}')
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['filename', 'score'])
        writer.writerows((name, score) for name, _, score in rows)


def prepare(args):
    splits = {split: read_annotations(getattr(args, f'{split}_txt'))
#                for split in ('train', 'val', 'test')}
                for split in ('train','val',)}
    seen_names, seen_sources = set(), set()
    for split, rows in splits.items():
        names = {row[0] for row in rows}
        sources = {row[1] for row in rows}
        if names & seen_names:
            raise ValueError(f'Videos shared across splits: {sorted(names & seen_names)[:5]}')
        if sources & seen_sources:
            print(f'WARNING: {split} shares source IDs with an earlier split: '
                  f'{sorted(sources & seen_sources)[:5]}. Supplied splits are preserved.', flush=True)
        seen_names.update(names)
        seen_sources.update(sources)
        video_dir = getattr(args, f'{split}_videos_dir') or args.videos_dir
        missing = [name for name in names if not (video_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f'{split}: {len(missing)} videos missing in {video_dir}; '
                                    f'examples: {sorted(missing)[:5]}. '
                                    f'Set --{split}-videos-dir to the correct directory.')
    for split, rows in splits.items():
        write_csv(args.output / 'annotations' / f'{split}.csv', rows)
        print(f'{split}: {len(rows)} videos, {len({r[1] for r in rows})} source IDs', flush=True)
    return splits


def complete(stage, folder, name):
    import numpy as np
    from PIL import Image
    stem = Path(name).with_suffix('')
    try:
        if stage in ('frames', 'original'):
            for i in range(8):
                with Image.open(folder / stem / f'{i:03d}.png') as frame:
                    if min(frame.size) < (384 if stage == 'frames' else 224):
                        return False
                    frame.verify()
            return True
        if stage == 'fastvqa':
            paths = [folder / f'{stem.name}.npy']
            dim = 768
        else:
            suffix, dim = ('fast', 256) if stage == 'slowfast' else ('LIQE', 495)
            paths = [folder / stem / f'feature_{i}_{suffix}_feature.npy' for i in range(8)]
        return all((feature := np.load(path, allow_pickle=False).squeeze()).shape == (dim,)
                   and np.isfinite(feature).all() for path in paths)
    except (OSError, ValueError, EOFError):
        return False


def run(script, *arguments, cwd=ROOT):
    command = [sys.executable, '-u', str(script), *map(str, arguments)]
    print('+ ' + ' '.join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def extract(args, splits):
    import torch
    import yaml
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Check torch.cuda.is_available() first.')
    print(f'GPU: {torch.cuda.get_device_name(0)}; torch cache: {os.environ["TORCH_HOME"]}', flush=True)
    stages = args.stages or list(STAGES if args.use_liqe else STAGES[:-1])
    if 'liqe' in stages and not args.use_liqe:
        raise ValueError('--stages liqe requires --use-liqe')
    for stage, weight in (('fastvqa', args.fastvqa_weights), ('liqe', args.liqe_weights)):
        if stage in stages and not weight.is_file():
            raise FileNotFoundError(f'{stage} checkpoint missing: {weight}')
    # Load FAST-VQA options from THIS repository: its evaluator returns features.
    with (ROOT / 'feature_extraction/FastVQA_feature/options/fast-b_NTIRE_UGC.yml').open() as f:
        template = yaml.safe_load(f)
    for stage in stages:
        for split, rows in splits.items():
            video_dir = getattr(args, f'{split}_videos_dir') or args.videos_dir
            split_dir = args.output / split
            if stage == 'liqe':
                extract_original_frames(args, split, rows, video_dir)
            folder = split_dir / {'frames': 'image_384p', 'slowfast': 'SlowFast',
                                  'fastvqa': 'FASTVQA', 'liqe': 'LIQE'}[stage]
            folder.mkdir(parents=True, exist_ok=True)
            pending = [r for r in rows if args.overwrite or not complete(stage, folder, r[0])]
            print(f'[{stage}/{split}] {len(pending)} pending / {len(rows)} total', flush=True)
            if not pending:
                continue
            pending_csv = args.output / 'annotations' / f'{split}_{stage}_pending.csv'
            write_csv(pending_csv, pending)
            if stage == 'frames':
                run(ROOT / 'frame_extraction/extract_frame_NTIREVideo_384p.py',
                    '--videos_dir', video_dir, '--filename_path', pending_csv, '--save_folder', folder)
            elif stage == 'slowfast':
                # pytorchvideo downloads SlowFast R50 on first use and reuses TORCH_HOME.
                run(ROOT / 'feature_extraction/extract_SlowFast_feature_VQA.py',
                    '--videos_dir', video_dir, '--datainfo_test', pending_csv,
                    '--feature_save_folder', str(folder) + os.sep, '--resize', 224,
                    '--num_workers', args.num_workers)
            elif stage == 'fastvqa':
                import copy
                opt = copy.deepcopy(template)
                dataset = next(iter(opt['data'].values()))
                dataset['args']['data_prefix'] = str(video_dir)
                # FusionDataset accepts a list; bypass its incompatible text parser
                # and its silent fallback to scanning the whole video directory.
                dataset['args']['anno_file'] = [
                    {'filename': str(video_dir / name), 'label': float(score)}
                    for name, _, score in pending]
                opt['data'] = {f'test-{split}': dataset}
                opt['num_workers'] = args.num_workers
                opt['test_load_path'] = str(args.fastvqa_weights)
                config = args.output / 'annotations' / f'{split}_fastvqa.yml'
                with config.open('w') as f:
                    yaml.safe_dump(opt, f, sort_keys=False)
                fast_root = ROOT / 'feature_extraction/FastVQA_feature'
                run(fast_root / 'extract_fastvqa_feature.py', '--opt', config,
                    '--save_path', folder, cwd=fast_root)
            else:
                run(ROOT / 'feature_extraction/extract_LIQE_feature_KVQ.py',
                    '--videos_dir_test', split_dir / 'image_original',
                    '--datainfo_test', pending_csv, '--feature_save_folder', str(folder) + os.sep,
                    '--model_path', args.liqe_weights)
            failed = [r[0] for r in rows if not complete(stage, folder, r[0])]
            if failed:
                raise RuntimeError(f'{stage}/{split}: invalid outputs: {failed[:5]}')
    print('Requested extraction stages completed and outputs verified.', flush=True)


def extract_original_frames(args, split, rows, video_dir):
    folder = args.output / split / 'image_original'
    pending = [r for r in rows if args.overwrite or not complete('original', folder, r[0])]
    if pending:
        annotation = args.output / 'annotations' / f'{split}_original_pending.csv'
        write_csv(annotation, pending)
        run(ROOT / 'frame_extraction/extract_frame_NTIREVideo_original.py',
            '--videos_dir', video_dir, '--filename_path', annotation, '--save_folder', folder)
    if any(not complete('original', folder, row[0]) for row in rows):
        raise RuntimeError(f'{split}: original frames missing or smaller than LIQE patch size 224')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--videos-dir', type=Path,
                        default=Path('/mnt/data/local-disk2/xyy/datasets/VDPVE/train'))
    for split in ('train', 'val', 'test'):
        parser.add_argument(f'--{split}-txt', type=Path, default=Path(DEFAULT_ANN) / f'{split}.txt')
        parser.add_argument(f'--{split}-videos-dir', type=Path)
    parser.add_argument('--output', type=Path,
                        default=Path('/mnt/data/local-disk2/xyy/datasets/VDPVE/rqvqa_features'))
    parser.add_argument('--fastvqa-weights', type=Path, default=ROOT / 'weights/FAST_VQA_B_1_4.pth')
    parser.add_argument('--liqe-weights', type=Path, default=ROOT / 'weights/LIQE.pt')
    parser.add_argument('--use-liqe', action='store_true')
    parser.add_argument('--stages', choices=STAGES, nargs='+')
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--prepare-only', action='store_true', help='Validate paths and convert annotations only')
    parser.add_argument('--overwrite', action='store_true', help='Recompute already valid outputs')
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())
    # Make imports independent of the shell working directory, including subprocesses.
    os.environ['PYTHONPATH'] = str(ROOT) + os.pathsep + os.environ.get('PYTHONPATH', '')
    os.environ.setdefault('TORCH_HOME', str(ROOT / 'weights/torch'))
    splits = prepare(args)
    if not args.prepare_only:
        extract(args, splits)


if __name__ == '__main__':
    main()
