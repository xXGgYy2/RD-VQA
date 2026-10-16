"""Extract frozen features, predict with a VDPVE checkpoint, and evaluate AIGC."""
import argparse
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.extract_vdpve import extract, run, write_csv

DATA = Path('/mnt/data/local-disk3/xyy/dataset/AIGC-test')
CHECKPOINT = 'RQ_VQA_noLIQE_VDPVE_plcc_NR_v0_epoch_8_SRCC_0.712248.pth'


def read_mos(path, videos_dir):
    rows, seen = [], set()
    with path.open(encoding='utf-8-sig') as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                name, score = line.strip().rsplit(maxsplit=1)
                if not math.isfinite(float(score)):
                    raise ValueError('non-finite MOS')
            except ValueError as error:
                raise ValueError(f'{path}:{line_number}: expected filename MOS') from error
            relative = Path(name)
            if (relative.is_absolute() or '..' in relative.parts or '\\' in name
                    or relative.suffix.lower() != '.mp4'):
                raise ValueError(f'Invalid relative MP4 path: {name}')
            if relative.stem in seen:
                raise ValueError(f'Duplicate video basename: {name}')
            if not (videos_dir / relative).is_file():
                raise FileNotFoundError(videos_dir / relative)
            seen.add(relative.stem)
            rows.append((name, relative.stem, score))
    if len(rows) < 2:
        raise ValueError('At least two labeled videos are required for evaluation')
    return rows


def evaluate(annotation, prediction, output, checkpoint):
    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr, pearsonr, kendalltau

    gt, pred = pd.read_csv(annotation), pd.read_csv(prediction)
    if (gt.filename.duplicated().any() or pred.filename.duplicated().any()
            or set(gt.filename) != set(pred.filename)):
        raise ValueError('Prediction filenames must match annotations exactly, without duplicates')
    joined = gt.merge(pred, on='filename', suffixes=('_mos', '_prediction'), validate='one_to_one')
    y, p = joined.score_mos.to_numpy(float), joined.score_prediction.to_numpy(float)
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('MOS or predictions contain non-finite values')
    metrics = dict(checkpoint=str(checkpoint), num_videos=len(joined),
                   SRCC=None, PLCC_raw=None, KRCC=None)
    if np.std(y) > 0 and np.std(p) > 0:
        metrics.update(SRCC=float(spearmanr(y, p)[0]), PLCC_raw=float(pearsonr(y, p)[0]),
                       KRCC=float(kendalltau(y, p)[0]))
    else:
        metrics['note'] = 'Correlation is undefined because MOS or predictions are constant.'
    joined.to_csv(output / 'predictions_with_mos.csv', index=False)
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False) + '\n')
    print(json.dumps(metrics, indent=2, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--videos-dir', type=Path, default=DATA / 'AIGC')
    parser.add_argument('--mos', type=Path, default=DATA / 'video_list_vqscore.txt')
    parser.add_argument('--output', type=Path, default=DATA / 'rqvqa_features')
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'ckpts/VDPVE_noLIQE' / CHECKPOINT)
    parser.add_argument('--fastvqa-weights', type=Path, default=ROOT / 'weights/FAST_VQA_B_1_4.pth')
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--overwrite', action='store_true', help='Re-extract valid cached features')
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    rows = read_mos(args.mos, args.videos_dir)
    annotation = args.output / 'annotations/test.csv'
    write_csv(annotation, rows)
    print(f'Validated {len(rows)} videos. Annotation: {annotation}', flush=True)
    if args.prepare_only:
        return
    os.environ['PYTHONPATH'] = str(ROOT) + os.pathsep + os.environ.get('PYTHONPATH', '')
    os.environ.setdefault('TORCH_HOME', str(ROOT / 'weights/torch'))
    args.test_videos_dir = None
    args.use_liqe = False
    args.liqe_weights = ROOT / 'weights/LIQE.pt'  # Unused by the selected stages.
    args.stages = ['frames', 'slowfast', 'fastvqa']
    extract(args, {'test': rows})
    result_dir = args.output / 'results' / args.checkpoint.stem
    result_dir.mkdir(parents=True, exist_ok=True)
    prediction = result_dir / 'predictions.csv'
    print('Starting inference with the specified VDPVE checkpoint (LIQE disabled).', flush=True)
    run(ROOT / 'test.py', '--pretrained_path', args.checkpoint,
        '--datainfo', annotation, '--videos_dir', args.output / 'test/image_384p',
        '--slowfast_dir', args.output / 'test/SlowFast',
        '--fastvqa_dir', args.output / 'test/FASTVQA',
        '--save_file', prediction, '--num_workers', args.num_workers)
    evaluate(annotation, prediction, result_dir, args.checkpoint)
    print(f'Completed. Results: {result_dir}', flush=True)


if __name__ == '__main__':
    main()
