import argparse
import atexit
import csv
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def validate_explicit_data(config):
    """Catch missing split paths before spending an epoch training."""
    filenames = []
    for split, annotation in (('train', config.datainfo), ('val', config.val_datainfo)):
        def folder(key):
            value = (getattr(config, 'val_' + key, None) if split == 'val' else None)
            value = value or getattr(config, key, None)
            if not value:
                raise ValueError(f'{split}: missing {key}')
            return Path(value)

        with open(annotation, encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ['filename', 'score']:
                raise ValueError(f'{annotation}: expected exactly filename,score columns')
            rows = list(reader)
        if len(rows) < 2:
            raise ValueError(f'{annotation}: at least two videos are required')
        names = [row['filename'] for row in rows]
        if len(names) != len(set(names)):
            raise ValueError(f'{annotation}: duplicate video filenames')
        if any(not math.isfinite(float(row['score'])) for row in rows):
            raise ValueError(f'{annotation}: MOS must be finite')
        for name in names:
            stem = Path(name).with_suffix('')
            if not name or Path(name).is_absolute() or '..' in stem.parts or '\\' in name:
                raise ValueError(f'{annotation}: invalid relative video path {name}')
            required = [folder('videos_dir') / stem / f'{i:03d}.png' for i in range(8)]
            required += [folder('slowfast_dir') / stem / f'feature_{i}_fast_feature.npy' for i in range(8)]
            # Preserve multi-dot stems, matching the dataset loader.
            fast = folder('fastvqa_dir') / (str(stem) + '.npy')
            if not fast.is_file():
                fast = folder('fastvqa_dir') / (stem.name + '.npy')
            required.append(fast)
            if config.use_liqe:
                required += [folder('liqe_dir') / stem / f'feature_{i}_LIQE_feature.npy' for i in range(8)]
            for path in required:
                if not path.is_file():
                    raise FileNotFoundError(f'{split} input missing: {path}')
        filenames.append(set(names))
        print(f'{split}: checked input paths for {len(names)} videos', flush=True)
    if filenames[0] & filenames[1]:
        raise ValueError('Training and validation CSVs share video filenames')


def parse_config(argv=None):
    parser = build_parser()
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument('--config')
    known, _ = probe.parse_known_args(argv)
    if known.config:
        import yaml
        with open(known.config, encoding='utf-8-sig') as handle:
            values = yaml.safe_load(handle)
        if not isinstance(values, dict):
            parser.error('Config must be a YAML mapping')
        feature_root = values.pop('feature_root', None)
        actions = {action.dest: action for action in parser._actions}
        for key, value in values.items():
            if key not in actions or key in ('help', 'config'):
                parser.error(f'Unknown config field: {key}')
            action = actions[key]
            if value is None:
                if action.default is not None:
                    parser.error(f'{key} cannot be null')
                continue
            if isinstance(value, str):
                if '${feature_root}' in value:
                    if not isinstance(feature_root, str):
                        parser.error('feature_root is required for ${feature_root}')
                    value = value.replace('${feature_root}', feature_root)
                value = os.path.expandvars(os.path.expanduser(value))
                values[key] = value
            if isinstance(action, (argparse._StoreTrueAction, argparse.BooleanOptionalAction)):
                if not isinstance(value, bool):
                    parser.error(f'{key} must be a YAML boolean (true/false)')
            elif action.nargs == '+':
                if not isinstance(value, list) or not value:
                    parser.error(f'{key} must be a nonempty list')
                if any(type(item) is not int for item in value):
                    parser.error(f'{key} must contain integers')
            elif action.type is int:
                if type(value) is not int:
                    parser.error(f'{key} must be an integer')
            elif action.type is float:
                try:
                    values[key] = float(value)
                except (TypeError, ValueError):
                    parser.error(f'{key} must be a number')
            elif not isinstance(value, str):
                parser.error(f'{key} must be a string; quote device IDs')
            if action.choices and values[key] not in action.choices:
                parser.error(f'Invalid {key}: {values[key]}')
        parser.set_defaults(**values)
    args = parser.parse_args(argv)
    if args.pretrained_path and args.finetune_path:
        parser.error('Use only one of pretrained_path (LSVQ) and finetune_path (RQ-VQA)')
    if args.early_stop_patience < 0 or not math.isfinite(args.early_stop_min_delta) or args.early_stop_min_delta < 0:
        parser.error('Early stopping patience and min_delta must be nonnegative and finite')
    if not math.isfinite(args.conv_base_lr) or args.conv_base_lr <= 0:
        parser.error('conv_base_lr must be positive and finite')
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error('weight_decay must be nonnegative and finite')
    for name in ('backbone_lr', 'head_lr'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f'{name} must be positive and finite')
    if args.warmup_epochs < 0 or args.freeze_backbone_epochs < 0:
        parser.error('warmup/freeze epochs must be nonnegative')
    if not math.isfinite(args.grad_clip_norm) or args.grad_clip_norm <= 0:
        parser.error('grad_clip_norm must be positive and finite')
    if not math.isfinite(args.min_lr_ratio) or not 0 <= args.min_lr_ratio <= 1:
        parser.error('min_lr_ratio must be between 0 and 1')
    if args.decay_interval < 1 or not math.isfinite(args.decay_ratio) or args.decay_ratio <= 0:
        parser.error('Invalid learning rate decay settings')
    if args.num_workers < 0:
        parser.error('num_workers must be nonnegative')
    if args.expected_world_size < 1:
        parser.error('expected_world_size must be positive')
    if args.amp_init_scale <= 0 or not math.isfinite(args.amp_init_scale):
        parser.error('amp_init_scale must be positive and finite')
    if args.max_nonfinite_steps < 0:
        parser.error('max_nonfinite_steps must be nonnegative')
    if args.database != 'NTIREVideo' and not args.val_datainfo:
        parser.error('Custom datasets require val_datainfo')
    if not args.database or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.database):
        parser.error('database must contain only letters, digits, underscores and hyphens')
    # Relative data/output/weight paths always refer to the repository root.
    for key, value in vars(args).items():
        if value and (key.endswith('_dir') or key.endswith('_path') or key in ('datainfo', 'val_datainfo', 'log_file')):
            path = Path(value).expanduser()
            setattr(args, key, str(path if path.is_absolute() else ROOT / path))
    return args


def setup_log(path):
    if not path:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, 'a', encoding='utf-8', buffering=1)
    atexit.register(handle.close)

    class Tee:
        def __init__(self, original):
            self.original = original

        def write(self, text):
            self.original.write(text)
            handle.write(text)
            return len(text)

        def flush(self):
            self.original.flush()
            if not handle.closed:
                handle.flush()

    sys.stdout, sys.stderr = Tee(sys.stdout), Tee(sys.stderr)


def build_parser():
    parser = argparse.ArgumentParser()
    # input parameters
    parser.add_argument('--database', default='NTIREVideo', help='Dataset label; custom datasets require --val_datainfo')
    parser.add_argument('--model_name', default='RQ_VQA', choices=['RQ_VQA'])
    parser.add_argument('--use_liqe', action=argparse.BooleanOptionalAction, default=False,
                        help='Include precomputed LIQE features')
    parser.add_argument('--datainfo', default='data/train_data.csv')
    parser.add_argument('--val_datainfo', default=None,
                        help='Explicit validation CSV (filename,score)')
    parser.add_argument('--videos_dir', default='image_384p')
    parser.add_argument('--slowfast_dir', default='features/SlowFast')
    parser.add_argument('--fastvqa_dir', default='features/FASTVQA')
    parser.add_argument('--liqe_dir', default=None)
    parser.add_argument('--val_videos_dir', default=None)
    parser.add_argument('--val_slowfast_dir', default=None)
    parser.add_argument('--val_fastvqa_dir', default=None)
    parser.add_argument('--val_liqe_dir', default=None)
    parser.add_argument('--conv_base_lr', type=float, default=1e-5)
    parser.add_argument('--decay_ratio', type=float, default=0.95)
    parser.add_argument('--decay_interval', type=int, default=2)
    parser.add_argument('--results_path', type=str)
    parser.add_argument('--print_samples', type=int, default = 0)
    parser.add_argument('--train_batch_size', type=int, default=2,
                        help='Per-GPU batch size; global batch is this times world size')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--resize', type=int, default=384)
    parser.add_argument('--crop_size', type=int, default=384)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--ckpt_path', type=str, default='ckpts')
    parser.add_argument('--pretrained_path', type=str, default=None)
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--motion', action='store_true')
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=None)
    parser.add_argument('--n_exp', type=int, default=1)
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--loss_type', default='plcc', choices=['plcc'])

    
    parser.add_argument('--config', help='YAML configuration; command-line values override it')
    parser.add_argument('--finetune_path', default=None, help='Full RQ-VQA checkpoint; same LIQE setting required')
    parser.add_argument('--early_stop_patience', type=int, default=0, help='0 disables early stopping')
    parser.add_argument('--early_stop_min_delta', type=float, default=0.0)
    parser.add_argument('--weight_decay', type=float, default=1e-7)
    parser.add_argument('--backbone_lr', type=float, default=1e-6)
    parser.add_argument('--head_lr', type=float, default=1e-5)
    parser.add_argument('--freeze_backbone_epochs', type=int, default=3)
    parser.add_argument('--warmup_epochs', type=int, default=3)
    parser.add_argument('--min_lr_ratio', type=float, default=0.01)
    parser.add_argument('--grad_clip_norm', type=float, default=1.0)
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--amp_dtype', choices=['bfloat16', 'float16'], default='bfloat16')
    parser.add_argument('--amp_init_scale', type=float, default=1024.0)
    parser.add_argument('--max_nonfinite_steps', type=int, default=5)
    parser.add_argument('--sync_batchnorm', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--expected_world_size', type=int, default=1)
    parser.add_argument('--visible_devices', default=None, help='Physical CUDA device IDs, e.g. 2 or 0,1')
    parser.add_argument('--log_file', default=None)
    return parser
