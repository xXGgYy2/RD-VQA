import csv
import hashlib
import math
import os
from collections import defaultdict
from pathlib import Path

SOURCE = Path('/mnt/data/local-disk3/xyy/dataset/RD-VQA-Feature/FineVD')
LABELS = Path('/mnt/data/local-disk3/xyy/dataset/blance-ABCDE/datasets/finevd')
OUTPUT = Path('/mnt/data/local-disk3/xyy/dataset/RD-VQA-Feature/FineVD_blance_ABCDE')

if not SOURCE.is_dir():
    raise SystemExit(f'原特征目录不存在: {SOURCE}')
for split in ('train', 'val', 'test'):
    if not (LABELS / f'{split}.txt').is_file():
        raise SystemExit(f'标签文件不存在: {LABELS / (split + ".txt")}')

# 搜索各旧划分下的完整特征，保留原目录用于定位。
index = defaultdict(list)
for frames in sorted(SOURCE.rglob('image_384p')):
    if not frames.is_dir():
        continue
    root = frames.parent
    for first in sorted(frames.rglob('000.png')):
        relative = first.parent.relative_to(frames)
        stem = relative.name
        files = [
            (first.parent / f'{i:03d}.png',
             Path('image_384p') / stem / f'{i:03d}.png')
            for i in range(8)
        ]
        files += [
            (root / 'SlowFast' / relative / f'feature_{i}_fast_feature.npy',
             Path('SlowFast') / stem / f'feature_{i}_fast_feature.npy')
            for i in range(8)
        ]
        fast = root / 'FASTVQA' / (str(relative) + '.npy')
        if not fast.is_file():
            fast = root / 'FASTVQA' / (stem + '.npy')
        files.append((fast, Path('FASTVQA') / (stem + '.npy')))
        if all(src.is_file() for src, _ in files):
            index[stem].append(files)

print(f'发现 {len(index)} 个具有完整特征的视频名', flush=True)

hash_cache = {}
def digest(path):
    path = path.resolve()
    if path not in hash_cache:
        h = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                h.update(block)
        hash_cache[path] = h.digest()
    return hash_cache[path]

plan = []
owners = {}
errors = []
for split in ('train', 'val', 'test'):
    label = LABELS / f'{split}.txt'
    count = 0
    with label.open(encoding='utf-8-sig', newline='') as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            fields = [x.strip() for x in next(csv.reader([line]))]
            if len(fields) != 4:
                raise ValueError(
                    f'{label}:{line_no}: 需要四列：绝对路径,-1,-1,mos'
                )
            video = Path(fields[0])
            if not video.is_absolute() or not video.suffix:
                raise ValueError(f'{label}:{line_no}: 视频路径格式错误')
            if not math.isfinite(float(fields[-1])):
                raise ValueError(f'{label}:{line_no}: MOS 无效')
            stem = video.stem
            if stem in owners:
                errors.append(
                    f'重复视频或同名冲突: {owners[stem]} <-> {split}: {video}'
                )
                continue
            owners[stem] = f'{split}: {video}'
            matches = index.get(stem, [])
            if not matches:
                errors.append(f'缺少完整特征: {split}: {video}')
                continue
            chosen = matches[0]
            conflict = False
            for other in matches[1:]:
                if any(
                    a.resolve() != b.resolve() and digest(a) != digest(b)
                    for (a, _), (b, _) in zip(chosen, other)
                ):
                    locations = ', '.join(str(m[0][0].parent) for m in matches)
                    errors.append(f'重复特征内容不一致: {stem}: {locations}')
                    conflict = True
                    break
            if conflict:
                continue
            for src, relative in chosen:
                src = src.resolve()
                dst = OUTPUT / split / relative
                if os.path.lexists(dst):
                    if not dst.is_symlink() or dst.resolve() != src:
                        errors.append(f'目标已存在且不是预期链接: {dst}')
                plan.append((src, dst))
            count += 1
    if count == 0:
        errors.append(f'{split}: 没有可用视频')
    print(f'{split}: 匹配 {count} 个视频', flush=True)

if errors:
    print('\n检查失败，本次未创建新链接：')
    for error in errors[:50]:
        print(' -', error)
    raise SystemExit(f'共 {len(errors)} 个问题，请先解决后重跑。')

for src, dst in plan:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not os.path.lexists(dst):
        dst.symlink_to(src)

print(f'完成：{len(plan)} 个特征文件链接')
print(f'新特征目录：{OUTPUT}')