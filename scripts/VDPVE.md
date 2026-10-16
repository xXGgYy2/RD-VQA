# VDPVE feature extraction

Run these commands inside your actual RQ-VQA repository on the Linux server.
Use this repository's FAST-VQA code: its evaluator returns a 768-dimensional
feature rather than a quality score.

```bash
conda activate rqvqa
python -m pip install 'numpy<2' 'timm==0.6.13' 'pytorchvideo==0.1.5' decord scikit-video pyyaml
# The bundled FAST-VQA module imports OpenAI CLIP even without LIQE.
python -m pip install ftfy regex tqdm 'git+https://github.com/openai/CLIP.git'

mkdir -p weights logs
wget -c --timeout=60 --tries=20 \
  'https://github.com/VQAssessment/FAST-VQA-and-FasterVQA/releases/download/v2.0.0/FAST_VQA_B_1_4.pth' \
  -O weights/FAST_VQA_B_1_4.pth

# Check paths and convert the three TXT files to filename,score CSVs first.
python scripts/extract_vdpve.py --prepare-only

# Default: frames + SlowFast + FAST-VQA for train, val, test, without LIQE.
set -o pipefail
CUDA_VISIBLE_DEVICES=0 python -u scripts/extract_vdpve.py 2>&1 | tee logs/vdpve_extract.log
```

SlowFast R50 weights download automatically at the first SlowFast stage. Unless
you set `TORCH_HOME`, the cache is `weights/torch` inside this repository. A failed
download stops the script; rerun after fixing connectivity. Valid feature outputs
are reused, but PyTorch's weight downloader does not guarantee partial download
resumption. There is no need to download Q-Align or the LSVQ initialization
checkpoint for feature extraction.

All three splits default to the video directory supplied by the user:
`/mnt/data/local-disk2/xyy/datasets/VDPVE/train`. If the validation/test files are
elsewhere, pass `--val-videos-dir` / `--test-videos-dir`; missing files cause an
error before extraction begins. The supplied split membership and MOS are kept
unchanged. Source IDs shared across splits produce a warning for review.

Outputs default to `/mnt/data/local-disk2/xyy/datasets/VDPVE/rqvqa_features`:

```text
annotations/train.csv, val.csv, test.csv
train/image_384p/<video>/000.png ... 007.png
train/SlowFast/<video>/feature_0_fast_feature.npy ... feature_7_fast_feature.npy
train/FASTVQA/<video>.npy
val/  (same layout)
test/ (same layout)
```

The original extractors may save more than eight positions; the current model
reads the first eight. SlowFast saves both pathways, while this RQ-VQA model reads
the 256-dimensional fast pathway. FAST-VQA contributes one 768-dimensional vector
per video. Resume checks validate images, feature dimensions and finite values;
use `--overwrite` or a different `--output` after changing source videos, weights
or extraction settings. Do not run simultaneous writers to the same output.

For optional LIQE (495-dimensional features), obtain LIQE.pt and run:

```bash
python -m pip install gdown
gdown --fuzzy 'https://drive.google.com/file/d/1GoKwUKNR-rvX11QbKRN8MuBZw2hXKHGh/view' -O weights/LIQE.pt
CUDA_VISIBLE_DEVICES=0 python -u scripts/extract_vdpve.py \
  --use-liqe --stages liqe --liqe-weights weights/LIQE.pt
```

This extracts original-resolution frames to each split's `image_original` and
features to `LIQE`. LIQE also loads/downloads OpenAI CLIP ViT-B/32 on first use.
Frames smaller than 224 pixels on either side are rejected by the output check.
To include LIQE from the first run, simply add `--use-liqe` to the default command.

Individual stages can be run via `--stages frames slowfast`, `--stages fastvqa`,
or `--use-liqe --stages liqe`. Default loader worker count is 1 to limit CPU RAM use.

This script only prepares inputs. The current training entry point still uses
the KVQ split and must be adapted to use these explicit VDPVE train/val CSVs before
training. Testing must use a newly trained checkpoint with the same LIQE setting.

Local lightweight checks (no model downloads):

```bash
python -m unittest discover -s tests -p test_vdpve_extraction.py -v
```
