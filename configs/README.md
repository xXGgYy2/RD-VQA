# Configured fine-tuning

```bash
torchrun --standalone --nproc_per_node=4 train.py --config configs/vdpve_no_liqe.yaml
# Background; log_file in YAML receives stdout and stderr.
nohup torchrun --standalone --nproc_per_node=4 train.py \
  --config configs/vdpve_no_liqe.yaml > /dev/null 2>&1 &
tail -f logs/VDPVE_noLIQE_earlystop.log
```

Install `pyyaml` if needed. The config path is relative to your shell directory;
paths inside the config are relative to the repository root. `${feature_root}`
avoids repeating the dataset root. Unknown fields and invalid types fail early.
Explicit splits are checked for missing frames/features before model initialization.
`train_batch_size` is per GPU. With four processes and batch 2, the global batch is
8; the PLCC loss gathers all eight predictions with autograd before computing the
loss. `expected_world_size` prevents accidentally launching a four-GPU config with
plain `python` and putting the whole batch on one card.
If batch 2 still exceeds memory, batch 1 per GPU is supported in DDP and gives a
global PLCC batch of 4. Do not use batch 1 in single-process training.
Command-line arguments override YAML values, for example `--epochs 2`.
For boolean fields, use the YAML true/false value (LIQE can also be disabled with
`--no-use_liqe`).

To fine-tune a new dataset, copy the YAML and change `database`, `feature_root`,
train/val CSVs and feature paths, `ckpt_path` and `log_file`. The CSV format is
`filename,score` with a header. Provide explicit train/val lists, separated by
source content; the script rejects overlapping video filenames but cannot infer
related source content from arbitrary filenames. Test data is not used for
training or early stopping. The current AIGC-test set is a test set; this change
does not automatically convert it into training data or generate new splits.
`configs/finevd_4gpu_template.yaml` is a four-GPU full-model fine-tuning template.

There are two distinct initialization choices:

```yaml
# Initialize Swin from the original LSVQ base checkpoint.
pretrained_path: weights/Swin_b_384_in22k_SlowFast_Fast_LSVQ.pth
finetune_path: null
```

```yaml
# Initialize ALL current RQ-VQA parameters from the trained VDPVE model.
pretrained_path: null
finetune_path: /mnt/data/local-disk3/xyy/RQ-VQA/ckpts/VDPVE_noLIQE/RQ_VQA_noLIQE_VDPVE_plcc_NR_v0_epoch_8_SRCC_0.712248.pth
```

`finetune_path` loads strictly, including optional DataParallel prefixes. LIQE
must match the source checkpoint. It starts a new optimizer and learning-rate
schedule, not a resume from the old epoch. Omitting both weights uses ImageNet
initialization and may download Swin weights. LSVQ loading only retains the Swin
branch; BoT and the new fusion regression head are newly initialized.

SlowFast, FAST-VQA and LIQE remain frozen offline feature extractors. Swin,
BoT and the regression head are trainable. Swin is frozen for
`freeze_backbone_epochs`, then unfrozen at `backbone_lr`; BoT and the regression
head use `head_lr`. Gradients are clipped by `grad_clip_norm`. Linear warm-up is
followed by cosine decay to `min_lr_ratio` of each group's base learning rate.
AMP reduces activation memory, and SyncBatchNorm combines BoT batch statistics
across DDP workers. Enabling LIQE requires precomputed
features for BOTH train and val.
L20 training defaults to BF16 because it has a wider numeric range than FP16.
With `amp_dtype: float16`, GradScaler starts at `amp_init_scale`; overflowed
steps are skipped and the scale is reduced instead of terminating immediately.

Early stopping monitors validation SRCC (higher is better): an improvement must
exceed `early_stop_min_delta` over the last meaningful best to reset the counter.
`early_stop_patience: 0` disables stopping. Non-finite SRCC counts as a failed
improvement. Checkpoint saving independently tracks EVERY strictly higher finite
SRCC, so a small improvement is still saved even if it does not reset patience.
The epoch limit always applies. Each experiment keeps its best checkpoint; saved
names include epoch and SRCC. Files from older runs are not automatically deleted;
use a new output directory for each run. `resolved_config.json` records the
effective settings. The existing test.py can load the saved .pth directly.

Logs append on reruns. With configured `log_file`, do not also redirect the shell
to that same file. GPU IDs in `visible_devices` are physical; `gpu_ids` are logical
indices within those visible devices.

```bash
python -m unittest discover -s tests -p test_training_config.py -v
python -m unittest discover -s tests -v
```
