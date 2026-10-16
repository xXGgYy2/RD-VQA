# -*- coding: utf-8 -*-
import os
if __name__ == '__main__':
    from train_options import parse_config
    config = parse_config()
    if config.visible_devices is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = config.visible_devices

import json
import math
import random
import time
from pathlib import Path
import numpy as np
import scipy.io as scio
import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from data_loader import BVQA_VideoDataset_RQ_VQA
from early_stopping import EarlyStopping
from utils import performance_fit, plcc_loss
import models


def ctx(cfg):
    world = int(os.getenv('WORLD_SIZE', '1'))
    rank = int(os.getenv('RANK', '0'))
    local = int(os.getenv('LOCAL_RANK', '0'))
    if world > 1:
        if not torch.cuda.is_available():
            raise RuntimeError('DDP requires CUDA')
        torch.cuda.set_device(local)
        dist.init_process_group('nccl')
        device = torch.device('cuda', local)
    elif torch.cuda.is_available():
        index = cfg.gpu_ids[0] if cfg.gpu_ids else 0
        torch.cuda.set_device(index)
        device = torch.device('cuda', index)
    else:
        device = torch.device('cpu')
    return world, rank, local, device


def main_only(rank, *args, **kwargs):
    if rank == 0:
        print(*args, **kwargs, flush=True)


def set_backbone(model, enabled):
    for parameter in model.feature_extraction.parameters():
        parameter.requires_grad = enabled


def optimizer_for(model, cfg):
    backbone = list(model.feature_extraction.parameters())
    ids = {id(p) for p in backbone}
    head = [p for p in model.parameters() if id(p) not in ids]
    return optim.Adam([
        {'params': backbone, 'lr': cfg.backbone_lr},
        {'params': head, 'lr': cfg.head_lr},
    ], weight_decay=cfg.weight_decay)


def make_scheduler(opt, cfg):
    def rate(epoch):
        if epoch < cfg.warmup_epochs:
            return max(1e-8, (epoch + 1) / cfg.warmup_epochs)
        total = max(1, cfg.epochs - cfg.warmup_epochs)
        progress = min(1., max(0., (epoch - cfg.warmup_epochs) / total))
        return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * .5 * (1 + math.cos(math.pi * progress))
    return optim.lr_scheduler.LambdaLR(opt, rate)


def global_loss(y, pred, distributed):
    if distributed:
        y = torch.cat(dist_nn.all_gather(y.float()))
        pred = torch.cat(dist_nn.all_gather(pred.float()))
    # utils.plcc_loss expects predictions first. The legacy training loop passed
    # these in reverse order, which made the asymmetric second term optimize MOS.
    return plcc_loss(pred, y)


def datasets(cfg, seed):
    tr = transforms.Compose([transforms.Resize(cfg.resize), transforms.RandomCrop(cfg.crop_size),
                             transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    va = transforms.Compose([transforms.Resize(cfg.resize), transforms.CenterCrop(cfg.crop_size),
                             transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    common = dict(data_dir=cfg.videos_dir, data_dir_3D=cfg.slowfast_dir,
                  data_dir_FASTVQA=cfg.fastvqa_dir, filename_path=cfg.datainfo,
                  crop_size=cfg.crop_size, seed=seed, use_liqe=cfg.use_liqe,
                  data_dir_LIQE=cfg.liqe_dir)
    if cfg.val_datainfo:
        train = BVQA_VideoDataset_RQ_VQA(transform=tr, database_name='explicit', **common)
        val = BVQA_VideoDataset_RQ_VQA(database_name='explicit',
            filename_path=cfg.val_datainfo, data_dir=cfg.val_videos_dir or cfg.videos_dir,
            data_dir_3D=cfg.val_slowfast_dir or cfg.slowfast_dir,
            data_dir_FASTVQA=cfg.val_fastvqa_dir or cfg.fastvqa_dir,
            data_dir_LIQE=cfg.val_liqe_dir or cfg.liqe_dir, transform=va,
            crop_size=cfg.crop_size, seed=seed, use_liqe=cfg.use_liqe)
    else:
        train = BVQA_VideoDataset_RQ_VQA(transform=tr, database_name='NTIREVideo_train', **common)
        val = BVQA_VideoDataset_RQ_VQA(transform=va, database_name='NTIREVideo_val', **common)
    if cfg.val_datainfo and set(train.video_names) & set(val.video_names):
        raise ValueError('Training and validation videos overlap')
    return train, val


def validate(model, loader, device):
    model.eval(); y = np.zeros(len(loader.dataset)); p = np.zeros(len(loader.dataset))
    with torch.no_grad():
        for i, (*inputs, mos, _) in enumerate(loader):
            p[i] = model(*[x.to(device, non_blocking=True) for x in inputs]).item(); y[i] = mos.item()
    return performance_fit(y, p)


def experiment(cfg, exp, world, rank, local, device):
    seed = (exp + 1) * cfg.random_seed if cfg.random_seed else exp
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed); random.seed(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    train, val = datasets(cfg, seed)
    sampler = DistributedSampler(train, num_replicas=world, rank=rank, shuffle=True, seed=seed) if world > 1 else None
    loader = torch.utils.data.DataLoader(train, batch_size=cfg.train_batch_size, sampler=sampler,
        shuffle=sampler is None, num_workers=cfg.num_workers, pin_memory=device.type == 'cuda',
        persistent_workers=cfg.num_workers > 0, drop_last=True)
    if not len(loader): raise ValueError('Per-GPU batch is too large for this training split')
    vloader = torch.utils.data.DataLoader(val, batch_size=1, shuffle=False, num_workers=cfg.num_workers,
        pin_memory=device.type == 'cuda', persistent_workers=cfg.num_workers > 0) if rank == 0 else None
    model = models.RQ_VQA(cfg.pretrained_path, use_liqe=cfg.use_liqe, pretrained=not bool(cfg.finetune_path))
    if cfg.finetune_path:
        from test import load_checkpoint
        load_checkpoint(model, cfg.finetune_path)
    model.to(device)
    if world > 1 and cfg.sync_batchnorm:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    raw = model
    opt = optimizer_for(raw, cfg); scheduler = make_scheduler(opt, cfg)
    if world > 1:
        model = DDP(model, device_ids=[local], output_device=local,
                    find_unused_parameters=cfg.freeze_backbone_epochs > 0)
    amp = cfg.amp and device.type == 'cuda'
    amp_dtype = torch.bfloat16 if cfg.amp_dtype == 'bfloat16' else torch.float16
    if amp and amp_dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError('This GPU/PyTorch build does not support bfloat16; set amp_dtype: float16')
    scaler_enabled = amp and amp_dtype is torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled, init_scale=cfg.amp_init_scale)
    stopper = EarlyStopping(cfg.early_stop_patience, cfg.early_stop_min_delta)
    best, best_path = -float('inf'), None
    for epoch in range(cfg.epochs):
        if sampler: sampler.set_epoch(epoch)
        enabled = epoch >= cfg.freeze_backbone_epochs; set_backbone(raw, enabled)
        model.train()
        if not enabled: raw.feature_extraction.eval()
        total = 0.; completed_steps = 0; skipped_steps = 0
        for inputs in loader:
            *features, labels, _ = inputs
            features = [x.to(device, non_blocking=True) for x in features]; labels = labels.to(device).float()
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp, dtype=amp_dtype):
                loss = global_loss(labels, model(*features), world > 1)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite loss at epoch {epoch + 1}; check MOS and features')
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            norm = torch.nn.utils.clip_grad_norm_(raw.parameters(), cfg.grad_clip_norm)
            finite = torch.tensor(int(torch.isfinite(norm)), device=device)
            if world > 1:
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite.item():
                skipped_steps += 1
                # FP16 GradScaler records overflow during unscale_ and reduces
                # its scale here. BF16 skips the bad update without corrupting weights.
                if scaler_enabled:
                    scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)
                main_only(rank, f'Warning: skipped non-finite gradient at epoch '
                                  f'{epoch + 1} (count={skipped_steps})')
                if skipped_steps > cfg.max_nonfinite_steps:
                    raise RuntimeError('Too many consecutive non-finite gradient steps')
                continue
            scaler.step(opt); scaler.update(); total += loss.item(); completed_steps += 1
            skipped_steps = 0
        scheduler.step()
        if completed_steps == 0:
            raise RuntimeError(f'No finite optimizer step completed in epoch {epoch + 1}')
        if world > 1:
            stat = torch.tensor([total, completed_steps], dtype=torch.float64, device=device); dist.all_reduce(stat)
            train_loss = (stat[0] / stat[1]).item()
        else: train_loss = total / completed_steps
        main_only(rank, f'Epoch {epoch+1}/{cfg.epochs}: loss={train_loss:.4f}, '
                        f'lr={[g["lr"] for g in opt.param_groups]}, backbone={enabled}, '
                        f'amp={cfg.amp_dtype if amp else "off"}')
        stop = False
        if rank == 0:
            plcc, srcc, krcc, rmse = validate(raw, vloader, device)
            print(f'Val SRCC={srcc:.4f}, KRCC={krcc:.4f}, PLCC={plcc:.4f}, RMSE={rmse:.4f}', flush=True)
            if np.isfinite(srcc) and srcc > best:
                best = srcc; path = Path(cfg.ckpt_path) / f'RQ_VQA_{"LIQE" if cfg.use_liqe else "noLIQE"}_{cfg.database}_plcc_NR_v{exp}_epoch_{epoch+1}_SRCC_{srcc:.6f}.pth'
                torch.save(raw.state_dict(), path)
                if best_path and Path(best_path).exists(): Path(best_path).unlink()
                best_path = str(path); print(f'Best checkpoint: {best_path}', flush=True)
            stop = stopper.step(srcc)
        if world > 1:
            flag = torch.tensor(int(stop), device=device); dist.broadcast(flag, 0); stop = bool(flag.item())
        if stop: break
    if world > 1: dist.barrier()
    return best_path if rank == 0 else None


def main(cfg):
    world, rank, local, device = ctx(cfg)
    if world != cfg.expected_world_size:
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()
        raise ValueError(f'Expected {cfg.expected_world_size} process(es), got {world}. '
                         f'Launch with torchrun --nproc_per_node={cfg.expected_world_size}.')
    if cfg.multi_gpu: raise ValueError('Use torchrun for DDP; multi_gpu must be false')
    if (cfg.train_batch_size < 1 or cfg.train_batch_size * world < 2 or
            cfg.crop_size != 384 or cfg.resize < 384):
        raise ValueError('PLCC requires global batch >=2; crop_size=384 and resize>=384')
    if cfg.warmup_epochs >= cfg.epochs: raise ValueError('warmup_epochs must be less than epochs')
    if cfg.freeze_backbone_epochs >= cfg.epochs: raise ValueError('freeze_backbone_epochs must be less than epochs')
    if rank == 0:
        from train_options import setup_log, validate_explicit_data
        setup_log(cfg.log_file)
        if cfg.val_datainfo: validate_explicit_data(cfg)
        Path(cfg.ckpt_path).mkdir(parents=True, exist_ok=True)
        Path(cfg.ckpt_path, 'resolved_config.json').write_text(
            json.dumps(dict(vars(cfg), world_size=world,
                            global_batch_size=cfg.train_batch_size * world),
                       indent=2, default=str))
    if world > 1: dist.barrier()
    paths = [experiment(cfg, i, world, rank, local, device) for i in range(cfg.n_exp)]
    if rank == 0:
        scio.savemat(Path(cfg.ckpt_path) / f'{cfg.model_name}_{cfg.database}_plcc.mat', {'save_model_name_all': paths})
    if world > 1: dist.destroy_process_group()


if __name__ == '__main__': main(config)
