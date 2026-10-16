# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torchvision import transforms

from data_loader import BVQA_VideoDataset
import models


def load_checkpoint(model, path):
    state_dict = torch.load(path, map_location='cpu')
    # Accept checkpoints saved with or without DataParallel.
    state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    weight = state_dict.get('quality.0.weight')
    if weight is None or weight.shape != model.quality[0].weight.shape:
        raise ValueError(f'{path}: incompatible checkpoint; retrain without Q-Align '
                         'and use the same --use_liqe setting as training')
    model.load_state_dict(state_dict)


def main(config):
    if config.use_liqe and not config.liqe_dir:
        raise ValueError('--liqe_dir is required with --use_liqe')
    if config.crop_size != 384 or config.resize < config.crop_size:
        raise ValueError('RQ_VQA requires --crop_size 384 and --resize >= 384')
    checkpoint_path = Path(config.pretrained_path)
    if checkpoint_path.is_dir():
        tag = 'LIQE' if config.use_liqe else 'noLIQE'
        model_files = sorted(checkpoint_path.glob(f'RQ_VQA_{tag}_*.pth'))
    else:
        model_files = [checkpoint_path] if checkpoint_path.is_file() else []
    if not model_files:
        raise ValueError('No matching RQ_VQA checkpoints found')
    if config.n_exp is not None:
        if not 1 <= config.n_exp <= len(model_files):
            raise ValueError('--n_exp must be between 1 and the number of matching checkpoints')
        model_files = model_files[:config.n_exp]

    transform = transforms.Compose([
        transforms.Resize(config.resize), transforms.CenterCrop(config.crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    testset = BVQA_VideoDataset(
        config.videos_dir, config.slowfast_dir, config.fastvqa_dir, config.datainfo,
        transform, config.crop_size, use_liqe=config.use_liqe, data_dir_LIQE=config.liqe_dir,
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=1, shuffle=False, num_workers=config.num_workers,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda' and config.gpu_ids:
        device = torch.device(f'cuda:{config.gpu_ids[0]}')
    # Full checkpoints already contain Swin weights; do not download ImageNet weights.
    model = models.RQ_VQA(use_liqe=config.use_liqe, pretrained=False).to(device)
    inference_model = model
    if device.type == 'cuda' and config.gpu_ids and len(config.gpu_ids) > 1:
        inference_model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)

    predictions = []
    for path in model_files:
        print(f'Loading {path}')
        load_checkpoint(model, path)
        inference_model.eval()
        scores = []
        with torch.no_grad():
            for *inputs, _ in test_loader:
                scores.append(inference_model(*[feature.to(device) for feature in inputs]).item())
        predictions.append(scores)

    result = pd.DataFrame({'filename': testset.video_names, 'score': np.mean(predictions, axis=0)})
    Path(config.save_file).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(config.save_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_path', required=True, help='Checkpoint file or directory')
    parser.add_argument('--save_file', default='results.csv')
    parser.add_argument('--use_liqe', action='store_true')
    parser.add_argument('--datainfo', default='data/test_data.csv')
    parser.add_argument('--videos_dir', default='test_image_384p')
    parser.add_argument('--slowfast_dir', default='features/SlowFast')
    parser.add_argument('--fastvqa_dir', default='features/FASTVQA')
    parser.add_argument('--liqe_dir', default=None)
    parser.add_argument('--resize', type=int, default=384)
    parser.add_argument('--crop_size', type=int, default=384)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--n_exp', type=int, default=None, help='Default: use all matching checkpoints')
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=None)
    parser.add_argument('--random_seed', type=int, default=8)
    config = parser.parse_args()

    torch.manual_seed(config.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)
    main(config)
