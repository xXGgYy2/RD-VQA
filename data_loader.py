import os

import pandas as pd
from PIL import Image

import torch
from torch.utils import data
import numpy as np
import cv2
import random


def _load_feature(path, dim):
    feature = torch.from_numpy(np.load(path)).float().squeeze()
    if feature.shape != (dim,):
        raise ValueError(f'{path}: expected {dim} features, got {tuple(feature.shape)}')
    return feature


class BVQA_VideoDataset(data.Dataset):
    """Return frames, SlowFast, FAST-VQA, optional LIQE, and video name."""
    def __init__(self, data_dir, data_dir_3D, data_dir_FASTVQA, filename_path,
                 transform, crop_size, use_liqe=False, data_dir_LIQE=None):
        if use_liqe and not data_dir_LIQE:
            raise ValueError('data_dir_LIQE is required when use_liqe=True')
        self.video_names = pd.read_csv(filename_path)['filename'].tolist()
        self.videos_dir = data_dir
        self.data_dir_3D = data_dir_3D
        self.data_dir_FASTVQA = data_dir_FASTVQA
        self.data_dir_LIQE = data_dir_LIQE
        self.use_liqe = use_liqe
        self.transform = transform
        self.crop_size = crop_size

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_name = self.video_names[idx]
        video_stem = os.path.splitext(video_name)[0]
        num_frames = 8
        frames = torch.empty(num_frames, 3, self.crop_size, self.crop_size)
        for i in range(num_frames):
            path = os.path.join(self.videos_dir, video_stem, f'{i:03d}.png')
            with Image.open(path) as frame:
                frames[i] = self.transform(frame.convert('RGB'))

        slowfast = torch.stack([
            _load_feature(os.path.join(self.data_dir_3D, video_stem,
                                      f'feature_{i}_fast_feature.npy'), 256)
            for i in range(num_frames)
        ])
        fastvqa_path = os.path.join(self.data_dir_FASTVQA, video_stem + '.npy')
        # The FAST-VQA extractor saves flat filenames; also accept mirrored CSV paths.
        if not os.path.isfile(fastvqa_path):
            fastvqa_path = os.path.join(self.data_dir_FASTVQA, os.path.basename(video_stem) + '.npy')
        fastvqa = _load_feature(fastvqa_path, 768).expand(num_frames, -1)
        inputs = (frames, slowfast, fastvqa)
        if self.use_liqe:
            liqe = torch.stack([
                _load_feature(os.path.join(self.data_dir_LIQE, video_stem,
                                          f'feature_{i}_LIQE_feature.npy'), 495)
                for i in range(num_frames)
            ])
            inputs += (liqe,)
        return (*inputs, video_name)


class BVQA_VideoDataset_RQ_VQA(BVQA_VideoDataset):
    """Labeled RQ-VQA data using the original KVQ content-group split."""
    def __init__(self, data_dir, data_dir_3D, data_dir_FASTVQA, filename_path,
                 transform, database_name, crop_size, seed=0,
                 use_liqe=False, data_dir_LIQE=None):
        super().__init__(data_dir, data_dir_3D, data_dir_FASTVQA, filename_path,
                         transform, crop_size, use_liqe, data_dir_LIQE)
        info = pd.read_csv(filename_path, header=0, names=['filename', 'score'],
                           index_col=False, encoding='utf-8-sig')
        if database_name in ('NTIREVideo_train', 'NTIREVideo_val'):
            # KVQ: 418 source groups, seven consecutive variants per group.
            if len(info) != 418 * 7:
                raise ValueError('The NTIREVideo train/val split requires 418 groups of 7 videos')
            groups = np.random.RandomState(seed).permutation(418)
            split = int(418 * 0.8)
            groups = groups[:split] if database_name == 'NTIREVideo_train' else groups[split:]
            indices = (groups[:, None] * 7 + np.arange(7)).reshape(-1)
            info = info.iloc[indices]
        elif database_name not in ('NTIREVideo', 'VDPVE', 'explicit'):
            raise ValueError(f'Unsupported database: {database_name}')
        self.video_names = info['filename'].tolist()
        self.score = info['score'].tolist()

    def __getitem__(self, idx):
        *inputs, video_name = super().__getitem__(idx)
        score = torch.tensor(float(self.score[idx]), dtype=torch.float32)
        return (*inputs, score, video_name)


class BVQA_VideoDataset_RQ_VQA_base_model(data.Dataset):
    """Read data from the original dataset for feature extraction"""
    def __init__(self, data_dir, data_dir_3D, filename_path, transform, database_name, crop_size, seed=0):
        super(BVQA_VideoDataset_RQ_VQA_base_model, self).__init__()


        column_names = ['filename','score']
        dataInfo = pd.read_csv(filename_path, header = 0, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
        video = dataInfo['filename'].tolist()
        score = dataInfo['score'].tolist()
        n = len(video)
        video_names = []
        for i in range(n):
            video_names.append(video[i])
        if database_name == 'NTIREVideo':
            self.video_names = video_names
            self.score = score
        else:
            dataInfo = pd.DataFrame(video_names)
            dataInfo['score'] = score
            dataInfo.columns = ['file_names', 'MOS']
            random.seed(seed)
            np.random.seed(seed)
            length = 418
            index_rd = np.random.permutation(length)
            train_index_ref = index_rd[0:int(length * 0.8)]
            # do not use the validation set
            val_index_ref = index_rd[int(length * 0.8):]
            train_index = []
            for i_ref in train_index_ref:
                for i_dis in range(7):
                    train_index.append(i_ref*7+i_dis)
            val_index = []
            for i_ref in val_index_ref:
                for i_dis in range(7):
                    val_index.append(i_ref*7+i_dis)

            print('train_index')
            print(train_index)
            print('val_index')
            print(val_index)
            if database_name == 'NTIREVideo_train':
                self.video_names = dataInfo.iloc[train_index]['file_names'].tolist()
                self.score = dataInfo.iloc[train_index]['MOS'].tolist()
            elif database_name == 'NTIREVideo_val':
                self.video_names = dataInfo.iloc[val_index]['file_names'].tolist()
                self.score = dataInfo.iloc[val_index]['MOS'].tolist()

        self.crop_size = crop_size
        self.videos_dir = data_dir
        self.data_dir_3D = data_dir_3D
        self.transform = transform
        self.length = len(self.video_names)
        self.database_name = database_name

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        video_name = self.video_names[idx]
        video_name_str = video_name[:-4]        
        

        video_score = torch.FloatTensor(np.array(float(self.score[idx])))

        path_name = os.path.join(self.videos_dir, video_name_str)

        video_channel = 3

        video_height_crop = self.crop_size
        video_width_crop = self.crop_size
       


        video_length_read = 8


        transformed_video = torch.zeros([video_length_read, video_channel, video_height_crop, video_width_crop])             


        for i in range(video_length_read):
            imge_name = os.path.join(path_name, '{:03d}'.format(i) + '.png')
            read_frame = Image.open(imge_name)
            read_frame = read_frame.convert('RGB')
            read_frame = self.transform(read_frame)
            transformed_video[i] = read_frame

        

        feature_folder_name = os.path.join(self.data_dir_3D, video_name_str)
        transformed_feature = torch.zeros([video_length_read, 256])
        for i in range(video_length_read):
            feature_3D = np.load(os.path.join(feature_folder_name, 'feature_' + str(int(i)) + '_fast_feature.npy'))
            feature_3D = torch.from_numpy(feature_3D)
            feature_3D = feature_3D.squeeze()
            transformed_feature[i] = feature_3D


        return transformed_video, transformed_feature, video_score, video_name


class BVQA_VideoDataset_SlowFast(data.Dataset):
    """Read data from the original dataset for feature extraction"""
    def __init__(self, data_dir, filename_path, transform, resize):
        super(BVQA_VideoDataset_SlowFast, self).__init__()


        dataInfo = pd.read_csv(filename_path)
        video = dataInfo['filename'].tolist()

        video_names = []
        for video_i in video:
            video_names.append(video_i)
        self.video_names = video_names



        self.transform = transform           
        self.videos_dir = data_dir
        self.resize = resize
        self.length = len(self.video_names)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        video_name = self.video_names[idx]
        video_name_str = video_name[:-4]

        # Accept both a directory containing videos directly and the original
        # NTIRE layout where videos are under ``test/``.
        filename = os.path.join(self.videos_dir, video_name)
        if not os.path.isfile(filename):
            filename = os.path.join(self.videos_dir, 'test', video_name)
        print(filename)

        video_capture = cv2.VideoCapture()
        video_capture.open(filename)
        cap=cv2.VideoCapture(filename)

        video_channel = 3
        
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))
        cap.release()
        if video_length <= 0 or not video_capture.isOpened():
            video_capture.release()
            raise ValueError(f'Cannot read video: {filename}')

        if video_frame_rate <= 0:
            video_capture.release()
            raise ValueError(f'Invalid frame rate: {filename}')
        video_clip = max(1, int(video_length/video_frame_rate))


        video_clip_min = 8

        video_length_clip = 32             

        transformed_frame_all = torch.zeros([video_length, video_channel, self.resize, self.resize])

        transformed_video_all = []
        
        video_read_index = 0
        for i in range(video_length):
            has_frames, frame = video_capture.read()
            if has_frames:
                read_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                read_frame = self.transform(read_frame)
                transformed_frame_all[video_read_index] = read_frame
                video_read_index += 1


        video_capture.release()
        if video_read_index == 0:
            raise ValueError(f'No decoded frames: {filename}')
        if video_read_index < video_length:
            for i in range(video_read_index, video_length):
                transformed_frame_all[i] = transformed_frame_all[video_read_index - 1]
 

        for i in range(video_clip):
            transformed_video = torch.zeros([video_length_clip, video_channel, self.resize, self.resize])
            if (i*video_frame_rate + video_length_clip) <= video_length:
                transformed_video = transformed_frame_all[i*video_frame_rate : (i*video_frame_rate + video_length_clip)]
            else:
                transformed_video[:(video_length - i*video_frame_rate)] = transformed_frame_all[i*video_frame_rate :]
                for j in range((video_length - i*video_frame_rate), video_length_clip):
                    transformed_video[j] = transformed_video[video_length - i*video_frame_rate - 1]
            transformed_video_all.append(transformed_video)

        if video_clip < video_clip_min:
            for i in range(video_clip, video_clip_min):
                transformed_video_all.append(transformed_video_all[video_clip - 1])
       
        return transformed_video_all, video_name_str
