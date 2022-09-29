import random
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import cv2 
import numpy as np
from get_masked_face_simple import get_masked_face_simple

import uuid
from albumentations import Compose, RandomBrightnessContrast, \
    HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, \
    ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur, Rotate

from transforms.albu import IsotropicResize

class DeepFakesDataset(Dataset):
    def __init__(self, videos, labels, video_name, tempdir, 
                image_size, mask_method, mask_number, picture_color, 
                aug_mode,
                mode = 'train'):
        self.x = videos
        self.y = torch.from_numpy(labels)
        self.video_name = video_name
        self.tempdir = tempdir
        self.image_size = image_size
        self.mask_method = mask_method
        self.mask_number = mask_number
        self.picture_color = picture_color
        self.aug_mode = aug_mode
        self.mode = mode
        self.n_samples = videos.shape[0]
    
    def create_train_transforms(self, size):
        return Compose([
            ImageCompression(quality_lower=60, quality_upper=100, p=0.2),
            GaussNoise(p=0.3),
            GaussianBlur(blur_limit=(1, 3), p=0.05),
            HorizontalFlip(),
            OneOf([
                IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
                IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
                IsotropicResize(max_side=size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
            ], p=1),
            PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
            OneOf([RandomBrightnessContrast(), FancyPCA(), HueSaturationValue()], p=0.4),
            ToGray(p=0.2),
            ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=5, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        ]
        )

    def create_train_transforms_smiple(self, size):
        return Compose([
            OneOf([
                IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
                IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
                IsotropicResize(max_side=size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
            ], p=1),
            PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
            ]
        )
        
    def create_val_transform(self, size):
        return Compose([
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
            PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
        ])

    def __getitem__(self, index):
        #video is a list of frames
        video = self.x[index]
        video = list(map(np.asarray, video))
        name = self.video_name[index]

        if self.mode == 'train':
            if self.aug_mode == 'yes':
                transform = self.create_train_transforms(self.image_size)
            else:
                transform = self.create_train_transforms_smiple(self.image_size)
        else:
            transform = self.create_val_transform(self.image_size)
                
        #unique = uuid.uuid4()
        
        video = list(map(lambda f: transform(image=f)['image'], video))
        #self.mode == 'train' and 
        if self.mask_method != 'none':# 目前的方法是对一个视频内的所有人脸图像采取同样的掩码区域。
            random_list = [i for i in range(0, 8)]
            random.shuffle(random_list)
            for i in range(0, len(video)):
                video[i] = get_masked_face_simple(
                video[i],
                self.tempdir,
                name + "_" + str(i),
                random_list=random_list, 
                mask_method=self.mask_method, 
                mask_number=self.mask_number)
            # video = list(map(lambda f: get_masked_face_simple(
            # input_img=f, 
            # random_list=random_list, 
            # mask_method=self.mask_method, 
            # mask_number=self.mask_number), video))
        
        if self.picture_color == 'gray':
            video = list(map(lambda f: cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), video))
            video = list(map(lambda f: cv2.cvtColor(f, cv2.COLOR_GRAY2RGB), video))
        
        #cv2.imwrite("data/dataset/aug_frames/"+str(unique)+"_"+str(index)+".png", video[0])
        
        # 将视频帧在通道维度拼接起来，并进行一些细节转换操作。
        # 最后video的shape为（通道数，帧数目，hight， wight）。
        video = np.concatenate(video, axis=-1)
        video = torch.from_numpy(video).permute(2, 0, 1).contiguous().float()
        video = video.view(-1,3,video.size(1),video.size(2)).permute(1,0,2,3)

        return video, self.y[index]


    def __len__(self):
        return self.n_samples

 