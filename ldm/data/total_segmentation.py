import glob
import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    RandRotate90d,
    RandZoomd,
    ToTensord,
    AsDiscrete,
    Spacingd
)
from monai.data.dataset import Dataset as MonaiDataset

import json
class TotalSegmentatorDataset(Dataset):
    """
    TotalSegmentatorDataset class.
    """

    def __init__(self,
                 data_path: str = None,
                 mode: str = None,
                 patch_size: list = (96, 96, 96)
                 ) -> None:
        """

        Args:
            data_path:
            mode:
        """
        self.data_path = data_path
        

        self.patch_size = patch_size
        assert mode in ["train", "val", "test", None]

        
        # Load the JSON file
        with open(os.path.join(self.data_path, 'split_data.json'), "r") as f:
            self.data_list = json.load(f)[mode]
            
            # Update each dictionary to include basedir in image and label
            for item in self.data_list:
                item['image'] = os.path.join(data_path, item['image'])
                item['label'] = os.path.join(data_path, item['label'])
                del item['split']
        self.mode = mode

        # self.images = sorted(glob.glob(os.path.join(self.data_path, "*/*ct*.nii.gz"), recursive=True))
        # self.labels = sorted(glob.glob(os.path.join(self.data_path, "*/*segmentations*.nii.gz"), recursive=True))
        # print(self.images[0])
        # print(self.data[0])
        
        # assert len(self.images) == len(self.labels)

        train_transform = Compose(
            [
                LoadImaged(keys=["image", "label"],
                           reader='NibabelReader'
                           ),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"],
                             axcodes="RAS"),
                # Spacingd(
                #     keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")
                # ),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-1024,
                    a_max=1024,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                CropForegroundd(keys=["image", "label"],
                                source_key="image",
                                k_divisible=self.patch_size),
                RandZoomd(keys=["image", "label"],
                          min_zoom=1.3, max_zoom=1.5,
                          mode=["area", "nearest"],
                          prob=0.3),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=self.patch_size,
                    pos=1,
                    neg=1,
                    num_samples=1,
                    image_key="image",
                    image_threshold=0,
                ),
                RandRotate90d(
                    keys=["image", "label"],
                    prob=0.1,
                    max_k=3,
                ),
                RandShiftIntensityd(keys=["image"],
                                    offsets=0.10,
                                    prob=0.20),
                ToTensord(keys=["image", "label"]),
            ]
        )

        valid_transform = Compose(
            [
                LoadImaged(keys=["image", "label"],
                           reader='NibabelReader'
                           ),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"],
                             axcodes="RAS"),
                Spacingd(   
                    keys=["image", "label"], pixdim=(2.0, 2.0, 2.0), mode=("bilinear", "nearest")
                ),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-1024,
                    a_max=1024,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True
                ),
                CropForegroundd(keys=["image", "label"],
                                source_key="image"),
                ToTensord(keys=["image", "label"]),
            ]
        )

        test_transform = Compose(
            [
                LoadImaged(keys=["image", "label"],
                           reader='NibabelReader'
                           ),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"],
                             axcodes="RAS"),
                Spacingd(   
                    keys=["image", "label"], pixdim=(2.5, 2.5, 2.5), mode=("bilinear", "nearest")
                ),
                ToTensord(keys=["image", "label"]),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-1024,
                    a_max=1024,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True
                ),
            ]
        )
        if mode == 'train':
            self.data = MonaiDataset(data = self.data_list, transform=train_transform)
        elif mode == "val":
            self.data = MonaiDataset(data = self.data_list, transform=valid_transform)
        elif mode == "test":
            self.data =  MonaiDataset(data = self.data_list, transform=test_transform)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]
    
# TotalSegmentatorDataset('/root/datasets/Totalsegmentator', mode='train')