# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

from monai import data, transforms
from torch.utils.data import Dataset
from monai.data import Dataset as MonaiDataset

def get_brats_seg_fold_dataset(json_path, fold=1, roi_size = [96, 96, 96], is_val = False):
        
    transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.CropForegroundd(
                keys=["image", "label"], source_key="image", k_divisible=roi_size
            ),
            transforms.RandSpatialCropd(
                keys=["image", "label"], roi_size=roi_size, random_size=False
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
        # Function to load specific fold data
    def load_fold_data(fold_idx, json_file):
        # Load the JSON file
        with open(json_file, 'r') as f:
            folds = json.load(f)
        
        # Access the specific fold
        fold_key = f'fold_{fold_idx}'
        
        if fold_key in folds:
            train_files = folds[fold_key]['val' if is_val else "train"]
            return train_files
        else:
            raise ValueError(f"Fold {fold_idx} does not exist in the provided file.")

    train_files = load_fold_data(fold, json_path)

    print(f"Files for fold {fold}: {len(train_files)}")


    return MonaiDataset(data=train_files, transform=val_transform if is_val else transform)
        
class BratsSegFoldDataset(Dataset):
    
    def __init__(self,data_path, fold=1, roi_size = [96, 96, 96]):
        super().__init__()
        self.data = get_brats_seg_fold_dataset(data_path, fold=fold, roi_size=roi_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]
        
class BratsSegValFoldDataset(Dataset):
    
    def __init__(self,data_path, fold=1, roi_size = [96, 96, 96]):
        super().__init__()
        self.data = get_brats_seg_fold_dataset(data_path, fold=fold, roi_size=roi_size, is_val=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]
    