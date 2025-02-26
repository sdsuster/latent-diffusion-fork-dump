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

import math
import os

import numpy as np
import torch

from monai import data, transforms
from monai.data import load_decathlon_datalist
import torch.distributed
import torch.utils
from numpy import random
import torch.utils.data

class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank : self.total_size : self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def get_dataset(data_dir, json_list, crop_size, test_mode = False, is_train = False, repeat = 0):
    data_dir = data_dir
    datalist_json = os.path.join(data_dir, json_list)
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")
            ),
            transforms.EnsureTyped(keys=["image", "label"], track_meta=False),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-175.0, a_max=250.0, b_min=0.0, b_max=1.0, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(crop_size[0], crop_size[1], crop_size[2]),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            transforms.RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=0.3),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.3),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-175.0, a_max=250.0, b_min=0.0, b_max=1.0, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            # transforms.Orientationd(keys=["image"], axcodes="RAS"),
            transforms.Spacingd(keys="image", pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-175.0, a_max=250.0, b_min=0.0, b_max=1.0, clip=True
            ),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    if test_mode:
        test_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)

                
        extended_files = test_files[:]

        for _ in range(repeat):  # Repeat 20 times
            shuffled = test_files[:]  # Copy the list
            random.shuffle(shuffled)  # Shuffle the copy
            extended_files.extend(shuffled)  # Add to final list
        test_ds = data.Dataset(data=extended_files, transform=test_transform)
        # test_sampler = Sampler(test_ds, shuffle=False) if args.distributed else None
        # test_loader = data.DataLoader(
        #     test_ds,
        #     batch_size=1,
        #     shuffle=False,
        #     num_workers=args.workers,
        #     sampler=test_sampler,
        #     pin_memory=True,
        #     persistent_workers=True,
        # )
        ds = test_ds
    else:
        if is_train:
            datalist = load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir)
            extended_files = datalist[:]

            for _ in range(repeat):  # Repeat 20 times
                shuffled = datalist[:]  # Copy the list
                random.shuffle(shuffled)  # Shuffle the copy
                extended_files.extend(shuffled)  # Add to final list
            # print(datalist[0])
            # exit()
            train_ds = data.Dataset(data=extended_files, transform=train_transform)
        # train_sampler = Sampler(train_ds) if distributed else None
        # train_loader = data.DataLoader(
        #     train_ds,
        #     batch_size=args.batch_size,
        #     shuffle=(train_sampler is None),
        #     num_workers=args.workers,
        #     sampler=train_sampler,
        #     pin_memory=True,
        # )
            ds = train_ds
        else:

            val_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
            extended_files = val_files[:]

            for _ in range(repeat):  # Repeat 20 times
                shuffled = val_files[:]  # Copy the list
                random.shuffle(shuffled)  # Shuffle the copy
                extended_files.extend(shuffled)  # Add to final list
            val_ds = data.Dataset(data=extended_files, transform=val_transform)
            # val_sampler = Sampler(val_ds, shuffle=False) if distributed else None
            # val_loader = data.DataLoader(
            #     val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=True
            # )
            ds = val_ds

    return ds
        
class BTCVSegTrainDataset(torch.utils.data.Dataset):
    
    def __init__(self, data_dir, json_list, crop_size = [96, 96, 96], repeat = 0):
        super().__init__()
        self.data = get_dataset(data_dir=data_dir, json_list=json_list, crop_size=crop_size, test_mode=False, is_train=True, repeat=repeat)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

class BTCVSegValDataset(torch.utils.data.Dataset):
    
    def __init__(self, data_dir, json_list  , crop_size = [96, 96, 96], repeat = 0):
        super().__init__()
        self.data = get_dataset(data_dir=data_dir, json_list=json_list, crop_size=crop_size, test_mode=False, is_train=False, repeat=repeat)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]
    
class BTCVSegTestDataset(torch.utils.data.Dataset):
    
    def __init__(self, data_dir, json_list  , crop_size = [96, 96, 96], repeat = 0):
        super().__init__()
        self.data = get_dataset(data_dir=data_dir, json_list=json_list, crop_size=crop_size, test_mode=True, is_train=False, repeat=repeat)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]
    
def generate_fold(training_folder):
    from sklearn.model_selection import KFold
    import json
    
    file_list = sorted(os.listdir(os.path.join(training_folder, 'img')))  # Sorting ensures consistent splits


    file_list = [
                    {
                        "image": [f"img/{name}"],
                        "label": f"label/label{name[3:]}"  # Assuming label names match image names
                    }
                    for name in file_list
                ]
    # Ensure randomness
    # np.random.shuffle(file_list)

    # Define 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    os.makedirs(training_folder, exist_ok=True)

    # Perform cross-validation and save splits
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(file_list)):
        fold_data = {
            "training": [file_list[i] for i in train_idx],
            "validation": [file_list[i] for i in val_idx],
            "label": {
                "0": "background",
                "1": "spleen",
                "2": "rkid",
                "3": "lkid",
                "4": "gall",
                "5": "eso",
                "6": "liver",
                "7": "sto",
                "8": "aorta",
                "9": "IVC",
                "10": "veins",
                "11": "pancreas",
                "12": "rad",
                "13": "lad"
            },
            "modality": {
                "0": "CT"
            },
        }

        json_path = os.path.join(training_folder, f"btcv_fold_{fold_idx + 1}.json")
        with open(json_path, "w") as f:
            json.dump(fold_data, f, indent=4)

        print(f"Saved fold {fold_idx + 1} to {json_path}")

if __name__ == "__main__":
    import argparse

    # Initialize parser
    parser = argparse.ArgumentParser(description="Example script with arguments")

    # Add arguments
    parser.add_argument("--training_folder", type=str, required=True)

    # Parse arguments
    args = parser.parse_args()

    generate_fold(args.training_folder)
