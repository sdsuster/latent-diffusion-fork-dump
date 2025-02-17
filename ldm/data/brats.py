import os

from monai import transforms
from monai.data import Dataset as MonaiDataset
from torch.utils.data import Dataset
import json
import torch


def get_brats_dataset(data_path, pad_size = [160, 160, 126], crop_size = [80, 80, 64], resize = None, is_val = False):
        
    transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"], allow_missing_keys=True),
            transforms.EnsureChannelFirstd(keys=["image"], allow_missing_keys=True),
            transforms.EnsureTyped(keys=["image"]),
            # transforms.Orientationd(keys=["image"], axcodes="RAI", allow_missing_keys=True),
            transforms.CropForegroundd(keys=["image"], allow_smaller=True, source_key="image", allow_missing_keys=True),
            # transforms.SpatialPadd(keys=["image"], spatial_size=pad_size, allow_missing_keys=True),
            
            transforms.Identityd(keys=["image"]) if resize is None else
                transforms.Resized( keys=["image"],
                    spatial_size=resize,
                    anti_aliasing=True, 
                ),
            transforms.Identityd(keys=["image"]) if crop_size is None else
                transforms.RandSpatialCropd( keys=["image"],
                    roi_size=crop_size,
                    random_center=True, 
                    random_size=False,
                ),
            transforms.RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
            transforms.RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
            transforms.RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
            transforms.NormalizeIntensityd(keys="image", nonzero=False, channel_wise=True),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            transforms.ScaleIntensityRangePercentilesd(keys=["image"], lower=0, upper=99.75, b_min=0, b_max=1),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"], allow_missing_keys=True),
            transforms.EnsureChannelFirstd(keys=["image"], allow_missing_keys=True),
            transforms.EnsureTyped(keys=["image"]),
            # transforms.Orientationd(keys=["image"], axcodes="RAI", allow_missing_keys=True),
            transforms.CropForegroundd(keys=["image"], allow_smaller=True, source_key="image", allow_missing_keys=True),
            transforms.SpatialPadd(keys=["image"], spatial_size=pad_size, allow_missing_keys=True),
            
            transforms.Identityd(keys=["image"]) if resize is None else
                transforms.Resized( keys=["image"],
                    spatial_size=resize,
                    anti_aliasing=True, 
                ),
            transforms.Identityd(keys=["image"]) if crop_size is None else
                transforms.RandSpatialCropd( keys=["image"],
                    roi_size=crop_size,
                    random_center=True, 
                    random_size=False,
                ),
            transforms.NormalizeIntensityd(keys="image", nonzero=False, channel_wise=True),
            transforms.ScaleIntensityRangePercentilesd(keys=["image"], lower=0, upper=99.75, b_min=0, b_max=1),
        ]
    )
    data = []
    for subject in os.listdir(data_path):
        sub_path = os.path.join(data_path, subject)
        if os.path.exists(sub_path) == False or os.path.isdir(sub_path) is False: continue
        for files in os.listdir(sub_path):
            if 'seg' in files:
                continue
            image = os.path.join(sub_path, files) 
            data.append({"image":image, "subject_id": subject})
                    
    print(f"num of images {data_path}:", len(data))

    return MonaiDataset(data=data, transform=val_transform if is_val else transform)

def get_brats_seg_dataset(data_path, pad_size = [160, 160, 126], crop_size = [160, 160, 126], resize = None, is_val = False):
        
    transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"], allow_missing_keys=True),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            # transforms.EnsureChannelFirstd(keys=["image"], allow_missing_keys=True),
            transforms.EnsureTyped(keys=["image", "label"]),
            # transforms.Orientationd(keys=["image"], axcodes="RAI", allow_missing_keys=True),
            transforms.CropForegroundd( keys=["image", "label"], source_key="image", allow_smaller=True, allow_missing_keys=True),
            # transforms.SpatialPadd(keys=["image", "label"], spatial_size=pad_size, allow_missing_keys=True),
            
            transforms.Identityd(keys=["image"]) if resize is None else
                transforms.Resized( keys=["image"],
                    spatial_size=resize,
                    anti_aliasing=True, 
                ),
            transforms.Identityd(keys=["label"]) if resize is None else
                transforms.Resized( keys=["label"],
                    mode='nearest-exact',
                    spatial_size=resize,
                    anti_aliasing=True, 
                ),
            transforms.Identityd(keys=["image", "label"]) if crop_size is None else
                transforms.RandSpatialCropd(keys=["image", "label"],
                    roi_size=crop_size,
                    random_center=True, 
                    random_size=False,
                ),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            # transforms.NormalizeIntensityd(keys="image", nonzero=False, channel_wise=True),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            transforms.ScaleIntensityRangePercentilesd(keys=["image"], lower=0, upper=99.75, b_min=0, b_max=1),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"], allow_missing_keys=True),
            transforms.EnsureChannelFirstd(keys=["image"], allow_missing_keys=True),
            transforms.EnsureTyped(keys=["image"]),
            # transforms.Orientationd(keys=["image"], axcodes="RAI", allow_missing_keys=True),
            transforms.CropForegroundd(keys=["image"], allow_smaller=True, source_key="image", allow_missing_keys=True),
            transforms.SpatialPadd(keys=["image"], spatial_size=pad_size, allow_missing_keys=True),
            
            transforms.Identityd(keys=["image"]) if resize is None else
                transforms.Resized( keys=["image"],
                    spatial_size=resize,
                    anti_aliasing=True, 
                ),
            transforms.Identityd(keys=["image"]) if crop_size is None else
                transforms.RandSpatialCropd( keys=["image"],
                    roi_size=crop_size,
                    random_center=True, 
                    random_size=False,
                ),
            transforms.NormalizeIntensityd(keys="image", nonzero=False, channel_wise=True),
            transforms.ScaleIntensityRangePercentilesd(keys=["image"], lower=0, upper=99.75, b_min=0, b_max=1),
        ]
    )
    data = []
    for subject in os.listdir(data_path):
        sub_path = os.path.join(data_path, subject)
        if os.path.exists(sub_path) == False or os.path.isdir(sub_path) is False: continue
        image = [
            os.path.join(sub_path, f'{subject}-t1c.nii.gz') ,
            os.path.join(sub_path, f'{subject}-t1n.nii.gz') ,
            os.path.join(sub_path, f'{subject}-t2f.nii.gz') ,
            os.path.join(sub_path, f'{subject}-t2w.nii.gz') ,
        ]
        seg = os.path.join(sub_path, f'modified_{subject}-seg.nii.gz')
        data.append({"image":image, "subject_id": subject, "label": seg})
                    
    print(f"num of images {data_path}:", len(data))

    return MonaiDataset(data=data, transform=val_transform if is_val else transform)

def get_brats_seg_fold_dataset(json_path, pad_size = [160, 160, 126], crop_size = [80, 80, 64], resize = None, is_val = False):
        
    transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"], allow_missing_keys=True),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            # transforms.EnsureChannelFirstd(keys=["label"], allow_missing_keys=True),
            transforms.EnsureTyped(keys=["image", "label"]),
            # transforms.Orientationd(keys=["image"], axcodes="RAI", allow_missing_keys=True),
            transforms.CropForegroundd( keys=["image", "label"], source_key="image", allow_smaller=True, allow_missing_keys=True),
            # transforms.SpatialPadd(keys=["image", "label"], spatial_size=pad_size, allow_missing_keys=True),
            
            transforms.Identityd(keys=["image", "label"]) if resize is None else
                transforms.Resized( keys=["image", "label"],
                    spatial_size=resize,
                    mode=('area', 'nearest'),
                    anti_aliasing=(True, False), 
                ),
            transforms.Identityd(keys=["image", "label"]) if crop_size is None else
                transforms.RandSpatialCropd(keys=["image", "label"],
                    roi_size=crop_size,
                    random_center=True, 
                    random_size=False,
                ),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            transforms.RandZoomd(keys=["image", "label"], prob=0.8, mode=('area', 'nearest'), keep_size=True),
            # transforms.Rand3DElasticd(keys=["image", "label"], sigma_range=(3, 6), magnitude_range=(1, 5), prob=0.75, mode=('bilinear', 'nearest')),
            transforms.RandRotated(keys=["image", "label"],  range_x=15, range_y=15, range_z=15, 
                                   prob=.8,padding_mode='zeros',
                                   mode=('bilinear', 'nearest')),
            transforms.NormalizeIntensityd(keys="image", nonzero=False, channel_wise=True),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            # transforms.ScaleIntensityRangePercentilesd(keys=["image"], lower=0, upper=99.75, b_min=0, b_max=1),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"], allow_missing_keys=True),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            # transforms.AddChanneld(keys="label", ),
            # transforms.EnsureChannelFirstd(keys=["label"], allow_missing_keys=True),
            transforms.EnsureTyped(keys=["image", "label"]),
            # transforms.Orientationd(keys=["image"], axcodes="RAI", allow_missing_keys=True),
            transforms.CropForegroundd( keys=["image", "label"], source_key="image", allow_smaller=True, allow_missing_keys=True),
            transforms.SpatialPadd(keys=["image", "label"], spatial_size=pad_size, allow_missing_keys=True),
            
            transforms.Identityd(keys=["image", "label"]) if resize is None else
                transforms.Resized( keys=["image", "label"],
                    spatial_size=resize,
                    mode=('area', 'nearest-exact'),
                    anti_aliasing=(True, False), 
                ),
            transforms.Identityd(keys=["image", "label"]) if crop_size is None else
                transforms.RandSpatialCropd(keys=["image", "label"],
                    roi_size=crop_size,
                    random_center=True, 
                    random_size=False,
                ),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            # transforms.ScaleIntensityRangePercentilesd(keys=["image"], lower=0, upper=99.75, b_min=0, b_max=1),
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
    fold_idx = 1  # Fold index to load (1 to 5)

    train_files = load_fold_data(fold_idx, json_path)

    print(f"Files for fold {fold_idx}: {len(train_files)}")


    return MonaiDataset(data=train_files, transform=val_transform if is_val else transform)
class BratsDataset(Dataset):
    
    def __init__(self,data_path, pad_size = [160, 160, 126], crop_size = [160, 160, 126], resize = None):
        super().__init__()
        self.data = get_brats_dataset(data_path, pad_size, crop_size, resize)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

class BratsValDataset(BratsDataset):
    
    def __init__(self,data_path, pad_size = [160, 160, 126], crop_size = [160, 160, 126], resize = None):
        self.data = get_brats_dataset(data_path, pad_size, crop_size, resize, is_val=True)

        
class BratsSegFoldDataset(Dataset):
    
    def __init__(self,data_path, pad_size = [160, 160, 126], crop_size = [80, 80, 64], resize = None):
        super().__init__()
        self.data = get_brats_seg_fold_dataset(data_path, pad_size, crop_size, resize)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]
        
class BratsSegValFoldDataset(Dataset):
    
    def __init__(self,data_path, pad_size = [160, 160, 126], crop_size = [80, 80, 64], resize = None):
        super().__init__()
        self.data = get_brats_seg_fold_dataset(data_path, pad_size, crop_size, resize, is_val=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]
    
class BratsSegDataset(Dataset):
    
    def __init__(self,data_path, pad_size = [160, 160, 126], crop_size = [160, 160, 126], resize = None):
        super().__init__()
        self.data = get_brats_seg_dataset(data_path, pad_size, crop_size, resize)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


class BratsDatasetTPU(BratsDataset):

    def __getitem__(self, i):
        example = {}
        meta = self.data[i]['image']
        example['image'] = meta.numpy()
        return example