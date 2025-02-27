import pandas as pd
import json
import os
from sklearn.model_selection import KFold
from monai import transforms
from monai.data.dataset import Dataset
import torch


cls_name = {"bkl": 0,
            "df": 1,
            "mel": 2,
            "vasc": 3,
            "bcc": 4,
            "nv": 5,
            "akiec": 6,
            }

def create_folds(df, json_path, n_splits=5):
    
    # Initialize KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Create directory if not exists
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    

    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np

    def get_class_weights(labels_list):
        """
        Compute class weights based on class frequencies.
        
        Args:
            labels (list or numpy array): List of class labels.
            
        Returns:
            dict: Dictionary of class weights {class_index: weight}
        """
        classes = np.unique(labels_list)
        class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels_list)
        
        return class_weights.tolist()
    
    # Split and save each fold
    folds = {}
    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):

        df_t = df.iloc[train_idx]
        class_weights = get_class_weights(df_t["dx"].tolist())

        folds[f'fold_{fold + 1}'] = {
            'train': df_t.to_dict(orient='records'),
            'val': df.iloc[val_idx].to_dict(orient='records'),
            'weights': class_weights
        }
    
    # Save to JSON
    with open(json_path, 'w') as f:
        json.dump(folds, f, indent=4)
    
    print(f"Folds saved to {json_path}")

def indexing(json_path, metadata):
    df = pd.read_csv(metadata, usecols=['image_id', 'dx'])

    # Copy 'image_id' to 'segmentation_id'
    df['segmentation_id'] = df['image_id']
    
    # Map values (example mapping function, modify as needed)
    df['image_id'] = df['image_id'].apply(lambda x: f"{x}.jpg")
    df['segmentation_id'] = df['segmentation_id'].apply(lambda x: f"{x}_segmentation.png")
    print(df['segmentation_id'])
    df['dx'] = df['dx'].apply(lambda x: cls_name[x])
    create_folds(df, json_path)

def get_transforms(is_train = True, crop_size = [400, 400], resize = None):
    """Define MONAI transform pipeline."""
    if is_train:
        return  transforms.Compose([
        transforms.LoadImaged(keys=['image', 'seg'],image_only=True),
        transforms.EnsureChannelFirstd(keys=["image", "seg"]),
        transforms.RandSpatialCropd(keys=["image", "seg"],
            roi_size=crop_size,
            random_center=True, 
            random_size=False,
        ),
        transforms.RandRotated(keys=["image", "seg"], range_x=0.2, prob=0.5),  # Random rotation up to ~11.5 degrees
        transforms.RandZoomd(keys=["image", "seg"], min_zoom=0.9, max_zoom=1.1, prob=0.5),  # Random zoom between 90% to 110%
        transforms.RandFlipd(keys=["image", "seg"], prob=0.35, spatial_axis=[0, 1]),
        transforms.Identityd(keys=["image", "seg"]) if resize is None else
            transforms.Resized( keys=["image", "seg"],
                mode=['nearest-exact', 'nearest-exact'],
                spatial_size=resize,
                anti_aliasing=True, 
            ),

        # transforms.RandAdjustContrastd(keys=["image"], gamma=(0.7, 1.3), prob=0.35),  # Adjust contrast randomly
        # transforms.RandHistogramShiftd(keys=["image"], num_control_points=5, prob=0.35),  # Simulates brightness changes
        # transforms.RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.35),  # Simulate hue shift
        transforms.ScaleIntensityRanged(keys=['image', "seg"], a_min = 0., a_max=255., b_min=0., b_max=1.),
        transforms.ToTensord(keys=['image', 'seg'])
    ])
    else:
        return transforms.Compose([
        transforms.LoadImaged(keys=['image', 'seg'], image_only=True, allow_missing_keys=True),
        transforms.EnsureChannelFirstd(keys=["image", "seg"], allow_missing_keys=True),
        # transforms.SpatialCropd(keys=["image", "seg"],
        #     roi_size=crop_size,
        #     random_center=False, 
        #     random_size=False,
        #      mode="center"
        # ),
        transforms.CenterSpatialCropd(keys=['image', 'seg'], roi_size=crop_size),
        transforms.Identityd(keys=["image", "seg"], allow_missing_keys=True) if resize is None else
            transforms.Resized( keys=["image", "seg"],
                mode=['nearest-exact', 'nearest-exact'],
                spatial_size=resize,
                anti_aliasing=True, 
                allow_missing_keys=True
            ),

        transforms.ScaleIntensityRanged(keys=['image', "seg"], a_min = 0., a_max=255., b_min=0., b_max=1.),
        transforms.ToTensord(keys=['image', 'seg'], allow_missing_keys=True)
    ])


def rgb_to_grayscale(img):
    """Convert RGB (C,H,W) to grayscale (1,H,W)"""
    if img.shape[0] == 3:  # Assuming input is (C,H,W) and C=3 for RGB
        return torch.mean(img, dim=0, keepdim=True)  # Average across channels
    return img  # If already grayscale, return as is

def get_seg_transforms(is_train = True, crop_size = [400, 400], resize = None):
    """Define MONAI transform pipeline."""
    if is_train:
        return  transforms.Compose([
        transforms.LoadImaged(keys=['image', 'seg'], image_only=True,),
        transforms.EnsureChannelFirstd(keys=["image", "seg"]),
        # transforms.Lambdad(keys=["image"], func=rgb_to_grayscale),  # Convert to grayscale
        transforms.RandSpatialCropd(keys=["image", "seg"],
            roi_size=crop_size,
            random_center=True, 
            random_size=False,
        ),
        transforms.RandRotated(keys=["image", "seg"], range_x=0.2, mode='nearest', prob=0.75),  # Random rotation up to ~11.5 degrees
        transforms.RandFlipd(keys=["image", "seg"], prob=0.75, spatial_axis=[0, 1]),
        transforms.Identityd(keys=["image", "seg"]) if resize is None else
            transforms.Resized( keys=["image", "seg"],
                mode=['area', 'nearest'],
                spatial_size=resize,
                anti_aliasing=True, 
            ),
        transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        transforms.ThresholdIntensityd(keys=["seg"], threshold=150, above=True, cval=1),  # Set values >150 to 1
        transforms.ThresholdIntensityd(keys=["seg"], threshold=150, above=False, cval=0), # Set values <=150 to 0
        transforms.ToTensord(keys=['image', 'seg'])
    ])
    else:
        return transforms.Compose([
        transforms.LoadImaged(keys=['image', 'seg'], image_only=True, allow_missing_keys=True),
        transforms.EnsureChannelFirstd(keys=["image", "seg"], allow_missing_keys=True),
        # transforms.Lambdad(keys=["image"], func=rgb_to_grayscale),  # Convert to grayscale
        transforms.Identityd(keys=["image", "seg"], allow_missing_keys=True) if resize is None else
            transforms.Resized( keys=["image", "seg"],
                mode=['area', 'nearest'],
                spatial_size=resize,
                anti_aliasing=True,
                allow_missing_keys=True
            ),

        transforms.ThresholdIntensityd(keys=["seg"], threshold=150, above=True, cval=1),  # Set values >150 to 1
        transforms.ThresholdIntensityd(keys=["seg"], threshold=150, above=False, cval=0), # Set values <=150 to 0
        transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        transforms.ToTensord(keys=['image', 'seg'], allow_missing_keys=True)
    ])


def create_monai_dataset(data, image_dir, crop_size = [400, 400], resize = None, is_train = True, is_seg = False):
    """Create a MONAI dataset from fold data."""
    transform_fn = get_seg_transforms  if is_seg else get_transforms
    transforms =  transform_fn(is_train=is_train, crop_size=crop_size, resize=resize) 
    
    return Dataset(
        data=[{ "image": os.path.join(image_dir, 'images', item["image_id"]), "seg": os.path.join(image_dir, 'segmentations', item["segmentation_id"]), "label": item["dx"] } for item in data['train' if is_train else 'val']],
        transform=transforms
    )

def load_fold(json_path, fold = 1):
    """Load a specific fold from the JSON file."""
    with open(json_path, 'r') as f:
        folds = json.load(f)
    
    if f"fold_{fold}" in folds:
        return folds[f"fold_{fold}"]
    else:
        raise ValueError(f"Fold {fold} not found in {json_path}")

class PAM10KDDataset(torch.utils.data.Dataset):
    
    def __init__(self, json_path, image_dir, is_train, crop_size=[400, 400], resize = None, fold = 1):
        super().__init__()
        data = load_fold(json_path, fold=fold)
        self.weight = data['weights']
        self.data = create_monai_dataset(data, image_dir = image_dir, is_train=is_train, crop_size=crop_size, resize=resize)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

class PAM10KSegDataset(torch.utils.data.Dataset):
    
    def __init__(self, json_path, image_dir, is_train, crop_size=[400, 400], resize = None, fold = 1):
        super().__init__()
        data = load_fold(json_path, fold=fold)
        self.weight = data['weights']
        self.data = create_monai_dataset(data, image_dir = image_dir, is_train=is_train, crop_size=crop_size, resize=resize, is_seg=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

if __name__ == "__main__":
    indexing('jsons/pam10kfolds.json', '/home/k8suser/jo/HAM/ham10000/HAM10000_metadata.csv')

    # ds = PAM10KDDataset(json_path='jsons/pam10kfolds.json', image_dir='/home/jovianto/dataset/ham10000/train/', is_train=True)

    # A = ds.__getitem__(50)

    # print(A['seg'])