import os

from monai import transforms
from monai.data import Dataset as MonaiDataset
from torch.utils.data import Dataset



def get_brats_dataset(data_path, pad_size = [160, 160, 126], crop_size = [160, 160, 126], resize = None):
        
    transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"], allow_missing_keys=True),
            transforms.EnsureChannelFirstd(keys=["image"], allow_missing_keys=True),
            transforms.EnsureTyped(keys=["image"]),
            transforms.Orientationd(keys=["image"], axcodes="RAI", allow_missing_keys=True),
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

    return MonaiDataset(data=data, transform=transform)
class BratsDataset(Dataset):
    
    def __init__(self,data_path, pad_size = [160, 160, 126], crop_size = [160, 160, 126], resize = None):
        super().__init__()
        self.data = get_brats_dataset(data_path, pad_size, crop_size, resize)

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