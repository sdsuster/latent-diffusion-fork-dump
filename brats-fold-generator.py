import os
import json
from sklearn.model_selection import KFold
def get_image_object(image_folder, subjects, i):
    image = [
        os.path.join(image_folder, subjects[i], f'{subjects[i]}-t1c.nii.gz') ,
        os.path.join(image_folder, subjects[i], f'{subjects[i]}-t1n.nii.gz') ,
        os.path.join(image_folder, subjects[i], f'{subjects[i]}-t2f.nii.gz') ,
        os.path.join(image_folder, subjects[i], f'{subjects[i]}-t2w.nii.gz') ,
    ]
    seg = os.path.join(image_folder, subjects[i], f'modified_{subjects[i]}-seg.nii.gz')
    return {"image":image, "subject_id": subjects[i], "label": seg}

# Function to split the dataset into 5 folds and save it to JSON
def create_five_fold_splits(image_folder, output_json):
    # Get list of image files from the folder
    subjects = [f for f in os.listdir(image_folder)]
    
    # Sort the images to ensure consistent splits
    subjects.sort()
    
    # Initialize KFold with 5 splits
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Dictionary to hold the folds
    folds = {}
    
    # Split the image files
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(subjects)):

        
        folds[f'fold_{fold_idx+1}'] = {
            'train': [get_image_object(image_folder, subjects, i) for i in train_idx],
            'val': [get_image_object(image_folder, subjects, i) for i in test_idx]
        }
    
    # Save the folds into a JSON file
    with open(output_json, 'w') as json_file:
        json.dump(folds, json_file, indent=4)
        
    print(f"5-fold splits saved to {output_json}")

# Usage
image_folder = '/home/hdd2/jo/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/'  # Replace with your image folder path
output_json = 'brats_folds.json'  # Output JSON file path

create_five_fold_splits(image_folder, output_json)