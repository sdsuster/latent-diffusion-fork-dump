import os
import nibabel as nib
import numpy as np

# Define the root directory where subfolders and files are located
root_directory = '/home/hdd2/jo/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/'  # Replace with the path to your folder

def delete_files_containing_word(root_folder, word):
    for foldername, subfolders, filenames in os.walk(root_folder):
        for filename in filenames:
            if word in filename:
                file_path = os.path.join(foldername, filename)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

delete_files_containing_word(root_directory, "modified")
# Iterate through all subdirectories and files
for subdir, _, files in os.walk(root_directory):
    for file in files:
        # Check if the filename contains "seg"
        if "seg" in file and file.endswith('.nii.gz'):  # Ensure it's a NIfTI file
            file_path = os.path.join(subdir, file)
            
            # Load the NIfTI file using nibabel
            nifti_image = nib.load(file_path)
            
            # Convert the NIfTI image to a NumPy array
            image_array = nifti_image.get_fdata()

            # Change values from 3 to 4 in the array
            image_array[image_array == 3] = 4
            
            # Create a new NIfTI image from the modified array
            new_nifti_image = nib.Nifti1Image(image_array, nifti_image.affine, nifti_image.header)
            
            # Define the path to save the modified file (optional: you can overwrite the original file)
            output_path = os.path.join(subdir, f"modified_{file}")
            
            # Save the modified NIfTI image
            nib.save(new_nifti_image, output_path)
            
            print(f"Processed and saved: {output_path}")