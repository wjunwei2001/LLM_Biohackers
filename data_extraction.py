import os
from tqdm import tqdm
import glob
import shutil
from PIL import Image
import random

dataset_path = '/home/ubuntu/Dataset_BUSI_with_GT/malignant'


def move_non_mask_files(source_folder, output_folder):
    # Create a new folder to store non-mask files
    os.makedirs(output_folder, exist_ok=True)

    # Get a list of all file paths in the source folder
    file_paths = glob.glob(os.path.join(source_folder, "*"))

    # Move non-mask files to the new folder
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        if "_mask" not in file_name:
            destination_path = os.path.join(output_folder, file_name)
            shutil.copy(file_path, destination_path)
    print(dataset_path)
    print("Non-mask files have been extracted to the 'non_mask_files' folder.")


move_non_mask_files(dataset_path,'/home/ubuntu/Dataset_BUSI_with_GT/malignant_1')
