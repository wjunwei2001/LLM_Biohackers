import os
from tqdm import tqdm
import glob
import shutil
from PIL import Image
import random
import cv2
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

def select_random_images(folder_path, n):
    image_filenames = os.listdir(folder_path)
    random.shuffle(image_filenames)
    selected_filenames = image_filenames[:n]
    return selected_filenames

# normal_images = select_random_images('.../Dataset_BUSI_with_GT/normal_correct/', 10)
# benign_images = select_random_images('.../Dataset_BUSI_with_GT/benign_correct/', 10)
malignant_images = select_random_images('Dataset_BUSI_with_GT/malignant_correct/', 600)

data_augmentation = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='reflect'
    # elastic_deformation=True
)   

for image_filename in malignant_images:
    class_path = 'Dataset_BUSI_with_GT/malignant_correct/'

    # Load image
    image_path = os.path.join(class_path, image_filename)
    image = cv2.imread(image_path)

    # Data augmentation
    augmented_images = data_augmentation.flow(np.expand_dims(image, axis=0), batch_size=1)

    # Save augmented images
    for i, augmented_image in enumerate(augmented_images):
        augmented_image = augmented_image[0].astype(np.uint8)
        output_filename = f"{os.path.splitext(image_filename)[0]}_augmented_{i}.png"
        output_path = os.path.join(class_path, output_filename)
        cv2.imwrite(output_path, augmented_image)

        if i == 2:
            break