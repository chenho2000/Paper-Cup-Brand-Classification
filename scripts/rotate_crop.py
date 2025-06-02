"""
Rotate and Crop Augmentation Script
-----------------------------------

This script performs data augmentation on images by:
1. Randomly rotating images -90 to 90 degrees
2. Randomly cropping images
3. adding contrast and brightness
4. horizontal flip 

The augmented images are added to the dataset to improve balance and training performance.

Target images:
- Images containing "proj", "project", "AUG", or "VID_2025" in their filenames
- Only Tim Hortons cup images are augmented since they are half the amount of cup images
"""

import albumentations as A
from PIL import Image
import os
import numpy as np
import cv2

images_dir = "images"
labels_dir = "labels"
images_augmented_dir = "images_augmented"
labels_augmented_dir = "labels_augmented"
    
# Create output directories if they don't exist
os.makedirs(images_augmented_dir, exist_ok=True)
os.makedirs(labels_augmented_dir, exist_ok=True)
    
# Get all image files that match the target substrings
image_files = [f for f in os.listdir(images_dir) 
                if f.endswith(('.jpg', '.jpeg', '.png')) 
                and any(substring in f for substring in ["proj", "project", "AUG", "VID_2025"])]
    
for image_file in image_files:
    # split the path in name and extension, name being at index 0
    base_name = os.path.splitext(image_file)[0]
    label_file = base_name + ".txt"
        
    # Check if label file contains class 0, if so skip this image since we only augment timmies
    label_path = os.path.join(labels_dir, label_file)
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            label_content = f.read().strip()
            if any(line.strip().startswith('0 ') for line in label_content.split('\n')):
                continue
    else:
        print("no such label file with same name as image file")
        continue
        
    # Define input and output paths
    image_path = os.path.join(images_dir, image_file)
    label_path = os.path.join(labels_dir, label_file)
    output_image_path = os.path.join(images_augmented_dir, f"{base_name}_AUG.jpg")
    output_label_path = os.path.join(labels_augmented_dir, f"{base_name}_AUG.txt")
        
    # Read the image in BGR format
    image = cv2.imread(image_path)
    # Convert to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
    # Read the YOLO label
    with open(label_path, 'r') as f:
        label_content = f.read().strip()
        
    # Parse YOLO label
    labels = []
    for line in label_content.split('\n'):
        if line:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            labels.append([class_id, x_center, y_center, width, height])
        
    # Define the augmentations
    transform = A.Compose([
        A.RandomCrop(width=int(image.shape[1] * 0.75), height=int(image.shape[0] * 0.75), p=0.6),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=90, p=0.65),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.4))
        
    # Apply transformations
    class_labels = [label[0] for label in labels]
    # Albumentations expects bboxes in yolo format [x_center, y_center, width, height, class]
    try:
        transformed = transform(image=image, 
                                bboxes=[[label[1], label[2], label[3], label[4]] 
                                        for label in labels], class_labels=class_labels)
    except ValueError:
        print("error with bbox")
        continue
        
    # Get the transformed image and bounding boxes
    transformed_image = transformed['image']
    transformed_bboxes = transformed['bboxes']
    transformed_image = transformed_image.astype(np.uint8)
        
    # Convert to PIL Image and save
    pil_image = Image.fromarray(transformed_image)
    pil_image.save(output_image_path)
        
    # Save the transformed YOLO labels
    with open(output_label_path, 'w') as f:
        for i, bbox in enumerate(transformed_bboxes):
            # After transformation, Albumentations returns bboxes as [x_center, y_center, width, height]            
            class_id = class_labels[i]
            x_center, y_center, width, height = bbox
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")


