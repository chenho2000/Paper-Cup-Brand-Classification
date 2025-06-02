import os
import shutil
import glob

def get_and_copy_images():
    # Define paths
    images_dir = "images"
    labels_dir = "labels"
    images_add_dir = "image_add"
    labels_add_dir = "label_add"
    
    # Create output directories if they don't exist
    os.makedirs(images_add_dir, exist_ok=True)
    os.makedirs(labels_add_dir, exist_ok=True)
    
    # Get all image files that match the target substrings
    image_files = [f for f in os.listdir(images_dir) 
                  if f.endswith(('.jpg', '.jpeg', '.png')) 
                  and any(substring in f for substring in ["proj", "project", "AUG", "VID_2025"])]
    
    print(f"Found {len(image_files)} matching images")
    
    # Process each image
    copied_count = 0
    for image_file in image_files:
        # Get base name without extension
        base_name = os.path.splitext(image_file)[0]
        label_file = base_name + ".txt"
        
        # Define paths
        image_path = os.path.join(images_dir, image_file)
        label_path = os.path.join(labels_dir, label_file)
        
        # Check if label contains only class 1
        with open(label_path, 'r') as f:
            label_content = f.read().strip()
            # Skip if any line starts with class 0
            if any(line.strip().startswith('0 ') for line in label_content.split('\n')):
                continue
        # Define destination paths
        dest_image_path = os.path.join(images_add_dir, image_file)
        dest_label_path = os.path.join(labels_add_dir, label_file)
        
        # Copy files
        shutil.copy2(image_path, dest_image_path)
        shutil.copy2(label_path, dest_label_path)
        copied_count += 1
    
    print(f"Successfully copied {copied_count} images and their labels with class 1")

if __name__ == "__main__":
    get_and_copy_images()
