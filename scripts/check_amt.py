import os
import glob

def check_image_label_correspondence():
    # Define paths to images and labels folders
    images_folder = "dataset/images"
    labels_folder = "dataset/labels"
    
    # Check if folders exist
    if not os.path.exists(images_folder):
        print(f"Error: {images_folder} does not exist")
        return
    
    if not os.path.exists(labels_folder):
        print(f"Error: {labels_folder} does not exist")
        return
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(images_folder, f"*{ext}")))
    
    # Get all label files
    label_files = glob.glob(os.path.join(labels_folder, "*.txt"))
    
    # Count images
    num_images = len(image_files)
    num_labels = len(label_files)
    
    print(f"Number of images found: {num_images}")
    print(f"Number of labels found: {num_labels}")
    
    # Check correspondence
    image_basenames = [os.path.splitext(os.path.basename(f))[0] for f in image_files]
    label_basenames = [os.path.splitext(os.path.basename(f))[0] for f in label_files]
    
    # Find missing labels
    missing_labels = set(image_basenames) - set(label_basenames)
    if missing_labels:
        print(f"Images without corresponding labels ({len(missing_labels)}):")
        for name in missing_labels:
            print(f"  - {name}")
    
    # Find missing images
    missing_images = set(label_basenames) - set(image_basenames)
    if missing_images:
        print(f"Labels without corresponding images ({len(missing_images)}):")
        for name in missing_images:
            print(f"  - {name}")
    
    # Check if perfect match
    if not missing_labels and not missing_images:
        print("Perfect match")

if __name__ == "__main__":
    check_image_label_correspondence()
