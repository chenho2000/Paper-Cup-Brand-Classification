# Remove labels that are empty along with the images

import os
import time

def remove_empty_labels():
    # list directories
    labels_dir = "labels"
    images_dir = "images"

    # Iterate through all files in the labels directory
    # This loop will check each label file to see if it's empty
    # and remove both the empty label file and its corresponding image
    for label in os.listdir(labels_dir):
        with open(os.path.join(labels_dir, label), "r") as f:
            # Check if the label file is empty
            content = f.readlines()
            if not content:
                # Get the base name without extension
                base_name = os.path.splitext(label)[0]
                image_file = base_name + ".jpg"
                
                # Define full paths
                label_path = os.path.join(labels_dir, label)
                image_path = os.path.join(images_dir, image_file)
                
                # Remove the empty label file
                os.remove(label_path)
                print(f"Removed empty label file: {label}")
                
                # Remove the corresponding image if it exists
                if os.path.exists(image_path):
                    os.remove(image_path)
                    print(f"Removed corresponding image: {image_file}")
                else:
                    print(f"Warning: Corresponding image not found: {image_file}")

if __name__ == "__main__":
    remove_empty_labels()

