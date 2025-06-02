import os
import glob
import time

def validate_yolo_labels():
    # Path to the labels directory
    labels_dir = 'dataset/labels'
    images_dir = 'dataset/images'
    
    # Initialize counters
    valid_files = 0
    invalid_files = 0
    error_files = []
    removed_files = []

    # Get all text files in the labels directory
    label_files = glob.glob(os.path.join(labels_dir, '*.txt'))
    
    for file_path in label_files:
        is_valid = True
        with open(file_path, 'r') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                
                try:
                    # Split the line into components
                    values = line.split()
                    
                    # Check if we have exactly 5 values (class_id, x, y, width, height)
                    if len(values) != 5:
                        is_valid = False
                        error_files.append(f"{file_path}: Line {line_num} - Invalid number of values")
                        
                        # Get base filename without extension
                        base_name = os.path.splitext(os.path.basename(file_path))[0]
                        image_path = os.path.join(images_dir, base_name + '.jpg')
                        
                        # Remove label file
                        file.close()

                        os.remove(file_path)
                        
                        # Remove corresponding image if it exists
                        if os.path.exists(image_path):
                            os.remove(image_path)
                            removed_files.append(f"Removed {file_path} and {image_path}")
                        else:
                            removed_files.append(f"Removed {file_path} (no corresponding image found)")
                        break
                    
                    # Check if class_id is an integer
                    class_id = int(values[0])
                    
                    # Check if coordinates are float and between 0 and 1
                    x, y, w, h = map(float, values[1:])
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                        is_valid = False
                        error_files.append(f"{file_path}: Line {line_num} - Coordinates must be between 0 and 1")
                        break
                        
                except ValueError:
                    is_valid = False
                    error_files.append(f"{file_path}: Line {line_num} - Invalid value format")
                    break
        
        if is_valid:
            valid_files += 1
        else:
            invalid_files += 1
    
    # Print results
    print(f"\nYOLO Format Validation Results:")
    print(f"Total files checked: {len(label_files)}")
    print(f"Valid files: {valid_files}")
    print(f"Invalid files: {invalid_files}")
    
    if error_files:
        print("\nDetailed errors:")
        for error in error_files:
            print(error)
            
    if removed_files:
        print("\nRemoved files:")
        for removed in removed_files:
            print(removed)

if __name__ == "__main__":
    validate_yolo_labels()
