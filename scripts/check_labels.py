import os
import glob

def count_labels():
    # Path to the labels directory
    labels_dir = 'dataset/labels'
    
    # Initialize counters
    class0_count = 0
    class1_count = 0
    both_classes_count = 0
    empty_files_count = 0
    invalid_files_count = 0
    total_files = 0
    
    # Get all text files in the labels directory
    label_files = glob.glob(os.path.join(labels_dir, '*.txt'))
    
    for file_path in label_files:
        total_files += 1
        has_class0 = False
        has_class1 = False
        has_content = False
        
        # Read the file
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                
                has_content = True
                # Check the first character (class)
                try:
                    class_id = line.split()[0]
                    if class_id == '0':
                        has_class0 = True
                    elif class_id == '1':
                        has_class1 = True
                    elif class_id not in ['0', '1']:
                        print(f"Found invalid class ID in {file_path}: {class_id}")
                except IndexError:
                    print(f"Found malformed line in {file_path}: {line}")
        
        # Update counters
        if not has_content:
            empty_files_count += 1
            print(f"Found empty file: {file_path}")
        elif has_class0 and has_class1:
            both_classes_count += 1
        elif has_class0:
            class0_count += 1
        elif has_class1:
            class1_count += 1
        else:
            invalid_files_count += 1
            print(f"Found file with no valid classes: {file_path}")
    
    # Print results
    print(f"\nTotal label files: {total_files}")
    print(f"Files with only class 0 (Cup): {class0_count}")
    print(f"Files with only class 1 (Timmies): {class1_count}")
    print(f"Files with both classes: {both_classes_count}")
    print(f"Empty files: {empty_files_count}")
    print(f"Invalid files: {invalid_files_count}")
    print(f"Sum of all categories: {class0_count + class1_count + both_classes_count + empty_files_count + invalid_files_count}")

if __name__ == "__main__":
    count_labels()
    print(os.getcwd())