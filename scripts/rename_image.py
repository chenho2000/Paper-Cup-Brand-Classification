import os

# Define the directory containing images
input_directory = 'images'  # Change this to your actual images folder path

# Starting number for renaming
start_number = 568

# Loop through all files in the input directory
for index, filename in enumerate(os.listdir(input_directory), start=start_number):
    # Check if the file is an image
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        # Create the new filename
        new_filename = f"syde_cup_{index}.jpg"
        
        # Define the full paths
        old_file_path = os.path.join(input_directory, filename)
        new_file_path = os.path.join(input_directory, new_filename)
        
        # Rename the file
        os.rename(old_file_path, new_file_path)
        print(f"Renamed '{filename}' to '{new_filename}'")