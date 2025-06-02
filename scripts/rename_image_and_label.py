import os

# Define the directories
images_directory = 'images'  # Change this to your actual images folder path
labels_directory = 'labels'  # Change this to your actual labels folder path

# Loop through all files in the images directory
for index, filename in enumerate(os.listdir(images_directory)):
    # Check if the file is a JPG image
    if filename.endswith('.jpg'):

        new_filename = f"syde_cup_{index}"

        # Define full paths
        old_image_path = os.path.join(images_directory, filename)
        new_image_path = os.path.join(images_directory, new_filename + '.jpg')
        old_label_path = os.path.join(labels_directory, filename[:-4] + '.txt') 
        new_label_path = os.path.join(labels_directory, new_filename + '.txt')

        # Rename the image file
        os.rename(old_image_path, new_image_path)
        os.rename(old_label_path, new_label_path)