import os
from PIL import Image

def downsample_image(input_path, size):
    """
    Downsample image and save to original path

    Args:
        input_path (str): Path to the image to downsample
        size (tuple): Size to downsample the image to
    
        return: None
    """
    with Image.open(input_path) as my_image:
        # Downsample the image using lanczos resampling, lanczos is essentially 
        # high quality resampling as it minimizes aliasing and blurring. It uses 
        # kernel size of 3 or more to consider neighbour pixels when calculating
        # new pixel. Could consider anti-aliasing as well.
        my_image = my_image.resize(size, Image.LANCZOS)
        # Save the downsampled image back to the original path
        my_image.save(input_path)

if __name__ == '__main__':
    my_path = 'drive-download-20250228T040106Z-001'
    my_size = (800, 600)

    # Iterate all files in the path my_path
    for filename in os.listdir(my_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            input_image = os.path.join(my_path, filename)
            downsample_image(input_image, my_size)
