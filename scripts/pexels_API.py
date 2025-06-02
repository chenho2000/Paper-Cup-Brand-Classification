# Script for scraping images from Pexels

# documentation: https://www.pexels.com/api/documentation/

import requests
import os
import time
from urllib.parse import urlparse
from pathlib import Path

API_key = "QwE3MPsSYSSXpus6xJC5sou5sKwlnmFxaRJHw9ytMX1QWkeWKBIepvck"

# Change to your directory
DIR = "scraped_images/pexel_images"
os.makedirs(DIR, exist_ok=True)

# CAN CHANGE
# Orientation is the orientation of the image
# size is the minimum size of scraped image which can be left as small
def search_pexels(query: str, per_page: int = 10, 
                  page: int = 1, size: str = "small", 
                  orientation: str = "square"):
    """
    Search for images on Pexels
    
    Parameters:
    - query: Search term
    - color: Color of the image
    - per_page: Number of results per page (max 80)
    - page: Page number for pagination
    - size: Image size to download 
    
    Returns:
    - JSON response from Pexels API
    """
    url = "https://api.pexels.com/v1/search"
    headers = {"Authorization": API_key}
    params = {"query": query,
              "per_page": per_page, "page": page,
              "size": size, "orientation": orientation}
    response = requests.get(url, headers=headers, params=params)
    return response.json()

def download_image(image_url, filename):
    """Download an image from image_url and save it to the specified filename"""
    response = requests.get(image_url)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        return True
    return False

def main():
    # Search parameters
    query = "paper cup"  #CHANGE THIS (what you search for)
    total_images = 30  #CHANGE THIS (how many total images to scrape)
    per_page = 10  #CHANGE THIS (how many images to scrape per page)
    size = "large"  #CHANGE THIS (size of your desired image)
    
    # Calculate number of pages needed
    pages = (total_images + per_page - 1) // per_page
    
    num = 0
    # keep track of downloaded id's so we can reject duplicates
    downloaded_ids = set()
    
    for page in range(1, pages + 1):
        print(f"page {page}")
        # results is parsed json
        results = search_pexels(query, per_page = per_page, page = page)
        
        if "photos" not in results:
            print(f"Error in API response: {results}")
            break
            
        for photo in results["photos"]:
            photo_id = photo["id"]

            if photo_id in downloaded_ids:
                print("skipping duplicate")
                continue

            # Get the image URL based on the requested size
            # Can get original size if desired
            if size == "original":
                image_url = photo["src"]["original"]
            else:
                image_url = photo["src"][size]
                
            # Extract image ID and create filename
            photo_id = photo["id"]
            # parse the url to get the Path
            parsed_url = urlparse(image_url)
            # From the path, get the file extension
            file_extension = Path(parsed_url.path).suffix
            filename = os.path.join(DIR, f"paper_cup_{photo_id}{file_extension}")
            
            # Download the image
            print(f"Downloading")
            if download_image(image_url, filename):
                downloaded_ids.add(photo_id)
                num += 1
                print(f"Downloaded {num}/{total_images}")
            else:
                print(f"Failed to download {image_url}")
            
            time.sleep(0.5)
            
            if num >= total_images:
                break

        time.sleep(1)
    
    print(f"Downloaded {num} images to {DIR}")
    
if __name__ == "__main__":
    main()