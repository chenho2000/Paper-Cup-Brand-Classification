from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import os
import time
import requests
import base64
from PIL import Image
from io import BytesIO
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# What you enter here will be searched for in
# Google Images
query = "Tim Horton Paper Cup"

# Creating a webdriver instance
driver = webdriver.Chrome("F:\Download\chromedriver-win64\chromedriver.exe")

# Maximize the screen
driver.maximize_window()

# Open Google Images in the browser
driver.get("https://www.reddit.com/search/?q=tim+horton+cups&type=media&cId=c08d59a0-7d21-403a-a8f6-3bed64159c0d&iId=fa405f15-0189-440f-98cb-bd76c3e8ae45")

# # Finding the search box
# box = driver.find_element("xpath", '//*[@id="APjFqb"]')

# # Type the search query in the search box
# box.send_keys(query)

# # Pressing enter
# box.send_keys(Keys.ENTER)


# Function for scrolling to the bottom of Google
# Images results
def scroll_to_bottom():

    last_height = driver.execute_script(
        "\
	return document.body.scrollHeight"
    )

    while True:
        driver.execute_script(
            "\
		window.scrollTo(0,document.body.scrollHeight)"
        )

        # waiting for the results to load
        # Increase the sleep time if your internet is slow
        time.sleep(3)

        new_height = driver.execute_script(
            "\
		return document.body.scrollHeight"
        )

        # click on "Show more results" (if exists)
        try:
            driver.find_element("CSS_SELECTOR", ".YstHxe input").click()

            # waiting for the results to load
            # Increase the sleep time if your internet is slow
            time.sleep(3)

        except:
            pass

        # checking if we have reached the bottom of the page
        if new_height == last_height:
            break

        last_height = new_height


# Calling the function

# NOTE: If you only want to capture a few images,
# there is no need to use the scroll_to_bottom() function.
scroll_to_bottom()


try:
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "img.h-full"))
    )
except:
    print("Images did not load. Exiting...")
    driver.quit()
    exit()

# Extract image URLs
image_elements = driver.find_elements(By.CSS_SELECTOR, "img.h-full")
print(f"Found {len(image_elements)} images. Starting download...")


def download_image(index, src):
    try:
        if src is None or src.strip() == "":
            return False  # Skip blank images

        # Handle Base64 images
        if src.startswith("data:image"):
            header, encoded = src.split(",", 1)
            img_data = base64.b64decode(encoded)
        else:
            img_data = requests.get(src, timeout=5).content

        # Check if image is blank (all white or transparent)
        img = Image.open(BytesIO(img_data))
        if img.getbbox() is None:
            return False  # Skip blank images

        # Save the image
        img.save(os.path.join(folder_name, f"re_image_{index}.jpg"))
        return True

    except Exception as e:
        return False


# Configuration
folder_name = "datasetV0"

# Create folder if it doesn't exist
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Download images in parallel using ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=8) as executor:
    tasks = []
    for index, img in enumerate(image_elements):
        src = img.get_attribute("src")
        tasks.append(executor.submit(download_image, index, src))

    downloaded_count = 0
    for task in tqdm(as_completed(tasks), total=len(tasks)):
        if task.result():
            downloaded_count += 1

print(f"Downloaded {downloaded_count} images into the '{folder_name}' folder.")

# Close the browser
driver.quit()
