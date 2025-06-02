import requests
import base64
import zipfile
import io

def kaggle_download_images():
    url = "https://www.kaggle.com/api/v1/datasets/download/mrgetshjtdone/vn-trash-classification?datasetVersionNumber=1"

    # encoding credentials
    username = "<ansonlai1210>"
    key = "<f69536f4c50ccd9cecfe45b9b0e27dc6>"
    creds = base64.b64encode(bytes(f"{username}:{key}", "ISO-8859-1")).decode("ascii")
    # Create header
    headers = {
        "Authorization": f"Basic {creds}"
    }

    # GET request
    response = requests.get(url, headers=headers)

    # Opening the file via zipfile and extracting all the files
    zip_file = zipfile.ZipFile(io.BytesIO(response.content))
    zip_file.extractall()


if __name__ == "__main__":
    kaggle_download_images()