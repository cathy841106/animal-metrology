import os
from zipfile import ZipFile

from config import Config
from utils.download import download_file, unzip_file

# ==== Config ====
# Download URLs
IMAGES_URL = Config.COCO_IMAGE_DOWNLOAD_URL
ANNOTATIONS_URL = Config.COCO_ANNOTATION_DOWNLOAD_URL

# Storage path 
BASE_DIR = Config.COCO_BASE_DIR
IMAGES_ZIP_PATH = os.path.join(BASE_DIR, os.path.basename(IMAGES_URL))
ANNOTATIONS_ZIP_PATH = os.path.join(BASE_DIR, os.path.basename(ANNOTATIONS_URL))
IMAGES_DIR = os.path.join(BASE_DIR, os.path.splitext(os.path.basename(IMAGES_URL))[0])
ANNOTATIONS_DIR = os.path.join(BASE_DIR, os.path.splitext(os.path.basename(ANNOTATIONS_URL))[0])


# ==== Create base directory if it doesn't exist ====
os.makedirs(BASE_DIR, exist_ok=True)


# ==== Download validation images ====
if not os.path.exists(IMAGES_DIR):
    # Download file from url
    download_file(IMAGES_URL, IMAGES_ZIP_PATH)
    
    # Unzip downloaded file
    print (f"Unzipping {os.path.basename(IMAGES_ZIP_PATH)} ...")
    unzip_file(IMAGES_ZIP_PATH, BASE_DIR)
    print (f"Unzipped to {IMAGES_DIR}")
else:
    print (f"{IMAGES_DIR} already exists. Skipping download.")


# ==== Download annotations ====
if not os.path.exists(ANNOTATIONS_DIR):
    # Download file from url
    download_file(ANNOTATIONS_URL, ANNOTATIONS_ZIP_PATH)
    
    # Only extract the 'annotations' folder
    with ZipFile(ANNOTATIONS_ZIP_PATH, 'r') as zip_ref:
        members = [m for m in zip_ref.namelist() if m.startswith("annotations/")]
        zip_ref.extractall(path=BASE_DIR, members=members)
    os.rename(os.path.join(BASE_DIR, "annotations"), ANNOTATIONS_DIR)
        
    print (f"Annotations extracted to {ANNOTATIONS_DIR}")
else:
    print (f"{ANNOTATIONS_DIR} already exists. Skipping download.")