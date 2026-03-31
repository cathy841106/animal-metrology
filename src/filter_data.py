import os
import shutil
from pycocotools.coco import COCO

from config import Config
from utils.coco_utils import (
    get_category_ids_by_supercategory,
    get_image_ids_by_categories,
    filter_images_by_min_annotations
)

# ==== Config ====
# Data paths 
IMG_DIR = Config.IMG_DIR
ANNOTATION_FILE = Config.ANNOTATION_FILE
EXPORT_DIR = Config.FILTERED_IMAGE_DIR

# Category to filter
FILTER_SUPERCATEGORY = ['animal']


# ==== Load COCO ====
coco = COCO(ANNOTATION_FILE)


# ==== Get category ids for animal category====
animal_cat_ids = get_category_ids_by_supercategory(
    coco, FILTER_SUPERCATEGORY
)


# ==== Collect all image ids that contain at least one animal ====
img_ids = get_image_ids_by_categories(coco, animal_cat_ids)


# ==== Filter images that contain at least 2 animals ====
selected_img_ids = filter_images_by_min_annotations(
    coco, img_ids, animal_cat_ids, min_count=2
)

print (f"Selected {len(selected_img_ids)} images containing at least 2 animals")


# ==== Export filtered images ====
os.makedirs(EXPORT_DIR, exist_ok=True)

for img_id in selected_img_ids:
    img_info = coco.loadImgs(img_id)[0]  # Get info dict
    src_path = os.path.join(IMG_DIR, img_info['file_name'])
    dst_path = os.path.join(EXPORT_DIR, img_info['file_name'])
    shutil.copy2(src_path, dst_path)  # Copy selected image to export directory

print (f"Exported {len(selected_img_ids)} images to {EXPORT_DIR}")