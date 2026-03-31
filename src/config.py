import os

class Config:
    # ==== App ====
    APP_ENV = os.getenv("APP_ENV")
    
    # ==== Paths ====
    DATA_DIR = os.getenv("DATA_DIR")
    COCO_BASE_DIR = os.path.join(DATA_DIR, os.getenv("COCO_DIR"))
    COCO_IMAGE_DOWNLOAD_URL = os.getenv("COCO_IMAGE_DOWNLOAD_URL")
    COCO_ANNOTATION_DOWNLOAD_URL = os.getenv("COCO_ANNOTATION_DOWNLOAD_URL")

    IMG_DIR_NAME = os.path.splitext(os.path.basename(COCO_IMAGE_DOWNLOAD_URL))[0]
    IMG_DIR = os.path.join(COCO_BASE_DIR, IMG_DIR_NAME)
    ANNOTATION_DIR_NAME = os.path.splitext(os.path.basename(COCO_ANNOTATION_DOWNLOAD_URL))[0]
    ANNOTATION_FILE = os.path.join(COCO_BASE_DIR, ANNOTATION_DIR_NAME, f"instances_{IMG_DIR_NAME}.json")

    OUTPUT_DIR = os.path.join(DATA_DIR, os.getenv("OUTPUT_DIR"))
    FILTERED_IMAGE_DIR = os.path.join(OUTPUT_DIR, os.getenv("FILTERED_IMAGE_DIR"))
    RESULT_DIR = os.path.join(OUTPUT_DIR, os.getenv("RESULT_DIR"))

    # ==== Model ====
    ULTRALYTICS_HOME = os.getenv("ULTRALYTICS_HOME")
    SEG_MODEL_PATH = os.path.join(ULTRALYTICS_HOME, os.getenv("SEG_MODEL_FILE"))
    POS_MODEL_PATH = os.path.join(ULTRALYTICS_HOME, os.getenv("POS_MODEL_FILE"))
    
    # ==== YOLO ====
    DEVICE = os.getenv("DEVICE", None)
    CONF = float(os.getenv("CONF", 0.25))
    IMGSZ = int(os.getenv("IMGSZ", 640))
