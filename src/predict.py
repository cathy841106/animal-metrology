import os
import itertools
import cv2
import numpy as np
import csv
from ultralytics import YOLO
from pycocotools.coco import COCO

from config import Config
from detector import run_segmentation, detect_eyes
from utils.distance import euclidean
from utils.coco_utils import (
    build_file_to_imgid, 
    get_category_ids_by_supercategory, 
    get_gt_bboxes
)
from utils.yolo_utils import xywh_to_xyxy, extract_bboxes
from utils.draw import draw_results
from utils.metrics import bbox_iou, compute_ious

# ==== Config ====
ANNOTATION_FILE = Config.ANNOTATION_FILE
 
FILTERED_IMAGE_DIR = Config.FILTERED_IMAGE_DIR       
RESULT_DIR = Config.RESULT_DIR                

SEG_MODEL_PATH = Config.SEG_MODEL_PATH      
POS_MODEL_PATH = Config.POS_MODEL_PATH 

DEVICE = Config.DEVICE
CONF = Config.CONF
IMGSZ = Config.IMGSZ


# ==== Create directory if it doesn't exist ====
os.makedirs(RESULT_DIR, exist_ok=True)


# ==== Load COCO annotations ====
coco = COCO(ANNOTATION_FILE)

# Build mapping from file_name to image_id
file_to_imgid = build_file_to_imgid(coco)

# Get category ids for supercategory:'animal'
animal_cat_ids = get_category_ids_by_supercategory(coco, ['animal'])


# ==== Load YOLO model ====
seg_model = YOLO(SEG_MODEL_PATH)
pos_model = YOLO(POS_MODEL_PATH)


# ==== Initialize containers for evaluation metrics ====
all_ious = []               # Store IoU for bbox accuracy


# ==== Iterate through selected images ====
for img_file in os.listdir(FILTERED_IMAGE_DIR):
    img_path = os.path.join(FILTERED_IMAGE_DIR, img_file)
    
    print (f"Predicting image file: {img_path}")
    
    # Read image
    image = cv2.imread(img_path)    
    if image is None:
        print (f"Cannot read image {img_path}")
        continue

    # Get COCO image id
    img_id = file_to_imgid.get(img_file, None)
    if img_id is None:
        print (f"COCO annotation not found for {img_file}")
        continue

    # Load ground truth annotations
    gt_bboxes = get_gt_bboxes(coco, img_id, animal_cat_ids)
    
    
    # ==== YOLO segmentation prediction ====
    results = run_segmentation(seg_model, img_path, imgsz=IMGSZ, conf=CONF, device=DEVICE)

    # Record the predicted bbox
    pred_bboxes_xyxy, pred_bboxes_xywh = extract_bboxes(results, animal_cat_ids)


    # ==== Detect eyes & calculate distances ====
    animals_pred = []
    for bbox in pred_bboxes_xyxy:
        # Detect eyes
        left, right = detect_eyes(pos_model, image, bbox, imgsz=IMGSZ, conf=CONF, device=DEVICE)
        
        # Calculate distances between left eye and right eye
        eye_dist = euclidean(left, right) if left and right else None  # Use euclidean distance
        animals_pred.append({
            "bbox": bbox,
            "left_eye": left,
            "right_eye": right,
            "eye_distance": eye_dist
        })

    # Calculate distances between right eyes for all animal pairs
    inter_animal_results = []
    
    for idx1, idx2 in itertools.combinations(range(len(animals_pred)), 2): # Generate all pairs of animals
        a1 = animals_pred[idx1]
        a2 = animals_pred[idx2]

        if a1["right_eye"] and a2["right_eye"]:  # Only compute if both animals have right_eye detected
            distance = euclidean(a1["right_eye"], a2["right_eye"])  # Use euclidean distance

            inter_animal_results.append({
                "animal_1_id": idx1 + 1,
                "animal_2_id": idx2 + 1,
                "a1_right_x": a1["right_eye"][0],
                "a1_right_y": a1["right_eye"][1],
                "a2_right_x": a2["right_eye"][0],
                "a2_right_y": a2["right_eye"][1],
                "distance": distance
            })

    # ==== IoU calculation ====
    # Compute IoU for each predicted bbox with corresponding ground truth
    ious = compute_ious(gt_bboxes, pred_bboxes_xywh)
    all_ious.extend(ious)
     
    
    # ==== Draw results on image ====
    out_img = draw_results(image.copy(), animals_pred)
    img_output_path = os.path.join(RESULT_DIR, img_file)
    cv2.imwrite(img_output_path, out_img)
    
    print (f"Exported predicted image to {img_output_path}")


    # ==== Save results to CSV ====
    # Save per-animal data
    csv_file_animals = os.path.join(RESULT_DIR, f"{os.path.splitext(img_file)[0]}_animals.csv")
    with open(csv_file_animals, "w", newline="") as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow([
            "animal_id", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
            "left_eye_x", "left_eye_y", "right_eye_x", "right_eye_y", "eye_distance"
        ])
        
        # Write data
        for idx, a in enumerate(animals_pred, 1):  # start IDs from 1
            le = a["left_eye"] if a["left_eye"] else (None, None)
            re = a["right_eye"] if a["right_eye"] else (None, None)
            x1, y1, x2, y2 = a["bbox"]
            writer.writerow([idx, x1, y1, x2, y2, le[0], le[1], re[0], re[1], a["eye_distance"]])

    print (f"Exported animal detection result to {csv_file_animals}")

    # Save inter-distance data
    csv_file_inter_distance = os.path.join(RESULT_DIR, f"{os.path.splitext(img_file)[0]}_inter_distance.csv")
    with open(csv_file_inter_distance, "w", newline="") as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow([
            "animal_1_id", "animal_2_id", "a1_right_x", "a1_right_y",
            "a2_right_x", "a2_right_y", "distance"
        ])

        # Write data
        for row in inter_animal_results:
            writer.writerow([
                row["animal_1_id"],
                row["animal_2_id"],
                row["a1_right_x"],
                row["a1_right_y"],
                row["a2_right_x"],
                row["a2_right_y"],
                row["distance"]
            ])
        
    print (f"Exported inter-distance result to {csv_file_inter_distance}")


# ==== Final statistics ====
print ("========= Final Statistics Results =========")

final_iou = (sum(all_ious) / len(all_ious)) if all_ious else 0
print (f"Final IoU: {final_iou:.4f}")
    
print ("============================================")