
def xywh_to_xyxy(box):
    """
    Convert bounding box from center format to corner format.
    
    Parameters:
        box: bounding box in [x_center, y_center, width, height]
        
    Returns:
        Bounding box in [x1, y1, x2, y2] format (int)
    """
    x_c, y_c, w, h = box
    x1 = int(x_c - w / 2)
    y1 = int(y_c - h / 2)
    x2 = int(x_c + w / 2)
    y2 = int(y_c + h / 2)
    return [x1, y1, x2, y2]

def extract_bboxes(results, cat_ids):
    """
    Extract bounding boxes from model results filtered by category IDs.
    
    Parameters:
        results: YOLO prediction result object
        cat_ids: list of category IDs to keep
        
    Returns:
        pred_bboxes_xyxy: list of bounding boxes in [x1, y1, x2, y2] format
        pred_bboxes_xywh: list of bounding boxes in [x_center, y_center, width, height] format
    """
    pred_bboxes_xyxy = []
    pred_bboxes_xywh = []
    
    for xywh, cls_id in zip(results.boxes.xywh, results.boxes.cls):
        cls_id = int(cls_id)
        if cls_id in cat_ids:  # Just use classes that are in specific category
            bbox = xywh_to_xyxy(xywh)
            pred_bboxes_xyxy.append(bbox)
            pred_bboxes_xywh.append(xywh)

    return pred_bboxes_xyxy, pred_bboxes_xywh