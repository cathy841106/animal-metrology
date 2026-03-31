
def bbox_iou(boxA, boxB):
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    
    Parameters:
        boxA: bounding box in [x, y, w, h] format
        boxB: bounding box in [x, y, w, h] format
        
    Returns:
        IoU score (float)
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    denom = boxAArea + boxBArea - interArea
    return interArea / denom if denom > 0 else 0

def compute_ious(gt_bboxes, pred_bboxes):
    """
    Compute IoU scores between ground truth and predicted bounding boxes (pairwise by index).
    
    Parameters:
        gt_bboxes: list of ground truth bounding boxes [x, y, w, h]
        pred_bboxes: list of predicted bounding boxes [x, y, w, h]
        
    Returns:
        List of IoU scores
    """
    ious = []
    for i, gt in enumerate(gt_bboxes):
        if i < len(pred_bboxes):
            ious.append(bbox_iou(gt, pred_bboxes[i]))
    return ious