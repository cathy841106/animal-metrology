import cv2
import numpy as np

def run_segmentation(seg_model, img_path, imgsz=640, conf=0.25, device=None):
    """
    Run segmentation model inference on a single image.

    Args:
        seg_model: Loaded YOLO segmentation model.
        img_path: Path to the input image.
        imgsz: Image size for inference. Default is 640.
        conf: Confidence threshold. Default is 0.25.
        device: Device to run inference on (e.g., cpu, cuda:0, 0, npu or npu:0), Default is None.

    Returns:
        results: Prediction result from the model.
    """
    
    results = seg_model.predict(img_path, imgsz=imgsz, conf=conf, device=device)[0]

    return results

def detect_eyes(model, image, bbox, imgsz=640, conf=0.25, device=None):
    """
    Detect left and right eye positions within a bounding box.
    
    Parameters:
        model: pose detection model
        image: Input image (numpy array)
        bbox: bounding box (xyxy format [x1, y1, x2, y2] )
        imgsz: Image size for inference. Default is 640.
        conf: Confidence threshold. Default is 0.25.
        device: Device to run inference on (e.g., cpu, cuda:0, 0, npu or npu:0), Default is None.
        
    Returns: 
        (left_eye, right_eye) in image coordinates or (None, None)
    """
    x1, y1, x2, y2 = map(int, bbox)
    roi = image[y1:y2, x1:x2]

    # Predict keypoints in the ROI
    results = model.predict(roi, imgsz=imgsz, conf=conf, device=device)

    # No detections
    if not results or len(results) == 0:
        return None, None

    # Take the first detection only
    result = results[0]
    if not hasattr(result, "keypoints") or result.keypoints is None:
        return None, None

    # Get keypoints for first detected animal/person
    keypoints = result.keypoints.xy  # shape: (num_people, num_keypoints, 2)

    if len(keypoints) == 0:
        return None, None

    # Take the first detected object
    first_kpts = keypoints[0].cpu().numpy()

    # COCO keypoint indices: 1 = left_eye, 2 = right_eye
    left_eye = first_kpts[1] + np.array([x1, y1])  # convert back to full image coords
    right_eye = first_kpts[2] + np.array([x1, y1])

    left_eye = tuple(left_eye.astype(int))
    right_eye = tuple(right_eye.astype(int))

    return left_eye, right_eye