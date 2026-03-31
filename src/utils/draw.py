import cv2

def draw_results(image, animals, eye_box_size=8):
    """
    Draw bounding boxes for animals and eyes on the image.
    
    Parameters:
        image: np.array, the original image
        animals: list of dicts with keys 'bbox', 'left_eye', 'right_eye'
        eye_box_size: int, half-size of the eye box (default=8)
        
    Returns:
        image with drawings
    """
    for a in animals:
        # Draw animal bbox
        x1, y1, x2, y2 = map(int, a["bbox"])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw eyes as small boxes
        for eye_key, color in [("left_eye", (255, 0, 0)), ("right_eye", (0, 0, 255))]:
            eye = a.get(eye_key)
            if eye:
                ex, ey = eye
                ex, ey = int(ex), int(ey)
                cv2.rectangle(image, 
                              (ex - eye_box_size, ey - eye_box_size),
                              (ex + eye_box_size, ey + eye_box_size),
                              color, 2)

    return image