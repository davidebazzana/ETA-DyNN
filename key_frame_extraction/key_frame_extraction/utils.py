import cv2 as cv
import numpy as np

def is_inside(point, rectangle):
    px, py = point
    (x1, y1), (x2, y2) = rectangle
    
    return x1 <= px <= x2 and y1 <= py <= y2

def scale_image(img, scale):
    h, w = img.shape[:2]
    return cv.resize(img, (int(w * scale), int(h * scale)))

def resize_to_fit(img, max_width, max_height):
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h)
    if scale < 1.0:  # Resize only if image is bigger
        img_resized = cv.resize(img, (int(w * scale), int(h * scale)))
    else:
        img_resized = img.copy()
        scale = 1.0
    return img_resized, scale

def map_bboxes_to_original(bboxes_resized, scale):
    """
    Maps bounding boxes from resized image back to original coordinates.

    Parameters:
        bboxes_resized (list of tuples): Bounding boxes as (x, y, w, h) in resized image
        scale (float): Scale factor used during resizing

    Returns:
        bboxes_original (list of tuples): Bounding boxes in original image coordinates
    """
    if scale == 0:
        raise ValueError("Scale must be non-zero")

    if isinstance(bboxes_resized, tuple):
        return tuple(int(i / scale) for i in bboxes_resized)
    
    return [
        tuple(int(i / scale) for i in item)
        if item is not None else None for item in bboxes_resized
    ]

def compute_iou(boxA, boxB, bb_format:str="x_y_max"):
    """Compute Intersection Over Union of two bounding boxes.

    The assumed bounding box format is: (x_min, y_min, x_max, y_max).
    The format (x_min, y_min, width, height) can be used but must be specified
    using the bb_format argument "width_height".
    """
    if bb_format == "width_height":
        # Convert (x, y, w, h) to (x_min, y_min, x_max, y_max)
        boxA = (boxA[0], boxA[1], boxA[0] + boxA[2], boxA[1] + boxA[3])
        boxB = (boxB[0], boxB[1], boxB[0] + boxB[2], boxB[1] + boxB[3])

    # Coordinates of intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute area of intersection
    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area = inter_width * inter_height

    # Compute areas of individual boxes
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute IoU
    iou = inter_area / float(boxA_area + boxB_area - inter_area) if (boxA_area + boxB_area - inter_area) != 0 else 0
    return iou
