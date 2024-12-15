import re
import cv2
import numpy as np

def sorted_alphanumeric(data):
    """Sort filenames in alphanumeric order."""
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)

def save_image_pairs(color, grayscale, predicted, save_path):
    """Save color, grayscale, and predicted images side by side."""
    combined = np.hstack((
        (color * 255).astype(np.uint8),
        (grayscale * 255).astype(np.uint8),
        (predicted * 255).astype(np.uint8)
    ))
    cv2.imwrite(save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
