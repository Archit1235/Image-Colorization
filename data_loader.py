import os
import cv2
import numpy as np
from tqdm import tqdm
from utils import sorted_alphanumeric

def load_and_preprocess_images(path, size, stop_filename=None, color=True):
    """Load and preprocess images from a given directory."""
    images = []
    files = sorted_alphanumeric(os.listdir(path))
    for file in tqdm(files):
        if file == stop_filename:
            break
        img = cv2.imread(os.path.join(path, file), 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if color else img
        img = cv2.resize(img, (size, size))
        img = img.astype('float32') / 255.0
        images.append(img)
    return np.array(images)
