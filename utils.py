import re
import cv2
import numpy as np
from PIL import Image

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

def load_image(image_path, size=160, grayscale=False):
    """
    Loads and preprocesses an image.
    
    Parameters:
        image_path (str): Path to the image.
        size (int): Target size for resizing (width and height).
        grayscale (bool): Whether to load the image in grayscale.
        
    Returns:
        np.array: Preprocessed image of shape (size, size, 3).
    """
    if grayscale:
        # Load in grayscale mode
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Image not found or invalid format: {image_path}")
        # Convert single channel grayscale to 3-channel grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        # Load in color mode
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Image not found or invalid format: {image_path}")
        # Convert BGR (OpenCV default) to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize the image
    image = cv2.resize(image, (size, size))
    
    # Normalize to range [0, 1]
    image = image.astype("float32") / 255.0
    
    return image


def save_image(image, output_path):
    """
    Save a NumPy array as an image.

    Parameters:
        image (np.array): Image data to save.
        output_path (str): Path to save the image.
    """
    image = (image * 255).astype(np.uint8)
    img = Image.fromarray(image)
    img.save(output_path)