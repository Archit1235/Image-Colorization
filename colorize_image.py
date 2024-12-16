import os
import numpy as np
import tensorflow as tf
from utils import load_image, save_image

SIZE = 160
MODEL_PATH = '.models/colorModel.h5'

def colorize_image(image_path, output_path):
    """Load a grayscale image, colorize it, and save the result."""
    model = tf.keras.models.load_model(MODEL_PATH)
    gray_image = load_image(image_path, SIZE, grayscale=True)
    gray_image = gray_image.reshape(1, SIZE, SIZE, 3)

    # Predict and save
    colorized_image = np.clip(model.predict(gray_image), 0.0, 1.0).reshape(SIZE, SIZE, 3)
    save_image(colorized_image, output_path)
    print(f"Colorized image saved to: {output_path}")

if __name__ == "__main__":
    test_image = './data/gray/sample.jpg'  # Replace with the grayscale image path
    output_image = './results/colorized_sample.jpg'
    colorize_image(test_image, output_image)
