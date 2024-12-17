import argparse
import os
from utils import load_image, save_image
from keras.models import load_model
import numpy as np

# Constants
SIZE = 160
MODEL_PATH = './models/model.keras'
DEFAULT_OUTPUT_DIR = './test/output'

# Load the pre-trained model
model = load_model(MODEL_PATH)

def colorize_image(input_path, output_path):
    """
    Colorizes a grayscale image and saves the result.
    
    Parameters:
        input_path (str): Path to the input grayscale image.
        output_path (str): Path to save the colorized output image.
    """
    # Load and preprocess the input grayscale image
    gray_image = load_image(input_path, size=SIZE, grayscale=True)
    
    # Add batch dimension for prediction
    gray_image = gray_image.reshape(1, SIZE, SIZE, 3)
    
    # Predict the colorized image
    predicted_image = model.predict(gray_image)
    
    # Clip predictions to [0, 1] range
    predicted_image = np.clip(predicted_image, 0.0, 1.0)
    
    # Remove batch dimension and save the colorized image
    predicted_image = predicted_image[0]
    save_image(predicted_image, output_path)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Colorize a grayscale image.")
    parser.add_argument("input_path", type=str, help="Path to the input grayscale image")
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save the colorized output image (optional)",
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Automatically generate output path if not provided
    if args.output_path is None:
        filename = os.path.basename(args.input_path)  # Get the filename from input path
        args.output_path = os.path.join(DEFAULT_OUTPUT_DIR, filename)
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Call the colorize_image function with CLI arguments
    colorize_image(args.input_path, args.output_path)
    print(f"Colorized image saved to: {args.output_path}")