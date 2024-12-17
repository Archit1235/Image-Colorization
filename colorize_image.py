import argparse
import os
from utils import load_image, save_image
from keras.models import load_model
import numpy as np

SIZE = 160
MODEL_PATH = './models/model.keras'
DEFAULT_OUTPUT_DIR = './test/output'

model = load_model(MODEL_PATH)

def colorize_image(input_path, output_path):
    gray_image = load_image(input_path, size=SIZE, grayscale=True)
    
    gray_image = gray_image.reshape(1, SIZE, SIZE, 3)
    
    predicted_image = model.predict(gray_image)
    
    predicted_image = np.clip(predicted_image, 0.0, 1.0)
    
    predicted_image = predicted_image[0]
    save_image(predicted_image, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Colorize a grayscale image.")
    parser.add_argument("input_path", type=str, help="Path to the input grayscale image")
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save the colorized output image (optional)",
    )
    
    args = parser.parse_args()
    
    if args.output_path is None:
        filename = os.path.basename(args.input_path)
        args.output_path = os.path.join(DEFAULT_OUTPUT_DIR, filename)
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    colorize_image(args.input_path, args.output_path)
    print(f"Colorized image saved to: {args.output_path}")