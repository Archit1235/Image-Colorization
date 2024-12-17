import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from data_loader import load_and_preprocess_images
from utils import save_image_pairs

SIZE = 160
MODEL_PATH = './models/model.keras'
COLOR_IMAGE_PATH = './data/color'
GRAY_IMAGE_PATH = './data/gray'
RESULTS_PATH = './results/evaluation'
BATCH_SIZE = 64

def load_test_data():
    """Load and preprocess test data."""
    color_images = load_and_preprocess_images(COLOR_IMAGE_PATH, SIZE, color=True)
    gray_images = load_and_preprocess_images(GRAY_IMAGE_PATH, SIZE, color=False)

    test_gray, test_color = gray_images[6000:], color_images[6000:]

    test_gray = np.reshape(test_gray, (len(test_gray), SIZE, SIZE, 3))
    test_color = np.reshape(test_color, (len(test_color), SIZE, SIZE, 3))
    return test_gray, test_color

def plot_sample_predictions(model, gray_images, color_images, num_samples=5):
    """Plot original, grayscale, and predicted images side by side."""
    os.makedirs(RESULTS_PATH, exist_ok=True)

    for i in range(num_samples):
        original = color_images[i]
        grayscale = gray_images[i]
        predicted = np.clip(model.predict(grayscale.reshape(1, SIZE, SIZE, 3)), 0.0, 1.0).reshape(SIZE, SIZE, 3)

        save_image_pairs(original, grayscale, predicted, os.path.join(RESULTS_PATH, f'prediction_{i}.png'))

    print(f"Sample predictions saved in {RESULTS_PATH}")

def plot_metric_histories(history_path):
    """Plot training and validation accuracy and loss from a saved history."""
    history = np.load(history_path, allow_pickle=True).item()
    epochs = range(1, len(history['loss']) + 1)

    plt.figure(figsize=(12, 6))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['loss'], 'r', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'b', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['accuracy'], 'r', label='Training Accuracy')
    plt.plot(epochs, history['val_accuracy'], 'b', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, 'metric_histories.png'))
    plt.show()

def calculate_psnr(original_images, predicted_images):
    """Calculate the Peak Signal-to-Noise Ratio (PSNR) for the predictions."""
    psnr_values = []
    for original, predicted in zip(original_images, predicted_images):
        psnr = tf.image.psnr(original, predicted, max_val=1.0).numpy()
        psnr_values.append(psnr)
    return psnr_values

def plot_psnr_distribution(psnr_values):
    """Plot a histogram of PSNR values."""
    plt.figure(figsize=(8, 6))
    plt.hist(psnr_values, bins=20, color='blue', edgecolor='black')
    plt.title('PSNR Distribution')
    plt.xlabel('PSNR (dB)')
    plt.ylabel('Frequency')
    plt.grid(axis='y')
    plt.savefig(os.path.join(RESULTS_PATH, 'psnr_distribution.png'))
    plt.show()

def calculate_ssim(original_images, predicted_images):
    """Calculate the Structural Similarity Index Measure (SSIM) for the predictions."""
    ssim_values = []
    for original, predicted in zip(original_images, predicted_images):
        ssim = tf.image.ssim(original, predicted, max_val=1.0).numpy()
        ssim_values.append(ssim)
    return ssim_values

def plot_ssim_distribution(ssim_values):
    """Plot a histogram of SSIM values."""
    plt.figure(figsize=(8, 6))
    plt.hist(ssim_values, bins=20, color='green', edgecolor='black')
    plt.title('SSIM Distribution')
    plt.xlabel('SSIM')
    plt.ylabel('Frequency')
    plt.grid(axis='y')
    plt.savefig(os.path.join(RESULTS_PATH, 'ssim_distribution.png'))
    plt.show()

def main():
    os.makedirs(RESULTS_PATH, exist_ok=True)

    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("Loading test data...")
    gray_images, color_images = load_test_data()

    print("Evaluating model on test data...")
    results = model.evaluate(gray_images, color_images, batch_size=BATCH_SIZE, verbose=1)
    print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

    print("Generating sample predictions...")
    plot_sample_predictions(model, gray_images, color_images, num_samples=5)

    print("Calculating PSNR...")
    predicted_images = np.clip(model.predict(gray_images), 0.0, 1.0)
    psnr_values = calculate_psnr(color_images, predicted_images)
    plot_psnr_distribution(psnr_values)
    print(f"Mean PSNR: {np.mean(psnr_values):.2f} dB")

    print("Calculating SSIM...")
    ssim_values = calculate_ssim(color_images, predicted_images)
    plot_ssim_distribution(ssim_values)
    print(f"Mean SSIM: {np.mean(ssim_values):.2f}")

    history_path = './results/training_history.npy'
    if os.path.exists(history_path):
        print("Plotting training and validation metrics...")
        plot_metric_histories(history_path)

if __name__ == "__main__":
    main()
