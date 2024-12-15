import os
import tensorflow as tf
import numpy as np
from data_loader import load_and_preprocess_images
from model import build_model
from utils import save_image_pairs

SIZE = 160
EPOCHS = 50
BATCH_SIZE = 64
COLOR_IMAGE_PATH = './data/color'
GRAY_IMAGE_PATH = './data/gray'
RESULTS_PATH = './results'
MODEL_PATH = './models'

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def configure_gpus():
    """Configure TensorFlow to use available GPUs with memory growth."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Configured {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
        except RuntimeError as e:
            print(f"Error configuring GPUs: {e}")

def main():
    configure_gpus()

    # Set up distributed training strategy
    strategy = tf.distribute.MirroredStrategy()
    print(f"Number of devices: {strategy.num_replicas_in_sync}")

    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_PATH, exist_ok=True)

    # Load Data
    print("Loading images...")
    color_images = load_and_preprocess_images(COLOR_IMAGE_PATH, SIZE, stop_filename='6000.jpg', color=True)
    gray_images = load_and_preprocess_images(GRAY_IMAGE_PATH, SIZE, stop_filename='6000.jpg', color=False)

    # Split Data
    train_gray, train_color = gray_images[:5500], color_images[:5500]
    test_gray, test_color = gray_images[5500:], color_images[5500:]

    # Reshape
    train_gray = np.reshape(train_gray, (len(train_gray), SIZE, SIZE, 3))
    train_color = np.reshape(train_color, (len(train_color), SIZE, SIZE, 3))
    test_gray = np.reshape(test_gray, (len(test_gray), SIZE, SIZE, 3))
    test_color = np.reshape(test_color, (len(test_color), SIZE, SIZE, 3))

    print(f'Train color image shape: {train_color.shape}')
    print(f'Test color image shape: {test_color.shape}')

    # Build and Compile Model
    with strategy.scope():
        model = build_model((SIZE, SIZE, 3))
        model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['accuracy'])
        model.summary()

    # Train Model
    model.fit(
        train_gray, train_color, 
        validation_data=(test_gray, test_color), 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE
    )

    # Evaluate Model
    model.evaluate(test_gray, test_color)

    # Save Model
    model.save(f'{MODEL_PATH}/model.keras')
    model.save_weights(f'{MODEL_PATH}/model.weights.h5')

    # Prediction and Saving Images
    for i in range(50, 58):
        predicted = np.clip(model.predict(test_gray[i].reshape(1, SIZE, SIZE, 3)), 0.0, 1.0).reshape(SIZE, SIZE, 3)
        save_image_pairs(test_color[i], test_gray[i], predicted, os.path.join(RESULTS_PATH, f'result_{i}.png'))

if __name__ == "__main__":
    main()
