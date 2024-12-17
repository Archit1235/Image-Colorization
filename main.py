import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_and_preprocess_images
from model import build_model
from utils import save_image_pairs

SIZE = 160
EPOCHS = 50
BATCH_SIZE = 64
LOGS_PATH = './results/logs'
MODEL_PATH = './models'
RESULTS_PATH = './results'
GRAY_IMAGE_PATH = './data/gray'
COLOR_IMAGE_PATH = './data/color'

os.makedirs(LOGS_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)

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

def plot_metrics(history):
    """Plot training and validation accuracy and loss."""
    epochs = range(1, len(history.history['loss']) + 1)

    # Loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history['loss'], 'r', label='Training Loss')
    plt.plot(epochs, history.history['val_loss'], 'b', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history['accuracy'], 'r', label='Training Accuracy')
    plt.plot(epochs, history.history['val_accuracy'], 'b', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, 'training_metrics.png'))
    plt.show()

def main():
    configure_gpus()

    # Set up distributed training strategy
    strategy = tf.distribute.MirroredStrategy()
    print(f"Number of devices: {strategy.num_replicas_in_sync}")

    # Load Data
    print("Loading images...")
    color_images = load_and_preprocess_images(COLOR_IMAGE_PATH, SIZE, color=True)
    gray_images = load_and_preprocess_images(GRAY_IMAGE_PATH, SIZE, color=False)

    # Split Data
    train_gray, train_color = gray_images[:6000], color_images[:6000]
    test_gray, test_color = gray_images[6000:], color_images[6000:]

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

    # Callbacks for TensorBoard
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOGS_PATH, histogram_freq=1)

    # Train Model
    history = model.fit(
        train_gray, train_color,
        validation_data=(test_gray, test_color),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[tensorboard_callback]
    )

    # Evaluate Model
    evaluation = model.evaluate(test_gray, test_color)
    print(f"Test Loss: {evaluation[0]}, Test Accuracy: {evaluation[1]}")

    # Save Model
    model.save(f'{MODEL_PATH}/model.keras')
    model.save_weights(f'{MODEL_PATH}/model.weights.h5')

    # Save Training Metrics
    plot_metrics(history)

    # Prediction and Saving Images
    for i in range(0, 1129):
        predicted = np.clip(model.predict(test_gray[i].reshape(1, SIZE, SIZE, 3)), 0.0, 1.0).reshape(SIZE, SIZE, 3)
        save_image_pairs(test_color[i], test_gray[i], predicted, os.path.join(RESULTS_PATH, f'result_{i}.png'))

if __name__ == "__main__":
    main()
