from keras import layers, models

def down(filters, kernel_size, apply_batch_normalization=True):
    """Create a downsampling block."""
    block = models.Sequential()
    block.add(layers.Conv2D(filters, kernel_size, padding='same', strides=2))
    if apply_batch_normalization:
        block.add(layers.BatchNormalization())
    block.add(layers.LeakyReLU())
    return block

def up(filters, kernel_size, dropout=False):
    """Create an upsampling block."""
    block = models.Sequential()
    block.add(layers.Conv2DTranspose(filters, kernel_size, padding='same', strides=2))
    if dropout:
        block.add(layers.Dropout(0.2))
    block.add(layers.LeakyReLU())
    return block

def build_model(input_shape):
    """Build the image colorization model."""
    inputs = layers.Input(shape=input_shape)
    # Downsampling
    d1 = down(128, (3, 3), False)(inputs)
    d2 = down(128, (3, 3), False)(d1)
    d3 = down(256, (3, 3), True)(d2)
    d4 = down(512, (3, 3), True)(d3)
    d5 = down(512, (3, 3), True)(d4)
    # Upsampling
    u1 = up(512, (3, 3))(d5)
    u1 = layers.concatenate([u1, d4])
    u2 = up(256, (3, 3))(u1)
    u2 = layers.concatenate([u2, d3])
    u3 = up(128, (3, 3))(u2)
    u3 = layers.concatenate([u3, d2])
    u4 = up(128, (3, 3))(u3)
    u4 = layers.concatenate([u4, d1])
    u5 = up(3, (3, 3))(u4)
    u5 = layers.concatenate([u5, inputs])
    output = layers.Conv2D(3, (2, 2), strides=1, padding='same')(u5)
    return models.Model(inputs=inputs, outputs=output)
