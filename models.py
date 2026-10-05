import tensorflow as tf
from tensorflow.keras import layers, models, Model

IMG_SIZE = (128, 128)
INPUT_SHAPE = (128, 128, 3)

def build_custom_cnn(input_shape=INPUT_SHAPE):
    """
    Baseline Custom CNN Model
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ], name="Custom_CNN")
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def build_alexnet(input_shape=INPUT_SHAPE):
    """
    AlexNet Architecture (adapted for 128x128 input)
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        # Conv Layer 1
        layers.Conv2D(64, (7, 7), strides=2, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((3, 3), strides=2, padding='same'),
        
        # Conv Layer 2
        layers.Conv2D(192, (5, 5), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((3, 3), strides=2, padding='same'),
        
        # Conv Layer 3, 4, 5
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((3, 3), strides=2, padding='same'),
        
        # Dense Classification Layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ], name="AlexNet")
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def _inception_block(x, f1, f3_r, f3, f5_r, f5, pool_proj):
    """
    GoogLeNet Inception Block
    """
    # 1x1 Conv branch
    b1 = layers.Conv2D(f1, (1, 1), padding='same', activation='relu')(x)
    
    # 1x1 Conv -> 3x3 Conv branch
    b2 = layers.Conv2D(f3_r, (1, 1), padding='same', activation='relu')(x)
    b2 = layers.Conv2D(f3, (3, 3), padding='same', activation='relu')(b2)
    
    # 1x1 Conv -> 5x5 Conv branch
    b3 = layers.Conv2D(f5_r, (1, 1), padding='same', activation='relu')(x)
    b3 = layers.Conv2D(f5, (5, 5), padding='same', activation='relu')(b3)
    
    # 3x3 MaxPool -> 1x1 Conv branch
    b4 = layers.MaxPooling2D((3, 3), strides=1, padding='same')(x)
    b4 = layers.Conv2D(pool_proj, (1, 1), padding='same', activation='relu')(b4)
    
    return layers.concatenate([b1, b2, b3, b4], axis=-1)


def build_googlenet(input_shape=INPUT_SHAPE):
    """
    GoogLeNet (Inception Architecture)
    """
    inputs = layers.Input(shape=input_shape)
    
    # Stem
    x = layers.Conv2D(64, (7, 7), strides=2, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(64, (1, 1), padding='same', activation='relu')(x)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
    
    # Inception Modules
    x = _inception_block(x, 64, 96, 128, 16, 32, 32)
    x = _inception_block(x, 128, 128, 192, 32, 64, 64)
    x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
    
    x = _inception_block(x, 192, 96, 208, 16, 48, 64)
    
    # Classification Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="GoogLeNet_Inception")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def build_vgg16(input_shape=INPUT_SHAPE, pretrained=True):
    """
    VGG-16 Architecture
    """
    weights = 'imagenet' if pretrained else None
    base_model = tf.keras.applications.VGG16(
        weights=weights,
        include_top=False,
        input_shape=input_shape
    )
    # Freeze base feature extractor layers for fine-tuning
    if pretrained:
        for layer in base_model.layers[:-4]:
            layer.trainable = False
            
    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="VGG16")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def build_vgg19(input_shape=INPUT_SHAPE, pretrained=True):
    """
    VGG-19 Architecture
    """
    weights = 'imagenet' if pretrained else None
    base_model = tf.keras.applications.VGG19(
        weights=weights,
        include_top=False,
        input_shape=input_shape
    )
    # Freeze base feature extractor layers for fine-tuning
    if pretrained:
        for layer in base_model.layers[:-4]:
            layer.trainable = False
            
    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="VGG19")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model
