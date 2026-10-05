import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

def main():
    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pothole-detection-dataset')
    
    print("=" * 60)
    print("  Pothole Detection using CNN")
    print("=" * 60)
    print(f"Dataset path: {dataset_dir}")
    print(f"TensorFlow version: {tf.__version__}")

    # 1. Data Preprocessing and Augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    print("\n[1/4] Loading training set...")
    training_set = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary',
        subset='training'
    )

    print("\n[2/4] Loading validation set...")
    validation_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary',
        subset='validation'
    )

    print(f"\nClass Indices: {training_set.class_indices}")
    # Invert class indices mapping (0 -> classA, 1 -> classB)
    index_to_class = {v: k for k, v in training_set.class_indices.items()}

    # 2. Building the CNN Architecture
    print("\n[3/4] Building CNN Model...")
    cnn = tf.keras.models.Sequential([
        # Convolutional layer 1 + MaxPooling
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
        
        # Convolutional layer 2 + MaxPooling
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
        
        # Flattening & Dense Layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])

    cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    cnn.summary()

    # 3. Training the Model
    epochs = 15
    print(f"\n[4/4] Training the CNN for {epochs} epochs...")
    history = cnn.fit(
        x=training_set,
        validation_data=validation_generator,
        epochs=epochs
    )

    # Save trained model weights
    model_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pothole_detector_model.keras')
    cnn.save(model_save_path)
    print(f"\nModel successfully saved to {model_save_path}")

    # 4. Running Inference on Test Images
    print("\n" + "=" * 60)
    print("  Testing Inference on Sample Images")
    print("=" * 60)

    test_samples = ['trial.jpeg', 'trial2.jpg']
    for sample_name in test_samples:
        sample_path = os.path.join(dataset_dir, sample_name)
        if os.path.exists(sample_path):
            test_img = load_img(sample_path, target_size=(64, 64))
            test_arr = img_to_array(test_img) / 255.0
            test_arr = np.expand_dims(test_arr, axis=0)
            
            raw_prob = cnn.predict(test_arr, verbose=0)[0][0]
            predicted_class_idx = 1 if raw_prob >= 0.5 else 0
            predicted_label = index_to_class.get(predicted_class_idx, 'unknown')
            confidence = raw_prob if predicted_class_idx == 1 else (1.0 - raw_prob)

            print(f"File: {sample_name} -> Predicted: {predicted_label.upper()} (Confidence: {confidence*100:.2f}%, Raw Sigmoid Output: {raw_prob:.4f})")
        else:
            print(f"Sample {sample_name} not found at {sample_path}")

if __name__ == '__main__':
    main()
