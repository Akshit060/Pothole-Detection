import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from models import build_custom_cnn, build_alexnet, build_googlenet, build_vgg16, build_vgg19, IMG_SIZE, INPUT_SHAPE

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, 'pothole-detection-dataset')
    models_dir = os.path.join(base_dir, 'saved_models')
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 70)
    print("  POTHOLE DETECTION BENCHMARK: 5 DEEP LEARNING ARCHITECTURES")
    print("  1. Custom CNN  |  2. AlexNet  |  3. GoogLeNet (Inception)")
    print("  4. VGG-16      |  5. VGG-19")
    print("=" * 70)
    print(f"Dataset path: {dataset_dir}")
    print(f"Target image size: {IMG_SIZE}")
    print(f"TensorFlow version: {tf.__version__}")

    # Data Generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    batch_size = 32
    training_set = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode='binary',
        subset='training',
        shuffle=True
    )

    validation_set = val_datagen.flow_from_directory(
        dataset_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )

    class_indices = training_set.class_indices
    idx_to_class = {v: k for k, v in class_indices.items()}
    print(f"Classes: {class_indices}")

    # Extract true validation labels for sklearn metrics
    val_labels = validation_set.classes

    # Define model builders
    model_builders = {
        'Custom CNN': build_custom_cnn,
        'AlexNet': build_alexnet,
        'GoogLeNet': build_googlenet,
        'VGG16': build_vgg16,
        'VGG19': build_vgg19
    }

    results = []
    histories = {}
    trained_models = {}

    epochs = 12

    for name, builder_func in model_builders.items():
        print("\n" + "-" * 70)
        print(f"  Training Model: {name}")
        print("-" * 70)
        
        start_time = time.time()
        model = builder_func(input_shape=INPUT_SHAPE)
        
        # Train
        history = model.fit(
            training_set,
            validation_data=validation_set,
            epochs=epochs,
            verbose=1
        )
        train_duration = time.time() - start_time
        
        # Save model
        save_path = os.path.join(models_dir, f"{name.lower().replace(' ', '_')}.keras")
        model.save(save_path)
        
        # Evaluate on validation set
        validation_set.reset()
        val_probs = model.predict(validation_set, verbose=0).flatten()
        val_preds = (val_probs >= 0.5).astype(int)
        
        acc = accuracy_score(val_labels, val_preds)
        prec = precision_score(val_labels, val_preds, zero_division=0)
        rec = recall_score(val_labels, val_preds, zero_division=0)
        f1 = f1_score(val_labels, val_preds, zero_division=0)
        try:
            auc = roc_auc_score(val_labels, val_probs)
        except Exception:
            auc = 0.5

        val_loss = history.history['val_loss'][-1]

        results.append({
            'Model': name,
            'Val Accuracy (%)': round(acc * 100, 2),
            'Val Loss': round(val_loss, 4),
            'Precision (%)': round(prec * 100, 2),
            'Recall (%)': round(rec * 100, 2),
            'F1-Score (%)': round(f1 * 100, 2),
            'ROC-AUC': round(auc, 4),
            'Train Time (s)': round(train_duration, 1)
        })

        histories[name] = history.history
        trained_models[name] = model

    # Create Comparison Table
    df_results = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print("  MODEL BENCHMARK COMPARISON RESULTS")
    print("=" * 70)
    print(df_results.to_string(index=False))
    df_results.to_csv(os.path.join(base_dir, 'model_benchmark_results.csv'), index=False)

    # Plot Comparison Charts
    plt.figure(figsize=(16, 12))
    
    # 1. Validation Accuracy Bar Chart
    plt.subplot(2, 2, 1)
    bars = plt.bar(df_results['Model'], df_results['Val Accuracy (%)'], color=['#3498db', '#9b59b6', '#2ecc71', '#e67e22', '#e74c3c'])
    plt.title('Validation Accuracy Comparison (%)', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 105)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 2. F1-Score & Precision/Recall Bar Chart
    plt.subplot(2, 2, 2)
    x = np.arange(len(df_results['Model']))
    width = 0.25
    plt.bar(x - width, df_results['Precision (%)'], width, label='Precision', color='#3498db')
    plt.bar(x, df_results['Recall (%)'], width, label='Recall', color='#2ecc71')
    plt.bar(x + width, df_results['F1-Score (%)'], width, label='F1-Score', color='#e74c3c')
    plt.title('Precision, Recall & F1-Score', fontsize=14, fontweight='bold')
    plt.xticks(x, df_results['Model'])
    plt.ylabel('Percentage (%)')
    plt.ylim(0, 105)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 3. Training & Validation Accuracy Curves Across Epochs
    plt.subplot(2, 2, 3)
    for name, hist in histories.items():
        plt.plot(hist['val_accuracy'], label=f"{name} (Val Acc)", linewidth=2)
    plt.title('Validation Accuracy Progression across Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 4. Training & Validation Loss Curves
    plt.subplot(2, 2, 4)
    for name, hist in histories.items():
        plt.plot(hist['val_loss'], label=f"{name} (Val Loss)", linewidth=2)
    plt.title('Validation Loss Progression across Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    chart_path = os.path.join(base_dir, 'model_comparison.png')
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"\nComparison charts saved to: {chart_path}")

    # Inference on Test Samples
    print("\n" + "=" * 70)
    print("  INFERENCE ON TEST SAMPLES ACROSS ALL MODELS")
    print("=" * 70)
    test_samples = ['trial.jpeg', 'trial2.jpg']
    for sample in test_samples:
        sample_path = os.path.join(dataset_dir, sample)
        if os.path.exists(sample_path):
            img = load_img(sample_path, target_size=IMG_SIZE)
            img_arr = img_to_array(img) / 255.0
            img_arr = np.expand_dims(img_arr, axis=0)
            print(f"\nTest Image: {sample}")
            for name, model in trained_models.items():
                prob = model.predict(img_arr, verbose=0)[0][0]
                pred_label = idx_to_class[1 if prob >= 0.5 else 0]
                conf = prob if prob >= 0.5 else (1.0 - prob)
                print(f"  [{name:12s}] -> Predicted: {pred_label.upper():7s} | Confidence: {conf*100:.2f}% (prob={prob:.4f})")

if __name__ == '__main__':
    main()
