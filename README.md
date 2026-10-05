# Pothole Detection using Deep Learning 🚗🕳️

A comprehensive Deep Learning benchmark for automated road pothole detection comparing **5 Convolutional Neural Network Architectures**:
1. **Custom Baseline CNN**
2. **AlexNet**
3. **GoogLeNet (Inception Architecture)**
4. **VGG-16**
5. **VGG-19**

---

## 📊 Benchmark & Comparative Results

All models were evaluated on the pothole dataset using a consistent 80/20 train/validation split with real-time data augmentation (zoom, shear, rotation, horizontal flip).

| Model Architecture | Validation Accuracy (%) | Validation Loss | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **VGG-16** | **97.78%** | **0.0661** | **96.97%** | **98.46%** | **97.71%** | **0.9985** |
| 🥈 **VGG-19** | **94.81%** | **0.1690** | **100.00%** | **89.23%** | **94.31%** | **0.9996** |
| 🥉 **Custom Baseline CNN** | **84.44%** | **0.4398** | **86.67%** | **80.00%** | **83.20%** | **0.9200** |
| 4️⃣ **GoogLeNet (Inception)** | **67.41%** | **0.6246** | **70.59%** | **55.38%** | **62.07%** | **0.7310** |
| 5️⃣ **AlexNet** | **62.22%** | **0.6301** | **65.91%** | **44.62%** | **53.21%** | **0.7257** |

---

## 📁 Repository Structure

```text
├── pothole-detection-dataset/      # Pothole and Normal road image dataset (683 images)
│   ├── normal/                     # Normal road images (352 images)
│   ├── potholes/                   # Road images with potholes (329 images)
│   ├── trial.jpeg                  # Sample test image
│   └── trial2.jpg                  # Sample test image
├── models.py                       # Modular model architecture definitions (CNN, AlexNet, GoogLeNet, VGG16, VGG19)
├── main.py                         # Single model training & inference script
├── train_all_models.py             # Full 5-model benchmark training & evaluation script
├── pothole-detection-cnn.ipynb     # Baseline CNN Jupyter Notebook with plots
├── pothole-detection-all-models.ipynb # Full Multi-Model Benchmark Jupyter Notebook (with Confusion Matrices & Graphs)
├── .gitignore                      # Git ignore rules
└── README.md                       # Project documentation & benchmark overview
```

---

## 🚀 Getting Started

### 1. Prerequisites & Virtual Environment

Create and activate a Python 3.11 virtual environment:
```bash
python -m venv .venv
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
```

Install the required dependencies:
```bash
pip install tensorflow opencv-python matplotlib seaborn scikit-learn pandas numpy pillow ipykernel
```

### 2. Running the Interactive Notebook

Launch Jupyter or open the notebook directly in your IDE:
```bash
jupyter notebook pothole-detection-all-models.ipynb
```

### 3. Running CLI Benchmark

To train all 5 models and generate the benchmark comparison tables & plots from the terminal:
```bash
python train_all_models.py
```
