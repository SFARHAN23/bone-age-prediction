# Bone Age Prediction from Hand Radiographs


> An automated deep learning system for predicting skeletal bone age from pediatric hand X-ray images using transfer learning with ResNet50.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Authors](#authors)
- [Acknowledgments](#acknowledgments)

## 🎯 Overview

Bone age assessment is a critical clinical procedure for evaluating skeletal maturity in pediatric patients. Traditional manual methods are time-consuming and suffer from inter-observer variability (12-18 months). This project develops an automated system using deep learning to predict bone age from hand X-rays with **MAE = 9.55 months** and **R² = 0.90**.

### Key Highlights
- 🔬 **Clinical-grade accuracy**: MAE within acceptable clinical range
- 🧠 **Dual-task learning**: Regression for continuous age + Classification for developmental stages
- ⚖️ **Fair predictions**: Minimal gender bias (0.78 months difference)
- 🔍 **Interpretable AI**: Grad-CAM visualizations confirm clinically relevant feature learning

## ✨ Features

- **Transfer Learning**: Pretrained ResNet50 backbone fine-tuned on medical imaging data
- **Data Augmentation**: Random flips, rotations (±20°), color jitter for robust training
- **Mixed Precision Training**: AMP (Automatic Mixed Precision) for faster computation
- **Model Interpretability**: Grad-CAM heatmaps showing anatomical focus areas
- **Comprehensive Evaluation**: Regression metrics (MAE, RMSE, R²) and classification metrics (Accuracy, F1, Kappa)
- **Architecture Comparison**: Benchmarked against DenseNet121 and MobileNetV2

## 📊 Dataset

**RSNA Pediatric Bone Age Dataset**
- **Total Images**: 12,611 hand X-ray images
- **Labels**: Bone age in months (0-228) + biological sex metadata
- **Split**: 70% training (8,827), 15% validation (1,892), 15% test (1,892)
- **Classes**: 4 developmental stages
  - Infant: 0-24 months
  - Child: 25-144 months
  - Adolescent: 145-204 months
  - Young Adult: >204 months

**Preprocessing Pipeline**:
1. Resize to 256×256 pixels (GPU memory optimization)
2. ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
3. Data augmentation (training only): horizontal flips, ±20° rotation, color jitter
4. RAM loading: All images loaded into memory (~2.5GB) for 10× faster training

## 🏗️ Model Architecture

### Regression Model
```
ResNet50 Backbone (pretrained ImageNet)
    ↓
Global Average Pooling (2048 features)
    ↓
Linear(2048 → 512) + ReLU + Dropout(0.3)
    ↓
Linear(512 → 1)
    ↓
Predicted Age (months)
```

### Classification Model
```
ResNet50 Backbone (pretrained ImageNet)
    ↓
Global Average Pooling (2048 features)
    ↓
Linear(2048 → 4)
    ↓
Softmax → 4 Developmental Stages
```

**Training Configuration**:
- **Optimizer**: Adam (lr=1e-4)
- **Loss Functions**: L1 Loss (regression), Cross-Entropy (classification)
- **Scheduler**: ReduceLROnPlateau (factor=0.1, patience=3)
- **Batch Size**: 32
- **Epochs**: 15
- **Hardware**: NVIDIA RTX 3050 Laptop GPU
- **Training Time**: ~70 minutes (41 min regression + 29 min classification)

## 📈 Results

### Regression Performance

| Metric | Value |
|--------|-------|
| **MAE** | **9.55 months** |
| **RMSE** | 12.37 months |
| **R² Score** | 0.90 |

**Gender-wise Analysis**:
| Gender | MAE (months) | Samples |
|--------|--------------|---------|
| Male | 9.20 | 5,940 |
| Female | 9.98 | 6,671 |
| **Difference** | **0.78** | — |

### Classification Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | **88%** |
| Quadratic Weighted Kappa | 0.78 |
| Precision (macro) | 0.79 |
| Recall (macro) | 0.73 |
| F1-Score (macro) | 0.75 |

### Model Comparison

| Architecture | MAE (months) | Accuracy (%) |
|--------------|--------------|--------------|
| **ResNet50** | **9.55** | **88** |
| DenseNet121 | 10.12 | 86 |
| MobileNetV2 | 11.87 | 83 |

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- GPU recommended (NVIDIA RTX 3050 or higher) but not required
- 16GB RAM minimum
- Google Colab (alternatively, for free GPU access)

### Quick Start (Recommended: Google Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/C0DER0712/bone-age-prediction/blob/main/bone-age-prediction.ipynb)

**No setup needed!** Simply:
1. Click the "Open in Colab" badge above
2. Upload the notebook to Google Colab
3. Run all cells sequentially
4. Colab provides free GPU (Tesla T4) and pre-installed libraries

### Local Setup (Optional)

If you prefer running locally:

1. **Clone the repository**
```bash
git clone https://github.com/C0DER0712/bone-age-prediction.git
cd bone-age-prediction
```

2. **Install dependencies**
```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn Pillow tqdm opencv-python
```

Or use:
```bash
pip install -r requirements.txt
```

3. **Download the dataset**
```bash
# Option 1: Auto-download via kagglehub (built into notebook)
# The notebook will automatically download RSNA Bone Age dataset on first run

# Option 2: Manual download from Kaggle
# Visit: https://www.kaggle.com/datasets/kmader/rsna-bone-age
# Download and extract to a folder, update path in notebook
```

4. **Run the notebook**
```bash
jupyter notebook bone-age-prediction.ipynb
```

### 💡 What Gets Generated

When you run the notebook, it will:
- **Download dataset** automatically (12,611 X-ray images, ~2.5GB)
- **Preprocess images** to 256×256 (saved to `processed_images_256/`)
- **Train models** for 15 epochs (~70 minutes on RTX 3050)
- **Save trained models**: 
  - `resnet50_regression.pth` (bone age prediction model)
  - `resnet50_classification.pth` (developmental stage model)
- **Generate visualizations**: training curves, confusion matrix, Grad-CAM heatmaps

### 📂 Project Structure

```
bone-age-prediction/
│
├── bone-age-prediction.ipynb    # Main Jupyter notebook (all code)
├── project-report.pdf      # Project report & results
├── requirements.txt            # Required libraries in python
├── README.md                    # This file
│
└── (Generated after running notebook)
    ├── data/
    │   ├── boneage-training-dataset/      # Downloaded X-rays
    │   └── boneage-training-dataset.csv   # Age labels
    ├── processed_images_256/              # Preprocessed images
    ├── resnet50_regression.pth            # Trained regression model
    └── resnet50_classification.pth        # Trained classification model
```

### ⚡ Quick Test

After setup, verify everything works:

```python
import torch
import torchvision

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

## 💻 Usage

### Training Models

Simply run all cells in `bone-age-prediction.ipynb` sequentially. The notebook contains:

1. **Data Preprocessing** (Cells 1-6)
   - Download & load RSNA dataset
   - Resize images to 256×256
   - Split into train/val/test (70/15/15)

2. **Model Training** (Cells 7-10)
   - ResNet50 regression model (continuous bone age)
   - ResNet50 classification model (4 developmental stages)
   - Mixed precision training with AMP

3. **Evaluation & Visualization** (Cells 11+)
   - Performance metrics (MAE, accuracy, confusion matrix)
   - Grad-CAM interpretability visualizations
   - Model comparison (ResNet50 vs DenseNet121 vs MobileNetV2)

### Using Pre-trained Models

If you want to skip training and use the saved models:

```python
import torch
from torchvision import models

# Load regression model
model = models.resnet50()
model.fc = torch.nn.Sequential(
    torch.nn.Linear(2048, 512),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.3),
    torch.nn.Linear(512, 1)
)
model.load_state_dict(torch.load('resnet50_regression.pth'))
model.eval()

# Predict bone age from new X-ray
from PIL import Image
from torchvision import transforms

img = Image.open('path/to/xray.png').convert('RGB')
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
img_tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    prediction = model(img_tensor).item()
    print(f"Predicted Bone Age: {prediction:.1f} months")
```

### Expected Runtime

| Task | Time (RTX 3050) | Time (Colab T4) |
|------|----------------|-----------------|
| Data preprocessing | ~5 min | ~5 min |
| Regression training (15 epochs) | ~41 min | ~30 min |
| Classification training (15 epochs) | ~29 min | ~22 min |
| Full notebook | ~70 min | ~55 min |






## 👥 Authors

- **Syed Farhan Syed Sathik Basha** (CS23B2039)
- **Bommireddy Raviteja Reddy** (CS23B2011)
- **Mohamed Amjad** (CS23B2013)

*Department of Computer Science and Engineering*  
*Indian Institute of Information Technology, Kancheepuram*

## 🙏 Acknowledgments

- **RSNA Bone Age Challenge** for providing the dataset
- **PyTorch** and **torchvision** teams for excellent deep learning frameworks
- **Kaiming He et al.** for ResNet architecture
- **Selvaraju et al.** for Grad-CAM methodology
- Course instructors for guidance on Pattern Recognition and Machine Learning



## 📚 References

1. Halabi, S. S., et al. (2019). *The RSNA Pediatric Bone Age Challenge.* Radiology, 290(2), 498–503.
2. He, K., et al. (2016). *Deep Residual Learning for Image Recognition.* IEEE CVPR, 770–778.
3. Selvaraju, R. R., et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks.* IEEE ICCV, 618–626.
4. Greulich, W. W., & Pyle, S. I. (1959). *Radiographic Atlas of Skeletal Development.* Stanford University Press.

---

⭐ **If you found this project helpful, please consider giving it a star!**

📧 For questions or collaborations, reach out via [GitHub Issues](https://github.com/C0DER0712/bone-age-prediction/issues)
