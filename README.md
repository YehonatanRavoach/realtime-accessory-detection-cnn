# Real-Time Accessory Detection with CNN (PyTorch)

Real-time **computer vision classification** project that detects whether a person is wearing **glasses**, a **hat**, or **no accessory** from a live webcam stream.  
The system is implemented using a lightweight **Convolutional Neural Network (CNN)** in PyTorch and achieves **99.6% test accuracy** while running at ~25 FPS on a standard laptop webcam.

---

## Project Overview

This project addresses a real-time image classification problem under practical constraints:  
limited data, low latency requirements, and robustness to lighting and pose variations.

The goal is to design a **fast, accurate, and simple CNN-based pipeline** capable of live inference without relying on heavy pretrained models.

Key characteristics:
- End-to-end pipeline: data → training → evaluation → live inference
- Designed for **real-time performance**
- Focus on preventing overfitting on a small dataset

---

## Project Structure
MidProject/   
│ ├── data/ ← train / val / test image folders (git-ignored)    
│ │ ├── train/   
│ │ ├── val/   
│ │ └── test/   
│ ├── saved_models/   
│ │ └── best_model.pth   
├── AccessoryCNN.ipynb ← single, self-contained notebook   
├── requirements.txt └── README.md  

*(No extra *src/* or *notebooks/* folders — everything lives in the notebook.)*


---

*All logic is contained in a single notebook for clarity and reproducibility.*

---

## Dataset & Preprocessing

- **Classes**: glasses / hat / no accessory
- **Raw data**: ~600 real images per class
- **Split**: 70% train / 15% validation / 15% test
- **Augmentation** (applied **after** the split to avoid data leakage):
  - Random crop
  - Horizontal flip
  - Color jitter
- **Training set size after augmentation**: ~1,260 images

---

## Model & Training

| Component | Details |
|---------|---------|
| Input size | 128 × 128 RGB |
| Architecture | 3 × (Conv → ReLU → MaxPool) + Fully Connected |
| Regularization | Dropout (0.3) |
| Optimizer | Adam (LR = 5e-4) |
| Epochs | 8 |
| Training strategy | “Slow and steady” — short training to convergence |

The model is intentionally kept small to ensure **low latency and stable real-time inference**.

---

## Results

| Split | Accuracy | Loss |
|------|----------|------|
| Validation | 99.26% | 0.3438 |
| Test | **99.63%** | 0.31 |

The final model generalizes well despite the small dataset, with clear separation between training, validation, and test sets.

---

## Running the Project

```bash
git clone https://github.com/YehonatanRavoach/realtime-accessory-detection-cnn.git
cd realtime-accessory-detection-cnn

python -m venv .venv
.venv\Scripts\activate   # Windows

pip install -r requirements.txt
jupyter notebook AccessoryCNN.ipynb
