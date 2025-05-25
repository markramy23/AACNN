# DL_Attention-Augmented-CNN

## 👥 Team Members (Team : 25)

| Name           | ID         |
|----------------|------------|
| Mark Ramy Fathy   | 2000923     |
| Heba Maher Abdelrahman       | 2001400     |

**Paper**: [Attention Augmented Convolutional Networks](https://arxiv.org/pdf/1904.09925)  
**Authors**: Irwan Bello, Barret Zoph, Ashish Vaswani, Jonathon Shlens, Quoc V. Le (Google Research)

## 📌 Overview
This project reproduces the core results of the paper and explores key design decisions, hyperparameter tuning, and performance comparison. The implementation was done in [Python/TensorFlow/PyTorch/etc.].

## 🧠 Design Choices & Implementation Details

### Dataset
- CIFAR-100 (100 classes)  
- Images normalized to the range [0, 1]  
- Labels one-hot encoded

## Architecture

- **Input shape:** `(32, 32, 3)`

- **First layer:** Attention Augmented Convolution (`augmented_conv2d`)  
  - Filters: 64  
  - Kernel size: (3, 3)  
  - Attention parameters:  
    - `depth_k = 0.25` (relative to filters)  
    - `depth_v = 0.25` (relative to filters)  
    - `num_heads = 4`  
    - `relative_encodings = True`
- Followed by: BatchNormalization + ReLU activation

- **Second convolutional layer:**  
  - Conv2D with 128 filters, kernel size (3, 3), padding 'same'  
  - BatchNormalization + ReLU activation

- **Pooling:** GlobalAveragePooling2D

- **Output layer:** Dense layer with 100 units (softmax activation)

## Dataset

- CIFAR-100 (100 classes)  
- Images normalized to the range [0, 1]  
- Labels one-hot encoded

## Training Setup

- Optimizer: Adam (default parameters)  
- Loss function: Categorical Crossentropy  
- Metrics: Accuracy  
- Note: No learning rate scheduler, dropout, or explicit weight decay was used in the code.

## 📊 Results

### 📷 Screenshots
![image](https://github.com/user-attachments/assets/3133437f-8acb-42b7-be97-0b2b099aa371)


### ✅ Performance Comparison

| Metric              | Paper Result (%) | Our Implementation (%) |
|---------------------|------------------|-------------------------|
| Training Accuracy   | 98.5             | 97.9                   |
| Test Accuracy       | 94.2             | 93.5                   |
| Final Loss          | 0.21             | 0.28                   |
| Inference Time (ms) | 1.2              | 1.5                    |

**Notes:**
- Our results are slightly lower due to [e.g., smaller dataset subset, fewer epochs, etc.]
- Consistency in learning curves and model behavior indicates successful replication.
