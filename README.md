# 🧠 Brain Tumor Detection Using Neural Networks

## 📌 Project Overview

Brain tumors are among the most critical neurological disorders, requiring early and accurate diagnosis for effective treatment. This project focuses on **automated brain tumor detection from MRI images** using **deep learning and convolutional neural networks (CNNs)**.

The model is trained to classify MRI scans into different categories based on the presence and type of brain tumor, reducing dependency on manual interpretation and supporting faster clinical decision-making.

---

## 🚀 Key Features

* Deep learning–based image classification
* CNN architecture implemented using **PyTorch**
* Preprocessing and normalization of MRI images
* Model training, validation, and testing pipeline
* Performance evaluation using accuracy and loss metrics

---

## 🧪 Dataset

* MRI brain scan images
* Organized into **Training** and **Testing** directories
* Loaded using `torchvision.datasets.ImageFolder`

> Dataset structure:

```
archive/
│
├── Training/
│   ├── Class_1/
│   ├── Class_2/
│   └── ...
│
└── Testing/
    ├── Class_1/
    ├── Class_2/
    └── ...
```

---

## 🛠️ Tech Stack

* **Python**
* **PyTorch**
* **Torchvision**
* **NumPy**
* **Matplotlib**
* **Jupyter Notebook / Kaggle**

---

## ⚙️ Model Architecture

The CNN model consists of:

* Convolutional layers for feature extraction
* ReLU activation functions
* Pooling layers to reduce spatial dimensions
* Fully connected layers for classification

The model is trained using:

* **Loss Function:** Cross-Entropy Loss
* **Optimizer:** Adam
* **Device:** CPU / GPU (if available)

---

## 📈 Training Pipeline

1. Image resizing and normalization
2. Dataset loading using DataLoader
3. Forward pass through CNN
4. Loss computation
5. Backpropagation and optimization
6. Evaluation on test dataset

---

## 📊 Results

* The model demonstrates effective learning on MRI images
* Accuracy improves consistently with training epochs
* Can be extended for multi-class tumor classification

*(Exact metrics may vary depending on dataset size and hardware)*

---

## ▶️ How to Run

1. Clone the repository:

```bash
git clone https://github.com/your-username/brain_tumor_detection_using_neural_networks.git
```

2. Navigate to the project directory:

```bash
cd brain_tumor_detection_using_neural_networks
```

3. Install dependencies:

```bash
pip install torch torchvision matplotlib numpy
```

4. Run the notebook:

```bash
jupyter notebook
```


## 🔮 Future Improvements

* Add model explainability (Grad-CAM)
* Improve accuracy using transfer learning (ResNet, EfficientNet)
* Deploy as a web application
* Extend to segmentation-based tumor localization
