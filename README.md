# Brain-Tumor-MRI-Image-Classification
A deep learning project for classifying brain MRI images into four categories: glioma, meningioma, pituitary tumor, and no tumor. The model uses transfer learning and fine-tuning techniques with data augmentation to improve accuracy and generalization. Designed for medical image analysis and early-stage tumor detection.



This project focuses on the classification of brain MRI images into one of four categories:
- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

It utilizes **transfer learning** with fine-tuning and **data augmentation** to achieve high accuracy in tumor detection from MRI scans.

---

## 📂 Dataset
- The dataset contains labeled MRI images for 4 classes.
- Preprocessed into `train`, `val`, and `test` directories with image size `(128x128)`.

---

## 📌 Model Architecture
- Base Model: `EfficientNetB0` (pretrained on ImageNet)
- Layers added on top:
  - GlobalAveragePooling
  - Dense + Dropout
  - Softmax output layer (4 classes)
- Fine-tuned top layers for better domain adaptation

---

## 🧪 Data Augmentation
Used on training set to prevent overfitting:
- Rotation
- Width & height shift
- Zoom
- Shear
- Horizontal flip

---

## 📈 Performance
- Validation Accuracy: ~90%
- Loss and accuracy curves indicate stable training with minimal overfitting.


## 📈 Accuracy & Loss Curves

![Model Accuracy and Loss](Screenshot%202025-07-29%20084415.png)


---

## 🛠️ How to Run
```bash
# Install dependencies
pip install tensorflow matplotlib numpy

# Run training
python train.py

# Evaluate model
python evaluate.py
