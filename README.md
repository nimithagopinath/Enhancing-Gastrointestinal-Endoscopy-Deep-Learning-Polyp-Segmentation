# Enhancing Gastrointestinal Endoscopy: Deep Learning Polyp Segmentation

Colorectal polyps are abnormal tissue growths in the colon or rectum, often benign but with the potential to become cancerous over time. Early detection through colonoscopy is critical, yet human error and polyp variability can lead to missed diagnoses. This project presents a deep learning-based approach to improve polyp segmentation in endoscopic images, thereby enhancing accuracy and efficiency in computer-aided colorectal cancer screening systems.

---

## 📌 Objective

To develop and compare two semantic segmentation architectures—**a standard UNet** and a **modified UNet with ResNet50 encoder**—for automatic polyp detection in colonoscopy images, aiding early diagnosis and intervention.

---

## 🗂️ Dataset

- **CVC-ClinicDB**: Official dataset used in the **MICCAI 2015 Sub-Challenge** on Automatic Polyp Detection in Colonoscopy Videos.
- The dataset consists of annotated endoscopic images with pixel-wise ground truth for polyp regions.

---

## 🧠 Methodology

- Implemented **UNet**, a widely used encoder-decoder architecture for medical image segmentation.
- Designed a **Modified UNet** by integrating a pre-trained **ResNet50** model into the encoder (transfer learning).
- Used **semantic segmentation** to classify each pixel as polyp or background.
- Performed **data augmentation** (rotation, flipping, etc.) to increase training robustness and generalization.

---

## 🛠️ Implementation

- **Libraries:** TensorFlow, Keras, OpenCV, Pandas, NumPy  
- **Preprocessing:**  
  - Image resizing and normalization using OpenCV  
  - Data augmentation to prevent overfitting  
- **Training:**  
  - Optimized using cross-entropy loss and gradient-based backpropagation  
  - Trained both UNet and Modified UNet architectures from scratch  
- **Transfer Learning:**  
  - ResNet50 weights (pre-trained on ImageNet) used to initialize encoder in Modified UNet

---

## 📈 Evaluation Metrics

- **Pixel-wise Accuracy**  
- **IoU (Intersection over Union)**  
- **Dice Coefficient**  
- **Precision / Recall / F1-score**  
- **Qualitative Visual Comparison**

---

## ✅ Results

| Model           | Accuracy | Notes                           |
|----------------|----------|----------------------------------|
| UNet            | 93%      | Baseline UNet with full training |
| Modified UNet   | 99%      | Used ResNet50 encoder (transfer learning) |

- The **Modified UNet outperformed** the standard model, demonstrating the benefit of transfer learning.
- Visual segmentation maps confirmed superior boundary detection and polyp localization.

---

## 🏁 Conclusion

This project highlights the potential of deep learning, particularly **semantic segmentation** with **transfer learning**, in improving polyp detection accuracy in gastrointestinal endoscopy. By achieving near-perfect segmentation results, this system serves as a strong foundation for real-time, AI-powered clinical diagnostic tools.

---

## 🚀 Future Work

- Expand to larger and more diverse endoscopy datasets
- Integrate real-time inference capability for clinical deployment
- Explore 3D segmentation and attention-based architectures

---

## 📎 License

This project is intended for academic and educational purposes. Please cite the original dataset and model architectures where applicable.

