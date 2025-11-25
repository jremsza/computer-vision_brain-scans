# Brain Tumor Classification: Transfer Learning Comparative Study

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Status](https://img.shields.io/badge/Status-Complete-green)

## 📌 Project Overview
This project implements a deep learning pipeline to classify brain MRI scans into four distinct categories: **Glioma**, **Meningioma**, **Pituitary**, and **Healthy**. 

The core objective was to evaluate the effectiveness of **Transfer Learning** against a baseline custom CNN. The study benchmarks industry-standard architectures (**VGG16**, **ResNet50**) and explores optimization techniques like Data Augmentation to mitigate overfitting in medical image analysis.

## 🎯 Key Results
After conducting 10 controlled experiments, the **VGG-16 model with Data Augmentation** achieved the best performance, demonstrating superior generalization compared to ResNet50 and the custom baseline.

| Model Architecture | Test Accuracy | Training Time | Verdict |
| :--- | :--- | :--- | :--- |
| **Custom CNN (Baseline)** | 94.5% | 47s | Fast but prone to overfitting. |
| **ResNet50** | 78.0% | 122s | Failed to converge effectively on this dataset. |
| **VGG-16 + Augmentation** | **96.3%** | 513s | **Best Performer (Production Candidate)** |

## 📂 Dataset
* **Source:** [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/rm1000/brain-tumor-mri-scans)
* **Classes:** 4 (Glioma, Meningioma, Pituitary, No Tumor)
* **Image Count:** 7,023 images
* **Preprocessing:**
  * Resizing to 224x224 (to match VGG/ResNet input requirements).
  * Normalization (0-1 scaling).
  * Data Augmentation (Rotation, Zoom, Horizontal Flip) applied to training sets.

## 🧪 Methodology
The project followed a rigorous experimental framework:
1.  **Baseline Establishment:** Built a 4-layer Custom CNN to set a performance floor.
2.  **Architecture Search:** Implemented Transfer Learning using weights pre-trained on ImageNet.
    * *Hypothesis:* Pre-trained feature extractors should outperform scratch-trained models on small medical datasets.
3.  **Optimization:** Addressed significant overfitting in VGG-16 by introducing a Data Augmentation pipeline, resulting in a **1.3% accuracy boost** and stabilized loss curves.

## 📊 Visualizations
*(Note: Images generated from the notebook)*

### Confusion Matrix (Best Model)
The model distinguishes "Healthy" scans with near-perfect precision. Slight confusion remains between Glioma and Meningioma tumor types.
![Confusion Matrix](images/confusion_matrix.png)

### Model Performance Comparison
![Performance Chart](images/model_comparison.png)

## 🛠️ Tech Stack
* **Core:** Python, Jupyter Notebook
* **Deep Learning:** TensorFlow, Keras
* **Data Manipulation:** NumPy, Pandas
* **Visualization:** Matplotlib, Seaborn
* **Computer Vision:** OpenCV

## 🔮 Future Work
* **Explainable AI (XAI):** Implement Grad-CAM to visualize the specific tumor regions driving the model's predictions.
* **Hyperparameter Tuning:** Utilize KerasTuner to optimize learning rates and dropout percentages automatically.
* **Deployment:** Containerize the model using Docker and serve via a Flask API.

## 👤 Author
**Jacob Remsza** *Data Scientist | Deep Learning Enthusiast* [Portfolio](https://jremsza-portfolio.netlify.app/) | [LinkedIn](https://linkedin.com/in/yourprofile)
