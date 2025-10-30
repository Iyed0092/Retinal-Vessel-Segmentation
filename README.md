# Retinal Vessel Segmentation with Vectorized Retina U-Net

This repository contains **another implementation of [Retina U-Net](https://github.com/orobix/retina-unet)** for retinal vessel segmentation, with fully vectorized operations for improved efficiency. The model is trained and evaluated on the **DRIVE dataset**, using a custom train/test split due to the absence of public test masks.

---

## 🔍 Project Highlights

- Another implementation of Retina U-Net with vectorized operations for faster training and inference.  
- Custom train/test split on the DRIVE dataset (80% train / 20% test) for evaluation.  
- Visual results including retinal images, ground truth masks, and model predictions.  
- Performance evaluation using **ROC** and **Precision-Recall curves**.

---

## 🖼️ Sample Results

### Example Segmentations

| Retinal Image | Ground Truth Mask | Model Prediction |
|---------------|-----------------|----------------|
| ![Image1](path/to/image1.png) | ![Mask1](path/to/mask1.png) | ![Prediction1](path/to/prediction1.png) |
| ![Image2](path/to/image2.png) | ![Mask2](path/to/mask2.png) | ![Prediction2](path/to/prediction2.png) |
| ![Image3](path/to/image3.png) | ![Mask3](path/to/mask3.png) | ![Prediction3](path/to/prediction3.png) |

---

### Performance Curves

#### ROC Curve
![ROC Curve](path/to/roc_curve.png)

#### Precision-Recall Curve
![Precision-Recall Curve](path/to/pr_curve.png)

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/retina-unet-vectorized.git
cd retina-unet-vectorized

# Install dependencies
pip install -r requirements.txt
