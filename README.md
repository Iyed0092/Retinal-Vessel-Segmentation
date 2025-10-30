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

## 🖼️ Sample Segmentations

### Example 1
![Example 1](https://raw.githubusercontent.com/Iyed0092/Retinal-Vessel-Segmentation/main/test/test_Original_GroundTruth_Prediction0.png)
*Retinal image, Ground Truth Mask, and Model Prediction stacked vertically.*


---

### Performance Curves

#### ROC Curve
![ROC Curve](https://raw.githubusercontent.com/Iyed0092/Retinal-Vessel-Segmentation/main/test/ROC.png)

#### Precision-Recall Curve
![Precision-Recall Curve](https://raw.githubusercontent.com/Iyed0092/Retinal-Vessel-Segmentation/main/test/Precision_recall.png)

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/retina-unet-vectorized.git
cd retina-unet-vectorized

# Install dependencies
pip install -r requirements.txt
