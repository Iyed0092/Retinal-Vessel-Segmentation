# Retinal Vessel Segmentation with Vectorized Retina U-Net

This repository contains **another implementation of [Retina U-Net](https://github.com/orobix/retina-unet)** for retinal vessel segmentation, with fully vectorized operations for improved efficiency.

---

## 📝 Dataset Context

The DRIVE dataset was originally released in a competition setting, so the **test image masks are not publicly available**.  
To evaluate the model, the original training set of 20 images was split into **10 training images and 10 test images**. This custom split allows for proper evaluation while maintaining consistency with the dataset structure.

---

## 🔍 Preprocessing Steps

Before feeding images into the network, the following preprocessing operations are applied:

1. **Gray-scale conversion** – convert color fundus images to gray-scale to focus on vessel structures.  
2. **Standardization** – normalize pixel intensities for consistent input.  
3. **Contrast-limited adaptive histogram equalization (CLAHE)** – enhance local contrast to make vessels more distinguishable.  
4. **Gamma adjustment** – correct overall brightness and contrast for better visibility of thin vessels.

---

## 🏋️ Training Methodology

The neural network is trained on **sub-images (patches)** extracted from the pre-processed full images:

- Each patch is **48x48 pixels**, randomly centered within the full image.  
- Patches partially or completely outside the Field Of View (FOV) are included, allowing the network to learn to discriminate the FOV border from blood vessels.  
- From the 20 DRIVE training images, **9500 patches per image** are randomly extracted, giving a total of **190,000 patches**.  
- The first **90%** of patches (171,000) are used for training, and the remaining **10%** (19,000) for validation.

### Neural Network Architecture

- Derived from **U-Net**.  
- **Loss function:** Cross-entropy.  
- **Optimizer:** Stochastic Gradient Descent (SGD).  
- **Activation:** ReLU after each convolution.  
- **Dropout:** 0.2 between consecutive convolution layers.  
- **Training:** 150 epochs, mini-batch size 32 patches.  

### Testing Procedure

- Tested on the 10 DRIVE test images (gold-standard masks as ground truth).  
- Only pixels within the **FOV** are considered.  
- **Vessel probability per pixel** is obtained by averaging multiple predictions:  
  - Multiple overlapping patches are extracted with a stride of 5 pixels.  
  - Each pixel probability is averaged across all predicted patches covering that pixel.

---

## 🔍 Project Highlights

- Vectorized Retina U-Net implementation for faster training and inference.  
- Custom train/test split on the DRIVE dataset.  
- Full preprocessing pipeline applied to enhance vessel visibility.  
- Visual results and performance evaluation with ROC and Precision-Recall curves.

---

## 🖼️ Sample Segmentations

### Example 1
<img src="https://raw.githubusercontent.com/Iyed0092/Retinal-Vessel-Segmentation/main/test/test_Original_GroundTruth_Prediction0.png" width="400"/>
<p>Retinal image, Ground Truth Mask, and Model Prediction stacked vertically.</p>

### Example 2
<img src="https://raw.githubusercontent.com/Iyed0092/Retinal-Vessel-Segmentation/main/test/test_Original_GroundTruth_Prediction1.png" width="400"/>
<p>Retinal image, Ground Truth Mask, and Model Prediction stacked vertically.</p>

### Example 3
<img src="https://raw.githubusercontent.com/Iyed0092/Retinal-Vessel-Segmentation/main/test/test_Original_GroundTruth_Prediction2.png" width="400"/>
<p>Retinal image, Ground Truth Mask, and Model Prediction stacked vertically.</p>

---

## 📊 Performance Curves

### ROC Curve
<img src="https://raw.githubusercontent.com/Iyed0092/Retinal-Vessel-Segmentation/main/test/ROC.png" width="400"/>

### Precision-Recall Curve
<img src="https://raw.githubusercontent.com/Iyed0092/Retinal-Vessel-Segmentation/main/test/Precision_recall.png" width="400"/>

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/retina-unet-vectorized.git
cd retina-unet-vectorized

# Install dependencies
pip install -r requirements.txt
