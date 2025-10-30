###################################################
#
#   Script to pre-process the original imgs
#
##################################################


import numpy as np
from PIL import Image
import cv2

from help_functions import *


#My pre processing (use for both training and testing!)
def my_PreProc(data):
    assert(len(data.shape)==4)
    assert (data.shape[1]==3)  #Use the original images
    #black-white conversion
    train_imgs = rgb2gray(data)
    #my preprocessing:
    train_imgs = dataset_normalized(train_imgs)
    train_imgs = clahe_equalized(train_imgs)
    train_imgs = adjust_gamma(train_imgs, 1.2)
    train_imgs = train_imgs/255.  #reduce to 0-1 range
    return train_imgs


#============================================================
#========= PRE PROCESSING FUNCTIONS ========================#
#============================================================

#==== histogram equalization

def histo_equalized(imgs):
    assert imgs.ndim == 4
    assert imgs.shape[1] == 1

    # Convert to uint8 if not already
    imgs_uint8 = imgs.astype(np.uint8)

    # Apply equalization using list comprehension
    equalized_list = [cv2.equalizeHist(imgs_uint8[i, 0]) for i in range(imgs.shape[0])]

    # Stack back and keep shape (N, 1, H, W)
    imgs_equalized = np.stack(equalized_list, axis=0)[:, np.newaxis, :, :]

    return imgs_equalized



# CLAHE (Contrast Limited Adaptive Histogram Equalization)
#adaptive histogram equalization is used. In this, image is divided into small blocks called "tiles" (tileSize is 8x8 by default in OpenCV). Then each of these blocks are histogram equalized as usual. So in a small area, histogram would confine to a small region (unless there is noise). If noise is there, it will be amplified. To avoid this, contrast limiting is applied. If any histogram bin is above the specified contrast limit (by default 40 in OpenCV), those pixels are clipped and distributed uniformly to other bins before applying histogram equalization. After equalization, to remove artifacts in tile borders, bilinear interpolation is applied


def clahe_equalized(imgs):
    assert imgs.ndim == 4
    assert imgs.shape[1] == 1

    # CLAHE object
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    # Convert to uint8
    imgs_uint8 = imgs.astype(np.uint8)

    # Apply CLAHE to each image using list comprehension
    equalized_list = [clahe.apply(imgs_uint8[i, 0]) for i in range(imgs.shape[0])]

    # Stack back to original shape (N, 1, H, W)
    imgs_equalized = np.stack(equalized_list, axis=0)[:, np.newaxis, :, :]

    return imgs_equalized



# ===== normalize over the dataset


def dataset_normalized(imgs):
    assert imgs.ndim == 4
    assert imgs.shape[1] == 1

    # Standardize the dataset
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs - imgs_mean) / imgs_std

    # Compute per-image min and max
    min_per_img = imgs_normalized.min(axis=(1,2,3), keepdims=True)
    max_per_img = imgs_normalized.max(axis=(1,2,3), keepdims=True)

    # Scale to [0, 255] per image
    imgs_normalized = (imgs_normalized - min_per_img) / (max_per_img - min_per_img) * 255

    return imgs_normalized



def adjust_gamma(imgs, gamma=1.0):
    assert imgs.ndim == 4
    assert imgs.shape[1] == 1

    invGamma = 1.0 / gamma
    # lookup table
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8")

    # Convert to uint8
    imgs_uint8 = imgs.astype(np.uint8)

    # Apply gamma correction to each image using list comprehension
    corrected_list = [cv2.LUT(imgs_uint8[i, 0], table) for i in range(imgs.shape[0])]

    # Stack back to original shape (N, 1, H, W)
    new_imgs = np.stack(corrected_list, axis=0)[:, np.newaxis, :, :]

    return new_imgs

