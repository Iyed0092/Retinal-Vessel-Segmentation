import h5py
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def load_hdf5(infile):
  with h5py.File(infile,"r") as f:  #"with" close the file after its nested commands
    return f["image"][()]

def write_hdf5(arr,outfile):
  with h5py.File(outfile,"w") as f:
    f.create_dataset("image", data=arr, dtype=arr.dtype)

#convert RGB image in black and white

def rgb2gray(rgb):
    assert rgb.ndim == 4  # 4D array
    assert rgb.shape[1] == 3  # RGB channels

    # Compute grayscale directly using vectorized operation
    bn_imgs = rgb[:, 0, :, :] * 0.299 + rgb[:, 1, :, :] * 0.587 + rgb[:, 2, :, :] * 0.114

    # Keep the same output shape (batch, 1, height, width)
    return bn_imgs[:, np.newaxis, :, :]


#group a set of images row per columns
def group_images(data, per_row):
    assert data.shape[0] % per_row == 0
    assert data.shape[1] in (1, 3)

    # Convert to (batch, H, W, C)
    data = np.transpose(data, (0, 2, 3, 1))
    
    n_rows = data.shape[0] // per_row
    H, W, C = data.shape[2], data.shape[3], data.shape[1]  # height, width, channels
    
    # Reshape to (n_rows, per_row, H, W, C)
    data_reshaped = data.reshape(n_rows, per_row, data.shape[1], data.shape[2], data.shape[3])
    
    # Concatenate images horizontally in each row
    stripes = [np.concatenate(row, axis=1) for row in data_reshaped]
    
    # Concatenate all stripes vertically
    totimg = np.concatenate(stripes, axis=0)
    
    return totimg



#visualize image (as PIL image, NOT as matplotlib!)
def visualize(data,filename):
    assert (len(data.shape)==3) #height*width*channels
    img = None
    if data.shape[2]==1:  #in case it is black and white
        data = np.reshape(data,(data.shape[0],data.shape[1]))
    if np.max(data)>1:
        img = Image.fromarray(data.astype(np.uint8))   #the image is already 0-255
    else:
        img = Image.fromarray((data*255).astype(np.uint8))  #the image is between 0-1
    img.save(filename + '.png')
    return img


#prepare the mask in the right shape for the Unet
def masks_Unet(masks):
    assert masks.ndim == 4
    assert masks.shape[1] == 1

    N, _, H, W = masks.shape
    masks_flat = masks.reshape(N, H * W)  # (N, H*W)

    # Create an empty array for one-hot encoding
    new_masks = np.zeros((N, H * W, 2), dtype=masks.dtype)

    # Vectorized assignment
    new_masks[np.arange(N)[:, None], np.arange(H*W)[None, :], 0] = (masks_flat == 0)
    new_masks[np.arange(N)[:, None], np.arange(H*W)[None, :], 1] = (masks_flat != 0)

    return new_masks.astype(masks.dtype)


def pred_to_imgs(pred, patch_height, patch_width, mode="original"):
    assert pred.ndim == 3
    assert pred.shape[2] == 2

    if mode == "original":
        # Take probability of class 1 directly
        pred_images = pred[:, :, 1]
    elif mode == "threshold":
        # Apply threshold 0.5
        pred_images = (pred[:, :, 1] >= 0.5).astype(pred.dtype)
    else:
        raise ValueError(f"mode '{mode}' not recognized, it can be 'original' or 'threshold'")

    # Reshape to (Npatches, 1, patch_height, patch_width)
    pred_images = pred_images.reshape(pred.shape[0], 1, patch_height, patch_width)
    return pred_images
