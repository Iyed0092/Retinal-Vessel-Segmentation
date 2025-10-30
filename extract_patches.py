import numpy as np
import random


from help_functions import load_hdf5
from help_functions import visualize
from help_functions import group_images

from pre_processing import my_PreProc


#To select the same images
# random.seed(10)

#Load the original data and return the extracted patches for training/testing
def get_data_training(DRIVE_train_imgs_original,
                      DRIVE_train_groudTruth,
                      patch_height,
                      patch_width,
                      N_subimgs,
                      inside_FOV):
    train_imgs_original = load_hdf5(DRIVE_train_imgs_original)
    train_masks = load_hdf5(DRIVE_train_groudTruth) #masks always the same
    # visualize(group_images(train_imgs_original[0:20,:,:,:],5),'imgs_train')#.show()  #check original imgs train


    train_imgs = my_PreProc(train_imgs_original)
    train_masks = train_masks/255.

    train_imgs = train_imgs[:,:,9:574,:]  #cut bottom and top so now it is 565*565
    train_masks = train_masks[:,:,9:574,:]  #cut bottom and top so now it is 565*565
    data_consistency_check(train_imgs,train_masks)

    #check masks are within 0-1
    assert(np.min(train_masks)==0 and np.max(train_masks)==1)

    print ("\ntrain images/masks shape:")
    print (train_imgs.shape)
    print ("train images range (min-max): " +str(np.min(train_imgs)) +' - '+str(np.max(train_imgs)))
    print ("train masks are within 0-1\n")

    #extract the TRAINING patches from the full images
    patches_imgs_train, patches_masks_train = extract_random(train_imgs,train_masks,patch_height,patch_width,N_subimgs,inside_FOV)
    data_consistency_check(patches_imgs_train, patches_masks_train)

    print ("\ntrain PATCHES images/masks shape:")
    print (patches_imgs_train.shape)
    print ("train PATCHES images range (min-max): " +str(np.min(patches_imgs_train)) +' - '+str(np.max(patches_imgs_train)))

    return patches_imgs_train, patches_masks_train#, patches_imgs_test, patches_masks_test


#Load the original data and return the extracted patches for training/testing
def get_data_testing(DRIVE_test_imgs_original, DRIVE_test_groudTruth, Imgs_to_test, patch_height, patch_width):
    ### test
    test_imgs_original = load_hdf5(DRIVE_test_imgs_original)
    test_masks = load_hdf5(DRIVE_test_groudTruth)

    test_imgs = my_PreProc(test_imgs_original)
    test_masks = test_masks/255.

    #extend both images and masks so they can be divided exactly by the patches dimensions
    test_imgs = test_imgs[0:Imgs_to_test,:,:,:]
    test_masks = test_masks[0:Imgs_to_test,:,:,:]
    test_imgs = paint_border(test_imgs,patch_height,patch_width)
    test_masks = paint_border(test_masks,patch_height,patch_width)

    data_consistency_check(test_imgs, test_masks)

    #check masks are within 0-1
    assert(np.max(test_masks)==1  and np.min(test_masks)==0)

    print ("\ntest images/masks shape:")
    print (test_imgs.shape)
    print ("test images range (min-max): " +str(np.min(test_imgs)) +' - '+str(np.max(test_imgs)))
    print ("test masks are within 0-1\n")

    #extract the TEST patches from the full images
    patches_imgs_test = extract_ordered(test_imgs,patch_height,patch_width)
    patches_masks_test = extract_ordered(test_masks,patch_height,patch_width)
    data_consistency_check(patches_imgs_test, patches_masks_test)

    print ("\ntest PATCHES images/masks shape:")
    print (patches_imgs_test.shape)
    print ("test PATCHES images range (min-max): " +str(np.min(patches_imgs_test)) +' - '+str(np.max(patches_imgs_test)))

    return patches_imgs_test, patches_masks_test




# Load the original data and return the extracted patches for testing
# return the ground truth in its original shape
def get_data_testing_overlap(DRIVE_test_imgs_original, DRIVE_test_groudTruth, Imgs_to_test, patch_height, patch_width, stride_height, stride_width):
    ### test
    test_imgs_original = load_hdf5(DRIVE_test_imgs_original)
    test_masks = load_hdf5(DRIVE_test_groudTruth)

    test_imgs = my_PreProc(test_imgs_original)
    test_masks = test_masks/255.
    #extend both images and masks so they can be divided exactly by the patches dimensions
    test_imgs = test_imgs[0:Imgs_to_test,:,:,:]
    test_masks = test_masks[0:Imgs_to_test,:,:,:]
    test_imgs = paint_border_overlap(test_imgs, patch_height, patch_width, stride_height, stride_width)

    #check masks are within 0-1
    assert(np.max(test_masks)==1  and np.min(test_masks)==0)

    print ("\ntest images shape:")
    print (test_imgs.shape)
    print ("\ntest mask shape:")
    print (test_masks.shape)
    print ("test images range (min-max): " +str(np.min(test_imgs)) +' - '+str(np.max(test_imgs)))
    print ("test masks are within 0-1\n")

    #extract the TEST patches from the full images
    patches_imgs_test = extract_ordered_overlap(test_imgs,patch_height,patch_width,stride_height,stride_width)

    print ("\ntest PATCHES images shape:")
    print (patches_imgs_test.shape)
    print ("test PATCHES images range (min-max): " +str(np.min(patches_imgs_test)) +' - '+str(np.max(patches_imgs_test)))

    return patches_imgs_test, test_imgs.shape[2], test_imgs.shape[3], test_masks


#data consinstency check
def data_consistency_check(imgs,masks):
    assert(len(imgs.shape)==len(masks.shape))
    assert(imgs.shape[0]==masks.shape[0])
    assert(imgs.shape[2]==masks.shape[2])
    assert(imgs.shape[3]==masks.shape[3])
    assert(masks.shape[1]==1)
    assert(imgs.shape[1]==1 or imgs.shape[1]==3)


#extract patches randomly in the full training images
#  -- Inside OR in full image
import numpy as np
import random

def extract_random(full_imgs, full_masks, patch_h, patch_w, N_patches, inside=True):
    if N_patches % full_imgs.shape[0] != 0:
        print("N_patches: please enter a multiple of number of images")
        exit()

    assert full_imgs.ndim == 4 and full_masks.ndim == 4
    assert full_imgs.shape[1] in (1, 3)
    assert full_masks.shape[1] == 1
    assert full_imgs.shape[2] == full_masks.shape[2] and full_imgs.shape[3] == full_masks.shape[3]

    N_imgs, C_img, img_h, img_w = full_imgs.shape
    C_mask = full_masks.shape[1]

    patches = np.empty((N_patches, C_img, patch_h, patch_w), dtype=full_imgs.dtype)
    patches_masks = np.empty((N_patches, C_mask, patch_h, patch_w), dtype=full_masks.dtype)

    patch_per_img = N_patches // N_imgs
    print("patches per full image:", patch_per_img)

    iter_tot = 0

    for i in range(N_imgs):
        # Generate candidate centers in bulk
        num_candidates = patch_per_img * 3  # generate 3× to ensure enough valid centers
        xs = np.random.randint(patch_w // 2, img_w - patch_w // 2 + 1, size=num_candidates)
        ys = np.random.randint(patch_h // 2, img_h - patch_h // 2 + 1, size=num_candidates)

        if inside:
            # Vectorized FOV check
            mask_valid = np.array([is_patch_inside_FOV(x, y, img_w, img_h, patch_h) for x, y in zip(xs, ys)])
            xs = xs[mask_valid]
            ys = ys[mask_valid]

        # Take exactly patch_per_img centers
        xs = xs[:patch_per_img]
        ys = ys[:patch_per_img]

        # Extract patches
        y_start = ys - patch_h // 2
        y_end = ys + patch_h // 2
        x_start = xs - patch_w // 2
        x_end = xs + patch_w // 2

        for k in range(patch_per_img):
            patches[iter_tot] = full_imgs[i, :, y_start[k]:y_end[k], x_start[k]:x_end[k]]
            patches_masks[iter_tot] = full_masks[i, :, y_start[k]:y_end[k], x_start[k]:x_end[k]]
            iter_tot += 1

    return patches, patches_masks



#check if the patch is fully contained in the FOV
def is_patch_inside_FOV(x,y,img_w,img_h,patch_h):
    x_ = x - int(img_w/2) # origin (0,0) shifted to image center
    y_ = y - int(img_h/2)  # origin (0,0) shifted to image center
    R_inside = 270 - int(patch_h * np.sqrt(2.0) / 2.0) #radius is 270 (from DRIVE db docs), minus the patch diagonal (assumed it is a square #this is the limit to contain the full patch in the FOV
    radius = np.sqrt((x_*x_)+(y_*y_))
    if radius < R_inside:
        return True
    else:
        return False


#Divide all the full_imgs in pacthes

def extract_ordered(full_imgs, patch_h, patch_w):
    assert full_imgs.ndim == 4  # (N, C, H, W)
    assert full_imgs.shape[1] in (1, 3)

    N, C, H, W = full_imgs.shape

    N_patches_h = H // patch_h
    if H % patch_h != 0:
        print(f"warning: {N_patches_h} patches in height, with about {H % patch_h} pixels left over")

    N_patches_w = W // patch_w
    if W % patch_w != 0:
        print(f"warning: {N_patches_w} patches in width, with about {W % patch_w} pixels left over")

    print(f"number of patches per image: {N_patches_h * N_patches_w}")

    # Crop full images to multiple of patch size
    full_imgs_cropped = full_imgs[:, :, :N_patches_h*patch_h, :N_patches_w*patch_w]

    # Reshape and transpose to extract patches
    patches = full_imgs_cropped.reshape(
        N, C, N_patches_h, patch_h, N_patches_w, patch_w
    ).transpose(0, 2, 4, 1, 3, 5).reshape(
        N * N_patches_h * N_patches_w, C, patch_h, patch_w
    )

    return patches



def paint_border_overlap(full_imgs, patch_h, patch_w, stride_h, stride_w):
    assert full_imgs.ndim == 4
    assert full_imgs.shape[1] in (1, 3)

    N, C, img_h, img_w = full_imgs.shape

    # Compute leftover pixels
    leftover_h = (img_h - patch_h) % stride_h
    leftover_w = (img_w - patch_w) % stride_w

    pad_h = stride_h - leftover_h if leftover_h != 0 else 0
    pad_w = stride_w - leftover_w if leftover_w != 0 else 0

    if pad_h > 0:
        print(f"\nthe side H is not compatible with stride {stride_h}")
        print(f"img_h {img_h}, patch_h {patch_h}, stride_h {stride_h}")
        print(f"(img_h - patch_h) MOD stride_h: {leftover_h}")
        print(f"So the H dim will be padded with additional {pad_h} pixels")
        full_imgs = np.pad(full_imgs, ((0,0),(0,0),(0,pad_h),(0,0)), mode='constant', constant_values=0)

    if pad_w > 0:
        print(f"the side W is not compatible with stride {stride_w}")
        print(f"img_w {img_w}, patch_w {patch_w}, stride_w {stride_w}")
        print(f"(img_w - patch_w) MOD stride_w: {leftover_w}")
        print(f"So the W dim will be padded with additional {pad_w} pixels")
        full_imgs = np.pad(full_imgs, ((0,0),(0,0),(0,0),(0,pad_w)), mode='constant', constant_values=0)

    print("new full images shape:\n", full_imgs.shape)
    return full_imgs


#Divide all the full_imgs in pacthes
from numpy.lib.stride_tricks import as_strided

def extract_ordered_overlap(full_imgs, patch_h, patch_w, stride_h, stride_w):
    assert full_imgs.ndim == 4
    assert full_imgs.shape[1] in (1, 3)

    N, C, H, W = full_imgs.shape
    assert (H - patch_h) % stride_h == 0 and (W - patch_w) % stride_w == 0

    n_h = (H - patch_h) // stride_h + 1
    n_w = (W - patch_w) // stride_w + 1
    N_patches_img = n_h * n_w
    N_patches_tot = N_patches_img * N

    print(f"Number of patches on h: {n_h}")
    print(f"Number of patches on w: {n_w}")
    print(f"number of patches per image: {N_patches_img}, totally for this dataset: {N_patches_tot}")

    # Compute strides for sliding windows
    s0, s1, s2, s3 = full_imgs.strides
    new_shape = (N, n_h, n_w, C, patch_h, patch_w)
    new_strides = (s0, stride_h*s2, stride_w*s3, s1, s2, s3)

    patches_view = as_strided(full_imgs, shape=new_shape, strides=new_strides)

    # Reshape to (N_patches_tot, C, patch_h, patch_w)
    patches = patches_view.reshape(N_patches_tot, C, patch_h, patch_w)

    return patches


def recompone_overlap(preds, img_h, img_w, stride_h, stride_w):
    assert preds.ndim == 4
    assert preds.shape[1] in (1, 3)

    N_patches, C, patch_h, patch_w = preds.shape
    N_patches_h = (img_h - patch_h) // stride_h + 1
    N_patches_w = (img_w - patch_w) // stride_w + 1
    N_patches_img = N_patches_h * N_patches_w

    print("N_patches_h:", N_patches_h)
    print("N_patches_w:", N_patches_w)
    print("N_patches_img:", N_patches_img)

    assert N_patches % N_patches_img == 0
    N_full_imgs = N_patches // N_patches_img
    print(f"According to the dimension inserted, there are {N_full_imgs} full images (of {img_h}x{img_w} each)")

    full_prob = np.zeros((N_full_imgs, C, img_h, img_w), dtype=preds.dtype)
    full_sum = np.zeros((N_full_imgs, C, img_h, img_w), dtype=preds.dtype)

    # Reshape preds to (N_full_imgs, N_patches_h, N_patches_w, C, patch_h, patch_w)
    preds_reshaped = preds.reshape(N_full_imgs, N_patches_h, N_patches_w, C, patch_h, patch_w)

    # Vectorized accumulation
    for i in range(N_patches_h):
        h_start = i * stride_h
        h_end = h_start + patch_h
        for j in range(N_patches_w):
            w_start = j * stride_w
            w_end = w_start + patch_w
            full_prob[:, :, h_start:h_end, w_start:w_end] += preds_reshaped[:, i, j]
            full_sum[:, :, h_start:h_end, w_start:w_end] += 1

    final_avg = full_prob / full_sum
    print(final_avg.shape)

    assert np.max(final_avg) <= 1.0
    assert np.min(final_avg) >= 0.0

    return final_avg


#Recompone the full images with the patches

def recompone(data, N_h, N_w):
    assert data.ndim == 4
    assert data.shape[1] in (1, 3)  # channel check

    N_patches, C, patch_h, patch_w = data.shape
    N_patches_per_img = N_h * N_w
    assert N_patches % N_patches_per_img == 0
    N_full_imgs = N_patches // N_patches_per_img

    # Reshape to (N_full_imgs, N_h, N_w, C, patch_h, patch_w)
    data_reshaped = data.reshape(N_full_imgs, N_h, N_w, C, patch_h, patch_w)

    # Transpose to (N_full_imgs, C, N_h, patch_h, N_w, patch_w)
    data_transposed = data_reshaped.transpose(0, 3, 1, 4, 2, 5)

    # Reshape to full images (N_full_imgs, C, N_h*patch_h, N_w*patch_w)
    full_recomp = data_transposed.reshape(N_full_imgs, C, N_h*patch_h, N_w*patch_w)

    return full_recomp



#Extend the full images because patch divison is not exact
def paint_border(data,patch_h,patch_w):
    assert (len(data.shape)==4)  #4D arrays
    assert (data.shape[1]==1 or data.shape[1]==3)  #check the channel is 1 or 3
    img_h=data.shape[2]
    img_w=data.shape[3]
    new_img_h = 0
    new_img_w = 0
    if (img_h%patch_h)==0:
        new_img_h = img_h
    else:
        new_img_h = ((int(img_h)/int(patch_h))+1)*patch_h
    if (img_w%patch_w)==0:
        new_img_w = img_w
    else:
        new_img_w = ((int(img_w)/int(patch_w))+1)*patch_w
    new_data = np.zeros((data.shape[0],data.shape[1],new_img_h,new_img_w))
    new_data[:,:,0:img_h,0:img_w] = data[:,:,:,:]
    return new_data


#return only the pixels contained in the FOV, for both images and masks
def pred_only_FOV(data_imgs, data_masks, original_imgs_border_masks):
    assert data_imgs.ndim == 4 and data_masks.ndim == 4
    assert data_imgs.shape[0] == data_masks.shape[0]
    assert data_imgs.shape[2] == data_masks.shape[2] and data_imgs.shape[3] == data_masks.shape[3]
    assert data_imgs.shape[1] == 1 and data_masks.shape[1] == 1

    N, C, H, W = data_imgs.shape

    new_pred_imgs = []
    new_pred_masks = []

    for i in range(N):
        # Precompute FOV mask once for this image
        fov_mask = np.zeros((H, W), dtype=bool)
        for y in range(H):
            for x in range(W):
                fov_mask[y, x] = inside_FOV_DRIVE(i, x, y, original_imgs_border_masks)

        # Select only pixels inside FOV
        new_pred_imgs.append(data_imgs[i, :, fov_mask])
        new_pred_masks.append(data_masks[i, :, fov_mask])

    # Concatenate all images
    new_pred_imgs = np.concatenate(new_pred_imgs, axis=0)
    new_pred_masks = np.concatenate(new_pred_masks, axis=0)

    return new_pred_imgs, new_pred_masks


#function to set to black everything outside the FOV, in a full image

def kill_border(data, original_imgs_border_masks):
    assert data.ndim == 4
    assert data.shape[1] in (1, 3)

    N, C, H, W = data.shape

    for i in range(N):
        # Build FOV mask for this image
        fov_mask = np.zeros((H, W), dtype=bool)
        for y in range(H):
            for x in range(W):
                fov_mask[y, x] = inside_FOV_DRIVE(i, x, y, original_imgs_border_masks)

        # Set pixels outside FOV to zero using vectorized indexing
        data[i, :, ~fov_mask] = 0.0

    return data



def inside_FOV_DRIVE(i, x, y, DRIVE_masks):
    assert (len(DRIVE_masks.shape)==4)  #4D arrays
    assert (DRIVE_masks.shape[1]==1)  #DRIVE masks is black and white
    # DRIVE_masks = DRIVE_masks/255.  #NOOO!! otherwise with float numbers takes forever!!

    if (x >= DRIVE_masks.shape[3] or y >= DRIVE_masks.shape[2]): #my image bigger than the original
        return False

    if (DRIVE_masks[i,0,y,x]>0):  #0==black pixels
        # print DRIVE_masks[i,0,y,x]  #verify it is working right
        return True
    else:
        return False
