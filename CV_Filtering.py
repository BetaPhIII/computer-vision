import cv2
import matplotlib as plt
import numpy as np
from scipy.signal import convolve2d

def box_filter(image_path, image, k):
    mask = np.ones((k,k), np.float32)/(k**2)
    pad_size = k//2
    padded_image = np.pad(image, pad_width=pad_size, mode='constant', constant_values=0)
    filtered = image.copy()
    shape_row = filtered.shape[0]
    shape_col = filtered.shape[1]

    for i in range(shape_row):
        for j in range(shape_col):
            subarray = padded_image[i:i+k, j:j+k]
            filtered[i,j] = np.sum(mask*subarray)

    return filtered

def box_filter_open_cv(image_path, image, k):
    ksize = (k,k)
    filtered = cv2.boxFilter(image,-1,ksize)
    return filtered

def sobel_x(image, k=3):
    mask = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
    pad_size = k//2
    padded_image = np.pad(image, pad_width=pad_size, mode='constant', constant_values=0)

    filtered = np.zeros_like(image, dtype=np.float64)
    shape_row = filtered.shape[0]
    shape_col = filtered.shape[1]

    for i in range(shape_row):
        for j in range(shape_col):
            subimage = padded_image[i:i+k, j:j+k]
            filtered[i,j] = np.sum(mask*subimage)

    return filtered

def sobel_y(image, k=3):
    mask = np.array([[1, 2, 1],[0, 0, 0],[-1, -2, -1]])
    pad_size = k//2
    padded_image = np.pad(image, pad_width=pad_size, mode='constant', constant_values=0)

    filtered = np.zeros_like(image, dtype=np.float64)
    shape_row = filtered.shape[0]
    shape_col = filtered.shape[1]

    for i in range(shape_row):
        for j in range(shape_col):
            subimage = padded_image[i:i+k, j:j+k]
            filtered[i,j] = np.sum(mask*subimage)

    return filtered

def sobel_x_CV2(image):
    cv2.Sobel(image,)

def sobel_y_CV2(image):
    cv2.Sobel(image,)

def gaussian_CV2(image):
    cv2.GaussianBlur(image)

def gradient_magnitude(image_x, image_y):
    mag = np.sqrt(image_x**2 + image_y**2)
    return mag

def normalize_image(image):
    return ((image - np.min(image)) / (np.max(image) - np.min(image)) * 255).astype(np.uint8)

if __name__ == "__main__":
    image_path = "dog.bmp"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # # no cv2 box filter
    # boxedNoCV2 = box_filter(image_path, image, 5)

    # # cv2 box filter
    # boxedCV2 = box_filter_open_cv(image_path, image, 5)

    sobelNoCV2_x = (sobel_x(image, 3))
    sobelNoCV2_y = (sobel_y(image, 3))
    sobelNoCV2_xy = normalize_image(gradient_magnitude(sobelNoCV2_x, sobelNoCV2_y))
    cv2.imshow("x",sobelNoCV2_x)
    cv2.imshow("y",sobelNoCV2_y)
    cv2.imshow("image",sobelNoCV2_xy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()