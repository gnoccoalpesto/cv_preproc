#!/usr/bin/env python3

import cv2
import numpy as np

def linearStretching(image, max_v=None, min_v=None, histogram=None):
    """
    TODO: remember that this requires histogram normalized over total pixel count (not "range" normalized),
        menaning that the hist is the pfm of the pixels' intensities
    original: dark, peak at low intensity (GL)
    stretched:
    w/ image's min max vals: low effect, almost same hsit
    w/ const min max vals: incresed effect, various peaks, space between them is little, poisson  like curve
    w/ dynamic(%) min max vals: huge effect, peaks space over full intensity range
    """
    # P out= 255 *( P in -Pmin)/(Pmax-Pmin)
    # Stretching with max and min of the image. Not effective when outliers
    if max_v is None:
        max_v = np.max(image)
    # Stretching with dynamic max and min based on percentiles.
    elif 'p' in str(max_v) and histogram is not None:
        max_v = findPercentileValue(histogram, int(max_v[1:]))
    if min_v is None:
        min_v = np.min(image)
    elif 'p' in str(min_v) and histogram is not None:
        min_v = findPercentileValue(histogram, int(min_v[1:]))
    # else: use constant values, passed as input
    image[image < min_v] = min_v
    image[image > max_v] = max_v
    return 255. / (max_v - min_v) * (image - min_v)

def findPercentileValue(histogram, percentile):
    s = 0
    idx = 0
    total_pixel = np.sum(histogram)
    while s < total_pixel * percentile / 100:
        s += histogram[idx]
        idx += 1
    return idx

def exponentialOperator(image, r=0.45):
    """
    Pout= 255 *(Pin /255)**r
    r<1 improves dark areas, reduces bright ones
    r>1 opposite
    spacing between peaks reduced toward higher intensities, poisson like
    good effect, reduced noise
    """
    return ((image / 255) ** r) * 255

def pfm(histogram):
    # hist equalization operator: spreads uniformly pxs intensities across whole range
    # PMF(i) = 1 / Npxs_in_image * sum{0, i}(h(k))
    # Pout = 255 * PMF(Pin)
    total_pixel = np.sum(histogram)
    pfm_val = []
    for i in range(256):
        pfm_i = np.sum(histogram[:i]) / total_pixel
        pfm_val.append(pfm_i)
    return np.asarray(pfm_val)

def equalization(image, histogram):
    """
    high effect, high noise, turtleshell curve, equally spaced peaks
    """
    # Calculating equalization look up table
    eq_op = pfm(histogram) * 255
    # Mapping each pixel value into the equalized one given the pfm
    return eq_op[image]

def meanFilter(image, k_size=5):
    # really high smoothing
    mean_kernel = np.ones([k_size, k_size]) / (k_size ** 2)
    #TODO: filter does not automatically flip kernel (for convolution); but not needed for symmetrical ker
    return cv2.filter2D(image, -1, mean_kernel)

def denoisingFilter(image):
    # less smoothing, good result
    denoising_kernel = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]]) / 16
    #TODO: filter does not automatically flip kernel (for convolution); but not needed for symmetrical ker
    return cv2.filter2D(image, -1, denoising_kernel)

def highPassFilter(image):
    # edge filtering / enhancing
    hp_kernel = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]])
    #TODO: filter does not automatically flip kernel (for convolution); but not needed for symmetrical ker
    return cv2.filter2D(image, -1, hp_kernel)

def gaussianFilter(image, sigma=1.5, k_size=None, dim_mode=''):
    # Higher sigmas should correspond to larger kernels
    if k_size is None:
        k_size = int(np.ceil((3 * sigma)) * 2 + 1)
    if dim_mode == '':
        return cv2.GaussianBlur(image, (k_size, k_size), sigma)
    g_kernel_1D = cv2.getGaussianKernel(k_size, sigma)
    if dim_mode == '1d' or dim_mode == '1D':
        #TODO: filter does not automatically flip kernel (for convolution); but not needed for symmetrical ker
        transpose_img = cv2.filter2D(image, -1, g_kernel_1D)
        return cv2.filter2D(transpose_img, -1, g_kernel_1D.transpose())
    #if dir_mode=='2D'
    g_kernel_2D = g_kernel_1D.dot(g_kernel_1D.transpose())
    #TODO: filter does not automatically flip kernel (for convolution); but not needed for symmetrical ker
    return cv2.filter2D(image, -1, g_kernel_2D)

def medianFilter(image, k_size=5):
    return cv2.medianBlur(image, k_size)

def bilateralFilter(image, k_size=5, sigma_color=75, sigma_space=75):
    # sigma: < 10  no effect,
    #     > 150  cartoonization
    # dimension: d = 5 real time applications
    # d = 9 offline applcations
    return cv2.bilateralFilter(image, k_size, sigma_color, sigma_space)

def sobelFilter(image):
    sobel_kernel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]]) * 1 / 4
    sobel_kernel_y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]]) * 1 / 4
    # Finding dI(x, y)/dx
    #TODO: filter does not automatically flip kernel (for convolution); but not needed for symmetrical ker
    dx = cv2.filter2D(image.astype(float), -1, sobel_kernel_x)
    dx = np.abs(dx)
    # Finding dI(x, y)/dy
    dy = cv2.filter2D(image.astype(float), -1, sobel_kernel_y)
    dy = np.abs(dy)
    # Finding gradient module pixel-wise
    return np.maximum(dx,dy)
