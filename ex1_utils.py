"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from typing import List
from PIL import Image
from numpy import *
import cv2
import numpy as np


LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> int:
    return 318925617


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: grayscale(1) or RGB(2)
    :return: The image np array
    """

    src = cv2.imread(filename)

    # gray format
    if representation == 1:
        image = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # rgb format
    if representation == 2:
        image = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)

    return image / (image.max() - image.min())


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: grayscale(1) or RGB(2)
    :return: None
    """

    if representation == 1:
        with open(filename, 'rb'):
            img = Image.open(filename).convert('L')
            img.show()

    if representation == 2:
        with open(filename, 'rb'):
            img = Image.open(filename).convert('RGB')
            img.show()
    pass


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    MAT = np.array([[0.299, 0.587, 0.114],
                    [0.596, -0.275, -0.321],
                    [0.212, -0.523, 0.311]])
    first_shape = imgRGB.shape
    trans_matrix = MAT.transpose()
    reshape_matrix = imgRGB.reshape(-1, 3)
    return np.matmul(reshape_matrix, trans_matrix).reshape(first_shape)


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:

    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """

    MAT = np.array([[0.299, 0.587, 0.114],
                    [0.596, -0.275, -0.321],
                    [0.212, -0.523, 0.311]])
    first_shape = imgYIQ.shape
    trans_matrix = np.linalg.inv(MAT).transpose()
    reshape_matrix = imgYIQ.reshape(-1, 3)
    return np.matmul(reshape_matrix, trans_matrix).reshape(first_shape)


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Equalizes the histogram of an image
    :param imgOrig: Original image
    :return: (imgEq,histOrg,histEQ)
    """
    if len(imgOrig.shape) == 3:
        rgb_format = True
        # transform to yiq
        yiqIm = transformRGB2YIQ(imgOrig)
        imgOrig = yiqIm[:, :, 0]

    else:
        rgb_format = False

    # Calculate the normalized
    imgOrig = cv2.normalize(imgOrig, None, 0, 255, cv2.NORM_MINMAX)
    imgOrig = imgOrig.astype('uint8')

    # Calculate the image histogram (range = [0, 255])
    histOrig = np.histogram(imgOrig.flatten(), bins=256)[0]

    # cumSum
    CumSum = np.cumsum(histOrig)
    imEq = CumSum[imgOrig]

    # Calculate the normalized Cumulative Sum (CumSum)
    imEq = cv2.normalize(imEq, None, 0, 255, cv2.NORM_MINMAX)
    imEq = imEq.astype('uint8')

    # Calculate the cumSum histogram
    histEq = np.histogram(imEq.flatten(), bins=256)[0]

    if rgb_format:
        yiqIm[:, :, 0] = imEq / (imEq.max() - imEq.min())
        imEq = transformYIQ2RGB(yiqIm)

    return imEq, histOrig, histEq


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """

    # rgb format
    if len(imOrig.shape) == 3:
        return quantizeImage_rgb(imOrig, nQuant, nIter)

    #gray format
    else:
        return quantizeImage_gray(imOrig, nQuant, nIter)


def quantizeImage_gray(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    # List[qImage_i]
    quantized_image_list = []

    # List[error_i]
    MSE = []

    new_imOrig = (imOrig * 255).astype(int)
    new_imOrig = new_imOrig.ravel()
    pixel_sum = np.zeros(256)

    for pix in range(256):
        pixel_sum[pix] = np.count_nonzero(new_imOrig == pix)

    # create image cumSum
    pixel_sum = pixel_sum.astype(int)
    pixel_cum_sum = (np.cumsum(pixel_sum)).astype(int)
    limit = [0]
    max_val = pixel_cum_sum[255]

    # size of every interval
    interval_size = int(max_val / nQuant)

    i = 1
    for ind in range(255) or i > nQuant:
        if interval_size * i < pixel_cum_sum[ind]:
            limit.append(ind - 1)
            i += 1

    # last limit is always 255
    limit.append(255)

    # find all limits
    intervals_middle = finding_Q(limit, pixel_sum)

    # first quantize
    temp_image = image_update(new_imOrig, limit, intervals_middle)
    # add mse
    MSE.append(mse(new_imOrig, temp_image, max_val))
    temp_image.resize(imOrig.shape[0], imOrig.shape[1])
    # add to final list
    quantized_image_list.append(temp_image)

    if nIter > 1:
        for time in range(nIter):
            # find the new limits
            limit = finding_Z(intervals_middle)
            # specific color of every interval
            intervals_middle = finding_Q(limit, pixel_sum)
            temp_image = image_update(new_imOrig, limit, intervals_middle)
            MSE.append(mse(new_imOrig, temp_image, max_val))
            temp_image.resize(imOrig.shape[0], imOrig.shape[1])
            quantized_image_list.append(temp_image)

    return quantized_image_list, MSE


def quantizeImage_rgb(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    # List[qImage_i]
    quantized_image_list = []

    # List[error_i]
    MSE = []

    # yiq
    YIQ_format_imOrig = transformRGB2YIQ(imOrig)
    new_imOrig = (YIQ_format_imOrig[:, :, 0] * 255).astype(int)
    new_imOrig = new_imOrig.ravel()

    # image histogram
    pixel_sum = np.zeros(256)
    for pix in range(256):
        pixel_sum[pix] = np.count_nonzero(new_imOrig == pix)
    pixel_sum = pixel_sum.astype(int)  # convert from float to int
    # camSum
    pixel_cum_sum = (np.cumsum(pixel_sum)).astype(int)
    limit = [0]

    numOfPix = pixel_cum_sum[255]
    numOfPixInCell = int(numOfPix / nQuant)
    # Initial division of boundaries - by almost identical division of pixels in each cell

    i = 1
    for ind in range(255) or i > nQuant:
        if numOfPixInCell * i < pixel_cum_sum[ind]:
            limit.append(ind - 1)
            i += 1

    # last limit is always 255
    limit.append(255)

    # find all limits
    intervals_middle = finding_Q(limit, pixel_sum)

    # first quantize
    temp_image = image_update(new_imOrig, limit, intervals_middle)

    # update lists
    MSE.append(mse(new_imOrig, temp_image, numOfPix))
    YIQ_format_imOrig[:, :, 0] = np.reshape(temp_image / 255, (imOrig.shape[0], imOrig.shape[1]))
    temp_image = transformYIQ2RGB(YIQ_format_imOrig)
    quantized_image_list.append(temp_image)

    if nIter > 1:
        for time in range(nIter):
            limit = finding_Z(intervals_middle)
            intervals_middle = finding_Q(limit, pixel_sum)
            temp_image = image_update(new_imOrig, limit, intervals_middle)
            MSE.append(mse(new_imOrig, temp_image, numOfPix))
            YIQ_format_imOrig[:, :, 0] = np.reshape(temp_image / 255, (imOrig.shape[0], imOrig.shape[1]))
            temp_image = transformYIQ2RGB(YIQ_format_imOrig)
            quantized_image_list.append(temp_image)

    return quantized_image_list, MSE


# create new array with the updates colors
def image_update(img: np.ndarray, limit: list, intervals_middle: list) -> np.ndarray:
    new_image = np.zeros(len(img))

    for ind in range(len(intervals_middle)):
        new_image[limit[len(intervals_middle) - ind] > img] = intervals_middle[len(intervals_middle) - ind - 1]
    return new_image.astype(int)


# find the limits / intervals
def finding_Z(intervals_middle: list[int]) -> list[int]:
    Z = [0]

    for i in range(len(intervals_middle) - 1):
        lim = int((intervals_middle[i] + intervals_middle[i + 1]) / 2)
        Z.append(lim)
    Z.append(255)
    return Z


# find the specific colors of intervals
def finding_Q(limit: list[int], pixel_sum: np.ndarray) -> list[int]:
    Q = []
    for i in range(len(limit) - 1):
        # for every interval
        my_pixel = limit[i]
        color_sum = 0
        sum = 0
        while my_pixel < limit[i + 1]:
            # for every pixel at the interval
            num = pixel_sum[my_pixel]
            my_pixel += 1
            color_sum += num
            sum += num * my_pixel
        Q.append(int(sum / color_sum))
    return Q


# calculation MSE error
def mse(prev_img: np.ndarray, img: np.ndarray, numPixels: int) -> float:
    return (np.square(((pow(prev_img - img, 2)).astype(int)).sum())) / numPixels


