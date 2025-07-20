
#pip install opencv-python
#!pip3 uninstall opencv-python
#!pip install opencv-python==4.8.0.74.
import cv2
import numpy as np


# preprocessing the image input
def preparation(input):
    clean = cv2.fastNlMeansDenoising(input)
    ret, tresh = cv2.threshold(clean, 127, 1, cv2.THRESH_BINARY_INV)
    img = crop(tresh)
    
    # flatten image image as an array
    flatten_img = cv2.resize(img, (40, 10), interpolation=cv2.INTER_AREA).flatten()

    # resize the input image
    resd = cv2.resize(img, (400, 100), interpolation=cv2.INTER_AREA)
    col = np.sum(resd, axis=0)  # sum of all columns
    row = np.sum(resd, axis=1)  # sum lines

    h, w = img.shape
    aspect = w / h

    return [*flatten_img, *col, *row, aspect]


def crop(img):
    points = cv2.findNonZero(img)
    x, y, w, h = cv2.boundingRect(points)
    return img[y: y+h, x: x+w]






