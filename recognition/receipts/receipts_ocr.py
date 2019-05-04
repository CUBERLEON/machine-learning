import cv2
import sys
import re
import pytesseract
import numpy as np
from natsort import natsorted
from skimage.morphology import watershed

from common.config import REMOTE_DATA_URL, REMOTE_DATA_DIR


def preprocess_v1(img, img_init_width=1800, img_width=1200):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)

    img_height = int(img.shape[0] / img.shape[1] * img_init_width)
    new_shape = (img_init_width, img_height)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_LINEAR)

    img = cv2.bilateralFilter(img, 3, 75, 75)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 10)

    img_height = int(img.shape[0] / img.shape[1] * img_width * 0.75)
    new_shape = (img_width, img_height)
    print(new_shape)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_LINEAR)

    return img


def preprocess_v2(img, img_init_width=1800, img_width=1200):
    img_height = int(img.shape[0] / img.shape[1] * img_init_width)
    new_shape = (img_init_width, img_height)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_LINEAR)

    img = cv2.fastNlMeansDenoising(img, h=9, templateWindowSize=13)

    img = cv2.bilateralFilter(img, 3, 75, 75)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 4)

    kernel = np.ones((2, 2), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)

    img_height = int(img.shape[0] / img.shape[1] * img_width * 0.75)
    new_shape = (img_width, img_height)
    print(new_shape)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_LINEAR)

    return img


def preprocess_v3(img, img_init_width=1800, img_width=1200):
    img_height = int(img.shape[0] / img.shape[1] * img_init_width)
    new_shape = (img_init_width, img_height)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_LINEAR)

    # img = cv2.medianBlur(img, 3)
    # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    # img = cv2.filter2D(img, -1, kernel)

    # img = cv2.bilateralFilter(img, 3, 75, 75)
    # img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 45, 10)
    # img = cv2.bilateralFilter(img, 9, 75, 75)
    # img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # img = cv2.medianBlur(img, 5)

    img = cv2.fastNlMeansDenoising(img, h=9, templateWindowSize=13)

    img = cv2.bilateralFilter(img, 3, 75, 75)
    # img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)[1]
    img_binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 4)

    kernel = np.ones((3, 3), np.uint8)
    img_binary = cv2.dilate(img_binary, kernel, iterations=1)
    img_binary = cv2.erode(img_binary, kernel, iterations=1)

    img = np.array(np.clip(img * 1.5, 0, 255), dtype=np.uint8)
    img[img_binary == 0] = 0
    kernel_3 = np.ones((3, 3), np.uint8)
    img_binary = cv2.erode(img_binary, kernel_3, iterations=2)
    # cv2.imshow('img_binary', img_binary)
    img[img_binary == 255] = 255

    # img_width = img.shape[1]
    img_height = int(img.shape[0] / img.shape[1] * img_width * 0.75)
    new_shape = (img_width, img_height)
    print(new_shape)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_LINEAR)

    return img


if __name__ == '__main__':
    images_dir = REMOTE_DATA_DIR / "receipts"

    for image_path in natsorted(images_dir.iterdir()):
        img_orig = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)
        # img = preprocess_v3(img_orig.copy())
        img = img_orig.copy()
        print(img.shape)

        cv2.imshow('image_orig', cv2.resize(img_orig, (600, 800)))
        cv2.moveWindow('image_orig', 1700, 0)
        cv2.imshow('image', img)
        cv2.moveWindow('image', 2300, 0)

        # '--oem 1' for using LSTM OCR Engine
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        text = pytesseract.image_to_string(img,
                                           config=f'-l eng+ukr+rus --oem 1 --psm 3')
        filtered_text = "\n".join(filter(lambda x: not re.match(r'^\s*$', x), text.splitlines()))
        print(filtered_text.lower()  )

        cv2.waitKey()

    cv2.destroyAllWindows()
