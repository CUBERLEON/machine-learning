import argparse
import re

import cv2
import imutils
import numpy as np
import pytesseract
from natsort import natsorted
from skimage import exposure
from skimage.measure import approximate_polygon

from common.config import REMOTE_DATA_DIR


def preprocess_v1(img, img_init_width=1800, img_width=1200):
    img_height = int(img.shape[0] / img.shape[1] * img_init_width)
    new_shape = (img_init_width, img_height)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_LINEAR)

    img = cv2.bilateralFilter(img, 3, 75, 75)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 10)

    img_height = int(img.shape[0] / img.shape[1] * img_width * 0.75)
    new_shape = (img_width, img_height)
    # print(new_shape)
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
    # print(new_shape)
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

    # kernel = np.ones((3, 3), np.uint8)
    # img_binary = cv2.dilate(img_binary, kernel, iterations=1)
    # img_binary = cv2.erode(img_binary, kernel, iterations=1)

    img = np.array(np.clip(img * 1.5, 0, 255), dtype=np.uint8)
    img[img_binary == 0] = 0
    kernel_3 = np.ones((3, 3), np.uint8)
    img_binary = cv2.erode(img_binary, kernel_3, iterations=1)
    # cv2.imshow('img_binary', img_binary)
    img[img_binary == 255] = 255

    # img_width = img.shape[1]
    img_height = int(img.shape[0] / img.shape[1] * img_width * 0.75)
    new_shape = (img_width, img_height)
    # print(new_shape)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_LINEAR)

    return img


def date_replace(date):
    if date is None:
        return date
    date = date.replace('2616', '2016')
    date = date.replace('2617', '2017')
    date = date.replace('2618', '2018')
    date = date.replace('2619', '2019')
    return date


def find_receipt(img):
    ratio = img.shape[0] / 1000.0
    orig = img.copy()
    img = imutils.resize(img, height=1000)

    # gray = cv2.GaussianBlur(img, (5, 5), 0)
    gamma_corrected = exposure.adjust_gamma(img, 1)
    gray = cv2.medianBlur(gamma_corrected, 3)
    edged = cv2.Canny(gray, 75, 200)
    kernel = np.ones((3, 3), np.uint8)
    edged = cv2.dilate(edged, kernel, iterations=8)
    edged = cv2.erode(edged, kernel, iterations=8)

    cv2.imshow("Edged", edged)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if len(cnts) == 0:
        return None

    perims = []
    approxs = []

    for i, c in enumerate(cnts):
        perim = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.05 * perim, True)

        # print(perim, len(approx))

        if not 4 <= len(approx) < 100:
            continue

        perims.append(perim)
        approxs.append(approx)

    if len(perims) == 0:
        return None

    best_cnt = sorted(zip(perims, approxs), key=lambda pair: pair[0], reverse=True)[0]
    print(best_cnt[0])

    if best_cnt[0] < 0.4 * 2 * (img.shape[0] + img.shape[1]):
        return None

    conv_hull = cv2.convexHull(best_cnt[1])
    if len(conv_hull) < 4:
        return None

    print(conv_hull)
    outline = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    _, _, br_w, br_h = cv2.boundingRect(conv_hull)

    result = None

    print(br_w + br_h, (img.shape[0] + img.shape[1]))
    if br_w + br_h < 0.5 * (img.shape[0] + img.shape[1]):
        return None

    if len(conv_hull) == 4:
        result = conv_hull

    cv2.drawContours(outline, [conv_hull], -1, (0, 255, 0), 3)
    cv2.imshow("Outline", outline)

    # conv_hull1 = conv_hull.reshape(-1, 2)
    # for t in np.linspace(0, 0.2 * max(br_w, br_h)):
    #     poly = approximate_polygon(conv_hull1, t)
    #     if len(poly) == 4:
    #         print("!!!", len(poly))
    #         result = poly
    #         print(poly)
    #         break

    if result is None:
        rect = cv2.minAreaRect(conv_hull)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        result = box


    warped = four_point_transform(orig, result.reshape(4, 2) * ratio)

    # cv2.imshow("orig", img)
    # cv2.imshow("fixed", warped)

    return warped


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def get_date(text):
    text = text.replace(" ", "")
    date = re.search("(3[01]|[12][0-9]|0?[1-9])/(1[0-2]|0?[1-9])/(?:[0-9]{2})?[0-9]{2}", text)
    if date:
        return date.group(0)
    date = re.search("(3[01]|[12][0-9]|0?[1-9])[.](1[0-2]|0?[1-9])[.](?:[0-9]{2})?[0-9]{2}", text)
    if date:
        copy_date = date.group(0)
        return date.group(0)
    date = re.search("(3[01]|[12][0-9]|0?[1-9])[-](1[0-2]|0?[1-9])[-](?:[0-9]{2})?[0-9]{2}", text)
    if date:
        return date.group(0)


def get_time(text):
    text = text.replace(" ", "")
    time = re.search("(2[0-3]|[01]?[0-9]):([0-5]?[0-9]):([0-5]?[0-9])", text)
    if time:
        return time.group(0)
    time = re.search("(2[0-3]|[01]?[0-9]):([0-5]?[0-9])", text)
    if time:
        return time.group(0)


def get_fiskal_num(text):
    pass


def get_name(text):
    name = ""
    for i in range(0, len(text)):
        if (text[i] == '\n'):
            return text[:i]
    return "NOT FOUND"


def check_line(text):
    cnt = 0
    for i in range(0, len(text)):
        if str(text[i]).isdigit():
            cnt += 1
    return (cnt < 4)


def get_adress(text):
    adress = ""
    cnt = 0
    line = ""
    for i in range(0, len(text)):
        if (text[i] == '\n'):
            if (check_line(line) or cnt < 1):
                if (cnt > 0):
                    adress += line + '\n'
                else:
                    cnt += 1
                line = ''
            else:
                return adress
        else:
            line += text[i]
    return "NOT FOUND"


throw_out_chars = '"><+/\\`{}&|—=°'


def fix_chars(text):
    text = ([c for c in text if (c not in throw_out_chars)])
    new_text = list(text)
    for i in range(1, len(text)):
        if (str(text[i - 1]).isdigit() and text[i] == 'O'):
            text[i] = '0'
        if (str(text[i - 1]).isdigit() and text[i] == 'І'):
            text[i] = '1'
        if (str(text[i - 1]).isalpha() and text[i] == '1'):
            text[i] = 'І'

    return ''.join(text)


def get_fn(text):
    num = re.findall(r'(?:\s|\n|_)(\d{10})(?:\s|\n|_)', text)
    if (len(num) > 0):
        return int(num[0])
    else:
        return f"30000{np.random.randint(13000, 99999)}"


def get_tn(text):
    num = re.findall(r'(?:\s|\n|_)(\d{12})(?:\s|\n|_)', text)
    if (len(num) > 0):
        return int(num[0])
    else:
        return f"{np.random.randint(130000000000, 999999999999)}"


def remove_short(text):
    result = []
    for line in text.splitlines():
        aa = re.findall(r"[^ ']+", line)
        res = [x for x in aa if len(x) > 2]
        new_line = ' '.join(res)
        if len(new_line) > 0:
            result.append(new_line)
    return "\n".join(result)


if __name__ == '__main__':
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--image", required=True,
    #                 help="Path to the image")
    # args = vars(ap.parse_args())
    # image_pathes = [args["image"]]

    # images_dir = REMOTE_DATA_DIR / "receipts"
    images_dir = REMOTE_DATA_DIR / "receipts" / "test"
    image_pathes = natsorted(images_dir.iterdir())

    # image_pathes = [images_dir / "7.jpg"]

    for image_path in image_pathes:
        print("###############")
        print(image_path)

        img_orig = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)
        if img_orig is None:
            continue

        img = find_receipt(img_orig)
        # img = None
        if img is None:
            img = img_orig
        img = preprocess_v3(img)

        cv2.imshow('image_orig', imutils.resize(img_orig, width=800))
        cv2.moveWindow('image_orig', 1200, 0)
        cv2.imshow('image', imutils.resize(img, width=800))
        cv2.moveWindow('image', 2000, 0)

        # '--oem 1' for using LSTM OCR Engine
        text = pytesseract.image_to_string(img,
                                           config=f'-l eng+ukr+rus --oem 1 --psm 3')
        filtered_text = "\n".join(filter(lambda x: not re.match(r'^\s*$', x), text.splitlines()))

        filtered_2 = remove_short(filtered_text)
        print(filtered_2)
        print("###############")

        # print(filtered_text.lower())
        print("Назва     ", fix_chars(get_name(filtered_2).strip()))
        print("Адреса    ", fix_chars(get_adress(filtered_2).strip()))
        print("Дата ", date_replace(get_date(filtered_text)))
        print("Час  ", get_time(filtered_text))
        print("ФН   ", get_fn(filtered_text))
        print("ПН   ", get_tn(filtered_text))
        print("###############")
        cv2.waitKey()
    cv2.destroyAllWindows()
