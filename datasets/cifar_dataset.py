import os

import numpy as np
from PIL import Image


def load_data(folder_path, labels, img_shape):
    x = []
    y = []
    for root, dirs, files in os.walk(folder_path):
        for basename in files:
            if basename.endswith(".png"):
                file_path = os.path.join(root, basename)
                _, label = os.path.split(root)
                label = labels.get(label)
                if label is None:
                    continue
                img = Image.open(file_path)
                img.load()
                img.thumbnail(np.array(img_shape))
                img = np.asarray(img, dtype=np.int16)
                x.append(img)
                y.append(label)
    x = np.asarray(x).reshape((-1, *img_shape, 3)) / 255
    y = np.asarray(y).reshape((-1, 1))
    return x, y
