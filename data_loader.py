## data_loader.py
import cv2
import os

import numpy as np


def compute_histograms(image_paths):
    histograms = []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hist = [cv2.calcHist([img], [i], None, [256], [0, 256]).flatten() / img.size for i in range(3)]
        histograms.append(np.concatenate(hist))
    return np.array(histograms)

def get_first_20_image_paths(directory, class_names):
    paths = []
    for class_name in class_names:
        class_paths = [os.path.join(directory, filename) for filename in os.listdir(directory)
                       if filename.startswith(class_name) and filename.endswith('.jpg')]
        paths.extend(class_paths[:20])
    return paths
