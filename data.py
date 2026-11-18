"""
Data loading. Images are read with OpenCV (BGR), resized to 256 x 256 and
scaled to [0, 1]. Masks are resized to the same size and binarised at 127
(50 % intensity; for DRISHTI-GS1 this is the majority of expert annotations).
"""
import os
import cv2
import numpy as np
import pandas as pd

IMG = 256


def find_mask(mask_dir, img_name):
    stem = os.path.splitext(img_name)[0]
    for ext in (".png", ".jpg", ".bmp", ""):
        p = os.path.join(mask_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None


def read_image(path, size=IMG):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.resize(img, (size, size)).astype(np.float32) / 255.0


def read_mask(path, size=IMG):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    return (cv2.resize(m, (size, size)) > 127).astype(np.float32)


def load_split(image_dir, mask_dir, csv_path, size=IMG):
    """Load all images and masks listed in a split CSV (column 'filename')."""
    names = pd.read_csv(csv_path)["filename"].tolist()
    X, Y, kept = [], [], []
    for n in names:
        mp = find_mask(mask_dir, n)
        if mp is None:
            print(f"skipping {n}: no mask")
            continue
        X.append(read_image(os.path.join(image_dir, n), size))
        Y.append(read_mask(mp, size)[..., None])
        kept.append(n)
    return np.array(X, np.float32), np.array(Y, np.float32), kept
