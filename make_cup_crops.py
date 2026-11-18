"""
Make Stage II (optic cup) training crops.

During TRAINING, the cup ROI is taken from the ground-truth optic disc mask
(so disc errors do not affect cup training). At TEST time the ROI comes from
the predicted disc (see vcdr.py).

For each image (512 x 512): bounding box of the GT disc + 15 px padding,
crop the image and the cup mask with the same box, save both.

Example:
  python make_cup_crops.py --image-dir full-fundus --disc-dir optic-disc \
      --cup-dir optic-cup --out-images cup_crops/images --out-masks cup_crops/masks
"""
import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm

from src.data import find_mask

FULL, PAD = 512, 15


def main(a):
    os.makedirs(a.out_images, exist_ok=True)
    os.makedirs(a.out_masks, exist_ok=True)
    names = [f for f in sorted(os.listdir(a.image_dir)) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif"))]
    done = 0
    for n in tqdm(names):
        dp, cp = find_mask(a.disc_dir, n), find_mask(a.cup_dir, n)
        if dp is None or cp is None:
            continue
        img = cv2.resize(cv2.imread(os.path.join(a.image_dir, n)), (FULL, FULL))
        disc = cv2.resize(cv2.imread(dp, cv2.IMREAD_GRAYSCALE), (FULL, FULL)) > 127
        cup = cv2.resize(cv2.imread(cp, cv2.IMREAD_GRAYSCALE), (FULL, FULL))
        ys, xs = np.where(disc)
        if len(ys) == 0:
            continue
        y0, y1 = max(0, ys.min() - PAD), min(FULL, ys.max() + PAD)
        x0, x1 = max(0, xs.min() - PAD), min(FULL, xs.max() + PAD)
        stem = os.path.splitext(n)[0]
        cv2.imwrite(os.path.join(a.out_images, n), img[y0:y1, x0:x1])
        cv2.imwrite(os.path.join(a.out_masks, stem + ".png"), cup[y0:y1, x0:x1])
        done += 1
    print(f"saved {done} crops")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image-dir", required=True)
    p.add_argument("--disc-dir", required=True)
    p.add_argument("--cup-dir", required=True)
    p.add_argument("--out-images", required=True)
    p.add_argument("--out-masks", required=True)
    main(p.parse_args())
