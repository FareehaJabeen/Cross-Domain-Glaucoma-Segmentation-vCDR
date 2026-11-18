"""
Two-stage inference and vCDR (fully automatic, no ground truth used for prediction).

  1. Original fundus image (512 x 512).
  2. Stage I: disc model on the 256 x 256 image; prediction mapped back to 512.
  3. Only the largest connected region of the predicted disc is kept.
  4. Bounding box of the predicted disc + 15 px padding (512 space).
  5. The SAME box is cropped from the image (-> cup model) and from the predicted disc.
  6. Both crops are resized to 256 x 256; cup and disc heights are measured
     in the same crop, so the vertical scale 256/H cancels:  vCDR = H_cup / H_disc.
  7. Reference vCDR from the original 512 ground-truth masks (binarised at 127).

Output: one CSV with per-image heights, vCDR (true, predicted) and absolute error.

Example:
  python vcdr.py --disc-model models/disc_main.h5 --cup-model models/cup_main.h5 \
      --image-dir refuge/images --disc-dir refuge/disc_masks --cup-dir refuge/cup_masks \
      --names splits/refuge_test.csv --out results/refuge_zs_main.csv
"""
import argparse
import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from src.data import find_mask

FULL, NET, PAD, THR, BATCH = 512, 256, 15, 0.5, 16


def to_net(img_bgr):
    return cv2.resize(img_bgr, (NET, NET)).astype(np.float32) / 255.0


def read_mask512(path):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return (cv2.resize(m, (FULL, FULL)) > 127).astype(np.uint8)


def largest_cc(mask):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 2:
        return mask
    return (lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)


def bbox(mask):
    ys, xs = np.where(mask > 0)
    return None if len(ys) == 0 else (ys.min(), ys.max(), xs.min(), xs.max())


def height(mask):
    ys = np.where(mask > 0)[0]
    return 0 if len(ys) == 0 else int(ys.max() - ys.min())


def predict(model, X):
    out = [np.asarray(model(X[i:i + BATCH], training=False)) for i in range(0, len(X), BATCH)]
    return np.concatenate(out)[..., 0]


def main(a):
    names = pd.read_csv(a.names)["filename"].tolist()
    disc_model = tf.keras.models.load_model(a.disc_model, compile=False)
    cup_model = tf.keras.models.load_model(a.cup_model, compile=False)

    imgs, nets, gds, gcs, kept = [], [], [], [], []
    for n in tqdm(names, desc="loading"):
        img = cv2.imread(os.path.join(a.image_dir, n))
        dp, cp = find_mask(a.disc_dir, n), find_mask(a.cup_dir, n)
        if img is None or dp is None or cp is None:
            print(f"skipping {n}")
            continue
        nets.append(to_net(img))
        imgs.append(cv2.resize(img, (FULL, FULL)))
        gds.append(read_mask512(dp)); gcs.append(read_mask512(cp)); kept.append(n)

    p_disc = predict(disc_model, np.array(nets))                     # Stage I
    rows, crops, disc_crops, idx = [], [], [], []
    for i, n in enumerate(kept):
        r = {"filename": n, "disc_h_true": height(gds[i]), "cup_h_true": height(gcs[i])}
        r["vcdr_true"] = r["cup_h_true"] / r["disc_h_true"] if r["disc_h_true"] else np.nan
        disc512 = largest_cc((cv2.resize(p_disc[i], (FULL, FULL)) > THR).astype(np.uint8))
        b = bbox(disc512)
        if b is not None:
            y0, y1 = max(0, b[0] - PAD), min(FULL, b[1] + PAD)
            x0, x1 = max(0, b[2] - PAD), min(FULL, b[3] + PAD)
            r.update(crop_y0=y0, crop_y1=y1, crop_x0=x0, crop_x1=x1)
            crops.append(to_net(imgs[i][y0:y1, x0:x1]))
            disc_crops.append(cv2.resize(disc512[y0:y1, x0:x1], (NET, NET), interpolation=cv2.INTER_NEAREST))
            idx.append(i)
        rows.append(r)

    p_cup = predict(cup_model, np.array(crops)) if crops else []    # Stage II
    for j, i in enumerate(idx):
        r = rows[i]
        r["disc_h_pred"] = height(disc_crops[j])
        r["cup_h_pred"] = height((p_cup[j] > THR).astype(np.uint8))
        r["vcdr_pred"] = r["cup_h_pred"] / r["disc_h_pred"] if r["disc_h_pred"] else np.nan

    df = pd.DataFrame(rows)
    df["abs_error"] = (df["vcdr_pred"] - df["vcdr_true"]).abs()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out, index=False)
    err = df["vcdr_pred"] - df["vcdr_true"]
    print(f"{a.out}: n={len(df)}  MAE={err.abs().mean():.4f} ± {err.abs().std():.4f}  "
          f"bias={err.mean():+.4f}  LoA=[{err.mean() - 1.96 * err.std():.3f}, {err.mean() + 1.96 * err.std():.3f}]")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--disc-model", required=True)
    p.add_argument("--cup-model", required=True)
    p.add_argument("--image-dir", required=True)
    p.add_argument("--disc-dir", required=True)
    p.add_argument("--cup-dir", required=True)
    p.add_argument("--names", required=True, help="CSV with a 'filename' column (test images)")
    p.add_argument("--out", required=True)
    main(p.parse_args())
