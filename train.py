"""
Train a disc (Stage I) or cup (Stage II) model on ORIGA-Light.

Settings (as in the paper): Adam, lr 1e-4, batch 4, up to 100 epochs,
ReduceLROnPlateau on validation loss (factor 0.1, patience 5, min 1e-10),
early stopping on validation Dice (patience 10, best weights restored).
Loss = Dice + 0.5 x Focal. Seed 42 (also run with 123 and 2026).

Disc model: full fundus images + disc masks.
Cup model:  GT-disc crops from make_cup_crops.py + cropped cup masks.

Example:
  python train.py --variant drb_irskip --image-dir full-fundus --mask-dir optic-disc \
      --train-csv splits/origa_train.csv --val-csv splits/origa_val.csv --out models/disc_main.h5
"""
import argparse
import os
import random
import numpy as np
import tensorflow as tf

from src.data import load_split
from src.losses import make_loss, dice_coefficient, iou_metric
from src.models import build_model, VARIANTS


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)


def callbacks(out):
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_dice_coefficient", mode="max", patience=10,
                                         restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5,
                                             min_lr=1e-10, verbose=1),
        tf.keras.callbacks.CSVLogger(os.path.splitext(out)[0] + "_history.csv"),
    ]


def main(a):
    set_seed(a.seed)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    Xtr, Ytr, _ = load_split(a.image_dir, a.mask_dir, a.train_csv)
    Xva, Yva, _ = load_split(a.image_dir, a.mask_dir, a.val_csv)
    print(f"train {len(Xtr)}  val {len(Xva)}")

    model = build_model(a.variant)
    model.compile(optimizer=tf.keras.optimizers.Adam(a.lr),
                  loss=make_loss(1.0, a.focal_weight),
                  metrics=[dice_coefficient, iou_metric])
    model.fit(Xtr, Ytr, validation_data=(Xva, Yva), epochs=a.epochs, batch_size=a.batch,
              callbacks=callbacks(a.out), verbose=1)
    model.save(a.out)
    print(f"saved {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="drb_irskip", choices=list(VARIANTS))
    p.add_argument("--image-dir", required=True)
    p.add_argument("--mask-dir", required=True)
    p.add_argument("--train-csv", required=True)
    p.add_argument("--val-csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--focal-weight", type=float, default=0.5, help="0 = Dice only, 1 = equal weights")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
