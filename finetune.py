"""
Limited fine-tuning on a target dataset (REFUGE or DRISHTI-GS1).

Starts from the ORIGA-trained model. ALL layers are updated (nothing frozen).
"Limited" means few epochs: up to 30, lr 1e-5, batch 4, same callbacks and
loss as train.py. Only the target development (train) split is used;
the official test split is never seen.

Example:
  python finetune.py --model models/disc_main.h5 --image-dir refuge/images \
      --mask-dir refuge/disc_masks --train-csv splits/refuge_ft_train.csv \
      --val-csv splits/refuge_ft_val.csv --out models/disc_refuge_ft.keras
"""
import argparse
import os
import tensorflow as tf

from src.data import load_split
from src.losses import make_loss, dice_coefficient, iou_metric
from train import set_seed, callbacks


def main(a):
    set_seed(a.seed)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    Xtr, Ytr, _ = load_split(a.image_dir, a.mask_dir, a.train_csv)
    Xva, Yva, _ = load_split(a.image_dir, a.mask_dir, a.val_csv)
    print(f"fine-tune train {len(Xtr)}  val {len(Xva)}")

    model = tf.keras.models.load_model(a.model, compile=False)
    for layer in model.layers:          # full fine-tuning: every layer is trainable
        layer.trainable = True
    model.compile(optimizer=tf.keras.optimizers.Adam(a.lr),
                  loss=make_loss(1.0, 0.5),
                  metrics=[dice_coefficient, iou_metric])
    model.fit(Xtr, Ytr, validation_data=(Xva, Yva), epochs=a.epochs, batch_size=a.batch,
              callbacks=callbacks(a.out), verbose=1)
    model.save(a.out)
    print(f"saved {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="ORIGA-trained model to start from")
    p.add_argument("--image-dir", required=True)
    p.add_argument("--mask-dir", required=True)
    p.add_argument("--train-csv", required=True)
    p.add_argument("--val-csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
