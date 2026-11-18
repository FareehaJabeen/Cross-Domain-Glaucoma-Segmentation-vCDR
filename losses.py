"""
Loss and metrics.

Hybrid loss used in the paper:  L = L_Dice + lambda * L_Focal,  lambda = 0.5
lambda was fixed before training and was not tuned on the external datasets.
Focal loss: gamma = 2, alpha = 0.25.
"""
import tensorflow as tf


def dice_coefficient(y_true, y_pred, smooth=1e-7):
    """Soft Dice (used inside the loss and as the early-stopping metric)."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    inter = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    total = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    return tf.reduce_mean((2.0 * inter + smooth) / (total + smooth))


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)


def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
    return tf.reduce_mean(-alpha_t * tf.pow(1.0 - pt, gamma) * tf.math.log(pt))


def make_loss(dice_weight=1.0, focal_weight=0.5):
    """Paper: make_loss(1.0, 0.5). Ablation: (1.0, 0.0) Dice only, (1.0, 1.0) equal."""
    def hybrid_loss(y_true, y_pred):
        return dice_weight * dice_loss(y_true, y_pred) + focal_weight * focal_loss(y_true, y_pred)
    return hybrid_loss


def iou_metric(y_true, y_pred, smooth=1.0):
    y_true = tf.cast(y_true > 0.5, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    inter = tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
    union = tf.reduce_sum(y_true, axis=(1, 2, 3)) + tf.reduce_sum(y_pred, axis=(1, 2, 3)) - inter
    return tf.reduce_mean((inter + smooth) / (union + smooth))
