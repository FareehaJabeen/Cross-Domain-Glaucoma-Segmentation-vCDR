"""
Network architectures used in the paper (Keras / TensorFlow).

All models share the same EfficientNet-B3 encoder (ImageNet weights) and the
same U-Net style decoder. They differ only in:
  - the bottleneck at 8x8:   plain conv | DRB (sequential dilated) | ASPP (parallel)
  - IR-Skip:                 a 1x1 projection of the input image joined at the last decoder stage
  - decoder dropout:         the baseline has none, all other variants use the same dropout

Model names used in the paper (Table 6):
  build_model("baseline")          -> Baseline
  build_model("aspp")              -> + ASPP
  build_model("drb")               -> + DRB            (new row in the revision)
  build_model("aspp_irskip")       -> + ASPP + IR-Skip
  build_model("drb_irskip")        -> Proposed (DRB + IR-Skip)
"""
import tensorflow as tf
from tensorflow.keras import layers as L
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.models import Model

SKIPS = ["block2a_expand_activation",   # 128 x 128
         "block3a_expand_activation",   # 64 x 64
         "block4a_expand_activation",   # 32 x 32
         "block6a_expand_activation"]   # 16 x 16
DEEPEST = "top_activation"              # 8 x 8


def conv_bn_relu(x, filters, kernel=3, dilation=1):
    x = L.Conv2D(filters, kernel, padding="same", dilation_rate=dilation)(x)
    x = L.BatchNormalization()(x)
    return L.Activation("relu")(x)


def plain_bottleneck(c5):
    return L.Dropout(0.3)(conv_bn_relu(c5, 512))


def drb_bottleneck(c5):
    """Dilated Residual Bottleneck: sequential dilations 1 -> 2 -> 4, outputs summed."""
    x1 = conv_bn_relu(c5, 512, dilation=1)
    x2 = conv_bn_relu(x1, 512, dilation=2)
    x3 = conv_bn_relu(x2, 512, dilation=4)
    return L.Dropout(0.3)(L.Add()([x1, x2, x3]))


def aspp_bottleneck(c5):
    """ASPP: parallel dilations 1, 2, 4 + 1x1 branch, concatenated and fused."""
    branches = [conv_bn_relu(c5, 512, dilation=d) for d in (1, 2, 4)]
    branches.append(conv_bn_relu(c5, 512, kernel=1))
    x = conv_bn_relu(L.concatenate(branches), 512, kernel=1)
    return L.Dropout(0.3)(x)


BOTTLENECKS = {"plain": plain_bottleneck, "drb": drb_bottleneck, "aspp": aspp_bottleneck}

VARIANTS = {   # name: (bottleneck, IR-Skip, decoder dropout)
    "baseline":    ("plain", False, False),
    "drb":         ("drb",   False, True),
    "aspp":        ("aspp",  False, True),
    "aspp_irskip": ("aspp",  True,  True),
    "drb_irskip":  ("drb",   True,  True),   # proposed
}


def build_model(variant="drb_irskip", input_size=(256, 256, 3), encoder_weights="imagenet"):
    bottleneck, irskip, decoder_dropout = VARIANTS[variant]
    inputs = L.Input(input_size)
    encoder = EfficientNetB3(include_top=False, weights=encoder_weights, input_tensor=inputs)
    c1, c2, c3, c4 = [encoder.get_layer(n).output for n in SKIPS]
    c5 = encoder.get_layer(DEEPEST).output

    x = BOTTLENECKS[bottleneck](c5)

    # decoder: 8 -> 16 -> 32 -> 64 -> 128 -> 256
    for filters, skip, drop in [(512, c4, 0.3), (256, c3, 0.2), (128, c2, 0.2), (64, c1, 0.1)]:
        x = L.Conv2DTranspose(filters, 2, strides=2, padding="same")(x)
        x = L.concatenate([x, skip])
        x = conv_bn_relu(x, filters)
        if decoder_dropout:
            x = L.Dropout(drop)(x)

    x = L.Conv2DTranspose(32, 2, strides=2, padding="same")(x)
    if irskip:
        # IR-Skip: learned 1x1 projection of the input image, joined at full resolution
        x = L.concatenate([x, L.Conv2D(32, 1, padding="same")(inputs)])
    x = conv_bn_relu(x, 32)
    outputs = L.Conv2D(1, 1, activation="sigmoid")(x)
    return Model(inputs, outputs, name=variant)
