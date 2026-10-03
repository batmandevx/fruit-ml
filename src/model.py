"""Transfer-learning classifier: ImageNet backbone ("backbone") + small head ("head").

The two parts are kept as named sub-models so explain.py can split the forward
pass at the last convolutional feature map for Grad-CAM. Every backbone here
takes raw [0, 255] pixels (preprocessing is built in)."""
import tensorflow as tf

A = tf.keras.applications
# name -> (constructor, default input size)
BACKBONES = {
    "mobilenetv3": (lambda **kw: A.MobileNetV3Large(include_preprocessing=True, **kw), 224),
    "efficientnet": (A.EfficientNetB0, 224),  # closest Keras counterpart of EfficientNet-Lite
    "efficientnetv2b0": (lambda **kw: A.EfficientNetV2B0(include_preprocessing=True, **kw), 260),
    "efficientnetv2b3": (lambda **kw: A.EfficientNetV2B3(include_preprocessing=True, **kw), 300),
    "efficientnetv2s": (lambda **kw: A.EfficientNetV2S(include_preprocessing=True, **kw), 384),
    # ConvNeXt's grouped convs always go through XLA in Keras, which the Metal plugin lacks: GPU/Colab only.
    "convnext_tiny": (lambda **kw: A.ConvNeXtTiny(include_preprocessing=True, **kw), 288),
    "convnext_small": (lambda **kw: A.ConvNeXtSmall(include_preprocessing=True, **kw), 288),
}


def build_backbone(name, img_size):
    base = BACKBONES[name][0](input_shape=(img_size, img_size, 3), include_top=False, weights="imagenet")
    return tf.keras.Model(base.input, base.output, name="backbone")


def build_model(num_classes, backbone_name, img_size, dropout=0.3):
    backbone = build_backbone(backbone_name, img_size)
    backbone.trainable = False
    head = tf.keras.Sequential(
        [
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(num_classes, kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
            # float32 softmax so mixed precision stays numerically safe
            tf.keras.layers.Activation("softmax", dtype="float32"),
        ],
        name="head",
    )
    inputs = tf.keras.Input((img_size, img_size, 3), name="image")
    # training=False keeps BatchNorm statistics frozen, also during fine-tuning.
    outputs = head(backbone(inputs, training=False))
    return tf.keras.Model(inputs, outputs, name=f"date_classifier_{backbone_name}")


def unfreeze_top(model, n_layers):
    """Unfreeze the last n_layers of the backbone (n_layers <= 0: all). BatchNorm stays frozen."""
    backbone = model.get_layer("backbone")
    backbone.trainable = True
    if n_layers > 0:
        for layer in backbone.layers[:-n_layers]:
            layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
