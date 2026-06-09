"""Deep-learning model architectures (Keras 3 / TensorFlow).

Implements the three networks discussed in the report:

* :func:`build_eegnet` — the standard compact EEGNet baseline.
* :func:`build_vareegnet` — EEGNet with a log-variance compression layer that
  replaces the temporal-average pooling + flatten, cutting the parameter count
  by ~55% relative to EEGNet.
* :func:`build_fusion_3cnn` — three parallel temporal-convolution branches with
  different kernel sizes, concatenated before a dense classification head.

All models expect input of shape ``(channels, samples, 1)`` and emit a softmax
over ``n_classes``.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.constraints import max_norm

from .config import EEGNetConfig, FusionCNNConfig


def _log_variance(x: tf.Tensor) -> tf.Tensor:
    """Log-variance pooling over the time axis.

    Input shape ``(batch, 1, time, filters)``; output ``(batch, filters)``.
    Computing variance over time and taking its log mirrors the CSP
    log-variance feature and is what makes varEEGNet compact: it collapses the
    entire temporal axis to a single scalar per filter.
    """
    var = tf.math.reduce_variance(x, axis=2)        # (batch, 1, filters)
    var = tf.squeeze(var, axis=1)                   # (batch, filters)
    return tf.math.log(tf.clip_by_value(var, 1e-6, 1e6))


def build_eegnet(
    n_channels: int,
    n_samples: int,
    n_classes: int,
    cfg: EEGNetConfig,
) -> models.Model:
    """Construct a standard EEGNet-8,2 model.

    Architecture follows Lawhern et al. (2018): a temporal convolution, a
    depthwise spatial convolution, then a separable convolution, each followed
    by batch norm, ELU activation, average pooling, and dropout.
    """
    inp = layers.Input(shape=(n_channels, n_samples, 1))

    # Block 1: temporal convolution then depthwise spatial convolution.
    x = layers.Conv2D(
        cfg.F1,
        (1, cfg.kern_length),
        padding="same",
        use_bias=False,
    )(inp)
    x = layers.BatchNormalization()(x)
    x = layers.DepthwiseConv2D(
        (n_channels, 1),
        use_bias=False,
        depth_multiplier=cfg.D,
        depthwise_constraint=max_norm(1.0),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("elu")(x)
    x = layers.AveragePooling2D((1, 4))(x)
    x = layers.Dropout(cfg.dropout)(x)

    # Block 2: separable convolution (depthwise + pointwise).
    x = layers.SeparableConv2D(
        cfg.F2,
        (1, 16),
        padding="same",
        use_bias=False,
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("elu")(x)
    x = layers.AveragePooling2D((1, 8))(x)
    x = layers.Dropout(cfg.dropout)(x)

    x = layers.Flatten()(x)
    out = layers.Dense(
        n_classes,
        kernel_constraint=max_norm(0.25),
    )(x)
    out = layers.Activation("softmax")(out)

    return models.Model(inputs=inp, outputs=out, name="EEGNet")


def build_vareegnet(
    n_channels: int,
    n_samples: int,
    n_classes: int,
    cfg: EEGNetConfig,
) -> models.Model:
    """Construct varEEGNet: EEGNet feature blocks with log-variance compression.

    varEEGNet keeps both EEGNet convolution blocks but replaces the temporal
    average-pooling + flatten that precedes the classifier with a single
    log-variance layer over time. Standard EEGNet flattens ``F2`` feature maps
    across many time bins into a large dense input; varEEGNet collapses each of
    the ``F2`` maps to one log-variance scalar (``K2 = F2`` features). This is
    the compression described in the report (feature vector ``K2 * NT/32 -> K2``)
    and yields the ~1,524-parameter model.
    """
    inp = layers.Input(shape=(n_channels, n_samples, 1))

    # Block 1: temporal convolution then depthwise spatial convolution.
    x = layers.Conv2D(
        cfg.F1,
        (1, cfg.kern_length),
        padding="same",
        use_bias=False,
    )(inp)
    x = layers.BatchNormalization()(x)
    x = layers.DepthwiseConv2D(
        (n_channels, 1),
        use_bias=False,
        depth_multiplier=cfg.D,
        depthwise_constraint=max_norm(1.0),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("elu")(x)
    x = layers.AveragePooling2D((1, 4))(x)
    x = layers.Dropout(cfg.dropout)(x)

    # Block 2: separable convolution producing F2 feature maps.
    x = layers.SeparableConv2D(
        cfg.F2,
        (1, 16),
        padding="same",
        use_bias=False,
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("elu")(x)

    # Log-variance compression over time -> (batch, F2) features, replacing the
    # average-pool + flatten of standard EEGNet. This is the parameter saving.
    x = layers.Lambda(_log_variance, name="log_variance")(x)
    # LayerNormalization (not BatchNormalization) on the compressed features:
    # log-variance outputs have a distribution whose batch moving-average
    # statistics transfer poorly to inference, which collapses predictions at
    # test time. LayerNorm normalises per-sample and avoids that failure mode.
    x = layers.LayerNormalization(name="logvar_norm")(x)
    x = layers.Dropout(cfg.dropout)(x)

    out = layers.Dense(
        n_classes,
        kernel_constraint=max_norm(0.25),
    )(x)
    out = layers.Activation("softmax")(out)

    return models.Model(inputs=inp, outputs=out, name="varEEGNet")


def _conv_branch(
    inp: tf.Tensor,
    n_filters: int,
    kernel_length: int,
    n_channels: int,
    dropout: float,
) -> tf.Tensor:
    """One temporal-convolution branch for the fusion network.

    A temporal convolution captures band-specific dynamics at the branch's
    kernel scale; a depthwise spatial convolution then mixes channels. Two
    average-pooling stages aggressively downsample the time axis so the
    flattened branch output stays small, keeping the fused model near the
    report's ~50k-parameter budget and fast to train.
    """
    x = layers.Conv2D(n_filters, (1, kernel_length), padding="same", use_bias=False)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("elu")(x)
    x = layers.AveragePooling2D((1, 4))(x)
    x = layers.DepthwiseConv2D((n_channels, 1), use_bias=False, depth_multiplier=1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("elu")(x)
    x = layers.AveragePooling2D((1, 8))(x)
    x = layers.Dropout(dropout)(x)
    # Global average pooling collapses the remaining time axis to one value per
    # filter, so each branch contributes only ``n_filters`` features. This keeps
    # the fused dense head small (~50k total params) and speeds up training.
    x = layers.GlobalAveragePooling2D()(x)
    return x


def build_fusion_3cnn(
    n_channels: int,
    n_samples: int,
    n_classes: int,
    cfg: FusionCNNConfig,
) -> models.Model:
    """Construct the Fusion 3CNNs model.

    Three parallel branches with different temporal kernel sizes learn
    complementary multi-scale features; their flattened outputs are
    concatenated and passed through a dense head with softmax output.
    """
    if not (len(cfg.branch_filters) == len(cfg.branch_kernels) == 3):
        raise ValueError("Fusion 3CNNs requires exactly three branches.")

    inp = layers.Input(shape=(n_channels, n_samples, 1))
    branches = [
        _conv_branch(inp, n_filters, kern, n_channels, cfg.dropout)
        for n_filters, kern in zip(cfg.branch_filters, cfg.branch_kernels)
    ]
    x = layers.Concatenate()(branches)
    x = layers.Dense(
        cfg.dense_units, activation="elu", kernel_regularizer=regularizers.l2(1e-3)
    )(x)
    x = layers.Dropout(cfg.dropout)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)

    return models.Model(inputs=inp, outputs=out, name="Fusion3CNNs")


def compile_model(model: models.Model, learning_rate: float) -> models.Model:
    """Compile a model with Adam and sparse categorical cross-entropy."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
