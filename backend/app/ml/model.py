import tensorflow as tf
from tensorflow.keras import Model, layers

from app.core.config import settings


def build_cnn_lstm_model(
    sequence_length: int = settings.sequence_length,
    image_size: int = settings.image_size,
) -> Model:
    base_cnn = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(image_size, image_size, 3),
        pooling="avg",
    )
    base_cnn.trainable = False
    for layer in base_cnn.layers[-20:]:
        layer.trainable = True

    inputs = layers.Input(shape=(sequence_length, image_size, image_size, 3))
    encoded = layers.TimeDistributed(base_cnn)(inputs)
    encoded = layers.TimeDistributed(layers.Dense(256, activation="relu"))(encoded)
    encoded = layers.Dropout(0.3)(encoded)
    temporal = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(encoded)
    temporal = layers.Bidirectional(layers.LSTM(64))(temporal)
    dense = layers.Dense(128, activation="relu")(temporal)
    dense = layers.Dropout(0.4)(dense)
    outputs = layers.Dense(1, activation="sigmoid")(dense)

    model = Model(inputs=inputs, outputs=outputs, name="hybrid_cnn_lstm")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="roc_auc"),
        ],
    )
    return model
def get_forensic_backbone() -> Model:
    """
    Returns a pre-trained MobileNetV2 model for feature extraction.
    Used for deep forensic analysis when the primary CNN-LSTM artifact is missing.
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(settings.image_size, settings.image_size, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    return base_model
