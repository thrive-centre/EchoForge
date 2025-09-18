import tensorflow as tf
from echoforge.utils.loader import download_model


def build_EchoSDNet(
    input_shape=(224, 224, 3),
    num_classes=2,
    pretrained=True,
    fine_tune_block="conv5",   # set to None to freeze all; "conv5" to unfreeze final stage
    learning_rate=1e-5,
    dropout_rate=0.5,
):
    """
    Builds or loads the EchoSDNet classifier.

    EchoSDNet uses a ResNet50 backbone (ImageNet weights, include_top=False) with a
    custom classification head: GAP -> Dense(128, ReLU) -> Dense(64, ReLU) ->
    Dropout(0.5) -> Output. By default, only the final residual stage ("conv5") is
    unfrozen for domain fine-tuning; batch norm layers remain frozen.

    Args:
        input_shape (tuple): Input image shape, default (224, 224, 3).
        num_classes (int): Number of output classes. Default 2 (Single vs Dual).
        pretrained (bool): If True, loads a packaged model from Hugging Face.
        fine_tune_block (str|None): Which backbone stage to unfreeze. Options:
            - "conv5" (default): unfreeze only conv5_x
            - None: freeze entire backbone
        learning_rate (float): Adam learning rate (default 1e-5).
        dropout_rate (float): Dropout rate in the head (default 0.5).

    Returns:
        tf.keras.Model: A compiled Keras model.
    """
    if pretrained:
        print("\nLoading pretrained EchoSDNet classifier from Hugging Face...")
        model_path = download_model("EchoSDNet")
        # Load compiled to preserve training config if saved; fallback compile afterwards if needed.
        try:
            model = tf.keras.models.load_model(model_path)
            return model
        except Exception:
            model = tf.keras.models.load_model(model_path, compile=False)
            _compile_echosdnet(model, num_classes, learning_rate)
            return model

    print("\nBuilding EchoSDNet architecture from scratch...")

    # --- Input & basic scaling (external resizing recommended) ---
    inputs = tf.keras.Input(shape=input_shape, name="input")
    x = tf.keras.layers.Rescaling(1.0 / 255.0, name="rescale_0_1")(inputs)

    # --- ResNet50 backbone (ImageNet), exclude top ---
    backbone = tf.keras.applications.ResNet50(
        include_top=False, weights="imagenet", input_tensor=x
    )

    # Freeze all layers first
    for layer in backbone.layers:
        layer.trainable = False

    # Optionally unfreeze the final stage (conv5) for domain adaptation
    if fine_tune_block == "conv5":
        for layer in backbone.layers:
            name = layer.name
            # Keep BatchNorm frozen (best practice during fine-tuning)
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
                continue
            if "conv5_block" in name:
                layer.trainable = True

    # --- Classification head as per your report ---
    y = tf.keras.layers.GlobalAveragePooling2D(name="gap")(backbone.output)
    y = tf.keras.layers.Dense(128, activation="relu", name="fc_128")(y)
    y = tf.keras.layers.Dense(64, activation="relu", name="fc_64")(y)
    y = tf.keras.layers.Dropout(dropout_rate, name="dropout")(y)

    if num_classes <= 1:
        # Rare, but allow pure binary scalar with sigmoid
        outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="pred")(y)
    else:
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="pred")(y)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="EchoSDNet")

    # --- Compile with appropriate loss/metrics ---
    _compile_echosdnet(model, num_classes, learning_rate)
    return model


def _compile_echosdnet(model: tf.keras.Model, num_classes: int, learning_rate: float):
    """Compile helper to set loss/metrics consistently."""
    if num_classes <= 1:
        loss = "binary_crossentropy"
        metrics = [
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ]
    elif num_classes == 2:
        # Softmax(2) + categorical CE keeps compatibility with multi-class code paths
        loss = "categorical_crossentropy"
        metrics = [
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ]
    else:
        loss = "categorical_crossentropy"
        metrics = ["accuracy"]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=metrics,
    )