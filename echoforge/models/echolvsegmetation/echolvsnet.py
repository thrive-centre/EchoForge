import os 
import tensorflow as tf
from echoforge.utils.loader import download_model

def conv_block(x, filters):
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    return x

def encoder_block(x, filters):
    f = conv_block(x, filters)
    p = tf.keras.layers.MaxPooling2D((2, 2))(f)
    return f, p

def decoder_block(x, skip, filters):
    x = tf.keras.layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.Concatenate()([x, skip])
    
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    return x

def build_echolvsnet(input_shape=(512, 512, 1), pretrained=True):
    """
    Builds or loads the EchoLVSNet U-Net architecture.

    Args:
        input_shape (tuple): Input image shape.
        pretrained (bool): Whether to load pretrained weights from Hugging Face.

    Returns:
        tf.keras.Model: A U-Net model for left ventricular segmentation.
    """
    if pretrained:
        print("Loading pretrained EchoLVSNet from Hugging Face...")
        model_path = download_model("EchoLVSNet")
        return tf.keras.models.load_model(model_path)

    print("Building EchoLVSNet architecture from scratch...")

    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Rescaling(1./255)(inputs)

    # Encoder
    f1, p1 = encoder_block(x, 32)
    f2, p2 = encoder_block(p1, 64)
    f3, p3 = encoder_block(p2, 128)
    f4, p4 = encoder_block(p3, 256)
    f5, p5 = encoder_block(p4, 512)

    # Bridge
    b = conv_block(p5, 1024)

    # Decoder
    d1 = decoder_block(b, f5, 512)
    d2 = decoder_block(d1, f4, 256)
    d3 = decoder_block(d2, f3, 128)
    d4 = decoder_block(d3, f2, 64)
    d5 = decoder_block(d4, f1, 32)

    # Output layer
    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', name='output')(d5)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="EchoLVSNet")
    return model
