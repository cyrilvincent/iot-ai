import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, ReLU, Conv2D, DepthwiseConv2D, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras import Model

def convolution_block(input_layer, strides, filters):
    x = DepthwiseConv2D(kernel_size=3, strides=strides, padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=filters, kernel_size=1, strides=1)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

def create_mobilenet(input_shape=(224, 224, 3), num_classes=2):
    input_img = Input(shape=input_shape)
    x = Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), padding='same')(input_img)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = convolution_block(x, filters=64, strides=1)
    x = convolution_block(x, filters=128, strides=2)
    x = convolution_block(x, filters=128, strides=1)
    x = convolution_block(x, filters=256, strides=2)
    x = convolution_block(x, filters=256, strides=1)
    x = convolution_block(x, filters=512, strides=2)
    for _ in range(5):
        x = convolution_block(x, filters=512, strides=1)
    x = convolution_block(x, filters=1024, strides=2)
    x = convolution_block(x, filters=1024, strides=1)
    x = GlobalAveragePooling2D()(x)
    out = Dense(units=num_classes, activation='softmax')(x)
    return Model(inputs=input_img, outputs=out)
