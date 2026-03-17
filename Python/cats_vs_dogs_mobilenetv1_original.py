import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense, ReLU, Conv2D, DepthwiseConv2D, GlobalAveragePooling2D, BatchNormalization


def convolution_block(input_layer, strides, filters):
    x = DepthwiseConv2D(kernel_size=3, strides=strides, padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=filters, kernel_size=1, strides=1)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


input_img = keras.layers.Input(shape=(96, 96, 3))
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
x = convolution_block(x, filters=1024, strides=1) # 3.2Mo en int8
x = GlobalAveragePooling2D()(x)
out = Dense(units=2, activation='softmax')(x)

model = keras.models.Model(inputs=input_img, outputs=out)

print(model.output_shape)   # doit être (None, 2)


model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

trainset = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255, validation_split=0.2) # shear_range=0.2, zoom_range=0.2,


batchSize = 32

trainGenerator = trainset.flow_from_directory(
        r'C:\cats_vs_dogs',
        target_size=(224, 224),
        subset='training',
        batch_size=batchSize)

validationGenerator = trainset.flow_from_directory(
        r'C:\cats_vs_dogs',
        target_size=(224, 224),
        subset='validation',
        batch_size=batchSize)

checkpointer = keras.callbacks.ModelCheckpoint(filepath='data/mobilenetv1/ckpt-{epoch:03d}-{accuracy:.3f}.h5', monitor='accuracy')

model.fit(
        trainGenerator,
        epochs=20,
        batch_size=batchSize,
        validation_data=validationGenerator,
        callbacks=[checkpointer]
)

model.save('data/mobilenetv1/cats_vs_dogs.h5')

# python h5_to_tflite.py ../../python/data/mobilenetv1/ckpt-001-0.652.h5 ../../python/data/mobilenetv1/mobilenet_int8.tflite 1 quant_img96/ 0to1
# python tflite2tmdl.py ../../python/data/mobilenetv1/mobilenet_int8.tflite ../../python/data/mobilenetv1/mobilenet_int8.tmdl int8 1 96,96,3 1


