import tensorflow as tf

import numpy as np

with np.load("data/mnist/mnist.npz", allow_pickle=True) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

# Normalize value to [0, 1]
x_train /= 255
x_test /= 255

# Transform labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Reshape the dataset into 4D array
x_train = x_train.reshape(x_train.shape[0], 28,28,1)
x_test = x_test.reshape(x_test.shape[0], 28,28,1)

# A Porter avec le CNN de demo_tf_cnn_mnist
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(8, (3, 3), padding='valid', strides=(2, 2), input_shape=(28, 28, 1), name='ftr0'))
model.add(tf.keras.layers.BatchNormalization(name="bn0"))
model.add(tf.keras.layers.Activation('relu', name="relu0"))

model.add(tf.keras.layers.Conv2D(12, (3, 3), padding='valid', strides=(2, 2), name='ftr1'))
model.add(tf.keras.layers.BatchNormalization(name="bn1"));
model.add(tf.keras.layers.Activation('relu', name="relu1"));

model.add(tf.keras.layers.Conv2D(18, (3,3), padding='valid', strides=(2, 2), name='ftr2'))
model.add(tf.keras.layers.BatchNormalization());
model.add(tf.keras.layers.Activation('relu'));

model.add(tf.keras.layers.GlobalAveragePooling2D(name='GAP'))
model.add(tf.keras.layers.Dense(10, name="fc1"))
model.add(tf.keras.layers.Activation('softmax', name="sm"))

model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["categorical_accuracy"])
model.summary()

hist = model.fit(x=x_train, y=y_train, epochs=10, batch_size=128, verbose=1, validation_data=(x_test, y_test), shuffle=True)

model.save('data/mnist/mnist_cnn.h5') # 97.5%

# python h5_to_tflite.py ../../python/data/mnist/mnist_cnn.h5 ../../python/data/mnist/mnist_cnn_fp32.tflite 0
# python tflite2tmdl.py ../../python/data/mnist/mnist_cnn_fp32.tflite ../../python/data/mnist/mnist_cnn_fp32.tmdl fp32 0 28,28,1 1
# python h5_to_tflite.py ../../python/data/mnist/mnist_cnn.h5 ../../python/data/mnist/mnist_cnn_int8.tflite 1 quant_img_mnist/ 0to1
# python tflite2tmdl.py ../../python/data/mnist/mnist_cnn_int8.tflite ../../python/data/mnist/mnist_cnn_int8.tmdl int8 1 28,28,1 1

