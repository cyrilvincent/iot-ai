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
model.add(tf.keras.layers.Conv2D(16, (3, 3), strides=(2, 2), input_shape=(28, 28, 1), padding="same")) # 28,28,16
model.add(tf.keras.layers.Activation('relu'))
# model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2))) # 14,14,16 remplacé par stride 2

model.add(tf.keras.layers.Conv2D(16, (3, 3), strides=(2, 2)))  # 10,10,16
model.add(tf.keras.layers.Activation('relu'))
# model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2))) # 5,5,16

model.add(tf.keras.layers.Conv2D(16, (3, 3), strides=(2, 2)))  # 4,4,16
model.add(tf.keras.layers.Activation('relu'))
# model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2))) # 2,2,16

#Dense
model.add(tf.keras.layers.Flatten()) # 64
model.add(tf.keras.layers.Dense(32))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Dense(10))
model.add(tf.keras.layers.Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
model.summary()

hist = model.fit(x=x_train,y=y_train, epochs=5, batch_size=16, validation_data=(x_test, y_test))

model.save('data/mnist/mnist_cnn_mlp.h5') # 97.5%

# Quantisation obligatoire
# python h5_to_tflite.py ../../python/data/mnist/mnist_cnn_mlp.h5 ../../python/data/mnist/mnist_cnn_mlp_int8.tflite 1 quant_img_mnist/ 0to1
# python tflite2tmdl.py ../../python/data/mnist/mnist_cnn_mlp_int8.tflite ../../python/data/mnist/mnist_cnn_mlp_int8.tmdl int8 1 28,28,1 1