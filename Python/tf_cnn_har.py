import tensorflow as tf

import numpy as np

x_train = np.array([[]])
x_test = np.array([[]])
y_train = np.array([])
y_test = np.array([])

# Normalize value to [0, 1]
x_train = x_train.reshape(16, 3*8, 1)  # 3*8*16=3*128
x_test = x_test.reshape(16, 3*8, 1)  # 3*8*16=3*128

# Transform labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# A Porter avec le CNN de demo_tf_cnn_mnist
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(8, (3, 3), padding='valid', strides=(2, 2), input_shape=(16, 24, 1), name='ftr0'))
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

hist = model.fit(x=x_train, y=y_train, epochs=10, batch_size=32, verbose=1, validation_data=(x_test, y_test), shuffle=True)

