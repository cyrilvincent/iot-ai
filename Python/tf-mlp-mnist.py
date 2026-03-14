import tensorflow as tf

import numpy as np

with np.load("data/mnist/mnist.npz", allow_pickle=True) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']

# Set numeric type to float32 from uint8
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

x_train /= 255
x_test /= 255

x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)

y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# sample = np.random.randint(60000, size=5000)
# x_train = x_train[sample]
# y_train = y_train[sample]

model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation="relu", input_shape=(x_train.shape[1],)),
    tf.keras.layers.Dense(10, activation="softmax"),
  ])
model.compile(loss="categorical_crossentropy", metrics=['accuracy'])
trained = model.fit(x_train, y_train, epochs=5, batch_size=10, validation_data=(x_test, y_test))
print(model.summary())

model.save("data/mnist/mnist_mlp.h5")

predicted = model.predict(x_test)
print(y_test[:10], predicted[:10], np.argmax(predicted[:10], axis=1))

import matplotlib.pyplot as plt
predicted = predicted.argmax(axis=1)
misclass = (y_test.argmax(axis=1) != predicted)
x_test = x_test.reshape((-1, 28, 28))
misclass_images = x_test[misclass,:,:]
misclass_predicted = predicted[misclass]

select = np.random.randint(misclass_images.shape[0], size=12)

for index, value in enumerate(select):
    plt.subplot(3,4,index+1)
    plt.axis('off')
    plt.imshow(misclass_images[value],cmap=plt.cm.gray_r,interpolation="nearest")
    plt.title('Predicted: %i' % misclass_predicted[value])

plt.show()

# cd ../TinyMaix/tools
# python h5_to_tflite.py ../../python/data/mnist/mnist_mlp.h5 ../../python/data/mnist/mnist_mlp_fp32.tflite 0
# python tflite2tmdl.py ../../python/data/mnist/mnist_mlp_fp32.tflite ../../python/data/mnist/mnist_mlp_fp32.tmdl fp32 1 784 10
# 328Ko KO

# python h5_to_tflite.py ../../python/data/mnist/mnist_mlp.h5 ../../python/data/mnist/mnist_mlp_int8.tflite 1 quant_img_mnist/ 0to1
# python tflite2tmdl.py ../../python/data/mnist/mnist_mlp_int8.tflite ../../python/data/mnist/mnist_mlp_int8.tmdl int8 0 784 10
# 80Ko OK