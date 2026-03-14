import sklearn.preprocessing as pp
import sklearn.model_selection as ms
import tensorflow as tf
import pandas
import numpy as np

tf.random.set_seed(42)
np.random.seed(42)

dataframe = pandas.read_csv("data/breast-cancer/data.csv", index_col="id")
y = dataframe.diagnosis
x = dataframe.drop("diagnosis", axis=1)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)

scaler = pp.MinMaxScaler()
scaler.fit(x)
xtrain = scaler.transform(xtrain)
xtest = scaler.transform(xtest)

np.savetxt('data/breast-cancer/scaler_cancer_min.csv',  scaler.data_min_,  delimiter=',')
np.savetxt('data/breast-cancer/scaler_cancer_max.csv', scaler.data_max_, delimiter=',')


ytrain = tf.keras.utils.to_categorical(ytrain, 2)
ytest = tf.keras.utils.to_categorical(ytest, 2)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(20, activation="relu", input_shape=(x.shape[1],)),
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(2, activation="softmax")  # sigmoid not supported by tinymaix
  ])

model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=['accuracy'])
model.summary()

history = model.fit(xtrain, ytrain, epochs=10, batch_size=5, validation_data=(xtest, ytest))
eval = model.evaluate(xtrain, ytrain)
print(eval)
print(f"Total accuracy: {history.history['val_accuracy'][-1]*100:.1f}%")

model.save("data/breast-cancer/cancer_mlp.h5")

# cd ../TinyMaix/tools
# python h5_to_tflite.py ../../python/data/breast-cancer/cancer_mlp.h5 ../../python/data/breast-cancer/cancer_mlp.tflite 0
# python tflite2tmdl.py ../../python/data/breast-cancer/cancer_mlp.tflite ../../python/data/breast-cancer/cancer_mlp.tmdl fp32 1 30 2
# 3.9Ko


