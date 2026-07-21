import sklearn.preprocessing as pp
import sklearn.model_selection as ms
import tensorflow as tf
import pandas as pd
import numpy as np

tf.random.set_seed(42)
np.random.seed(42)

df = pd.read_csv('data/heart/data_cleaned_up.csv')
x = df.drop('num', axis=1)
y = df['num'].values

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)

scaler = pp.StandardScaler()
scaler.fit(x)
xtrain = scaler.transform(xtrain)
xtest = scaler.transform(xtest)

# ytrain = tf.keras.utils.to_categorical(ytrain)
# ytest = tf.keras.utils.to_categorical(ytest)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(20, activation="relu", input_shape=(x.shape[1],)),
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(1, activation="relu")  # sigmoid not supported by tinymaix
  ])

model.compile(loss="mse", optimizer="rmsprop", metrics=['accuracy'])
model.summary()

history = model.fit(xtrain, ytrain, epochs=10, batch_size=5, validation_data=(xtest, ytest))
eval = model.evaluate(xtrain, ytrain)
print(eval)
print(f"Total accuracy: {history.history['val_accuracy'][-1]*100:.1f}%")

model.save("data/heart/heart_mlp.h5")


