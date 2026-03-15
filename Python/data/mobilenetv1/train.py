import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from model import create_mobilenet
import tensorflow_datasets as tfds
from utils import preprocess

# Load the Cats vs. Dogs dataset
(train_ds, test_ds), ds_info = tfds.load('cats_vs_dogs', split=['train[:80%]', 'train[80%:]'], with_info=True, as_supervised=True,
                                         data_dir="C:/tensorflow_datasets")

# Apply preprocessing
train_ds = train_ds.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)
test_ds = tf.keras.utils.to_categorical(test_ds, 2)

# Create the model
mobilenet_tf = create_mobilenet(input_shape=(224, 224, 3), num_classes=2)
mobilenet_tf.summary()

# Compile the model
mobilenet_tf.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# Set up early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model with early stopping
history = mobilenet_tf.fit(train_ds, epochs=20, validation_data=test_ds, callbacks=[early_stopping])

# Save the trained model
mobilenet_tf.save('mobilenet_model.h5')

# Evaluate the model
test_loss, test_acc = mobilenet_tf.evaluate(test_ds)
print(f'Test accuracy: {test_acc}')
