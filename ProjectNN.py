import pathlib
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
import matplotlib.pyplot as plt
import os

# Allow our system to store the data properly
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Need this to run correctly. Otherwise, it will have overflow problems.
# Data location
data_dir = 'C:\Users\Andre\OneDrive\Pictures\lawns'
# Setting up the paths for data or something... not sure.
data_dir = pathlib.Path(data_dir)
# Training data
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(256, 256),
    batch_size=50)
# Validation Data
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(256, 256),
    batch_size=64)

# Print the different folder names that contain data
class_names = train_ds.class_names
print(class_names)

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# The NN Note: Maybe having .02 dropout is still too much. May try .01 or .02... I could be wrong... So, yeah. This
# is troublesome
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1. / 255),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.15),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.15),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.15),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.15),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.15),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.15),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.15),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(2)  # Number of classes
])
# callbacks = [EarlyStopping(monitor='val_accuracy', patience=3)]
# Start the training
model.compile(optimizer='adam', loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# Store the results
history = model.fit(train_ds, validation_data=val_ds, epochs=125)

# Save the model
model.save('saved_model/my_model')

# The various types of results
y_vloss = history.history['val_loss']
y_loss = history.history['loss']
y_acc = history.history['accuracy']
y_vacc = history.history['val_accuracy']
# Plotting for loss
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(np.arange(len(y_vloss)), y_vloss, marker='.', c='red')
ax1.plot(np.arange(len(y_loss)), y_loss, marker='.', c='blue')
ax1.grid()
plt.setp(ax1, xlabel='epoch', ylabel='loss')
# Plotting for accuracy
ax2.plot(np.arange(len(y_vacc)), y_vacc, marker='.', c='red')
ax2.plot(np.arange(len(y_acc)), y_acc, marker='.', c='blue')
ax2.grid()
plt.setp(ax2, xlabel='epoch', ylabel='accuracy')
# Show Plot
plt.show()
