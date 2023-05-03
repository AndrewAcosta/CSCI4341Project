from tkinter import *
import pathlib
from tkinter import filedialog
import numpy as np
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk
import tensorflow as tf
import keras
import os

# declare the window
win = Tk()  # Screen - Dice Roller
# set win title
win.title("Check my Lawn!")
# set window width and height
win.geometry("500x500")

# Explains purpose of button
l1 = Label(win, text='Choose image then predict', width=30)
l1.grid(row=1, column=1, columnspan=4)

# The upload button
# uploadButton = Button(win, text='Upload File', width=20, command=lambda: upload_file())
# uploadButton.grid(row=3, column=1)

# Write the path - Option 2
# path = Entry(win)
# path.grid(row=5, column=2)

# Choose a png - Option 1
getFilepathButton = Button(win, text='Choose image', width=20, command=lambda: open_file())
getFilepathButton.grid(row=2, column=1)
# Holds path to png
label = Label(win, text="")
label.grid(row=3, column=1)
# The Predict button
predictButton = Button(win, text='Do I need to Mow?', width=20, command=lambda: predict())
predictButton.grid(row=4, column=1)
# Shows the result
result = Label(win, text="Result is: (Choose image and click predict)")
result.grid(row=5, column=1)


def open_file():
    file = filedialog.askopenfile(mode='r', filetypes=[('image files', ('.png', '.jpg'))])
    if file:
        filepath = os.path.abspath(file.name)
        label.configure(text=f"{str(filepath)}")


def predict():
    # Data location
    data_dir = 'Lawns/'
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
    class_names = train_ds.class_names  # Types of lawns
    model = tf.keras.models.load_model('savedModels/my_model')  # Loads the saved model
    lawn_path = label.cget("text")  # ***** Will probably need to change this *****
    picture_height, picture_width = 256, 256
    img = keras.utils.load_img(
        lawn_path, target_size=(picture_height, picture_width)  # *** Will probably need to change this... maybe ***
    )
    img_array = keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)  # We need to load the model first... But first we need to create it
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )
    resultText = "This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score))
    result.configure(text=f"{resultText}")


win.mainloop()  # Keep the window open
