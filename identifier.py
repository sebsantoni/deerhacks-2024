import numpy as np 
import pandas as pd 
from pathlib import Path 
import os.path 

import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn.model_selection import train_test_split 

import tensorflow as tf

from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from PIL import Image
import io


# Function to check if the image file is valid
def is_valid_image(file_path):
    try:
        with Image.open(file_path) as img:
            # Attempting to load the image ensures it's a valid image file
            return True
    except Exception as e:
        # If an exception occurs, the file is not a valid image
        print(f"Invalid image file: {e}")
        return False


if __name__ == '__main__':
    # variables
    filePath = './fruit_data/Training' # the path of the files to train on
    sample_size = 50 # the number of images to sample from each class
    num_epochs = 3 # the number of epochs for which to train the model
    num_classes = 60 # the number of classes/categories from which to predict

    # data frame setup

    image_dir = Path(filePath)

    # returns the class name (target) for each file path
    filepaths = list(image_dir.glob(r'**/*.jpg')) # will need to change based on the actual path
    labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

    filepaths = pd.Series(filepaths, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')

    images = pd.concat([filepaths, labels], axis=1)

    # want to sample sample_size=100 images from each category for performance

    category_samples = []
    for category in images['Label'].unique():
        category_slice = images.query("Label == @category")
        category_samples.append(category_slice.sample(sample_size, random_state=1)) # can remove seed later

    image_df =  pd.concat(category_samples, axis=0).sample(frac=1.0, random_state=1).reset_index(drop=True) # can remove seed later



    # train-test split
    train_df, test_df =  train_test_split(image_df, train_size=0.7, shuffle=True, random_state = 1) # can remove seed later

    # create generators  using mobilenet_v2 preprocessing function
    train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
        validation_split=0.2
    )

    test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    )

    train_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col = 'Label',
        target_size=(224, 224), #img dimensions
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        seed=42, # can change later
        subset='training'
    )

    val_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col = 'Label',
        target_size=(224, 224), #img dimensions
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        seed=42, # can change later
        subset='validation'
    )

    test_images = test_generator.flow_from_dataframe(
        dataframe=test_df,
        x_col='Filepath',
        y_col = 'Label',
        target_size=(224, 224), #img dimensions
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=False,
    )

    # modeling
    pretrained_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg',
    )

    pretrained_model.trainable = False

    inputs = pretrained_model.input
    # classification layers
    x = tf.keras.layers.Dense(128, activation='relu')(pretrained_model.output)
    x = tf.keras.layers.Dense(128, activation='relu')(x)

    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x) # number of classes

    model = tf.keras.Model(inputs, outputs)

    print(model.summary())

    # training
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        train_images,
        validation_data=val_images,
        epochs=num_epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            )
        ]
    )

    # results
    results = model.evaluate(test_images, verbose=0)
    print("Test Accuracy: {:.2f}%".format(results[1] * 100))

    # predictions
    predictions = np.argmax(model.predict(test_images), axis=1)
    cm = confusion_matrix(test_images.labels, predictions)
    clr = classification_report(test_images.labels, predictions)

    # figures

    plt.figure(figsize=(30,30))
    sns.heatmap(cm, annot=True, fmt='g', vmin=0, cmap='Blues', cbar=False)
    plt.xticks(ticks=np.arange(num_classes) + 0.5, labels=test_images.class_indices, rotation=90) # number of classes
    plt.yticks(ticks=np.arange(num_classes) + 0.5, labels=test_images.class_indices, rotation=0) # number of classes
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
    print("Classification Report:\n------------------\n", clr)


    def load_and_preprocess_image(image_path):
        img = Image.open(image_path)
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array

    # image prediction function
    def make_prediction(image_path):
        new_image = load_and_preprocess_image(image_path)

        # Make prediction
        predictions = model.predict(new_image)

        # Decode predictions to get the class label
        predicted_class = np.argmax(predictions)

        class_label = list(train_images.class_indices.keys())[predicted_class]
        return class_label

    print("making predictions...")
    # image_path_apple = './fruit_data/Test/Apple/228_100.jpg'
    # image_path_orange = './fruit_data/Test/Orange/33_100.jpg'
    # print("apple?: ", make_prediction(image_path_apple))
    # print("orange?: ", make_prediction(image_path_orange))

    directory = './user_data'
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if is_valid_image(f):
            print(make_prediction(f))

    
