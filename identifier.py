import numpy as np 
import pandas as pd 
from pathlib import Path 
import os.path 

import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn.model_selection import train_test_split 

import tensorflow as tf

from sklearn.metrics import confusion_matrix, classification_report

# variables
filePath # the path of the files to train on
sample_size = 100 # the number of images to sample from each class
num_epochs = 100 # the number of epochs for which to train the model
num_classes = 101 # the number of classes/categories from which to predict

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
sns.heatmap(cm, annot=True, fmt='g', cmin=0, cmap='Blues', cbar=False)
plt.xticks(ticks=np.arange(num_classes) + 0.5, labels=test_images.class_indices, rotation=90) # number of classes
plt.yticks(ticks=np.arange(num_classes) + 0.5, labels=test_images.class_indices, rotation=0) # number of classes
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
print("Classification Report:\n------------------\n", clr)

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from PIL import Image
import numpy as np

def load_and_preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224)) 
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Example usage:
image_path = '/kaggle/input/food41/images/churros/1078896.jpg'
new_image = load_and_preprocess_image(image_path)

# Make prediction
predictions = model.predict(new_image)

# Decode predictions to get the class label
predicted_class = np.argmax(predictions)

class_label = list(train_images.class_indices.keys())[predicted_class]
print("Predicted Class:", class_label)