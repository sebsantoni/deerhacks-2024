import pandas as pd
from pathlib import Path 
import os.path 

from sklearn.model_selection import train_test_split 

import tensorflow as tf

filePath = './fruit_data/Training' # the path of the files to train on
sample_size = 50 # the number of images to sample from each class
num_epochs = 100 # the number of epochs for which to train the model
num_classes = 60 # the number of classes/categories from which to predict
testDir = './Apple_Photos' # directory from which user data is pulled for the model to use

# data frame setup
image_dir = Path(filePath)

# returns the class name (target) for each file path
filepaths = list(image_dir.glob(r'**/*.jpg')) # will need to change based on the actual path
labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

filepaths = pd.Series(filepaths, name='Filepath').astype(str)
labels = pd.Series(labels, name='Label')

images = pd.concat([filepaths, labels], axis=1)

# want to sample *sample_size* images from each category for performance

category_samples = []
for category in images['Label'].unique():
    category_slice = images.query("Label == @category")
    category_samples.append(category_slice.sample(sample_size, random_state=1)) # can remove seed later

image_df = pd.concat(category_samples, axis=0).sample(frac=1.0, random_state=1).reset_index(drop=True) # can remove seed later

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

