import numpy as np
import os.path 
import csv
import statistics

import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from PIL import Image

from globals import train_images, test_images, num_classes, testDir


def ecoScoreCalc(csv_file_path, search_string):
    with open(csv_file_path, 'r', newline='') as file:
        reader = csv.reader(file)

        wasFound = False

        # assuming the first row contains headers, skip it if needed
        # headers = next(reader, None)

        for row_num, row in enumerate(reader, start=1):
            if search_string.lower() == row[0].lower():
                print(f"String '{search_string}' found.")
                wasFound = True
                break

        if not (wasFound):
            print(search_string, "was not found.")
            return wasFound, -1, -1, -1  # if it was found, water score, carbon score, average score

        # second part - find footprints
        waterValue = float(row[1])
        # print("The water footprint of a(n)", search_string, "is:", waterValue, "liters/kg")

        carbonValue = float(row[2])
        # print("The carbon footprint of a(n)", search_string, "is:", carbonValue, "kg/kg")

        # third part - find how eco friendly it is
        target_value = waterValue

        # extract the specified column
        column_values = [float(row[1]) for row in reader if float(row[1]) != 0]

        if (target_value == 0):
            print("Water footprint data not available")
            waterValue = -1
        else:
            mean_value = statistics.mean(column_values)

            # calculate and print the percentage difference for the target value
            waterValue = round((((waterValue - mean_value) / 1.8109406130140314) + 919) / 300)
            if (waterValue > 5):
                waterValue = 5
            waterValue = 5 - waterValue
            print("The water eco-score for a(n)", search_string, "is", waterValue, "out of 5")

        file.seek(0)  # reset the file pointer to the beginning
        target_value = float(carbonValue)

        if (target_value == 0):
            print("Carbon data not available")
            carbonValue = -1
        else:
            # extract the specified column
            column_values = [float(row[2]) for row in reader if float(row[2]) != 0]

            mean_value = statistics.mean(column_values)

            ## Calculate and print the percentage difference for each value
            # for index, value in enumerate(column_values, start=1):
            #    percentage_diff = calculate_carbon_score(value, mean_value, std_dev_value)
            #    if(percentage_diff > 5):
            #        percentage_diff = 5
            #    percentage_diff = 5 - percentage_diff
            #    print(f"Percentage Difference from Mean = {round(percentage_diff)}")

            # Calculate and print the percentage difference for the target value
            carbonValue = round((carbonValue - mean_value + 0.47) * 14.652014652)
            if (carbonValue > 5):
                carbonValue = 5
                carbonValue = 5 - carbonValue
            print("The carbon eco-score for a(n)", search_string, "is", carbonValue, "out of 5")

        if (waterValue == -1 or carbonValue == -1):
            averageValue = -1
        else:
            averageValue = round((waterValue + carbonValue) / 2)
            print("The overall eco-score for a(n)", search_string, "is", averageValue, "out of 5")
  
def load_and_preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# image prediction function
def make_prediction(image_path, model, train_images):
    new_image = load_and_preprocess_image(image_path)

    # Make prediction
    predictions = model.predict(new_image)

    # Decode predictions to get the class label
    predicted_class = np.argmax(predictions)

    class_label = list(train_images.class_indices.keys())[predicted_class]
    return class_label

def evaluate_model(model, test_images):
    # results
    results = model.evaluate(test_images, verbose=0)
    print("Test Accuracy: {:.2f}%".format(results[1] * 100))

    # predictions
    predictions = np.argmax(model.predict(test_images, model, train_images), axis=1)
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

    print("making predictions...")
    # image_path_apple = './fruit_data/Test/Apple/228_100.jpg'
    # image_path_orange = './fruit_data/Test/Orange/33_100.jpg'
    # print("apple?: ", make_prediction(image_path_apple))
    # print("orange?: ", make_prediction(image_path_orange))

# Function to check if the image file is valid
def is_valid_image(file_path):
    try:
        with Image.open(file_path) as img:
            # Attempting to load the image ensures it's a valid image file
            return True
    except Exception as e:
        # If an exception occurs, the file is not a valid image
        # print(f"Invalid image file: {e}")
        return False


if __name__ == '__main__':

    # f = sys.argv[0] # path of image to predict is passed in as an argument

    model = tf.keras.models.load_model('./model')

#     file = open('./user_data/predictions.txt', 'a')
#     file.write(make_prediction(f, model, train_images))
#     file.close()

#    if is_valid_image(f):
#         print(make_prediction(f, model, train_images))


    csv_file_path =  "./fruit_carbon_and_water_footprint_data.csv"
    
    for filename in os.listdir(testDir):
        f = os.path.join(testDir, filename)
        # checking if it is a file
        if is_valid_image(f):
            prediction = make_prediction(f, model, train_images)
            ecoScoreCalc(csv_file_path, prediction)

    # evaluate_model(model, test_images)
