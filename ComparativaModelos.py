import os
import numpy as np
from xml.dom import minidom
from tensorflow.keras.preprocessing import image

# Libraries from Keras to import models.
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_ResNet50
from tensorflow.keras.applications.resnet50 import decode_predictions as decode_predictions_ResNet50

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_input_VGG19
from tensorflow.keras.applications.vgg19 import decode_predictions as decode_predictions_VGG19

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_input_InceptionV3
from tensorflow.keras.applications.inception_v3 import decode_predictions as decode_predictions_InceptionV3

# Global variables
N_IMAGE_TO_USE = 30000
BATCHES = 1000
N_TOP = [1, 2, 5]

pathImages = "/content/drive/MyDrive/TFM/ImageNet/ValidationData/Imagenes/"
pathLabels = "/content/drive/MyDrive/TFM/ImageNet/ValidationData/labelsFixed.xml"
expectedLabels = 0


"""
    List of models to create.
"""
models = [
    {
        "name": "ResNet50",
        "model": 0,
        "constructor": ResNet50,
        "preproc": preprocess_input_ResNet50,
        "decoder": decode_predictions_ResNet50,
        "width": 224,
        "height": 224,
        "correct": [],
    },
    {
        "name": "VGG19",
        "model": 0,
        "constructor": VGG19,
        "preproc": preprocess_input_VGG19,
        "decoder": decode_predictions_VGG19,
        "width": 224,
        "height": 224,
        "correct": [],
    },
    {
        "name": "InceptionV3",
        "model": 0,
        "constructor": InceptionV3,
        "preproc": preprocess_input_InceptionV3,
        "decoder": decode_predictions_InceptionV3,
        "width": 299,
        "height": 299,
        "correct": [],
    }
]


"""
  This functions load the images from a list and rescale them.
"""
def loadImages(files, width, height):
    global pathImages
    images = []
    for fileName in files:
        img = image.load_img(pathImages + fileName, target_size=(width, height))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        images.append(img)

    images = np.vstack(images)

    return images


"""
    This function creates a list with the expected labels to use
"""
def extractLabels():
    global pathLabels
    data = []
    annotations = minidom.parse(pathLabels).getElementsByTagName("annotation")

    for annotation in annotations:
        data.append(
            {
                "filename": str(
                    annotation.getElementsByTagName("filename")[0].firstChild.data
                )
                + ".JPEG",
                "label": annotation.getElementsByTagName("name")[0].firstChild.data,
            }
        )

    return data


def findLabel(fileName):
    global expectedLabels
    return list(filter(lambda label: label["filename"] == fileName, expectedLabels))[0][
        "label"
    ]


"""
  This functions compare the predicted labels with the expected ones, returning the number of right predictions.
"""
def compareLabels(predicted, filenames):
    correct = 0.0
    for index, predict in enumerate(predicted):
        labels = [item[0] for item in predict]
        if findLabel(filenames[index]) in labels:
            correct += 1

    return correct


"""
  This functions split the list of filenames of the images to reduce RAM usage
"""
def getCurrentImages(images, index):
    startIndex = index * BATCHES
    endIndex = (index + 1) * BATCHES
    return images[startIndex:endIndex]


"""
  This functions assign the model variables to initialize the "imagenet" weights
"""
def createModels():
    global models
    for model in models:
        model["model"] = model["constructor"](weights="imagenet")


def printAccuracy(model, ratio, spacer):
    print(f"{spacer}- {model['name']}: ", end="\t")
    for index, top in enumerate(N_TOP):
        print(f"TOP {top}: {(model['correct'][index] * ratio):.2f}%", end="\t")
    print("")


"""
  This functions load the images in batches and predict the labels for all the models
"""
def runPredictions():
    global models, expectedLabels, N_IMAGE_TO_USE, BATCHES, N_TOP
    filesImages = sorted(os.listdir(pathImages))[0:N_IMAGE_TO_USE]

    # Reset counts before prediction
    for model in models:
        model["correct"] = []
        for top in N_TOP:
            model["correct"].append(0.0)

    for index in range(0, int(N_IMAGE_TO_USE / BATCHES)):
        print(f"Iteration {index} of {int(N_IMAGE_TO_USE/BATCHES)}")

        # Ratio to print partial results
        ratio = 100.0 / ((index + 1) * BATCHES)

        # Split fileNames of images
        currentImages = getCurrentImages(filesImages, index)

        print("   Partial Results:")
        for model in models:
            # Load Images
            images = loadImages(currentImages, model["width"], model["height"])

            # Preprocess Images
            images = model["preproc"](images)

            # Predict and compare result with expected labels
            preds = model["model"].predict(images)

            for index, _top in enumerate(N_TOP):
                model["correct"][index] += compareLabels(
                    model["decoder"](preds, top=_top), currentImages)

            printAccuracy(model, ratio, spacer="     ")

    ratio = 100.0 / N_IMAGE_TO_USE
    print(f"\nFinal Results:")
    for model in models:
        printAccuracy(model, ratio, spacer="  ")


if __name__ == "__main__":
    # Create all models needed using the models dict
    createModels()

    # Obtain list of expected labels
    expectedLabels = extractLabels()

    # Calculate predictions and print the results
    runPredictions()
