#Importing the required libraries
from pyimagesearch import config
from imutils import paths
import os
import random
import shutil

def build_dataset():
    imagePaths = list(paths.list_images(config.ORIG_INPUT_DATASET))
    random.seed(42)
    random.shuffle(imagePaths)

    i = int(len(imagePaths) * config.TRAIN_SPLIT)
    trainPaths = imagePaths[:i]
    testPaths = imagePaths[i:]

    i = int(len(trainPaths) * config.VAL_SPLIT)
    valPaths = trainPaths[:i]
    trainPaths = trainPaths[i:]

    datasets = [
        ("training", trainPaths, config.TRAIN_PATH),
        ("validation", valPaths, config.VAL_PATH),
        ("testing", testPaths, config.TEST_PATH)
    ]

    for (dtype, imagePaths, baseOutput) in datasets:
        print("Creating {} split".format(dtype))

        #checking for existing directory and if not then creating one
        if not os.path.exists(baseOutput):
            print("Creating {} directory".format(baseOutput))
            os.makedirs(baseOutput)

        for inputPath in imagePaths:
            filename = inputPath.split(os.path.sep)[-1]
            label = filename[-5:-4]

            labelPath = os.path.sep.join([baseOutput, label])
            if not os.path.exists(labelPath):
                print("Creating {} directory".format(labelPath))
                os.makedirs(labelPath)

            p = os.path.sep.join([labelPath, filename])
            shutil.copy2(inputPath, p)

if __name__ == '__main__':
    build_dataset()