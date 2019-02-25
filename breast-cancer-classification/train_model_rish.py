# Importing the required packages
import os
import argparse
import matplotlib
matplotlib.use("Agg")

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adagrad
from keras.utils import np_utils
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from pyimagesearch.onconet import Onconet
from pyimagesearch import config
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--plot", default="plot.png")
args = vars(ap.parse_args())

NUM_EPOCHS = 40
INIT_LR = 1e-2
BS = 32

trainPaths = list(paths.list_images(config.TRAIN_PATH))
totalTrain = len(trainPaths)
totalVal = len(list(paths.list_images(config.VAL_PATH)))
totalTest = len(list(paths.list_images(config.TEST_PATH)))

trainLabels = [int(p.split(os.path.sep)[-2]) for p in trainPaths]
trainLabels = np_utils.to_categorical(trainLabels)
classTotals = trainLabels.sum(axis=0)
classWeight = classTotals.max() / classTotals

trainAug = ImageDataGenerator(
    rescale=1 / 255.0,
    rotation_range=20,
    zoom_range=0.05,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.05,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

valAug = ImageDataGenerator(rescale=1 / 255.0)

trainGen = trainAug.flow_from_directory(
    config.TRAIN_PATH,
    target_size=(48, 48),
    color_mode="rgb",
    class_mode="categorical",
    shuffle=True,
    batch_size=BS
)

valGen = valAug.flow_from_directory(
    config.VAL_PATH,
    target_size=(48, 48),
    color_mode="rgb",
    class_mode="categorical",
    shuffle=False,
    batch_size=BS
)

testGen = valAug.flow_from_directory(
    config.TEST_PATH,
    target_size=(48, 48),
    color_mode="rgb",
    class_mode="categorical",
    shuffle=False,
    batch_size=BS
)

model = Onconet.build(width=48, height=48, depth=3, classes=2)
opt = Adagrad(lr=INIT_LR, decay=INIT_LR/ NUM_EPOCHS)
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

H = model.fit_generator(
    trainGen,
    steps_per_epoch=totalTrain,
    validation_data=valGen,
    validation_steps=totalVal,
    class_weight=classWeight,
    epochs=NUM_EPOCHS
)

print("[INFO] evaluating network...")
testGen.reset()
predIdxs = model.predict_generator(testGen,	steps=(totalTest // BS) + 1)


predIdxs = np.argmax(predIdxs, axis=1)

print(classification_report(testGen.classes, predIdxs, target_names=testGen.class_indices.keys()))

cm = confusion_matrix(testGen.classes, predIdxs)
total = sum(sum(cm))
acc = (cm[0, 0] + cm[1, 1]) / total
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

print(cm)
print("acc: {:.4f}".format(acc))
print("sensitivity: {:.4f}".format(sensitivity))
print("specificity: {:.4f}".format(specificity))

N = NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])