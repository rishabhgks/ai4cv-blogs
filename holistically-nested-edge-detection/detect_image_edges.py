# Importing the required libraries
import argparse
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", type=str, required=True)
ap.add_argument("-i", "--image", type=str, required=True)
args = vars(ap.parse_args())

class CropLayer(object):
    def __init__(self, params, blobs):
        self.startX = 0
        self.startY = 0
        self.endX = 0
        self.endY = 0

    def getMemoryShapes(self, inputs):
        (inputShape, targetShape) = (inputs[0], inputs[1])
        (batchSize, numChannels) = (inputShape[0], inputShape[1])
        (H, W) = (targetShape[2], targetShape[3])

        self.startX = int((inputShape[3] - targetShape[3]) / 2)
        self.startY = int((inputShape[2] - targetShape[2]) / 2)
        self.endX = self.startX + W
        self.endY = self.startY + H

        return [[batchSize, numChannels, H, W]]

    def forward(self, inputs):
        return [inputs[0][:, :, self.startY:self.endY, self.startX:self.endX]]


print("[Status] loading the edge detector.......")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"], "hed_pretrained_bsds.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

cv2.dnn_registerLayer("Crop", CropLayer)

image = cv2.imread(args["image"])
(H, W) = image.shape[:2]

print("[Status] Performing Canny Edge Detection......")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
canny = cv2.Canny(blurred, 30, 150)

blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(H, W), mean=(104.00698793, 116.66876762, 122.67891434),
                             swapRB=False, crop=False)

print("[Status] Performing HED.......")
net.setInput(blob)
hed = net.forward()
hed = cv2.resize(hed[0, 0], (W, H))
hed = (255 * hed).astype("uint8")

cv2.imshow("Input", image)
cv2.imshow("Canny", canny)
cv2.imshow("HED", hed)
cv2.waitKey(0)
