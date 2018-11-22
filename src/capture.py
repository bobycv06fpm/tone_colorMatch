import numpy as np
import cv2
import thresholdMask
from landmarkPoints import Landmarks

class Capture:

    def __init__(self, name, image, metadata, mask=None):
        self.name = name
        self.image = np.clip(image, 0, 255).astype('uint8')
        self.metadata = metadata
        self.landmarks = Landmarks(self.metadata['faceLandmarksSource'], self.metadata['faceLandmarks'], image.shape)
        self.mask = thresholdMask.getClippedMask(image)

        if mask is not None:
            self.mask = np.logical_or(self.mask, mask)

    def getClippedImage(self):
        return np.clip(self.image, 0, 255).astype('uint8')

    def show(self, wait=True, passedImage=None):
        image = passedImage if passedImage is not None else self.image
        clippedImage = np.clip(image, 0, 255).astype('uint8')

        ratio = 3
        smallImage = cv2.resize(clippedImage, (0, 0), fx=1/ratio, fy=1/ratio)

        cv2.imshow(self.name, smallImage)
        if wait:
            cv2.waitKey(0)

    def showMasked(self, wait=True):
        masked = np.copy(self.image)
        masked[self.mask] = [0, 0, 0]

        self.show(wait, masked)
