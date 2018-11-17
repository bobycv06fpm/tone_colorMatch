import numpy as np
import cv2
import thresholdMask
from landmarkPoints import Landmarks

class Capture:

    def __init__(self, name, image, metadata):
        self.name = name
        self.image = image
        self.metadata = metadata
        self.landmarks = Landmarks(self.metadata['faceLandmarksSource'], self.metadata['faceLandmarks'], image.shape)
        self.mask = thresholdMask.getClippedMask(self.image, 1, 1)

    def show(self, wait=True):
        ratio = 3
        smallImage = cv2.resize(self.image, (0, 0), fx=1/ratio, fy=1/ratio)
        cv2.imshow(self.name, smallImage)
        if wait:
            cv2.waitKey(0)


