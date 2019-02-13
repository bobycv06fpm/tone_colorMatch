import numpy as np
import cv2
import thresholdMask
import colorTools
import saveStep
from landmarkPoints import Landmarks

class Capture:

    def __init__(self, name, image, metadata, mask=None):
        self.name = name
        #self.image = np.clip(image, 0, 255).astype('uint8')
        #self.image = image.astype('int32')
        #colorTools.whitebalance_from_asShot_to_d65(image.astype('uint16'), metadata['whiteBalance']['x'], metadata['whiteBalance']['y'])
        self.image = image.astype('uint16')
        self.metadata = metadata
        self.landmarks = Landmarks(self.metadata['faceLandmarksSource'], self.metadata['faceLandmarks'], image.shape)
        self.mask = thresholdMask.getClippedMask(image)
        self.whiteBalance = self.metadata['whiteBalance']

        if mask is not None:
            self.mask = np.logical_or(self.mask, mask)

    def blurredImage(self):
        #return cv2.GaussianBlur(self.image, (5, 5), 0)
        return cv2.medianBlur(self.image.astype('uint16'), 5)

    def whiteBalanceImageToD65(self):
        self.image = colorTools.whitebalance_from_asShot_to_d65(self.image.astype('uint16'), self.whiteBalance['x'], self.whiteBalance['y'])

    def getAsShotWhiteBalance(self):
        return [self.whiteBalance['x'], self.whiteBalance['y']]

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

    def showImageWithLandmarks(self, wait=True, tag=''):
        img = np.clip(self.image, 0, 255).astype('uint8')
        for point in self.landmarks.landmarkPoints:
            cv2.circle(img, (point[0], point[1]), 5, (0, 0, 255), -1)

        ratio = 3
        smallImage = cv2.resize(img, (0, 0), fx=1/ratio, fy=1/ratio)

        cv2.imshow(self.name + tag, smallImage)
        if wait:
            cv2.waitKey(0)

