import numpy as np
import cv2
import thresholdMask
import colorTools
import saveStep
from landmarkPoints import Landmarks

class Capture:

    def __init__(self, name, image, metadata, mask=None):
        self.name = name
        self.image = np.clip(image, 0, 255).astype('uint8')
        self.metadata = metadata
        self.landmarks = Landmarks(self.metadata['faceLandmarksSource'], self.metadata['faceLandmarks'], image.shape)
        self.mask = thresholdMask.getClippedMask(image)
        self.whiteBalance = self.metadata['whiteBalance']
        #print('WB :: ' + str(self.whiteBalance))

        if mask is not None:
            self.mask = np.logical_or(self.mask, mask)

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

    def showImageWithLandmarks(self):
        img = np.copy(self.image)
        for point in self.landmarks.landmarkPoints:
            cv2.circle(img, (point[0], point[1]), 5, (0, 0, 255), -1)

        ratio = 3
        smallImage = cv2.resize(img, (0, 0), fx=1/ratio, fy=1/ratio)

        cv2.imshow(self.name, smallImage)
        cv2.waitKey(0)

    def saturationDiff(self):
        img = np.copy(self.image)
        for point in self.landmarks.landmarkPoints:
            cv2.circle(img, (point[0], point[1]), 5, (0, 0, 255), -1)
        
        x, y, w, h = self.landmarks.getEyeStripBB()

        img = img[y:y + h, x:x + w]

        blur = 101
        #img = cv2.GaussianBlur(img, (blur, blur), 0)
        img = cv2.bilateralFilter(img,20,300,300)


        hsvImage = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
        saturations = hsvImage[:, :, 1]

        #saturations = cv2.Sobel(saturations, cv2.CV_64F, 1, 0, ksize=5)
        #saturations = cv2.Laplacian(saturations, cv2.CV_64F)

        #hues = hsvImage[:, :, 0]
        #hues = (hues / 255) * 360
        #hueMask = np.logical_or((hues < 60), (hues > 330))
        #print('hues :: ' + str(hues))
        #print('hues mask :: ' + str(hueMask))

        #saturationMask = saturations > 150

        #mask = np.logical_or(saturationMask, hueMask)

        saturations[hueMask] = 255

        ratio = 2
        smallImage = cv2.resize(saturations, (0, 0), fx=1/ratio, fy=1/ratio)
        cv2.imshow(self.name + ' Saturation', smallImage)
        cv2.waitKey(0)
        print('Saturations :: ' + str(saturations))

