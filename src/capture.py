import numpy as np
import cv2
import thresholdMask
import colorTools
import saveStep
from landmarkPoints import Landmarks

class Capture:

    def __init__(self, image, metadata, mask=None):
        self.name = '{}_{}_Flash'.format(metadata["flashSettings"]["area"], metadata["flashSettings"]["areas"])
        self.flashSettings = metadata["flashSettings"]
        self.flashRatio = self.flashSettings["area"] / self.flashSettings["areas"]
        self.isNoFlash = True if metadata["flashSettings"]["area"] == 0 else False
        #self.image = np.clip(image, 0, 255).astype('uint8')
        #self.image = image.astype('int32')
        #colorTools.whitebalance_from_asShot_to_d65(image.astype('uint16'), metadata['whiteBalance']['x'], metadata['whiteBalance']['y'])
        if metadata["imageTransforms"]["isGammaSBGR"] is False:
            self.image = colorTools.convert_sBGR_to_linearBGR_float_fast(image)
            print('{} :: sBGR -> Linear'.format(self.name))
        else:
            self.image = image / 255

        self.metadata = metadata
        self.landmarks = Landmarks(self.metadata['faceLandmarksSource'], self.metadata['faceLandmarks'], image.shape)
        self.mask = thresholdMask.getClippedMask(image)
        self.whiteBalance = self.metadata['whiteBalance']

        if mask is not None:
            self.mask = np.logical_or(self.mask, mask)

    def getFormattedImage(self):
        return np.clip(self.image * 255, 0, 255).astype('uint8')

    def getLargestValue(self):
        return np.max(self.getFormattedImage())

    def blurredImage(self):
        return cv2.GaussianBlur(self.getFormattedImage(), (5, 5), 0)
        #return cv2.medianBlur(self.image.astype('uint16'), 5)

    #def scaleToValue(self, value):
    #    if value < 255:
    #        return

    #    self.image = self.image * (255 / value)

    def whiteBalanceImageToD65(self):
        self.image = colorTools.whitebalance_from_asShot_to_d65(self.image, self.whiteBalance['x'], self.whiteBalance['y'])

    def getWhiteBalancedImageToD65(self):
        whiteBalanced = colorTools.whitebalance_from_asShot_to_d65(self.image, self.whiteBalance['x'], self.whiteBalance['y'])
        return whiteBalanced#np.clip(whiteBalanced * 255, 0, 255).astype('uint8')

    def getAsShotWhiteBalance(self):
        return [self.whiteBalance['x'], self.whiteBalance['y']]

    #def getClippedImage(self):
    #    return np.clip(self.image, 0, 255).astype('uint8')

    def show(self):
        ratio = 3
        smallImage = cv2.resize(self.getFormattedImage(), (0, 0), fx=1/ratio, fy=1/ratio)
        cv2.imshow(self.name, smallImage)
        cv2.waitKey(0)

    def calculateNoise(self):
        blurSize = 5
        blurred = cv2.medianBlur(self.getFormattedImage(), blurSize)
        #blurred = cv2.blur(image.astype('uint16'), (blurSize, blurSize))
        #blurred = cv2.GaussianBlur(image.astype('uint16'), (blurSize, blurSize), 0)

        blurred[blurred == 0] = 1
        #blurredLuminance[blurredLuminance == 0] = 1

        #noise = (np.abs(image.astype('int32') - blurred.astype('int32')) * 50).astype('uint8')
        noise = (np.abs(self.getFormattedImage().astype('int32') - blurred.astype('int32'))).astype('uint8')
        #noise = (np.clip(capture.image.astype('int32') - blurred.astype('int32'), 0, 255) * 50).astype('uint8')
        #noise = ((np.abs(capture.image.astype('int32') - blurred.astype('int32')) / capture.image) * 5000).astype('uint8')

        #blurSize2 = 55
        #noiseBlurred = cv2.GaussianBlur(noise, (blurSize2, blurSize2), 0)

        return noise
        #saveStep.saveReferenceImageBGR(noise, '{}Noise'.format(capture.name))

    def showMasked(self, wait=True):
        masked = np.copy(self.getFormattedImage())
        masked[self.mask] = [0, 0, 0]

        self.show(wait, masked)

    def showImageWithLandmarks(self, wait=True, tag=''):
        img = self.getFormattedImage()
        for point in self.landmarks.landmarkPoints:
            cv2.circle(img, (point[0], point[1]), 5, (0, 0, 255), -1)

        ratio = 3
        smallImage = cv2.resize(img, (0, 0), fx=1/ratio, fy=1/ratio)

        cv2.imshow(self.name + tag, smallImage)
        if wait:
            cv2.waitKey(0)

