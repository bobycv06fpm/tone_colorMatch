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
        self.isGammaSBGR = True#metadata['faceImageTransforms']["isGammaSBGR"]

        self.scaleRatio = metadata['faceImageTransforms']["scaleRatio"] if "scaleRatio" in metadata['faceImageTransforms'] else 1
        print("Scale Ratio :: {}".format(self.scaleRatio))
        #self.image = image
        self.faceImage, self.leftEyeImage, self.rightEyeImage = image
        #if metadata["imageTransforms"]["isGammaSBGR"] is False:
        #    self.image = colorTools.convert_sBGR_to_linearBGR_float_fast(image)
        #    print('{} :: sBGR -> Linear'.format(self.name))
        #else:
        #    self.image = image / 255

        self.metadata = metadata
        
        self.leftEyeBB = np.array(self.metadata['leftEyeImageTransforms']['bbInParent']) if 'bbInParent' in self.metadata['leftEyeImageTransforms'] else None
        print("Scale Ratio :: {}".format(self.leftEyeBB))
        self.rightEyeBB = np.array(self.metadata['rightEyeImageTransforms']['bbInParent']) if 'bbInParent' in self.metadata['rightEyeImageTransforms'] else None
        print("Scale Ratio :: {}".format(self.rightEyeBB))

        self.landmarks = Landmarks(self.metadata['faceLandmarksSource'], self.metadata['faceImageTransforms']['landmarks'], [self.leftEyeBB, self.rightEyeBB], self.faceImage.shape)

        self.faceMask = thresholdMask.getClippedMask(self.faceImage)
        self.leftEyeMask = thresholdMask.getClippedMask(self.leftEyeImage)
        self.rightEyeMask = thresholdMask.getClippedMask(self.rightEyeImage)

        self.whiteBalance = self.metadata['whiteBalance']
        self.isBlurry = False

        if mask is not None:
            self.mask = np.logical_or(self.mask, mask)

    def getFormattedFaceImage(self):
        return np.clip(self.faceImage, 0, 255).astype('uint8')

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
        self.faceImage = colorTools.whitebalance_from_asShot_to_d65(self.faceImage, self.whiteBalance['x'], self.whiteBalance['y'])
        self.leftEyeImage = colorTools.whitebalance_from_asShot_to_d65(self.leftEyeImage, self.whiteBalance['x'], self.whiteBalance['y'])
        self.rightEyeImage = colorTools.whitebalance_from_asShot_to_d65(self.rightEyeImage, self.whiteBalance['x'], self.whiteBalance['y'])

    def getWhiteBalancedImageToD65(self):
        faceImageWB = colorTools.whitebalance_from_asShot_to_d65(self.faceImage, self.whiteBalance['x'], self.whiteBalance['y'])
        leftEyeImageWB = colorTools.whitebalance_from_asShot_to_d65(self.leftEyeImage, self.whiteBalance['x'], self.whiteBalance['y'])
        rightEyeImageWB = colorTools.whitebalance_from_asShot_to_d65(self.rightEyeImage, self.whiteBalance['x'], self.whiteBalance['y'])
        return [faceImageWB, leftEyeImageWB, rightEyeImageWB]#np.clip(whiteBalanced * 255, 0, 255).astype('uint8')

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
        img = self.getFormattedFaceImage()
        for point in self.landmarks.landmarkPoints:
            print("POINT :: {}".format(point))
            cv2.circle(img, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)

        #ratio = 3
        #smallImage = cv2.resize(img, (0, 0), fx=1/ratio, fy=1/ratio)

        cv2.imshow(self.name + tag, img)
        if wait:
            cv2.waitKey(0)

