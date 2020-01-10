""" Capture class holds a full face image, left eye image, and right eye image along with relavant information parsed from the metadata"""
import numpy as np
import thresholdMask
from landmarkPoints import Landmarks
from logger import getLogger


logger = getLogger(__name__, 'app')

class Capture:
    """Hold full face, left eye, right eye images and metadata, including landmarking information"""

    def __init__(self, image, metadata, mask=None):
        self.name = '{}_{}_Flash'.format(metadata["flashSettings"]["area"], metadata["flashSettings"]["areas"])

        self.faceImage, self.leftEyeImage, self.rightEyeImage = image
        self.metadata = metadata

        self.flashSettings = metadata["flashSettings"]
        self.flashRatio = self.flashSettings["area"] / self.flashSettings["areas"]
        self.isNoFlash = self.flashRatio == 0

        #Scale Ratio represents the scaling factor for FaceImage from its original resolution
        self.scaleRatio = metadata['faceImageTransforms']["scaleRatio"] if "scaleRatio" in metadata['faceImageTransforms'] else 1

        self.leftEyeBB = np.array(self.metadata['leftEyeImageTransforms']['bbInParent']) if ('bbInParent' in self.metadata['leftEyeImageTransforms']) else None
        self.rightEyeBB = np.array(self.metadata['rightEyeImageTransforms']['bbInParent']) if ('bbInParent' in self.metadata['rightEyeImageTransforms']) else None

        self.landmarks = Landmarks(self.metadata['faceLandmarksSource'], self.metadata['faceImageTransforms']['landmarks'], self.faceImage.shape)

        self.faceMask = thresholdMask.getClippedMask(self.faceImage)
        self.leftEyeMask = thresholdMask.getClippedMask(self.leftEyeImage)
        self.rightEyeMask = thresholdMask.getClippedMask(self.rightEyeImage)

        self.whiteBalance = [self.metadata['whiteBalance']['x'], self.metadata['whiteBalance']['y']]

        self.isSharpest = False #Is the sharpest image in the capture set. Set later
        self.isBlurry = False

        if mask is not None:
            self.mask = np.logical_or(self.mask, mask)
