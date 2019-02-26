import numpy as np
import cv2
import extractMask

class FaceRegions:

    def __init__(self, capture):
        self.capture = capture

        self.leftCheekPolygon = capture.landmarks.getLeftCheekPoints()
        self.rightCheekPolygon = capture.landmarks.getRightCheekPoints()
        self.chinPolygon = capture.landmarks.getChinPoints()
        self.foreheadPolygon = capture.landmarks.getForeheadPoints()

        self.leftCheekPoints, self.leftCheekCleanRatio = extractMask.extractPolygonPoints(capture.image, capture.mask, self.leftCheekPolygon)
        self.rightCheekPoints, self.rightCheekCleanRatio = extractMask.extractPolygonPoints(capture.image, capture.mask, self.rightCheekPolygon)
        self.chinPoints, self.chinCleanRatio = extractMask.extractPolygonPoints(capture.image, capture.mask, self.chinPolygon)
        self.foreheadPoints, self.foreheadCleanRatio = extractMask.extractPolygonPoints(capture.image, capture.mask, self.foreheadPolygon)

        self.leftCheekMedian = np.median(self.leftCheekPoints, axis=0)
        self.rightCheekMedian = np.median(self.rightCheekPoints, axis=0)
        self.chinMedian = np.median(self.chinPoints, axis=0)
        self.foreheadMedian = np.median(self.foreheadPoints, axis=0)

    def maxSubpixelValue(self):
        return np.max([np.max(region) for region in [self.leftCheekPoints, self.rightCheekPoints, self.chinPoints, self.foreheadPoints]])

    def getMaskedImage(self):
        return extractMask.getMaskedImage(self.capture.image, self.capture.mask, [self.leftCheekPolygon, self.rightCheekPolygon, self.chinPolygon, self.foreheadPolygon])

    def getRegionPoints(self):
        return np.array([self.leftCheekPoints, self.rightCheekPoints, self.chinPoints, self.foreheadPoints])

    def getRegionMedians(self):
        return np.array([self.leftCheekMedian, self.rightCheekMedian, self.chinMedian, self.foreheadMedian])

    def getRegionCleanRatios(self):
        return np.array([self.leftCheekCleanRatio, self.rightCheekCleanRatio, self.chinCleanRatio, self.foreheadCleanRatio])

