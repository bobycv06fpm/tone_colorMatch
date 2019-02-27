import numpy as np
import cv2
import extractMask
import colorTools
import colorsys

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

    def getRegionMapValue(self):
        value = {}
        value['left'] = [float(value) for value in self.leftCheekMedian]
        value['right'] = [float(value) for value in self.rightCheekMedian]
        value['chin'] = [float(value) for value in self.chinMedian]
        value['forehead'] = [float(value) for value in self.foreheadMedian]
        return value

    def maxSubpixelValue(self):
        return np.max([np.max(region) for region in [self.leftCheekPoints, self.rightCheekPoints, self.chinPoints, self.foreheadPoints]])

    def getMaskedImage(self):
        return extractMask.getMaskedImage(self.capture.image, self.capture.mask, [self.leftCheekPolygon, self.rightCheekPolygon, self.chinPolygon, self.foreheadPolygon])

    def getRegionPoints(self):
        return np.array([self.leftCheekPoints, self.rightCheekPoints, self.chinPoints, self.foreheadPoints])

    def getRegionMedians(self):
        return np.array([self.leftCheekMedian, self.rightCheekMedian, self.chinMedian, self.foreheadMedian])

    def getRegionMedianHSV(self):
        return [colorsys.rgb_to_hsv(r, g, b) for b, g, r in self.getRegionMedians()]

    def getRegionHSV(self):
        #Need to match CV2 result with colorsys result H: [0-1] S: [0-1] V: [0-255] so divide by [255, 255, 1]
        return [cv2.cvtColor(np.array([regionPoints]).astype('uint8'), cv2.COLOR_BGR2HSV_FULL)[0] / np.array([255, 255, 1]) for regionPoints in self.getRegionPoints()]

    def getRegionMedianLuminance(self):
        return [colorTools.getRelativeLuminance([regionMedian]) for regionMedian in self.getRegionMedians()]

    def getRegionLuminance(self):
        return [colorTools.getRelativeLuminance(regionPoints) for regionPoints in self.getRegionPoints()]

    def getRegionCleanRatios(self):
        return np.array([self.leftCheekCleanRatio, self.rightCheekCleanRatio, self.chinCleanRatio, self.foreheadCleanRatio])

    def getNumberOfRegions(self):
        return 4

