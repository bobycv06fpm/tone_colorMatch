import numpy as np
import cv2
import extractMask
import colorTools
import colorsys
from matplotlib import pyplot as plt

class FaceRegions:

    def __init__(self, capture, mask):
        self.mask = mask
        self.capture = capture

        self.leftCheekPolygon = capture.landmarks.getLeftCheekPoints()
        self.rightCheekPolygon = capture.landmarks.getRightCheekPoints()
        self.chinPolygon = capture.landmarks.getChinPoints()
        self.foreheadPolygon = capture.landmarks.getForeheadPoints()

        self.mask = np.logical_and(np.logical_not(capture.faceMask), self.mask)

        self.leftCheekPoints, self.leftCheekCleanRatio = extractMask.extractPolygonPoints(capture.faceImage, self.mask, self.leftCheekPolygon)
        self.rightCheekPoints, self.rightCheekCleanRatio = extractMask.extractPolygonPoints(capture.faceImage, self.mask, self.rightCheekPolygon)
        self.chinPoints, self.chinCleanRatio = extractMask.extractPolygonPoints(capture.faceImage, self.mask, self.chinPolygon)
        self.foreheadPoints, self.foreheadCleanRatio = extractMask.extractPolygonPoints(capture.faceImage, self.mask, self.foreheadPolygon)

        #print('{} - Converting Face Regions to Linear -'.format(capture.name))
        self.linearLeftCheekPoints = colorTools.convert_sBGR_to_linearBGR_float_fast(self.leftCheekPoints)
        self.linearRightCheekPoints = colorTools.convert_sBGR_to_linearBGR_float_fast(self.rightCheekPoints)
        self.linearChinPoints = colorTools.convert_sBGR_to_linearBGR_float_fast(self.chinPoints)
        self.linearForeheadPoints = colorTools.convert_sBGR_to_linearBGR_float_fast(self.foreheadPoints)

        #self.linearLeftCheekMedian = np.median(self.linearLeftCheekPoints, axis=0)
        self.linearLeftCheekMean = np.mean(self.linearLeftCheekPoints, axis=0)
        #self.linearRightCheekMedian = np.median(self.linearRightCheekPoints, axis=0)
        self.linearRightCheekMean = np.mean(self.linearRightCheekPoints, axis=0)
        #self.linearChinMedian = np.median(self.linearChinPoints, axis=0)
        self.linearChinMean = np.mean(self.linearChinPoints, axis=0)
        #self.linearForeheadMedian = np.median(self.linearForeheadPoints, axis=0)
        self.linearForeheadMean = np.mean(self.linearForeheadPoints, axis=0)

    def getRegionMapValue(self):
        value = {}
        value['left'] = [float(value) for value in self.linearLeftCheekMedian]
        value['right'] = [float(value) for value in self.linearRightCheekMedian]
        value['chin'] = [float(value) for value in self.linearChinMedian]
        value['forehead'] = [float(value) for value in self.linearForeheadMedian]
        return value

    def maxSubpixelValue(self):
        return np.max([np.max(region) for region in [self.linearLeftCheekPoints, self.linearRightCheekPoints, self.linearChinPoints, self.linearForeheadPoints]])

    def getMaskedImage(self):
        return extractMask.getMaskedImage(self.capture.faceImage, self.capture.faceMask, [self.leftCheekPolygon, self.rightCheekPolygon, self.chinPolygon, self.foreheadPolygon])

    def getRegionPoints(self):
        return np.array([self.linearLeftCheekPoints, self.linearRightCheekPoints, self.linearChinPoints, self.linearForeheadPoints])

    def getNumPixelsPerRegion(self):
        regionsPoints = self.getRegionPoints()
        return [len(regionPoints) for regionPoints in regionsPoints]

    def getRegionMedians(self):
        #return np.array([self.linearLeftCheekMedian, self.linearRightCheekMedian, self.linearChinMedian, self.linearForeheadMedian])
        return np.array([self.linearLeftCheekMean, self.linearRightCheekMean, self.linearChinMean, self.linearForeheadMean])

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

