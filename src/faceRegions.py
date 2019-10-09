import numpy as np
import cv2
import extractMask
import colorTools
import colorsys
from matplotlib import pyplot as plt

class FaceRegions:

    def __init__(self, capture):
        self.capture = capture

        self.leftCheekPolygon = capture.landmarks.getLeftCheekPoints()
        self.rightCheekPolygon = capture.landmarks.getRightCheekPoints()
        self.chinPolygon = capture.landmarks.getChinPoints()
        self.foreheadPolygon = capture.landmarks.getForeheadPoints()

        #print(self.capture.faceImage)
        #linear = colorTools.convert_sBGR_to_linearBGR_float_fast(self.capture.faceImage / 255)
        #hsv = colorTools.naiveBGRtoHSV(linear)
        #hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
        #
        #masked_hsv = extractMask.getMaskedImage(hsv, self.capture.faceMask, [self.leftCheekPolygon, self.rightCheekPolygon, self.foreheadPolygon])#, self.chinPolygon])

        #sat = masked_hsv[:, :, 1]
        #hue = masked_hsv[:, :, 0]
        #val = masked_hsv[:, :, 2]

        ##sat = cv2.GaussianBlur(sat, (5, 5), 0)
        ##hue = cv2.GaussianBlur(hue, (5, 5), 0)

        #masked_region_rough = sat != 0
        #minSat = np.min(sat[masked_region_rough])
        #minHue = np.min(hue[masked_region_rough])
        #minVal = np.min(val[masked_region_rough])
        #maxSat = np.max(sat)
        #maxHue = np.max(hue)
        #maxVal = np.max(val)

        #sat[masked_region_rough] = (sat[masked_region_rough] - minSat) / (maxSat - minSat)
        #hue[masked_region_rough] = (hue[masked_region_rough] - minHue) / (maxHue - minHue)
        #val[masked_region_rough] = (val[masked_region_rough] - minVal) / (maxVal - minVal)

        ##mix = sat + (1 - val)
        #mix = sat - val
        #maxMix = np.max(mix)
        #minMix = np.min(mix[masked_region_rough])
        #mix[masked_region_rough] = (mix[masked_region_rough] - minMix) / (maxMix - minMix)


        ##plt.hist(sat[masked_region_rough].ravel(),256)
        ##plt.hist(hue[masked_region_rough].ravel(),256)
        ##plt.hist(val[masked_region_rough].ravel(),256)
        #plt.hist(mix[masked_region_rough].ravel(),256)
        #plt.show()
        ##plt.hist(hue[i*3].ravel(),256)

        #joint = np.hstack([sat, val, mix])
        ##cv2.imshow('masked Sat', sat)
        ##cv2.imshow('masked Hue', hue)
        ##cv2.imshow('masked Val', val)
        ##cv2.imshow('masked Mix', mix)
        #cv2.imshow('masked Joint', joint)
        #cv2.waitKey(0)

        self.leftCheekPoints, self.leftCheekCleanRatio = extractMask.extractPolygonPoints(capture.faceImage, capture.faceMask, self.leftCheekPolygon)
        self.rightCheekPoints, self.rightCheekCleanRatio = extractMask.extractPolygonPoints(capture.faceImage, capture.faceMask, self.rightCheekPolygon)
        self.chinPoints, self.chinCleanRatio = extractMask.extractPolygonPoints(capture.faceImage, capture.faceMask, self.chinPolygon)
        self.foreheadPoints, self.foreheadCleanRatio = extractMask.extractPolygonPoints(capture.faceImage, capture.faceMask, self.foreheadPolygon)

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

