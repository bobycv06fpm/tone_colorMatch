"A class to simplify handling face regions"""
import numpy as np
import extractMask
import colorTools

class FaceRegions:
    """Simplifies accessing face regions. Wraps Capture and mask"""

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

        self.linearLeftCheekPoints = colorTools.convert_sBGR_to_linearBGR_float(self.leftCheekPoints)
        self.linearRightCheekPoints = colorTools.convert_sBGR_to_linearBGR_float(self.rightCheekPoints)
        self.linearChinPoints = colorTools.convert_sBGR_to_linearBGR_float(self.chinPoints)
        self.linearForeheadPoints = colorTools.convert_sBGR_to_linearBGR_float(self.foreheadPoints)

        self.linearLeftCheekMean = np.mean(self.linearLeftCheekPoints, axis=0)
        self.linearRightCheekMean = np.mean(self.linearRightCheekPoints, axis=0)
        self.linearChinMean = np.mean(self.linearChinPoints, axis=0)
        self.linearForeheadMean = np.mean(self.linearForeheadPoints, axis=0)

    def getRegionMapValue(self):
        """Returns the mean value for each face region in a map"""
        value = {}
        value['left'] = [float(value) for value in self.linearLeftCheekMean]
        value['right'] = [float(value) for value in self.linearRightCheekMean]
        value['chin'] = [float(value) for value in self.linearChinMean]
        value['forehead'] = [float(value) for value in self.linearForeheadMean]
        return value

    def getMaskedImage(self):
        """Returns the face image masked by the face region polygons"""
        return extractMask.getMaskedImage(self.capture.faceImage, self.capture.faceMask, [self.leftCheekPolygon, self.rightCheekPolygon, self.chinPolygon, self.foreheadPolygon])

    def __getRegionPoints(self):
        return np.array([self.linearLeftCheekPoints, self.linearRightCheekPoints, self.linearChinPoints, self.linearForeheadPoints])

    def getRegionMeans(self):
        """Returns the mean for each facial region in an array"""
        return np.array([self.linearLeftCheekMean, self.linearRightCheekMean, self.linearChinMean, self.linearForeheadMean])
