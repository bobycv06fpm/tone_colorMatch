import utils
import colorsys
import numpy as np
import cv2

def getMaskedImage(image, clippedMask, polygons):
    regionsMask = np.zeros(clippedMask.shape, dtype='uint8')

    for polygon in polygons:
        hull = cv2.convexHull(polygon)
        regionsMask = cv2.fillConvexPoly(regionsMask, hull, 1)

    regionsMask = regionsMask.astype('bool')
    filteredRegionsMask = np.logical_and(regionsMask, np.logical_not(clippedMask))
    maskedImage = np.where(filteredRegionsMask[..., None], image, 1)

    return maskedImage


def extractPolygonPoints(image, clippedMask, polygon):
    regionMask = np.zeros(clippedMask.shape, dtype='uint8')

    hull = cv2.convexHull(polygon)
    regionMask = cv2.fillConvexPoly(regionMask, hull, 1)
    regionMask = regionMask.astype('bool')
    clippedRegionMask = np.logical_and(regionMask, np.logical_not(clippedMask))

    unfilteredRegionPoints = image[regionMask]
    filteredRegionPoints = image[clippedRegionMask]

    cleanClippedRatio = filteredRegionPoints.size / unfilteredRegionPoints.size

    cutoff = 0.1
    if cleanClippedRatio < cutoff:
        raise NameError('Not enough clean non-clipped pixels. Ratio :: ' + str(cleanClippedRatio))

    return [filteredRegionPoints, cleanClippedRatio]

