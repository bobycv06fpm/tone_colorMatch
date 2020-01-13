"""Functions that help extract masks defined by a bitmask or a polygon"""
import numpy as np
import cv2

def getMaskedImage(image, clippedMask, polygons):
    """Returns the masked image incorporating a bitmask and polygons"""
    regionsMask = np.zeros(clippedMask.shape, dtype='uint8')

    for polygon in polygons:
        hull = cv2.convexHull(polygon)
        regionsMask = cv2.fillConvexPoly(regionsMask, hull, 1)

    regionsMask = regionsMask.astype('bool')
    filteredRegionsMask = np.logical_and(regionsMask, np.logical_not(clippedMask))
    maskedImage = np.where(filteredRegionsMask[..., None], image, 0)

    return maskedImage#.astype('uint8')


def extractPolygonPoints(image, mask, polygon):
    """Returns the points contained in the polygon incorporating the bitmask"""
    regionMask = np.zeros(mask.shape, dtype='uint8')

    hull = cv2.convexHull(polygon)
    regionMask = cv2.fillConvexPoly(regionMask, hull, 1)
    regionMask = regionMask.astype('bool')
    filteredRegionMask = np.logical_and(regionMask, mask)

    unfilteredRegionPoints = image[regionMask]
    filteredRegionPoints = image[filteredRegionMask]

    cleanClippedRatio = filteredRegionPoints.size / unfilteredRegionPoints.size

    return [filteredRegionPoints, cleanClippedRatio]
