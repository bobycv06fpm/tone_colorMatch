import utils
import saveStep
import colorsys
import numpy as np
import cv2

def maskPolygons(mask, polygons):
    polyMask = np.copy(mask.astype('uint8'))
    polyMask.fill(0)

    for polygon in polygons:
        hull = cv2.convexHull(polygon)
        polyMask = cv2.fillConvexPoly(polyMask, hull, 1)

    return np.logical_and(polyMask.astype('bool'), mask)


def extractMask(username, imageName, polygons, capture):
    image = capture.image
    clippedMask = capture.mask

    mask = cv2.split(image)[0].copy()
    mask.fill(0)

    for polygon in polygons:
        hull = cv2.convexHull(polygon)
        mask = cv2.fillConvexPoly(mask, hull, 1)

    mask = mask.astype('bool')
    region_mask_point = image[mask]

    if clippedMask is not None:
        mask = np.logical_and(mask, np.logical_not(clippedMask))

    masked_image = np.where(mask[..., None], image, 0)
    masked_points = image[mask]

    clippedPixelRatio = masked_points.size / region_mask_point.size
    print('Clipping Ratio :: ' + str(clippedPixelRatio))
    if clippedPixelRatio < .2:
        raise NameError('Not enough clean non-clipped pixels. Ratio :: ' + str(clippedPixelRatio))

    sumOfUnscaledPixels = np.sum(masked_points, axis=0)
    averageOfUnscaledPixels = sumOfUnscaledPixels / len(masked_points)
    averageIntensityOfUnscaledPixels = sum(averageOfUnscaledPixels) / 3
    
    if imageName is not None:
        saveStep.logMeasurement(username, imageName, 'Flash Contribution', str(averageIntensityOfUnscaledPixels))
        saveStep.saveReferenceImageBGR(username, imageName, masked_image, 'masked')

    return [np.array(masked_points), averageIntensityOfUnscaledPixels]

