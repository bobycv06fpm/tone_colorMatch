import utils
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


def extractMask(capture, polygons, saveStep):
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
    #if clippedPixelRatio < .2:
    if clippedPixelRatio < .01:
        raise NameError('Not enough clean non-clipped pixels. Ratio :: ' + str(clippedPixelRatio))

    sumOfUnscaledPixels = np.sum(masked_points, axis=0)
    averageOfUnscaledPixels = sumOfUnscaledPixels / len(masked_points)
    averageIntensityOfUnscaledPixels = sum(averageOfUnscaledPixels) / 3
    
    #cv2.imshow('masked', masked_image)
    #cv2.waitKey(0)
    saveStep.saveReferenceImageBGR(masked_image, capture.name + '_masked')

    return [np.array(masked_points), averageIntensityOfUnscaledPixels]

