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


def extractMask(capture, saveStep):
    forehead = capture.landmarks.getForeheadPoints()
    leftCheek = capture.landmarks.getLeftCheekPoints()
    rightCheek = capture.landmarks.getRightCheekPoints()
    chin = capture.landmarks.getChinPoints()

    polygons = [forehead, leftCheek, rightCheek, chin]
    cheekPolygons = [leftCheek, rightCheek]

    image = capture.image
    clippedMask = capture.mask

    mask = cv2.split(image)[0].copy()
    mask.fill(0)
    cheekMask = mask.copy()

    for polygon in polygons:
        hull = cv2.convexHull(polygon)
        mask = cv2.fillConvexPoly(mask, hull, 1)

    for cheekPolygon in cheekPolygons:
        hull = cv2.convexHull(cheekPolygon)
        cheekMask = cv2.fillConvexPoly(cheekMask, hull, 1)


    mask = mask.astype('bool')
    region_mask_point = image[mask]

    cheekMask = cheekMask.astype('bool')
    cheek_mask_point = image[cheekMask]

    if clippedMask is not None:
        mask = np.logical_and(mask, np.logical_not(clippedMask))
        cheekMask = np.logical_and(cheekMask, np.logical_not(clippedMask))

    masked_image = np.where(mask[..., None], image, 0)
    masked_points = image[mask]

    cheek_masked_image = np.where(cheekMask[..., None], image, 0)
    cheek_masked_points = image[cheekMask]

    clippedPixelRatio = masked_points.size / region_mask_point.size
    print('Clipping Ratio :: ' + str(clippedPixelRatio))
    #if clippedPixelRatio < .2:
    if clippedPixelRatio < .01:
        raise NameError('Not enough clean non-clipped pixels. Ratio :: ' + str(clippedPixelRatio))

    #img = cv2.resize(cheek_masked_image.astype('uint8'), (0, 0), fx=1/3, fy=1/3)
    #cv2.imshow('cheek masked', img)
    #cv2.waitKey(0)
    saveStep.saveReferenceImageBGR(masked_image, capture.name + '_masked')

    return [np.array(masked_points), np.array(cheek_masked_image)]

