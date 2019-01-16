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
    #cheekPolygons = [leftCheek, rightCheek]

    image = capture.image
    clippedMask = capture.mask

    mask = cv2.split(image)[0].copy()
    mask.fill(0)
    leftCheekMask = mask.copy()
    rightCheekMask = mask.copy()

    for polygon in polygons:
        hull = cv2.convexHull(polygon)
        mask = cv2.fillConvexPoly(mask, hull, 1)

    leftCheekHull = cv2.convexHull(leftCheek)
    leftCheekMask = cv2.fillConvexPoly(leftCheekMask, leftCheekHull, 1)

    rightCheekHull = cv2.convexHull(rightCheek)
    rightCheekMask = cv2.fillConvexPoly(rightCheekMask, rightCheekHull, 1)

    mask = mask.astype('bool')
    #region_mask_point = image[mask]

    leftCheekMask = leftCheekMask.astype('bool')
    unmaskedLeftCheekPoints = image[leftCheekMask]

    rightCheekMask = rightCheekMask.astype('bool')
    unmaskedRightCheekPoints = image[rightCheekMask]

    if clippedMask is not None:
        mask = np.logical_and(mask, np.logical_not(clippedMask))
        leftCheekMask = np.logical_and(leftCheekMask, np.logical_not(clippedMask))
        rightCheekMask = np.logical_and(rightCheekMask, np.logical_not(clippedMask))

    maskedImage = np.where(mask[..., None], image, 0)
    maskedPoints = image[mask]

    leftCheekPoints = image[leftCheekMask]
    rightCheekPoints = image[rightCheekMask]

    #Base Clipping on cheeks for now...
    #clippedPixelRatio = (leftCheekPoints.size + rightCheekPoints.size) / (unmaskedLeftCheekPoints.size + unmaskedRightCheekPoints.size)
    leftCheekClippedPixelRatio = leftCheekPoints.size / unmaskedLeftCheekPoints.size
    print('LEFT CHEEK Clipping Ratio :: ' + str(leftCheekClippedPixelRatio))
    rightCheekClippedPixelRatio = rightCheekPoints.size / unmaskedRightCheekPoints.size
    print('RIGHT CHEEK Clipping Ratio :: ' + str(rightCheekClippedPixelRatio))

    cutoff = 0.2
    if leftCheekClippedPixelRatio < cutoff:
        raise NameError('LEFT: Not enough clean non-clipped pixels. Ratio :: ' + str(leftCheekClippedPixelRatio))

    if rightCheekClippedPixelRatio < cutoff:
        raise NameError('RIGHT: Not enough clean non-clipped pixels. Ratio :: ' + str(rightCheekClippedPixelRatio))

    #img = cv2.resize(cheek_masked_image.astype('uint8'), (0, 0), fx=1/3, fy=1/3)
    #cv2.imshow('cheek masked', img)
    #cv2.waitKey(0)
    saveStep.saveReferenceImageBGR(maskedImage, capture.name + '_masked')

    return [np.array(maskedPoints), np.array(leftCheekPoints), np.array(rightCheekPoints)]

