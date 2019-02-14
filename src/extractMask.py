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

def extractMask(capture, linearityError, saveStep):
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

    chinMask = mask.copy()
    foreheadMask = mask.copy()
    leftCheekMask = mask.copy()
    rightCheekMask = mask.copy()

    for polygon in polygons:
        hull = cv2.convexHull(polygon)
        mask = cv2.fillConvexPoly(mask, hull, 1)

    leftCheekHull = cv2.convexHull(leftCheek)
    leftCheekMask = cv2.fillConvexPoly(leftCheekMask, leftCheekHull, 1)

    rightCheekHull = cv2.convexHull(rightCheek)
    rightCheekMask = cv2.fillConvexPoly(rightCheekMask, rightCheekHull, 1)

    chinHull = cv2.convexHull(chin)
    chinMask = cv2.fillConvexPoly(chinMask, chinHull, 1)

    foreheadHull = cv2.convexHull(forehead)
    foreheadMask = cv2.fillConvexPoly(foreheadMask, foreheadHull, 1)

    mask = mask.astype('bool')
    #region_mask_point = image[mask]

    imageNoise = capture.calculateNoise()
    leftCheekMask = leftCheekMask.astype('bool')
    unmaskedLeftCheekPoints = image[leftCheekMask]
    leftCheekNoise = imageNoise[leftCheekMask]

    rightCheekMask = rightCheekMask.astype('bool')
    unmaskedRightCheekPoints = image[rightCheekMask]
    rightCheekNoise = imageNoise[rightCheekMask]

    chinMask = chinMask.astype('bool')
    unmaskedChinPoints = image[chinMask]
    chinNoise = imageNoise[chinMask]

    foreheadMask = foreheadMask.astype('bool')
    unmaskedForeheadPoints = image[foreheadMask]
    foreheadNoise = imageNoise[foreheadMask]

    leftNoise = np.mean(leftCheekNoise)
    rightNoise = np.mean(rightCheekNoise)
    chinNoise = np.mean(chinNoise)
    foreheadNoise = np.mean(foreheadNoise)

    print('NOISE :: Left :: {} | Right :: {} | Chin :: {} | Forehead :: {}'.format(leftNoise, rightNoise, chinNoise, foreheadNoise))

    if clippedMask is not None:
        mask = np.logical_and(mask, np.logical_not(clippedMask))
        leftCheekMask = np.logical_and(leftCheekMask, np.logical_not(clippedMask))
        rightCheekMask = np.logical_and(rightCheekMask, np.logical_not(clippedMask))
        chinMask = np.logical_and(chinMask, np.logical_not(clippedMask))
        foreheadMask = np.logical_and(foreheadMask, np.logical_not(clippedMask))

    maskedImage = np.where(mask[..., None], image, 0)
    maskedPoints = image[mask]

    leftCheekPoints = image[leftCheekMask]
    leftCheekMedianLinearityError = np.median(linearityError[leftCheekMask])

    rightCheekPoints = image[rightCheekMask]
    rightCheekMedianLinearityError = np.median(linearityError[rightCheekMask])

    chinPoints = image[chinMask]
    chinMedianLinearityError = np.median(linearityError[chinMask])

    foreheadPoints = image[foreheadMask]
    foreheadMedianLinearityError = np.median(linearityError[foreheadMask])

    linearityErrorString = "Left Cheek Linearity Error :: {} | Right Cheek Linearity Error :: {} | Chin Linearity Error :: {} | Forehead Linearity Error :: {}"
    print(linearityErrorString.format(leftCheekMedianLinearityError, rightCheekMedianLinearityError, chinMedianLinearityError, foreheadMedianLinearityError))

    #Base Clipping on cheeks for now...
    #clippedPixelRatio = (leftCheekPoints.size + rightCheekPoints.size) / (unmaskedLeftCheekPoints.size + unmaskedRightCheekPoints.size)
    leftCheekClippedPixelRatio = leftCheekPoints.size / unmaskedLeftCheekPoints.size
    print('LEFT CHEEK Clipping Ratio :: ' + str(leftCheekClippedPixelRatio))

    rightCheekClippedPixelRatio = rightCheekPoints.size / unmaskedRightCheekPoints.size
    print('RIGHT CHEEK Clipping Ratio :: ' + str(rightCheekClippedPixelRatio))

    chinClippedPixelRatio = chinPoints.size / unmaskedChinPoints.size
    print('CHIN Clipping Ratio :: ' + str(chinClippedPixelRatio))

    foreheadClippedPixelRatio = foreheadPoints.size / unmaskedForeheadPoints.size
    print('FOREHEAD Clipping Ratio :: ' + str(foreheadClippedPixelRatio))

    #cutoff = 0.2
    cutoff = 0.1
    #cutoff = 0.05
    if leftCheekClippedPixelRatio < cutoff:
        raise NameError('LEFT: Not enough clean non-clipped pixels. Ratio :: ' + str(leftCheekClippedPixelRatio))

    if rightCheekClippedPixelRatio < cutoff:
        raise NameError('RIGHT: Not enough clean non-clipped pixels. Ratio :: ' + str(rightCheekClippedPixelRatio))

    if chinClippedPixelRatio < cutoff:
        raise NameError('CHIN: Not enough clean non-clipped pixels. Ratio :: ' + str(chinClippedPixelRatio))

    if foreheadClippedPixelRatio < cutoff:
        raise NameError('FOREHEAD: Not enough clean non-clipped pixels. Ratio :: ' + str(foreheadClippedPixelRatio))

    #img = cv2.resize(cheek_masked_image.astype('uint8'), (0, 0), fx=1/3, fy=1/3)
    #cv2.imshow('cheek masked', img)
    #cv2.waitKey(0)
    saveStep.saveReferenceImageBGR(maskedImage.astype('uint8'), capture.name + '_masked')

    return [np.array(maskedPoints), [np.array(leftCheekPoints), leftCheekMedianLinearityError, leftCheekClippedPixelRatio], [np.array(rightCheekPoints), rightCheekMedianLinearityError, rightCheekClippedPixelRatio], [np.array(chinPoints), chinMedianLinearityError, chinClippedPixelRatio], [np.array(foreheadPoints), foreheadMedianLinearityError, foreheadClippedPixelRatio]]

