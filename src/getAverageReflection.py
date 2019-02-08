import utils
import saveStep
import cv2
import numpy as np
import colorTools
import alignImages
import cropTools
import colorsys

def getEyeCrops(capture):
    (lx, ly, w, h) = capture.landmarks.getLeftEyeBB()
    leftEye = capture.image[ly:ly+h, lx:lx+w]
    leftEyeMask = capture.mask[ly:ly+h, lx:lx+w]

    (rx, ry, w, h) = capture.landmarks.getRightEyeBB()
    rightEye = capture.image[ry:ry+h, rx:rx+w]
    rightEyeMask = capture.mask[ry:ry+h, rx:rx+w]

    return [[leftEye, leftEyeMask, [lx, ly]], [rightEye, rightEyeMask, [rx, ry]]]

#def getEyeRegionCrops(capture):
#    (x, y, w, h) = capture.landmarks.getLeftEyeRegionBB()
#    leftEye = capture.image[y:y+h, x:x+w]
#
#    (x, y, w, h) = capture.landmarks.getRightEyeRegionBB()
#    rightEye = capture.image[y:y+h, x:x+w]
#
#    return [leftEye, rightEye]

def blur(img):
    return cv2.GaussianBlur(img, (9, 9), 0)
    #return cv2.bilateralFilter(img,15,75,75)
    #return cv2.medianBlur(img, 9)

def erode(img):
    kernel = np.ones((5, 5), np.uint16)
    #kernel = np.ones((9, 9), np.uint16)
    morph = cv2.morphologyEx(img, cv2.MORPH_CROSS, kernel)
    return morph

def getReflectionBB(mask):
    img = mask.astype('uint8') * 255
    im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    if not areas:
        print("NO REFLECTION FOUND")
        raise NameError('NO REFLECTION FOUND')

    max_index = np.argmax(areas)
    contour = contours[max_index]

    return cv2.boundingRect(contour)

def maskReflectionBB(noFlash, halfFlash, fullFlash):
    #noFlash = colorTools.whitebalance_from_asShot_to_d65(noFlash.astype('int32'), x, y)
    #halfFlash = colorTools.whitebalance_from_asShot_to_d65(halfFlash.astype('int32'), x, y)
    #fullFlash = colorTools.whitebalance_from_asShot_to_d65(fullFlash.astype('int32'), x, y)

    #cv2.imshow('half flash', halfFlash.astype('uint8'))
    #cv2.waitKey(0)
    #ogHalfFlash = np.sum(halfFlash, axis=2)
    #ogNoFlash = np.sum(noFlash, axis=2)

    #noFlashGrey = np.sum(blur(noFlash), axis=2)
    #halfFlashGrey = np.sum(blur(halfFlash), axis=2)
    #fullFlashGrey = np.sum(blur(fullFlash), axis=2)

    ogHalfFlash = colorTools.getRelativeLuminanceImage(halfFlash).astype('uint16')
    noFlashGrey = colorTools.getRelativeLuminanceImage(blur(noFlash)).astype('uint16')
    halfFlashGrey = colorTools.getRelativeLuminanceImage(blur(halfFlash)).astype('uint16')
    fullFlashGrey = colorTools.getRelativeLuminanceImage(blur(fullFlash)).astype('uint16')

    #cv2.imshow('half flash grey', halfFlashGrey.astype('uint8'))
    #cv2.imshow('full flash grey', fullFlashGrey.astype('uint8'))

    halfFlashGrey = np.clip(halfFlashGrey.astype('int32') - noFlashGrey, 0, np.iinfo(np.uint16).max).astype('uint16')
    fullFlashGrey = np.clip(fullFlashGrey.astype('int32') - noFlashGrey, 0, np.iinfo(np.uint16).max).astype('uint16')

    #cv2.imshow('half flash grey Sub', halfFlashGrey.astype('uint8'))
    #cv2.imshow('full flash grey Sub', fullFlashGrey.astype('uint8'))

    halfFlashGrey = erode(halfFlashGrey)
    fullFlashGrey = erode(fullFlashGrey)

    #cv2.imshow('half flash grey Eroded', halfFlashGrey.astype('uint8'))
    #cv2.imshow('full flash grey Eroded', fullFlashGrey.astype('uint8'))
    #cv2.imshow('no flash Grey Blurred', (np.clip(noFlashGrey, 0, 255)).astype('uint8'))
    #cv2.imshow('half flash Grey Blurred', (np.clip(halfFlashGrey, 0, 255)).astype('uint8'))
    #cv2.imshow('full flash Grey Blurred', (np.clip(fullFlashGrey, 0, 255)).astype('uint8'))
    #cv2.waitKey(0)
    #cv2.imshow('Grey Eroded', (fullFlashGrey / 3).astype('uint8'))

    noFlashMin = np.min(noFlashGrey)
    noFlashMask = noFlashGrey > (noFlashMin + 10)

    halfFlashMedian = np.median(halfFlashGrey)
    halfFlashStd = np.std(halfFlashGrey)
    halfFlashUpper = halfFlashMedian + (3 * halfFlashStd)
    halfFlashMask = halfFlashGrey > halfFlashUpper

    fullFlashMedian = np.median(fullFlashGrey)
    fullFlashStd = np.std(fullFlashGrey)
    fullFlashUpper = fullFlashMedian + (2 * fullFlashStd)
    fullFlashMask = fullFlashGrey > fullFlashUpper

    #cv2.imshow('half flash mask', halfFlashMask.astype('uint8') * 255)
    #cv2.imshow('full flash mask', fullFlashMask.astype('uint8') * 255)

    roughFlashMask = np.logical_and(halfFlashMask, fullFlashMask)
    roughFlashMask = np.logical_and(roughFlashMask, np.logical_not(noFlashMask))

    #cv2.imshow('rough flash mask', roughFlashMask.astype('uint8') * 255)
    #temp = halfFlash.copy()
    #temp[np.logical_not(roughFlashMask)] = [255, 0, 255]
    #cv2.imshow('...', temp.astype('uint8'))
    #cv2.waitKey(0)

    #cv2.imshow('rough flash mask', roughFlashMask.astype('uint8') * 255)

    #cv2.imshow('no flash mask', (noFlashMask.astype('uint8') * 255))
    #cv2.imshow('half flash mask', (halfFlashMask.astype('uint8') * 255))
    #cv2.imshow('full flash mask', (fullFlashMask.astype('uint8') * 255))

    #cv2.imshow('rough flash mask', (roughFlashMask.astype('uint8') * 255))
    (x, y, w, h) = getReflectionBB(roughFlashMask)
    roughReflectionCrop = ogHalfFlash[y:y+h, x:x+w]
    #cv2.imshow('rough flash crop', (roughReflectionCrop / 3).astype('uint8'))

    #averageValueByColumn = np.sum(roughReflectionCrop, axis=0) / roughReflectionCrop.shape[0]
    medianValueByColumn = np.median(roughReflectionCrop, axis=0)
    #np.flip(averageValueByColumn, 0)
    #averageValueByRow = np.sum(roughReflectionCrop, axis=1) / roughReflectionCrop.shape[1]
    medianValueByRow = np.median(roughReflectionCrop, axis=1)
    #np.flip(averageValueByRow, 0)


    tolerance = 0.4
    columnMedianCuttoff = np.median(medianValueByColumn)
    columnMedianCuttoff -= (tolerance * columnMedianCuttoff)

    rowMedianCuttoff = np.median(medianValueByRow)
    rowMedianCuttoff -= (tolerance * rowMedianCuttoff)

    #print('Median Value By Column :: ' + str(medianValueByColumn) + ' | ' + str(columnMedianCuttoff))
    #print('Median Value By Row :: ' + str(medianValueByRow) + ' | ' + str(rowMedianCuttoff))

    xMask = medianValueByColumn >= columnMedianCuttoff
    yMask = medianValueByRow >= rowMedianCuttoff

    #print('X Mask :: ' + str(xMask))
    #print('Y Mask :: ' + str(yMask))

    xMask = xMask.reshape(1, xMask.shape[0])
    yMask = yMask.reshape(yMask.shape[0], 1)

    refinedMask = np.dot(yMask, xMask)

    #cv2.imshow('rough flash crop', (roughReflectionCrop / 2).astype('uint8'))

    #temp = halfFlash.copy()
    #temp2 = halfFlash.copy()
    #temp[np.logical_not(refinedMask)] = [255, 0, 255]
    #temp2[refinedMask] = [0, 0, 0]
    #cv2.imshow('...', temp.astype('uint8'))
    #cv2.imshow('...', temp2.astype('uint8'))
    #cv2.waitKey(0)

    #refinedMask = roughReflectionCrop > 100
    (x1, y1, w1, h1) = getReflectionBB(refinedMask)
    refinedReflectionCrop = roughReflectionCrop[y1:y1+h1, x1:x1+w1]

    #cv2.imshow('refined flash crop', (refinedReflectionCrop / 3).astype('uint8'))
    #cv2.waitKey(0)
    return [(x + x1), (y + y1), w1, h1]


#def stretchBW(image):
#    #median = np.median(image)
#    #sd = np.std(image)
#    #lower = median - (3 * sd)
#    #lower = lower if lower > 0 else 0
#    #upper = median + (3 * sd)
#    #upper = upper if upper < 256 else 255
#
#    lower = np.min(image)
#    upper = np.max(image)
#
#    #print('MEDIAN :: ' + str(median))
#    #print('SD :: ' + str(sd))
#    #print('LOWER :: ' + str(lower))
#    #print('UPPER :: ' + str(upper))
#
#    #bounds = np.copy(gray)
#    #bounds[bounds < lower] = lower
#    #bounds[bounds > upper] = upper
#
#    numerator = (image - lower).astype('int32')
#    denominator = (upper - lower).astype('int32')
#    #stretched = (numerator.astype('int32') / denominator.astype('int32'))
#    stretched = (numerator / denominator)
#    #stretched = np.clip(stretched * 255, 0, 255).astype('uint8')
#    stretched = np.clip(stretched * 255, 0, 255).astype('uint8')
#    return stretched

#def getEyeWidths(fullFlashCapture, leftEyeOffsets, leftEyeGreyReflectionMask, rightEyeOffsets, rightEyeGreyReflectionMask):
#    halfFlashEyeStripCoords = np.array(fullFlashCapture.landmarks.getEyeStripBB())
#
#    eyeStripCoordDiff_left = np.array(fullFlashLeftEyeCoord) - halfFlashEyeStripCoords[0:2]
#    eyeStripCoordDiff_right = np.array(fullFlashRightEyeCoord) - halfFlashEyeStripCoords[0:2]
#
#    (x, y, w, h) = halfFlashEyeStripCoords
#    halfFlashEyeStripXStart = x
#    halfFlashEyeStripXEnd = x + w
#    halfFlashEyeStrip = fullFlashCapture.image[y:y+h, x:x+w]
#
#    eyeStripCoordDiff_left += leftEyeOffsets[2]
#    eyeStripCoordDiff_right += rightEyeOffsets[2]

def getAnnotatedEyeStrip(leftReflectionBB, rightReflectionBB, capture):
    x, y, w, h = np.array(capture.landmarks.getEyeStripBB())

    start_x = x
    end_x = x + w

    eyeStrip = capture.image[y:y+h, x:x+w]

    eyeStripCoordDiff_left = np.array(halfFlashLeftEyeCoord) - halfFlashEyeStripCoords[0:2]
    eyeStripCoordDiff_right = np.array(halfFlashRightEyeCoord) - halfFlashEyeStripCoords[0:2]

#Note: both parent and child offsets should originally be measured to the same origin
def calculateRelativeOffset(parentOffset, childOffset):
    return childOffset[0:2] - parentOffset[0:2]

def calculateRepresentativeReflectionPoint(reflectionPoints):
    return np.median(reflectionPoints, axis=0) # Maybe change to only take median of top 10% of brightnesses?

def extractReflectionPoints(reflectionBB, eyeCrop, eyeMask):
    [x, y, w, h] = reflectionBB
    reflectionCrop = eyeCrop[y:y+h, x:x+w]
    reflectionMask = eyeMask[y:y+h, x:x+w]

    reflectionPoints = reflectionCrop[reflectionMask]
    cleanPixelRatio = reflectionPoints.shape[0] / (reflectionMask.shape[0] * reflectionMask.shape[1])

    representativeReflectionPoint = calculateRepresentativeReflectionPoint(reflectionPoints)

    return [representativeReflectionPoint, cleanPixelRatio]


def getAverageScreenReflectionColor(noFlashCapture, halfFlashCapture, fullFlashCapture, saveStep):
    [[noFlashLeftEyeCrop, noFlashLeftEyeMask, noFlashLeftEyeCoord], [noFlashRightEyeCrop, noFlashRightEyeMask, noFlashRightEyeCoord]] = getEyeCrops(noFlashCapture)
    [[halfFlashLeftEyeCrop, halfFlashLeftEyeMask, halfFlashLeftEyeCoord], [halfFlashRightEyeCrop, halfFlashRightEyeMask, halfFlashRightEyeCoord]] = getEyeCrops(halfFlashCapture)
    [[fullFlashLeftEyeCrop, fullFlashLeftEyeMask, fullFlashLeftEyeCoord], [fullFlashRightEyeCrop, fullFlashRightEyeMask, fullFlashRightEyeCoord]] = getEyeCrops(fullFlashCapture)

    [leftEyeOffsets, [noFlashLeftEyeCrop, halfFlashLeftEyeCrop, fullFlashLeftEyeCrop]] = alignImages.cropAndAlignEyes(noFlashLeftEyeCrop, halfFlashLeftEyeCrop, fullFlashLeftEyeCrop)
    [rightEyeOffsets, [noFlashRightEyeCrop, halfFlashRightEyeCrop, fullFlashRightEyeCrop]] = alignImages.cropAndAlignEyes(noFlashRightEyeCrop, halfFlashRightEyeCrop, fullFlashRightEyeCrop)

    [noFlashLeftEyeMask, halfFlashLeftEyeMask, fullFlashLeftEyeMask] = cropTools.cropImagesToOffsets([noFlashLeftEyeMask, halfFlashLeftEyeMask, fullFlashLeftEyeMask], np.array(leftEyeOffsets))
    [noFlashRightEyeMask, halfFlashRightEyeMask, fullFlashRightEyeMask] = cropTools.cropImagesToOffsets([noFlashRightEyeMask, halfFlashRightEyeMask, fullFlashRightEyeMask], np.array(rightEyeOffsets))
    

    leftReflectionBB = maskReflectionBB(noFlashLeftEyeCrop, halfFlashLeftEyeCrop, fullFlashLeftEyeCrop)

    halfLeftPoint, halfLeftCleanRatio = extractReflectionPoints(leftReflectionBB, halfFlashLeftEyeCrop, halfFlashLeftEyeMask)
    fullLeftPoint, fullLeftCleanRatio = extractReflectionPoints(leftReflectionBB, fullFlashLeftEyeCrop, fullFlashLeftEyeMask)

    rightReflectionBB = maskReflectionBB(noFlashRightEyeCrop, halfFlashRightEyeCrop, fullFlashRightEyeCrop)

    halfRightPoint, halfRightCleanRatio = extractReflectionPoints(rightReflectionBB, halfFlashRightEyeCrop, halfFlashRightEyeMask)
    fullRightPoint, fullRightCleanRatio = extractReflectionPoints(rightReflectionBB, fullFlashRightEyeCrop, fullFlashRightEyeMask)

    print('LEFT CLEAN RATIO HALF VS FULL ::  {} | {} '.format(halfLeftCleanRatio, fullLeftCleanRatio))
    print('RIGHT CLEAN RATIO HALF VS FULL :: {} | {} '.format(halfRightCleanRatio, fullRightCleanRatio))


    [leftRightPoint, leftLeftPoint] = halfFlashCapture.landmarks.getLeftEyeWidthPoints()
    [rightRightPoint, rightLeftPoint] = halfFlashCapture.landmarks.getRightEyeWidthPoints()

    (x, y, w, h) = halfFlashEyeStripCoords

    leftRightPoint -= [x, y]
    leftLeftPoint -= [x, y]

    rightRightPoint -= [x, y]
    rightLeftPoint -= [x, y]

    halfFlashEyeStrip = halfFlashEyeStrip.astype('uint8').copy()
    cv2.circle(halfFlashEyeStrip, (leftRightPoint[0], leftRightPoint[1]), 5, (0, 255, 0), -1)
    cv2.circle(halfFlashEyeStrip, (leftLeftPoint[0], leftLeftPoint[1]), 5, (0, 255, 0), -1)
    cv2.circle(halfFlashEyeStrip, (rightRightPoint[0], rightRightPoint[1]), 5, (0, 255, 0), -1)
    cv2.circle(halfFlashEyeStrip, (rightLeftPoint[0], rightLeftPoint[1]), 5, (0, 255, 0), -1)
    cv2.rectangle(halfFlashEyeStrip, leftReflectionP1, leftReflectionP2, (0, 0, 255), 1)
    cv2.rectangle(halfFlashEyeStrip, rightReflectionP1, rightReflectionP2, (0, 0, 255), 1)
    saveStep.saveReferenceImageBGR(halfFlashEyeStrip, 'eyeStrip_half')

    if leftClipRatio < .8:
        print("TOO MUCH CLIPPING!")
        raise NameError('Not enough clean non-clipped pixels in left eye reflections')

    if rightClipRatio < .8:
        print("TOO MUCH CLIPPING!")
        raise NameError('Not enough clean non-clipped pixels in right eye reflections')

    leftEyeWidth = leftRightPoint[0] - leftLeftPoint[0]
    rightEyeWidth = rightRightPoint[0] - rightLeftPoint[0]

    #print('Left Eye Width :: ' + str(leftEyeWidth))
    #print('Right Eye Width :: ' + str(rightEyeWidth))


    averageEyeWidth = (leftEyeWidth + rightEyeWidth) / 2

    #maxEyeWidth = max([rightEyeWidth, leftEyeWidth])

    print('RIGHT EYE WIDTH :: ' + str(rightEyeWidth))
    print('LEFT EYE WIDTH :: ' + str(leftEyeWidth))
    print('AVERAGE EYE WIDTH :: ' + str(averageEyeWidth))
    #print('MAX EYE WIDTH :: ' + str(maxEyeWidth))

    #blur = 5
    #leftEyeSlitDiff = cv2.GaussianBlur(leftEyeSlitDiff, (blur, blur), 0)
    #rightEyeSlitDiff = cv2.GaussianBlur(rightEyeSlitDiff, (blur, blur), 0)

    #threshold = 64
    #leftEyeSlitDiff = (leftEyeSlitDiff > threshold).astype('uint8') * 255
    #rightEyeSlitDiff = (rightEyeSlitDiff > threshold).astype('uint8') * 255

    #leftEyeSlitStack = np.vstack((leftEyeSlitL.astype('uint8'), leftEyeSlitS.astype('uint8'), leftEyeSlitDiff1, leftEyeSlitDiff2, leftEyeSlitDiff3))
    #rightEyeSlitStack = np.vstack((rightEyeSlitL.astype('uint8'), rightEyeSlitS.astype('uint8'), rightEyeSlitDiff1, rightEyeSlitDiff2, rightEyeSlitDiff3))

    #cv2.imshow('Eye Mask Comparison', np.hstack((rightEyeSlitStack, leftEyeSlitStack)))
    #cv2.waitKey(0)

    #valuesDiff = np.abs((rightReflectionMedian - leftReflectionMedian))
    averageMedian = np.round((leftReflectionMedian + rightReflectionMedian) / 2).astype('uint16')
    print('AVERAGE MEDIAN :: ' + str(averageMedian))

    leftReflectionMedian = colorTools.whitebalanceBGRPoints(leftReflectionMedian, averageMedian)
    rightReflectionMedian = colorTools.whitebalanceBGRPoints(rightReflectionMedian, averageMedian)

    leftReflectionLuminance = colorTools.getRelativeLuminance([leftReflectionMedian])[0]
    rightReflectionLuminance = colorTools.getRelativeLuminance([rightReflectionMedian])[0]

    print('left reflection median :: ' + str(leftReflectionMedian))
    print('left reflection luminance :: ' + str(leftReflectionLuminance))
    print('right reflection median :: ' + str(rightReflectionMedian))
    print('right reflection luminance :: ' + str(rightReflectionLuminance))
    leftReflectionHLS = colorsys.rgb_to_hls(leftReflectionMedian[2] / 255, leftReflectionMedian[1] / 255, leftReflectionMedian[0] / 255)
    rightReflectionHLS = colorsys.rgb_to_hls(rightReflectionMedian[2] / 255, rightReflectionMedian[1] / 255, rightReflectionMedian[0] / 255)

    #print('rightReflectionMedian :: ' + str(rightReflectionMedian))
    #print('right HLS :: ' + str(rightReflectionHLS))
    #print('leftReflectionMedian :: ' + str(leftReflectionMedian))
    #print('left HLS :: ' + str(leftReflectionHLS))

    hueDiff = np.abs(leftReflectionHLS[0] - rightReflectionHLS[0])
    satDiff = np.abs(leftReflectionHLS[2] - rightReflectionHLS[2])

    print('HUE and SAT diff :: ' + str(hueDiff) + ' | ' + str(satDiff)) 



    leftReflectionArea = (leftReflectionWidth / averageEyeWidth) * (leftReflectionHeight / averageEyeWidth)
    rightReflectionArea = (rightReflectionWidth / averageEyeWidth) * (rightReflectionHeight / averageEyeWidth)

    reflectionWidthRatio = max(leftReflectionWidth, rightReflectionWidth) / min(leftReflectionWidth, rightReflectionWidth)
    reflectionHeightRatio = max(leftReflectionHeight, rightReflectionHeight) / min(leftReflectionHeight, rightReflectionHeight)

    #if (max(leftReflectionArea, rightReflectionArea) / min(leftReflectionArea, rightReflectionArea)) > 1.25:
    if (reflectionWidthRatio > 1.25) or (reflectionHeightRatio > 1.25):
        raise NameError('Reflection Sizes are too different!')

    #averageArea = (leftReflectionArea + rightReflectionArea) / 2

    #averageValue = (leftReflectionValue + rightReflectionValue) / 2
    #fluxish = averageArea * averageValue

    #leftReflectionLuminosity = leftReflectionHLS[1]
    #rightReflectionLuminosity = rightReflectionHLS[1]

    #leftFluxish = averageArea * leftReflectionLuminosity
    #leftFluxish = leftReflectionArea * leftReflectionLuminosity
    leftFluxish = leftReflectionArea * leftReflectionLuminance
    print('LEFT FLUXISH :: ' + str(leftFluxish) + ' | AREA :: ' + str(leftReflectionArea) + ' | LUMINOSITY :: ' + str(leftReflectionLuminance))

    #rightFluxish = averageArea * rightReflectionLuminosity
    #rightFluxish = rightReflectionArea * rightReflectionLuminosity
    rightFluxish = rightReflectionArea * rightReflectionLuminance
    print('RIGHT FLUXISH :: ' + str(rightFluxish) + ' | AREA :: ' + str(rightReflectionArea) + ' | LUMINOSITY :: ' + str(rightReflectionLuminance))

    return [averageMedian, leftReflectionLuminance, leftFluxish, rightReflectionLuminance, rightFluxish]
