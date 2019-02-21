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

    return np.array([[leftEye, leftEyeMask, [lx, ly]], [rightEye, rightEyeMask, [rx, ry]]])

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

def maskReflectionBB2(noFlash, halfFlash, fullFlash):
    noFlashGrey = np.mean(noFlash, axis=2)
    halfFlashGrey = np.mean(halfFlash, axis=2)
    fullFlashGrey = np.mean(fullFlash, axis=2)

    #combined = np.hstack([noFlashGrey, halfFlashGrey, fullFlashGrey])

    #fullFlashGrey - halfFlashGrey - noFlashGrey

    #highDiff = np.abs(fullFlashGrey - halfFlashGrey)
    lowDiff = np.abs(2 * noFlashGrey - halfFlashGrey)

    #cv2.imshow('Low Diff', lowDiff.astype('uint8'))

    noFlashGreyClean = noFlashGrey - lowDiff
    halfFlashGreyClean = halfFlashGrey - lowDiff
    fullFlashGreyClean = fullFlashGrey - lowDiff

    #combinedClean = np.hstack([noFlashGreyClean, halfFlashGreyClean, fullFlashGreyClean])

    #beforeAndAfter = np.vstack([combined, combinedClean])
    #cv2.imshow('before and after', np.clip(beforeAndAfter, 0, 255).astype('uint8'))

    firstIncrease = np.clip(halfFlashGrey - noFlashGrey, 0, 255)
    secondIncrease = np.clip(fullFlashGrey - halfFlashGrey, 0, 255)

    cv2.imshow('first', firstIncrease.astype('uint8'))
    cv2.imshow('second', secondIncrease.astype('uint8'))


    cv2.waitKey(0)


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

    cv2.imshow('half flash grey Sub', halfFlashGrey.astype('uint8'))
    cv2.imshow('full flash grey Sub', fullFlashGrey.astype('uint8'))
    #cv2.waitKey(0)

    halfFlashGrey = erode(halfFlashGrey)
    fullFlashGrey = erode(fullFlashGrey)

    cv2.imshow('half flash grey Eroded', halfFlashGrey.astype('uint8'))
    cv2.imshow('full flash grey Eroded', fullFlashGrey.astype('uint8'))
    #cv2.imshow('no flash Grey Blurred', (np.clip(noFlashGrey, 0, 255)).astype('uint8'))
    #cv2.imshow('half flash Grey Blurred', (np.clip(halfFlashGrey, 0, 255)).astype('uint8'))
    #cv2.imshow('full flash Grey Blurred', (np.clip(fullFlashGrey, 0, 255)).astype('uint8'))
    cv2.waitKey(0)
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

def cropToBB(image, bb):
    [x, y, w, h] = bb
    return image[y:y+h, x:x+w]

def getAnnotatedEyeStrip(leftReflectionBB, leftOffsetCoords, rightReflectionBB, rightOffsetCoords, capture):
    eyeStripBB = np.array(capture.landmarks.getEyeStripBB())

    eyeWidthPoints = np.append(capture.landmarks.getLeftEyeWidthPoints(), capture.landmarks.getRightEyeWidthPoints(), axis=0)

    eyeWidthPoints -= eyeStripBB[0:2]
    leftOffsetCoords -= eyeStripBB[0:2]
    rightOffsetCoords -= eyeStripBB[0:2]

    leftReflectionP1 = leftOffsetCoords + leftReflectionBB[0:2]
    leftReflectionP2 = leftReflectionP1 + leftReflectionBB[2:4]
    leftReflectionP1 = tuple(leftReflectionP1)
    leftReflectionP2 = tuple(leftReflectionP2)

    rightReflectionP1 = rightOffsetCoords + rightReflectionBB[0:2]
    rightReflectionP2 = rightReflectionP1 + rightReflectionBB[2:4]
    rightReflectionP1 = tuple(rightReflectionP1)
    rightReflectionP2 = tuple(rightReflectionP2)

    eyeStrip = np.copy(cropToBB(capture.image.astype('uint8'), eyeStripBB))

    for [x, y] in eyeWidthPoints:
        cv2.circle(eyeStrip, (x, y), 5, (0, 255, 0), -1)

    cv2.rectangle(eyeStrip, leftReflectionP1, leftReflectionP2, (0, 0, 255), 1)
    cv2.rectangle(eyeStrip, rightReflectionP1, rightReflectionP2, (0, 0, 255), 1)

    return eyeStrip

#Note: both parent and child offsets should originally be measured to the same origin
def calculateRelativeOffset(parentOffset, childOffset):
    return childOffset[0:2] - parentOffset[0:2]

def calculateRepresentativeReflectionPoint(reflectionPoints):
    return np.median(reflectionPoints, axis=0) # Maybe change to only take median of top 10% of brightnesses?

def extractReflectionPoints(reflectionBB, eyeCrop, eyeMask):
    [x, y, w, h] = reflectionBB
    reflectionCrop = eyeCrop[y:y+h, x:x+w]
    reflectionMask = eyeMask[y:y+h, x:x+w]

    reflectionPoints = reflectionCrop[np.logical_not(reflectionMask)]

    if (reflectionMask.shape[0] == 0) or (reflectionMask.shape[1] == 0):
        raise NameError('Zero width eye reflection')

    cleanPixelRatio = reflectionPoints.shape[0] / (reflectionMask.shape[0] * reflectionMask.shape[1])

    representativeReflectionPoint = calculateRepresentativeReflectionPoint(reflectionPoints)

    if cleanPixelRatio < 0.8:
        raise NameError('Not enough clean non-clipped pixels in eye reflections')

    return [representativeReflectionPoint, cleanPixelRatio]

def getEyeWidth(capture):
    [leftP1, leftP2] = capture.landmarks.getLeftEyeWidthPoints()
    [rightP1, rightP2] = capture.landmarks.getRightEyeWidthPoints()

    leftEyeWidth = max(leftP1[0], leftP2[0]) - min(leftP1[0], leftP2[0])
    rightEyeWidth = max(rightP1[0], rightP2[0]) - min(rightP1[0], rightP2[0])

    return (leftEyeWidth + rightEyeWidth) / 2

def getAverageScreenReflectionColor(noFlashCapture, halfFlashCapture, fullFlashCapture, saveStep):
    [[noFlashLeftEyeCrop, noFlashLeftEyeMask, noFlashLeftEyeCoord], [noFlashRightEyeCrop, noFlashRightEyeMask, noFlashRightEyeCoord]] = getEyeCrops(noFlashCapture)
    [[halfFlashLeftEyeCrop, halfFlashLeftEyeMask, halfFlashLeftEyeCoord], [halfFlashRightEyeCrop, halfFlashRightEyeMask, halfFlashRightEyeCoord]] = getEyeCrops(halfFlashCapture)
    [[fullFlashLeftEyeCrop, fullFlashLeftEyeMask, fullFlashLeftEyeCoord], [fullFlashRightEyeCrop, fullFlashRightEyeMask, fullFlashRightEyeCoord]] = getEyeCrops(fullFlashCapture)

    print('Left No, Half Full :: {}, {}, {}'.format(noFlashLeftEyeCrop.shape, halfFlashLeftEyeCrop.shape, fullFlashLeftEyeCrop.shape))
    print('Right No, Half Full :: {}, {}, {}'.format(noFlashRightEyeCrop.shape, halfFlashRightEyeCrop.shape, fullFlashRightEyeCrop.shape))

    [leftEyeOffsets, [noFlashLeftEyeCrop, halfFlashLeftEyeCrop, fullFlashLeftEyeCrop]] = alignImages.cropAndAlignEyes(noFlashLeftEyeCrop, halfFlashLeftEyeCrop, fullFlashLeftEyeCrop)
    [rightEyeOffsets, [noFlashRightEyeCrop, halfFlashRightEyeCrop, fullFlashRightEyeCrop]] = alignImages.cropAndAlignEyes(noFlashRightEyeCrop, halfFlashRightEyeCrop, fullFlashRightEyeCrop)

    [noFlashLeftEyeMask, halfFlashLeftEyeMask, fullFlashLeftEyeMask] = cropTools.cropImagesToOffsets([noFlashLeftEyeMask, halfFlashLeftEyeMask, fullFlashLeftEyeMask], np.array(leftEyeOffsets))
    [noFlashRightEyeMask, halfFlashRightEyeMask, fullFlashRightEyeMask] = cropTools.cropImagesToOffsets([noFlashRightEyeMask, halfFlashRightEyeMask, fullFlashRightEyeMask], np.array(rightEyeOffsets))
    

    #leftEye = np.hstack([noFlashLeftEyeCrop, halfFlashLeftEyeCrop, fullFlashLeftEyeCrop])
    #rightEye = np.hstack([noFlashRightEyeCrop, halfFlashRightEyeCrop, fullFlashRightEyeCrop])
    ##cv2.imshow('left', leftEye.astype('uint8'))
    ##cv2.imshow('right', rightEye.astype('uint8'))
    ##cv2.waitKey(0)

    ##leftDiff = np.abs(fullFlashLeftEyeCrop - halfFlashLeftEyeCrop - noFlashLeftEyeCrop)
    #leftDiff = np.clip((-1) * (halfFlashLeftEyeCrop - (2 * noFlashLeftEyeCrop)), 0, 255)
    ##rightDiff = np.abs(fullFlashRightEyeCrop - halfFlashRightEyeCrop - noFlashRightEyeCrop)
    #rightDiff = np.clip(halfFlashRightEyeCrop - (2 * noFlashRightEyeCrop), 0, 255)
    #cv2.imshow('left diff', leftDiff.astype('uint8'))
    #cv2.imshow('right diff', rightDiff.astype('uint8'))

    #leftEyeSub = np.hstack([noFlashLeftEyeCrop - leftDiff, halfFlashLeftEyeCrop - leftDiff, fullFlashLeftEyeCrop - leftDiff])
    #rightEyeSub = np.hstack([noFlashRightEyeCrop - rightDiff, halfFlashRightEyeCrop - rightDiff, fullFlashRightEyeCrop - rightDiff])

    #leftDiffStack = np.vstack([leftEye, leftEyeSub])
    #rightDiffStack = np.vstack([rightEye, rightEyeSub])

    #cv2.imshow('left', np.clip(leftDiffStack, 0, 255).astype('uint8'))
    #cv2.imshow('right', np.clip(rightDiffStack, 0, 255).astype('uint8'))
    #cv2.waitKey(0)




    leftReflectionBB = maskReflectionBB2(noFlashLeftEyeCrop, halfFlashLeftEyeCrop, fullFlashLeftEyeCrop)
    rightReflectionBB = maskReflectionBB2(noFlashRightEyeCrop, halfFlashRightEyeCrop, fullFlashRightEyeCrop)

    halfFlashLeftEyeCoord += leftEyeOffsets[1]
    fullFlashLeftEyeCoord += leftEyeOffsets[2]

    halfFlashRightEyeCoord += rightEyeOffsets[1]
    fullFlashRightEyeCoord += rightEyeOffsets[2]

    halfFlashEyeStrip = getAnnotatedEyeStrip(leftReflectionBB, halfFlashLeftEyeCoord, rightReflectionBB, halfFlashRightEyeCoord, halfFlashCapture)
    fullFlashEyeStrip = getAnnotatedEyeStrip(leftReflectionBB, fullFlashLeftEyeCoord, rightReflectionBB, fullFlashRightEyeCoord, fullFlashCapture)

    saveStep.saveReferenceImageLinearBGR(halfFlashEyeStrip, 'eyeStrip_half')
    saveStep.saveReferenceImageLinearBGR(fullFlashEyeStrip, 'eyeStrip_full')

    halfLeftReflectionPoint, halfLeftCleanRatio = extractReflectionPoints(leftReflectionBB, halfFlashLeftEyeCrop, halfFlashLeftEyeMask)
    fullLeftReflectionPoint, fullLeftCleanRatio = extractReflectionPoints(leftReflectionBB, fullFlashLeftEyeCrop, fullFlashLeftEyeMask)

    halfRightReflectionPoint, halfRightCleanRatio = extractReflectionPoints(rightReflectionBB, halfFlashRightEyeCrop, halfFlashRightEyeMask)
    fullRightReflectionPoint, fullRightCleanRatio = extractReflectionPoints(rightReflectionBB, fullFlashRightEyeCrop, fullFlashRightEyeMask)

    print('LEFT CLEAN RATIO HALF VS FULL ::  {} | {} '.format(halfLeftCleanRatio, fullLeftCleanRatio))
    print('RIGHT CLEAN RATIO HALF VS FULL :: {} | {} '.format(halfRightCleanRatio, fullRightCleanRatio))

    averageHalfReflection = np.round((halfLeftReflectionPoint + halfRightReflectionPoint) / 2).astype('uint16')
    averageFullReflection = np.round((fullLeftReflectionPoint + fullRightReflectionPoint) / 2).astype('uint16')

    print('AVERAGE HALF REFLECTION :: ' + str(averageHalfReflection))
    print('AVERAGE FULL REFLECTION :: ' + str(averageFullReflection))

    leftHalfReflection = colorTools.whitebalanceBGRPoints(halfLeftReflectionPoint, averageHalfReflection)
    rightHalfReflection = colorTools.whitebalanceBGRPoints(halfRightReflectionPoint, averageHalfReflection)

    leftFullReflection = colorTools.whitebalanceBGRPoints(fullLeftReflectionPoint, averageFullReflection)
    rightFullReflection = colorTools.whitebalanceBGRPoints(fullRightReflectionPoint, averageFullReflection)

    leftHalfReflectionLuminance = colorTools.getRelativeLuminance([leftHalfReflection])[0]
    rightHalfReflectionLuminance = colorTools.getRelativeLuminance([rightHalfReflection])[0]

    leftFullReflectionLuminance = colorTools.getRelativeLuminance([leftFullReflection])[0]
    rightFullReflectionLuminance = colorTools.getRelativeLuminance([rightFullReflection])[0]

    print('left half reflection median :: ' + str(leftHalfReflection))
    print('left half reflection luminance :: ' + str(leftHalfReflectionLuminance))
    print('right half reflection median :: ' + str(rightHalfReflection))
    print('right half reflection luminance :: ' + str(rightHalfReflectionLuminance))

    print('left full reflection median :: ' + str(leftFullReflection))
    print('left full reflection luminance :: ' + str(leftFullReflectionLuminance))
    print('right full reflection median :: ' + str(rightFullReflection))
    print('right full reflection luminance :: ' + str(rightFullReflectionLuminance))

    #leftReflectionHLS = colorsys.rgb_to_hls(leftReflectionMedian[2] / 255, leftReflectionMedian[1] / 255, leftReflectionMedian[0] / 255)
    #rightReflectionHLS = colorsys.rgb_to_hls(rightReflectionMedian[2] / 255, rightReflectionMedian[1] / 255, rightReflectionMedian[0] / 255)

    #hueDiff = np.abs(leftReflectionHLS[0] - rightReflectionHLS[0])
    #satDiff = np.abs(leftReflectionHLS[2] - rightReflectionHLS[2])

    #print('HUE and SAT diff :: ' + str(hueDiff) + ' | ' + str(satDiff)) 

    eyeWidth = getEyeWidth(fullFlashCapture)

    if eyeWidth == 0:
        raise NameError('Zero value Eye Width')

    leftReflectionWidth, leftReflectionHeight = leftReflectionBB[2:4] / eyeWidth
    rightReflectionWidth, rightReflectionHeight = rightReflectionBB[2:4] / eyeWidth

    leftReflectionArea = leftReflectionWidth * leftReflectionHeight
    rightReflectionArea = rightReflectionWidth * rightReflectionHeight

    if min(leftReflectionWidth, rightReflectionWidth) == 0:
        raise NameError('Zero value reflection Width')

    if min(leftReflectionHeight, rightReflectionHeight) == 0:
        raise NameError('Zero value reflection Height')

    reflectionWidthRatio = max(leftReflectionWidth, rightReflectionWidth) / min(leftReflectionWidth, rightReflectionWidth)
    reflectionHeightRatio = max(leftReflectionHeight, rightReflectionHeight) / min(leftReflectionHeight, rightReflectionHeight)

    if (reflectionWidthRatio > 1.25) or (reflectionHeightRatio > 1.25):
        raise NameError('Reflection Sizes are too different!')

    leftFluxish = leftReflectionArea * leftHalfReflectionLuminance * 2 #2x because we are using half
    rightFluxish = rightReflectionArea * rightHalfReflectionLuminance * 2 #2x because we are using half

    print('LEFT FLUXISH :: ' + str(leftFluxish) + ' | AREA :: ' + str(leftReflectionArea) + ' | LUMINOSITY :: ' + str(leftHalfReflectionLuminance * 2))
    print('RIGHT FLUXISH :: ' + str(rightFluxish) + ' | AREA :: ' + str(rightReflectionArea) + ' | LUMINOSITY :: ' + str(rightHalfReflectionLuminance * 2))

    leftReflectionValues = [halfLeftReflectionPoint, fullLeftReflectionPoint]
    rightReflectionValues = [halfRightReflectionPoint, fullRightReflectionPoint]

    return [averageHalfReflection, leftHalfReflectionLuminance * 2, leftFluxish, rightHalfReflectionLuminance * 2, rightFluxish, leftReflectionValues, rightReflectionValues]
