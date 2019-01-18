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

def getAverageScreenReflectionColor(noFlashCapture, halfFlashCapture, fullFlashCapture, saveStep):
    [[noFlashLeftEyeCrop, noFlashLeftEyeMask, noFlashLeftEyeCoord], [noFlashRightEyeCrop, noFlashRightEyeMask, noFlashRightEyeCoord]] = getEyeCrops(noFlashCapture)
    [[halfFlashLeftEyeCrop, halfFlashLeftEyeMask, halfFlashLeftEyeCoord], [halfFlashRightEyeCrop, halfFlashRightEyeMask, halfFlashRightEyeCoord]] = getEyeCrops(halfFlashCapture)
    [[fullFlashLeftEyeCrop, fullFlashLeftEyeMask, fullFlashLeftEyeCoord], [fullFlashRightEyeCrop, fullFlashRightEyeMask, fullFlashRightEyeCoord]] = getEyeCrops(fullFlashCapture)

    #[noFlashLeftEyeRegionCrop, noFlashRightEyeRegionCrop] = getEyeRegionCrops(noFlashCapture)
    #[halfFlashLeftEyeRegionCrop, halfFlashRightEyeRegionCrop] = getEyeRegionCrops(halfFlashCapture)
    #[fullFlashLeftEyeRegionCrop, fullFlashRightEyeRegionCrop] = getEyeRegionCrops(fullFlashCapture)

    [leftEyeOffsets, [noFlashLeftEyeCrop, halfFlashLeftEyeCrop, fullFlashLeftEyeCrop]] = alignImages.cropAndAlignEyes(noFlashLeftEyeCrop, halfFlashLeftEyeCrop, fullFlashLeftEyeCrop)
    [rightEyeOffsets, [noFlashRightEyeCrop, halfFlashRightEyeCrop, fullFlashRightEyeCrop]] = alignImages.cropAndAlignEyes(noFlashRightEyeCrop, halfFlashRightEyeCrop, fullFlashRightEyeCrop)

    #[noFlashLeftEyeRegionCrop, halfFlashLeftEyeRegionCrop, fullFlashLeftEyeRegionCrop] = cropTools.cropImagesToOffsets([noFlashLeftEyeRegionCrop, halfFlashLeftEyeRegionCrop, fullFlashLeftEyeRegionCrop], np.array(leftEyeOffsets))
    #[noFlashLeftEyeRegionCrop, halfFlashLeftEyeRegionCrop, fullFlashLeftEyeRegionCrop] = [noFlashLeftEyeCrop, halfFlashLeftEyeCrop, fullFlashLeftEyeCrop]
    #[noFlashRightEyeRegionCrop, halfFlashRightEyeRegionCrop, fullFlashRightEyeRegionCrop] = cropTools.cropImagesToOffsets([noFlashRightEyeRegionCrop, halfFlashRightEyeRegionCrop, fullFlashRightEyeRegionCrop], np.array(rightEyeOffsets))
    #[noFlashRightEyeRegionCrop, halfFlashRightEyeRegionCrop, fullFlashRightEyeRegionCrop] = [noFlashRightEyeCrop, halfFlashRightEyeCrop, fullFlashRightEyeCrop]

    [noFlashLeftEyeMask, halfFlashLeftEyeMask, fullFlashLeftEyeMask] = cropTools.cropImagesToOffsets([noFlashLeftEyeMask, halfFlashLeftEyeMask, fullFlashLeftEyeMask], np.array(leftEyeOffsets))
    [noFlashRightEyeMask, halfFlashRightEyeMask, fullFlashRightEyeMask] = cropTools.cropImagesToOffsets([noFlashRightEyeMask, halfFlashRightEyeMask, fullFlashRightEyeMask], np.array(rightEyeOffsets))
    
    #leftEyeBB = fullFlashCapture.landmarks.getLeftEyeBB()
    #rightEyeBB = fullFlashCapture.landmarks.getRightEyeBB()

    #leftRemainder = np.abs((2 * halfFlashLeftEyeRegionCrop) - (fullFlashLeftEyeRegionCrop + noFlashLeftEyeRegionCrop))
    #rightRemainder = np.abs((2 * halfFlashRightEyeRegionCrop) - (fullFlashRightEyeRegionCrop + noFlashRightEyeRegionCrop))

    #leftRemainderMask = np.max(leftRemainder, axis=2) > 6
    #leftRemainderMask = leftRemainderMask.astype('uint8') * 255
    ##leftRemainderMask = np.stack((leftRemainderMask, leftRemainderMask, leftRemainderMask), axis=-1)
    #rightRemainderMask = np.max(rightRemainder, axis=2) > 6
    #rightRemainderMask = rightRemainderMask.astype('uint8') * 255
    ##rightRemainderMask = np.stack((rightRemainderMask, rightRemainderMask, rightRemainderMask), axis=-1)

    ##cv2.imshow('left linearity', np.hstack((leftRemainder, leftRemainderMask)))
    ##cv2.imshow('right linearity', np.hstack((rightRemainder, rightRemainderMask)))
    ##cv2.waitKey(0)
    #saveStep.saveReferenceImageBGR(leftRemainderMask, 'Left Remainder Eye Mask')
    #saveStep.saveReferenceImageBGR(rightRemainderMask, 'Right Remainder Eye Mask')

    halfFlashEyeStripCoords = np.array(halfFlashCapture.landmarks.getEyeStripBB())
    (x, y, w, h) = halfFlashEyeStripCoords
    halfFlashEyeStripXStart = x
    halfFlashEyeStripXEnd = x + w
    #halfFlashEyeStrip = fullFlashCapture.image[y:y+h, x:x+w]
    halfFlashEyeStrip = halfFlashCapture.image[y:y+h, x:x+w]

    eyeStripCoordDiff_left = np.array(halfFlashLeftEyeCoord) - halfFlashEyeStripCoords[0:2]
    eyeStripCoordDiff_right = np.array(halfFlashRightEyeCoord) - halfFlashEyeStripCoords[0:2]

    #(wb_x, wb_y) = halfFlashCapture.getAsShotWhiteBalance()
    #FOR REFLECITON
    [x, y, leftReflectionWidth, leftReflectionHeight] = maskReflectionBB(noFlashLeftEyeCrop, halfFlashLeftEyeCrop, fullFlashLeftEyeCrop)
    leftReflectionP1 = (x + eyeStripCoordDiff_left[0], y + eyeStripCoordDiff_left[1])
    leftReflectionP2 = (x + leftReflectionWidth + eyeStripCoordDiff_left[0], y + leftReflectionHeight + eyeStripCoordDiff_left[1])

    leftEyeReflection = halfFlashLeftEyeCrop[y:y+leftReflectionHeight, x:x+leftReflectionWidth]
    leftEyeMask = np.logical_not(halfFlashLeftEyeMask[y:y+leftReflectionHeight, x:x+leftReflectionWidth])


    #leftHighMask = np.max(leftEyeReflection, axis=2) < 253
    leftLowMask = np.min(leftEyeReflection, axis=2) >= 2
    leftEyeMask = np.logical_and(leftEyeMask, leftLowMask)

    leftEyePoints = leftEyeReflection[leftEyeMask]
    leftClipRatio = leftEyePoints.shape[0] / (leftEyeMask.shape[0] * leftEyeMask.shape[1])
    print('LEFT CLIP RATIO :: ' + str(leftClipRatio))

    leftReflectionMedian = np.median(leftEyePoints, axis=0) * 2 #Multiply by 2 because we got the value from the half flash
    #END FOR REFLECITON

    #FOR REFLECTION
    [x, y, rightReflectionWidth, rightReflectionHeight] = maskReflectionBB(noFlashRightEyeCrop, halfFlashRightEyeCrop, fullFlashRightEyeCrop)
    rightReflectionP1 = (x + eyeStripCoordDiff_right[0], y + eyeStripCoordDiff_right[1])
    rightReflectionP2 = (x + rightReflectionWidth + eyeStripCoordDiff_right[0], y + rightReflectionHeight + eyeStripCoordDiff_right[1])

    rightEyeReflection = halfFlashRightEyeCrop[y:y+rightReflectionHeight, x:x+rightReflectionWidth]
    rightEyeMask = np.logical_not(halfFlashRightEyeMask[y:y+rightReflectionHeight, x:x+rightReflectionWidth])

    #rightHighMask = np.max(rightEyeReflection, axis=2) < 253
    rightLowMask = np.min(rightEyeReflection, axis=2) >= 2
    rightEyeMask = np.logical_and(rightEyeMask, rightLowMask)

    rightEyePoints = rightEyeReflection[rightEyeMask]
    rightClipRatio = rightEyePoints.shape[0] / (rightEyeMask.shape[0] * rightEyeMask.shape[1])
    print('RIGHT CLIP RATIO :: ' + str(rightClipRatio))

    rightReflectionMedian = np.median(rightEyePoints, axis=0) * 2 #Multiply by 2 because we got the value from the half flash
    rightReflectionValue = np.max(rightReflectionMedian)
    #END FOR REFLECTION

    [leftRightPoint, leftLeftPoint] = fullFlashCapture.landmarks.getLeftEyeWidthPoints()
    [rightRightPoint, rightLeftPoint] = fullFlashCapture.landmarks.getRightEyeWidthPoints()

    (x, y, w, h) = halfFlashEyeStripCoords

    leftRightPoint -= [x, y]
    leftLeftPoint -= [x, y]

    rightRightPoint -= [x, y]
    rightLeftPoint -= [x, y]

    cv2.circle(halfFlashEyeStrip, (leftRightPoint[0], leftRightPoint[1]), 5, (0, 255, 0), -1)
    cv2.circle(halfFlashEyeStrip, (leftLeftPoint[0], leftLeftPoint[1]), 5, (0, 255, 0), -1)
    cv2.circle(halfFlashEyeStrip, (rightRightPoint[0], rightRightPoint[1]), 5, (0, 255, 0), -1)
    cv2.circle(halfFlashEyeStrip, (rightLeftPoint[0], rightLeftPoint[1]), 5, (0, 255, 0), -1)
    cv2.rectangle(halfFlashEyeStrip, leftReflectionP1, leftReflectionP2, (0, 0, 255), 1)
    cv2.rectangle(halfFlashEyeStrip, rightReflectionP1, rightReflectionP2, (0, 0, 255), 1)
    saveStep.saveReferenceImageBGR(halfFlashEyeStrip.astype('uint8'), 'eyeStrip')

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
    print('right reflection median :: ' + str(rightReflectionMedian))
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

    return [averageMedian, leftFluxish, rightFluxish]
