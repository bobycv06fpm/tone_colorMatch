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

    (rx, ry, w, h) = capture.landmarks.getRightEyeBB()
    rightEye = capture.image[ry:ry+h, rx:rx+w]

    return [[leftEye, [lx, ly]], [rightEye, [rx, ry]]]

def getEyeRegionCrops(capture):
    (x, y, w, h) = capture.landmarks.getLeftEyeRegionBB()
    leftEye = capture.image[y:y+h, x:x+w]

    (x, y, w, h) = capture.landmarks.getRightEyeRegionBB()
    rightEye = capture.image[y:y+h, x:x+w]

    return [leftEye, rightEye]

def blur(img):
    return cv2.GaussianBlur(img, (9, 9), 0)
    #return cv2.bilateralFilter(img,15,75,75)
    #return cv2.medianBlur(img, 9)

def erode(img):
    kernel = np.ones((5, 5), np.uint8)
    morph = cv2.morphologyEx(img, cv2.MORPH_CROSS, kernel)
    return morph




def getReflectionBB(mask):
    img = mask.astype('uint8') * 255
    im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    contour = contours[max_index]

    return cv2.boundingRect(contour)

def maskReflectionBB(noFlash, halfFlash, fullFlash):
    ogHalfFlash = np.sum(halfFlash, axis=2)
    #ogNoFlash = np.sum(noFlash, axis=2)

    noFlashGrey = np.sum(blur(noFlash), axis=2)
    halfFlashGrey = np.sum(blur(halfFlash), axis=2)
    fullFlashGrey = np.sum(blur(fullFlash), axis=2)

    #cv2.imshow('Grey Blurred', (fullFlashGrey / 3).astype('uint8'))

    halfFlashGrey = np.clip(halfFlashGrey.astype('int32') - noFlashGrey, 0, (256 * 3))
    fullFlashGrey = np.clip(fullFlashGrey.astype('int32') - noFlashGrey, 0, (256 * 3))

    halfFlashGrey = erode(halfFlashGrey)
    fullFlashGrey = erode(fullFlashGrey)
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

    roughFlashMask = np.logical_and(halfFlashMask, fullFlashMask)
    roughFlashMask = np.logical_and(roughFlashMask, np.logical_not(noFlashMask))

    (x, y, w, h) = getReflectionBB(roughFlashMask)
    roughReflectionCrop = ogHalfFlash[y:y+h, x:x+w]

    averageValueByColumn = np.sum(roughReflectionCrop, axis=0) / roughReflectionCrop.shape[0]
    #np.flip(averageValueByColumn, 0)
    averageValueByRow = np.sum(roughReflectionCrop, axis=1) / roughReflectionCrop.shape[1]
    #np.flip(averageValueByRow, 0)


    tolerance = 0.4
    columnAverageCuttoff = np.median(averageValueByColumn)
    columnAverageCuttoff -= (tolerance * columnAverageCuttoff)

    rowAverageCuttoff = np.median(averageValueByRow)
    rowAverageCuttoff -= (tolerance * rowAverageCuttoff)

    #print('Average Value By Column :: ' + str(averageValueByColumn) + ' | ' + str(columnAverageCuttoff))
    #print('Average Value By Row :: ' + str(averageValueByRow) + ' | ' + str(rowAverageCuttoff))

    xMask = averageValueByColumn >= columnAverageCuttoff
    yMask = averageValueByRow >= rowAverageCuttoff

    #print('X Mask :: ' + str(xMask))
    #print('Y Mask :: ' + str(yMask))

    xMask = xMask.reshape(1, xMask.shape[0])
    yMask = yMask.reshape(yMask.shape[0], 1)

    refinedMask = np.dot(yMask, xMask)

    #print('Refined Mask :: ' + str(refinedMask))

    #refinedMask = roughReflectionCrop > 100
    (x1, y1, w1, h1) = getReflectionBB(refinedMask)
    refinedReflectionCrop = roughReflectionCrop[y1:y1+h1, x1:x1+w1]

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
#    fullFlashEyeStripCoords = np.array(fullFlashCapture.landmarks.getEyeStripBB())
#
#    eyeStripCoordDiff_left = np.array(fullFlashLeftEyeCoord) - fullFlashEyeStripCoords[0:2]
#    eyeStripCoordDiff_right = np.array(fullFlashRightEyeCoord) - fullFlashEyeStripCoords[0:2]
#
#    (x, y, w, h) = fullFlashEyeStripCoords
#    fullFlashEyeStripXStart = x
#    fullFlashEyeStripXEnd = x + w
#    fullFlashEyeStrip = fullFlashCapture.image[y:y+h, x:x+w]
#
#    eyeStripCoordDiff_left += leftEyeOffsets[2]
#    eyeStripCoordDiff_right += rightEyeOffsets[2]

def getAverageScreenReflectionColor(noFlashCapture, halfFlashCapture, fullFlashCapture, saveStep):
    [[noFlashLeftEyeCrop, noFlashLeftEyeCoord], [noFlashRightEyeCrop, noFlashRightEyeCoord]] = getEyeCrops(noFlashCapture)
    [[halfFlashLeftEyeCrop, halfFlashLeftEyeCoord], [halfFlashRightEyeCrop, halfFlashRightEyeCoord]] = getEyeCrops(halfFlashCapture)
    [[fullFlashLeftEyeCrop, fullFlashLeftEyeCoord], [fullFlashRightEyeCrop, fullFlashRightEyeCoord]] = getEyeCrops(fullFlashCapture)

    [noFlashLeftEyeRegionCrop, noFlashRightEyeRegionCrop] = getEyeRegionCrops(noFlashCapture)
    [halfFlashLeftEyeRegionCrop, halfFlashRightEyeRegionCrop] = getEyeRegionCrops(halfFlashCapture)
    [fullFlashLeftEyeRegionCrop, fullFlashRightEyeRegionCrop] = getEyeRegionCrops(fullFlashCapture)

    [leftEyeOffsets, [noFlashLeftEyeCrop, halfFlashLeftEyeCrop, fullFlashLeftEyeCrop]] = alignImages.cropAndAlignEyes(noFlashLeftEyeCrop, halfFlashLeftEyeCrop, fullFlashLeftEyeCrop)
    [rightEyeOffsets, [noFlashRightEyeCrop, halfFlashRightEyeCrop, fullFlashRightEyeCrop]] = alignImages.cropAndAlignEyes(noFlashRightEyeCrop, halfFlashRightEyeCrop, fullFlashRightEyeCrop)

    [noFlashLeftEyeRegionCrop, halfFlashLeftEyeRegionCrop, fullFlashLeftEyeRegionCrop] = cropTools.cropImagesToOffsets([noFlashLeftEyeRegionCrop, halfFlashLeftEyeRegionCrop, fullFlashLeftEyeRegionCrop], np.array(leftEyeOffsets))
    #[noFlashLeftEyeRegionCrop, halfFlashLeftEyeRegionCrop, fullFlashLeftEyeRegionCrop] = [noFlashLeftEyeCrop, halfFlashLeftEyeCrop, fullFlashLeftEyeCrop]
    [noFlashRightEyeRegionCrop, halfFlashRightEyeRegionCrop, fullFlashRightEyeRegionCrop] = cropTools.cropImagesToOffsets([noFlashRightEyeRegionCrop, halfFlashRightEyeRegionCrop, fullFlashRightEyeRegionCrop], np.array(rightEyeOffsets))
    #[noFlashRightEyeRegionCrop, halfFlashRightEyeRegionCrop, fullFlashRightEyeRegionCrop] = [noFlashRightEyeCrop, halfFlashRightEyeCrop, fullFlashRightEyeCrop]

    
    leftEyeBB = fullFlashCapture.landmarks.getLeftEyeBB()
    rightEyeBB = fullFlashCapture.landmarks.getRightEyeBB()

    leftRemainder = np.clip(np.abs((2 * halfFlashLeftEyeRegionCrop.astype('int32')) - (fullFlashLeftEyeRegionCrop.astype('int32') + noFlashLeftEyeRegionCrop.astype('int32'))), 0, 255).astype('uint8')
    rightRemainder = np.clip(np.abs((2 * halfFlashRightEyeRegionCrop.astype('int32')) - (fullFlashRightEyeRegionCrop.astype('int32') + noFlashRightEyeRegionCrop.astype('int32'))), 0, 255).astype('uint8')

    leftRemainderMask = np.max(leftRemainder, axis=2) > 6
    leftRemainderMask = leftRemainderMask.astype('uint8') * 255
    #leftRemainderMask = np.stack((leftRemainderMask, leftRemainderMask, leftRemainderMask), axis=-1)
    rightRemainderMask = np.max(rightRemainder, axis=2) > 6
    rightRemainderMask = rightRemainderMask.astype('uint8') * 255
    #rightRemainderMask = np.stack((rightRemainderMask, rightRemainderMask, rightRemainderMask), axis=-1)

    #cv2.imshow('left linearity', np.hstack((leftRemainder, leftRemainderMask)))
    #cv2.imshow('right linearity', np.hstack((rightRemainder, rightRemainderMask)))
    #cv2.waitKey(0)
    saveStep.saveReferenceImageBGR(leftRemainderMask, 'Left Remainder Eye Mask')
    saveStep.saveReferenceImageBGR(rightRemainderMask, 'Right Remainder Eye Mask')

    fullFlashEyeStripCoords = np.array(fullFlashCapture.landmarks.getEyeStripBB())
    (x, y, w, h) = fullFlashEyeStripCoords
    fullFlashEyeStripXStart = x
    fullFlashEyeStripXEnd = x + w
    #fullFlashEyeStrip = fullFlashCapture.image[y:y+h, x:x+w]
    fullFlashEyeStrip = halfFlashCapture.image[y:y+h, x:x+w]

    eyeStripCoordDiff_left = np.array(fullFlashLeftEyeCoord) - fullFlashEyeStripCoords[0:2]
    eyeStripCoordDiff_right = np.array(fullFlashRightEyeCoord) - fullFlashEyeStripCoords[0:2]

    #FOR REFLECITON
    [x, y, leftReflectionWidth, leftReflectionHeight] = maskReflectionBB(noFlashLeftEyeCrop, halfFlashLeftEyeCrop, fullFlashLeftEyeCrop)
    leftReflectionP1 = (x + eyeStripCoordDiff_left[0], y + eyeStripCoordDiff_left[1])
    leftReflectionP2 = (x + leftReflectionWidth + eyeStripCoordDiff_left[0], y + leftReflectionHeight + eyeStripCoordDiff_left[1])

    leftEyeReflection = halfFlashLeftEyeCrop[y:y+leftReflectionHeight, x:x+leftReflectionHeight]

    print('LEFT EYE REFLECTION :: ' + str(leftEyeReflection))

    leftHighMask = np.max(leftEyeReflection, axis=2) < 253
    #leftLowMask = np.min(leftEyeReflection, axis=2) >= 2
    leftLowMask = np.max(leftEyeReflection, axis=2) >= 2

    leftEyeMask = np.logical_and(leftHighMask, leftLowMask)
    leftEyePoints = leftEyeReflection[leftEyeMask]
    leftClipRatio = leftEyePoints.shape[0] / (leftEyeMask.shape[0] * leftEyeMask.shape[1])
    print('LEFT CLIP RATIO :: ' + str(leftClipRatio))
    if leftClipRatio < .9:
        print("TOO MUCH CLIPPING!")
        raise NameError('Not enough clean non-clipped pixels in left eye reflections')

    leftReflectionMedian = np.median(leftEyePoints, axis=0) * 2 #Multiply by 2 because we got the value from the half flash
    #END FOR REFLECITON

    #FOR REFLECTION
    [x, y, rightReflectionWidth, rightReflectionHeight] = maskReflectionBB(noFlashRightEyeCrop, halfFlashRightEyeCrop, fullFlashRightEyeCrop)
    rightReflectionP1 = (x + eyeStripCoordDiff_right[0], y + eyeStripCoordDiff_right[1])
    rightReflectionP2 = (x + rightReflectionWidth + eyeStripCoordDiff_right[0], y + rightReflectionHeight + eyeStripCoordDiff_right[1])

    rightEyeReflection = halfFlashRightEyeCrop[y:y+rightReflectionHeight, x:x+rightReflectionWidth]

    rightHighMask = np.max(rightEyeReflection, axis=2) < 253
    rightLowMask = np.min(rightEyeReflection, axis=2) >= 2

    rightEyeMask = np.logical_and(rightHighMask, rightLowMask)
    rightEyePoints = rightEyeReflection[rightEyeMask]
    rightClipRatio = rightEyePoints.shape[0] / (rightEyeMask.shape[0] * rightEyeMask.shape[1])
    print('RIGHT CLIP RATIO :: ' + str(rightClipRatio))
    if rightClipRatio < .9:
        print("TOO MUCH CLIPPING!")
        raise NameError('Not enough clean non-clipped pixels in right eye reflections')

    rightReflectionMedian = np.median(rightEyePoints, axis=0) * 2 #Multiply by 2 because we got the value from the half flash
    rightReflectionValue = np.max(rightReflectionMedian)
    #END FOR REFLECTION
#
#    
#    rightEyeX = eyeStripCoordDiff_right[0]
#    eyeDistance = int((leftEyeX - rightEyeX) / 3)
#
#    print('Eyes left, right :: ' + str(leftEyeX) + ' ' + str(rightEyeX))
#    leftEyeLeftEdge = (leftEyeX + eyeDistance) if (leftEyeX + eyeDistance) < fullFlashEyeStrip.shape[1] else fullFlashEyeStrip.shape[1]
#    leftEyeRightEdge = (leftEyeX - eyeDistance) if (leftEyeX - eyeDistance) > 0 else 0
#
#    rightEyeLeftEdge = (rightEyeX + eyeDistance) if (rightEyeX + eyeDistance) < fullFlashEyeStrip.shape[1] else fullFlashEyeStrip.shape[1]
#    rightEyeRightEdge = (rightEyeX - eyeDistance) if (rightEyeX - eyeDistance) > 0 else 0
#    #print('Edges left, right :: ' + str(leftEdge) + ' ' + str(rightEdge))
#
#    #eyeSlitCrop = fullFlashEyeStrip[eyeSlitTop - 50:eyeSlitBottom + 50, rightEdge:leftEdge]
#
#    #getEyeWidths(fullFlashCapture, leftEyeOffsets[2], leftEyeGreyReflectionMask, rightEyeOffsets[2], rightEyeGreyReflectionMask)
#
#    print('Eye Slip Top :: ' + str(eyeSlitTop))
#    print('Eye Slip Bottom :: ' + str(eyeSlitBottom))
#
#    print('Left Eye Right Edge :: ' + str(leftEyeRightEdge))
#    print('Left Eye Left Edge :: ' + str(leftEyeLeftEdge))
#
#    print('Right Eye Right Edge :: ' + str(rightEyeRightEdge))
#    print('Right Eye Left Edge :: ' + str(rightEyeLeftEdge))
#
#    margin = 30
#    #marginMultiplier = np.array([-1, -1, 2, 2])
#    marginMultiplier = np.array([-1, 0, 2, 0])
#    marginValues = margin * marginMultiplier
#
#    #[[leftEyeLeftX, leftEyeLeftY], [leftEyeRightX, leftEyeRightY]] = leftEyeWidthPoints
#
#    #print('Left Eye Width Points :: ' + str(leftEyeWidthPoints))
#    #print('Right Eye Width Points :: ' + str(rightEyeWidthPoints))
#
#    #leftEyeStripCoords = [leftEyeLeftX - margin, leftEyeLeftY - margin]
#    #leftEyeSlit = np.copy(fullFlashEyeStrip[eyeSlitTop - margin:eyeSlitBottom + margin, leftEyeLeftEdge:leftEyeRightEdge])
#
#    #print('X From -> To :: ' + str(leftEyeRightX - margin) + ' -> ' + str(leftEyeLeftX + margin))
#    #print('Y From -> To :: ' + str(leftEyeLeftY - margin) + ' -> ' + str(leftEyeLeftY + margin))
#
#    leftEyeBB += marginValues
#
#    print('Total Width :: ' + str(fullFlashEyeStrip.shape))
#    print('Full Flash Eye Strip :: ' + str(fullFlashEyeStrip.shape))
#    print('Left BB :: ' + str(leftEyeBB))
#    #leftEyeSlit = np.copy(fullFlashEyeStrip[min(leftEyeLeftY, leftEyeRightY) - margin:max(leftEyeLeftY, leftEyeRightY) + margin, leftEyeRightX - margin:leftEyeLeftX + margin])
#    leftEyeSlit = np.copy(fullFlashEyeStrip[leftEyeBB[1]:leftEyeBB[1] + leftEyeBB[3], leftEyeBB[0]:leftEyeBB[0] + leftEyeBB[2]])
#    leftEyeSlitMiddle = int(leftEyeSlit.shape[1]/2)
#    leftEyeSlit[:, (leftEyeSlitMiddle - int(leftEyeSlitMiddle / 3)):(leftEyeSlitMiddle + int(leftEyeSlitMiddle / 3))] = 0
#
#    cv2.imshow('left eye slit', leftEyeSlit)
#
#
#    #[[rightEyeLeftX, rightEyeLeftY], [rightEyeRightX, rightEyeRightY]] = rightEyeWidthPoints
#    rightEyeBB += marginValues
#    print('Right BB :: ' + str(rightEyeBB))
#
#    rightEyeStripCoords = [rightEyeLeftEdge, eyeSlitTop - margin]
#    #rightEyeSlit = np.copy(fullFlashEyeStrip[eyeSlitTop - margin:eyeSlitBottom + margin, rightEyeLeftEdge:rightEyeRightEdge])
#    #rightEyeSlit = np.copy(fullFlashEyeStrip[min(rightEyeLeftY, rightEyeRightY) - margin:max(rightEyeLeftY, rightEyeRightY) + margin, rightEyeRightX - margin:rightEyeLeftX + margin])
#    rightEyeSlit = np.copy(fullFlashEyeStrip[rightEyeBB[1]:rightEyeBB[1] + rightEyeBB[3], rightEyeBB[0]:rightEyeBB[0] + rightEyeBB[2]])
#    rightEyeSlitMiddle = int(rightEyeSlit.shape[1]/2)
#    rightEyeSlit[:, (rightEyeSlitMiddle - int(rightEyeSlitMiddle / 3)):(rightEyeSlitMiddle + int(rightEyeSlitMiddle / 3))] = 0
#
#    cv2.imshow('left eye slit', leftEyeSlit)
#    cv2.imshow('right eye slit', rightEyeSlit)
#    cv2.waitKey(0)
#
#
#    #eyeSlitCrop = fullFlashEyeStrip[eyeSlitTop - 50:eyeSlitBottom + 50, rightEdge:leftEdge]
#    #eyeSlitCrop = fullFlashEyeStrip[eyeSlitTop - 50:eyeSlitBottom + 50, rightEyeX:leftEyeX]
#
#
#    #eyeSlitCrop = cv2.GaussianBlur(eyeSlitCrop, (51, 51), 0)
#    leftEyeSlitHLS = cv2.cvtColor(leftEyeSlit, cv2.COLOR_BGR2HLS_FULL)
#    rightEyeSlitHLS = cv2.cvtColor(rightEyeSlit, cv2.COLOR_BGR2HLS_FULL)
#
#    #print('Eye Slit HLS :: ' + str(eyeSlitHLS))
#    #eyeSlitH = eyeSlitHLS[:, :, 0]
#    #eyeSlitH = cv2.Sobel(eyeSlitH, cv2.CV_64F, 1, 1, ksize=5)
#    #eyeSlitH = (eyeSlitH < 26).astype('int32') * 255
#    #eyeSlitHOriginal = eyeSlitHLS[:, :, 0]
#    #eyeSlitH = np.copy(eyeSlitHOriginal).astype('int32')
#    #eyeSlitHigh = np.clip(eyeSlitH - 85, 0, 255) #Shift Hue by 1/3 of range to move red/yellow away from Hue boundary
#    #eyeSlitLow = np.clip(eyeSlitH + 170, 0, 255)
#
#    #eyeSlitH[eyeSlitHOriginal >= 85] = eyeSlitHigh[eyeSlitHOriginal >= 85]
#    #eyeSlitH[eyeSlitHOriginal < 85] = eyeSlitLow[eyeSlitHOriginal < 85]
#
#    #hueBound = 230
#    leftEyeSlitH = (leftEyeSlitHLS[:, :, 0]).astype('int32')
#    #leftEyeSlitH[leftEyeSlitH > hueBound] = 0
#    #leftEyeSlitH = stretchBW(leftEyeSlitH)
#    rightEyeSlitH = (rightEyeSlitHLS[:, :, 0]).astype('int32')
#    #rightEyeSlitH[rightEyeSlitH > hueBound] = 0
#    #rightEyeSlitH = stretchBW(rightEyeSlitH)
#
#
#
#    leftEyeSlitL = (leftEyeSlitHLS[:, :, 1]).astype('int32')
#    rightEyeSlitL = (rightEyeSlitHLS[:, :, 1]).astype('int32')
#    leftEyeSlitS = (leftEyeSlitHLS[:, :, 2]).astype('int32')
#    rightEyeSlitS = (rightEyeSlitHLS[:, :, 2]).astype('int32')
#
#    leftEyeSlitDiff = np.clip(leftEyeSlitS - leftEyeSlitL, 0, 255).astype('uint8')
#    rightEyeSlitDiff = np.clip(rightEyeSlitS - rightEyeSlitL, 0, 255).astype('uint8')
#    #eyeSlitTest = cv2.GaussianBlur(eyeSlitTest, (51, 51), 0)
#
#    #med = np.median(eyeSlitTest)
#    #sd = np.std(eyeSlitTest)
#
#    #lowerbound = med #+ (2 * sd)
#
#    #eyeSlitTest4 = (eyeSlitTest > lowerbound).astype('uint8') * 255
#
#
#    #eyeSlitTest2 = cv2.Sobel(eyeSlitTest, cv2.CV_16U, 1, 1, ksize=5)
#    #eyeSlitTest3 = cv2.GaussianBlur(eyeSlitTest2, (5, 5), 0)
#
#    #cv2.imshow('Eye Slit Crop H', eyeSlitH.astype('uint8'))
#    #cv2.imshow('Eye Slit Crop L', eyeSlitL.astype('uint8'))
#    #cv2.imshow('Eye Slit Crop S', eyeSlitS.astype('uint8'))
#    #cv2.imshow('Eye Slit Crop Test', eyeSlitTest.astype('uint8'))
#   # cv2.imshow('Eye Slit Crop Test 2', eyeSlitTest2.astype('uint8'))
#   # cv2.imshow('Eye Slit Crop Test 3', eyeSlitTest3.astype('uint8'))
#    #cv2.imshow('Eye Slit Crop Test 4', eyeSlitTest4.astype('uint8'))
#
#
#    #cv2.imshow('left no stretch', leftEyeSlitDiff)
#    #cv2.imshow('right no stretch', rightEyeSlitDiff)
#
#    leftEyeSlitDiff1 = stretchBW(leftEyeSlitDiff)
#    rightEyeSlitDiff1 = stretchBW(rightEyeSlitDiff)
#    
#    #margin = 20
#    #leftCenter = int(leftEyeSlitDiff.shape[0] / 2)
#    #cv2.line(leftEyeSlitDiff, (0, leftCenter), (leftEyeSlitDiff.shape[1], leftCenter), (255, 255, 255))
#    #cv2.line(leftEyeSlitDiff, (0, leftCenter + margin), (leftEyeSlitDiff.shape[1], leftCenter + margin), (255, 255, 255))
#    #cv2.line(leftEyeSlitDiff, (0, leftCenter - margin), (leftEyeSlitDiff.shape[1], leftCenter - margin), (255, 255, 255))
#
#    #rightCenter = int(rightEyeSlitDiff.shape[0] / 2)
#    #cv2.line(rightEyeSlitDiff, (0, rightCenter), (rightEyeSlitDiff.shape[1], rightCenter), (255, 255, 255))
#    #cv2.line(rightEyeSlitDiff, (0, rightCenter + margin), (rightEyeSlitDiff.shape[1], rightCenter + margin), (255, 255, 255))
#    #cv2.line(rightEyeSlitDiff, (0, rightCenter - margin), (rightEyeSlitDiff.shape[1], rightCenter - margin), (255, 255, 255))
#
#    #cv2.imshow('Left Slit Crop', leftEyeSlitDiff1)
#    #cv2.imshow('Right Slit Crop', rightEyeSlitDiff1)
#
#    threshold = 20
#    leftEyeSlitDiff2 = (leftEyeSlitDiff1 > threshold).astype('uint8') * 255
#    rightEyeSlitDiff2 = (rightEyeSlitDiff1 > threshold).astype('uint8') * 255
#
#    #cv2.imshow('Left Slit Crop mask', leftEyeSlitDiff)
#    #cv2.imshow('Right Slit Crop mask', rightEyeSlitDiff)
#
#    #kernel = np.ones((7, 5), np.uint8)
#    kernel = np.ones((3, 5), np.uint8)
#    #kernel = np.ones((7, 3), np.uint8)
#    #algo = cv2.MORPH_CLOSE
#
#    leftEyeSlitDiff3 = leftEyeSlitDiff2
#    rightEyeSlitDiff3 = rightEyeSlitDiff2
#
#    leftEyeSlitDiff3 = cv2.morphologyEx(leftEyeSlitDiff3, cv2.MORPH_OPEN, kernel)
#    rightEyeSlitDiff3 = cv2.morphologyEx(rightEyeSlitDiff3, cv2.MORPH_OPEN, kernel)
#
#    algo = cv2.MORPH_CROSS
#    leftEyeSlitDiff3 = cv2.morphologyEx(leftEyeSlitDiff3, algo, kernel)
#    rightEyeSlitDiff3 = cv2.morphologyEx(rightEyeSlitDiff3, algo, kernel)
#
#    margin = 40
#    leftCenter = int(leftEyeSlitDiff.shape[0] / 2)
##    cv2.line(leftEyeSlitDiff3, (0, leftCenter), (leftEyeSlitDiff3.shape[1], leftCenter), (255, 255, 255))
##    cv2.line(leftEyeSlitDiff3, (0, leftCenter + margin), (leftEyeSlitDiff3.shape[1], leftCenter + margin), (255, 255, 255))
##    cv2.line(leftEyeSlitDiff3, (0, leftCenter - margin), (leftEyeSlitDiff3.shape[1], leftCenter - margin), (255, 255, 255))
##
#    rightCenter = int(rightEyeSlitDiff.shape[0] / 2)
##    cv2.line(rightEyeSlitDiff3, (0, rightCenter), (rightEyeSlitDiff3.shape[1], rightCenter), (255, 255, 255))
##    cv2.line(rightEyeSlitDiff3, (0, rightCenter + margin), (rightEyeSlitDiff3.shape[1], rightCenter + margin), (255, 255, 255))
##    cv2.line(rightEyeSlitDiff3, (0, rightCenter - margin), (rightEyeSlitDiff3.shape[1], rightCenter - margin), (255, 255, 255))
#
#    leftEyeRows = leftEyeSlitDiff3[leftCenter - margin:leftCenter + margin]
#    leftEyeRows = leftEyeRows == 0
#    rightEyeRows = rightEyeSlitDiff3[rightCenter - margin:rightCenter + margin]
#    rightEyeRows = rightEyeRows == 0
#
#    #cv2.imshow('left eye rows', leftEyeRows.astype('uint8')*255)
#    #cv2.imshow('right eye rows', rightEyeRows.astype('uint8')*255)
#
#
#    leftEyeReflectionBB = getReflectionBB(leftEyeRows)
#    cv2.line(leftEyeSlitDiff1, (leftEyeReflectionBB[0], leftEyeReflectionBB[1]), (leftEyeReflectionBB[0], leftEyeReflectionBB[1] + leftEyeReflectionBB[3]), (255, 255, 255))
#    cv2.line(leftEyeSlitDiff1, (leftEyeReflectionBB[0] + leftEyeReflectionBB[2], leftEyeReflectionBB[1]), (leftEyeReflectionBB[0] + leftEyeReflectionBB[2], leftEyeReflectionBB[1] + leftEyeReflectionBB[3]), (255, 255, 255))
#    print('left eye bb :: ' + str(leftEyeReflectionBB))
#    if (leftEyeReflectionBB[0] == 0) or ((leftEyeReflectionBB[0] + leftEyeReflectionBB[2]) == leftEyeRows.shape[1]):
#        leftEyeWidth = 0
#    else:
#        leftEyeWidth = leftEyeReflectionBB[2]
#
    #leftRightPoint = np.array(leftEyeStripCoords) + np.array([leftEyeReflectionBB[0], leftEyeReflectionBB[1] + leftEyeReflectionBB[3]])
    #leftLeftPoint = np.array(leftEyeStripCoords) + np.array([leftEyeReflectionBB[0] + leftEyeReflectionBB[2], leftEyeReflectionBB[1] + leftEyeReflectionBB[3]])
    [leftRightPoint, leftLeftPoint] = fullFlashCapture.landmarks.getLeftEyeWidthPoints()

#    rightEyeReflectionBB = getReflectionBB(rightEyeRows)
#    cv2.line(rightEyeSlitDiff1, (rightEyeReflectionBB[0], rightEyeReflectionBB[1]), (rightEyeReflectionBB[0], rightEyeReflectionBB[1] + rightEyeReflectionBB[3]), (255, 255, 255))
#    cv2.line(rightEyeSlitDiff1, (rightEyeReflectionBB[0] + rightEyeReflectionBB[2], rightEyeReflectionBB[1]), (rightEyeReflectionBB[0] + rightEyeReflectionBB[2], rightEyeReflectionBB[1] + rightEyeReflectionBB[3]), (255, 255, 255))
#    print('right eye bb :: ' + str(rightEyeReflectionBB))
#    if (rightEyeReflectionBB[0] == 0) or ((rightEyeReflectionBB[0] + rightEyeReflectionBB[2]) == rightEyeRows.shape[1]):
#        rightEyeWidth = 0
#    else:
#        rightEyeWidth = rightEyeReflectionBB[2]

    #rightRightPoint = np.array(rightEyeStripCoords) + np.array([rightEyeReflectionBB[0], rightEyeReflectionBB[1] + rightEyeReflectionBB[3]])
    #rightLeftPoint = np.array(rightEyeStripCoords) + np.array([rightEyeReflectionBB[0] + rightEyeReflectionBB[2], rightEyeReflectionBB[1] + rightEyeReflectionBB[3]])

    [rightRightPoint, rightLeftPoint] = fullFlashCapture.landmarks.getRightEyeWidthPoints()

    (x, y, w, h) = fullFlashEyeStripCoords

    leftRightPoint -= [x, y]
    leftLeftPoint -= [x, y]

    rightRightPoint -= [x, y]
    rightLeftPoint -= [x, y]

    #print('LEFT RIGHT POINT' + str(leftEyeReflectionBB[0]))
    #print('LEFT LEFT POINT' + str(leftEyeReflectionBB[0] + leftEyeReflectionBB[2]))

    #print('RIGHT RIGHT POINT' + str(rightEyeReflectionBB[0]))
    #print('RIGHT LEFT POINT' + str(rightEyeReflectionBB[0] + rightEyeReflectionBB[2]))

    cv2.circle(fullFlashEyeStrip, (leftRightPoint[0], leftRightPoint[1]), 5, (0, 255, 0), -1)
    cv2.circle(fullFlashEyeStrip, (leftLeftPoint[0], leftLeftPoint[1]), 5, (0, 255, 0), -1)
   # if leftEyeReflectionBB[0] != 0:
   #     cv2.circle(fullFlashEyeStrip, (leftRightPoint[0], leftRightPoint[1]), 5, (0, 255, 0), -1)
   # else:
   #     cv2.circle(fullFlashEyeStrip, (leftRightPoint[0], leftRightPoint[1]), 5, (0, 0, 255), -1)
    
   # if (leftEyeReflectionBB[0] + leftEyeReflectionBB[2]) != leftEyeRows.shape[1]:
   #     cv2.circle(fullFlashEyeStrip, (leftLeftPoint[0], leftLeftPoint[1]), 5, (0, 255, 0), -1)
   # else:
   #     cv2.circle(fullFlashEyeStrip, (leftLeftPoint[0], leftLeftPoint[1]), 5, (0, 0, 255), -1)


    cv2.circle(fullFlashEyeStrip, (rightRightPoint[0], rightRightPoint[1]), 5, (0, 255, 0), -1)
    cv2.circle(fullFlashEyeStrip, (rightLeftPoint[0], rightLeftPoint[1]), 5, (0, 255, 0), -1)
    #if rightEyeReflectionBB[0] != 0:
    #    cv2.circle(fullFlashEyeStrip, (rightRightPoint[0], rightRightPoint[1]), 5, (0, 255, 0), -1)
    #else:
    #    cv2.circle(fullFlashEyeStrip, (rightRightPoint[0], rightRightPoint[1]), 5, (0, 0, 255), -1)
    #
    #if (rightEyeReflectionBB[0] + rightEyeReflectionBB[2]) != rightEyeRows.shape[1]:
    #    cv2.circle(fullFlashEyeStrip, (rightLeftPoint[0], rightLeftPoint[1]), 5, (0, 255, 0), -1)
    #else:
    #    cv2.circle(fullFlashEyeStrip, (rightLeftPoint[0], rightLeftPoint[1]), 5, (0, 0, 255), -1)

    cv2.rectangle(fullFlashEyeStrip, leftReflectionP1, leftReflectionP2, (0, 0, 255), 1)
    cv2.rectangle(fullFlashEyeStrip, rightReflectionP1, rightReflectionP2, (0, 0, 255), 1)

    saveStep.saveReferenceImageBGR(fullFlashEyeStrip, 'eyeStrip')
#    cv2.imshow('full flash eye strip', fullFlashEyeStrip)
#    cv2.waitKey(0)


    #NOTE: USING MAX MIGHT BE MORE ACCURATE....
    #averageEyeWidth = int(round((rightEyeWidth + leftEyeWidth) / 2))

    leftEyeWidth = leftRightPoint[0] - leftLeftPoint[0]
    rightEyeWidth = rightRightPoint[0] - rightLeftPoint[0]

    print('Left Eye Width :: ' + str(leftEyeWidth))
    print('Right Eye Width :: ' + str(rightEyeWidth))


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
    print('left reflection median :: ' + str(leftReflectionMedian))
    leftReflectionHLS = colorsys.rgb_to_hls(leftReflectionMedian[2] / 255, leftReflectionMedian[1] / 255, leftReflectionMedian[0] / 255)
    rightReflectionHLS = colorsys.rgb_to_hls(rightReflectionMedian[2] / 255, rightReflectionMedian[1] / 255, rightReflectionMedian[0] / 255)

    print('rightReflectionMedian :: ' + str(rightReflectionMedian))
    print('right HLS :: ' + str(rightReflectionHLS))
    print('leftReflectionMedian :: ' + str(leftReflectionMedian))
    print('left HLS :: ' + str(leftReflectionHLS))

    hueDiff = np.abs(leftReflectionHLS[0] - rightReflectionHLS[0])
    satDiff = np.abs(leftReflectionHLS[2] - rightReflectionHLS[2])

    print('HUE and SAT diff :: ' + str(hueDiff) + ' | ' + str(satDiff)) 


    averageMedian = (leftReflectionMedian + rightReflectionMedian) / 2

    leftReflectionArea = (leftReflectionWidth / averageEyeWidth) * (leftReflectionHeight / averageEyeWidth)
    rightReflectionArea = (rightReflectionWidth / averageEyeWidth) * (rightReflectionHeight / averageEyeWidth)

    if (max(leftReflectionArea, rightReflectionArea) / min(leftReflectionArea, rightReflectionArea)) > 1.25:
        raise NameError('Reflection Sizes are too different!')

    #averageArea = (leftReflectionArea + rightReflectionArea) / 2

    #averageValue = (leftReflectionValue + rightReflectionValue) / 2
    #fluxish = averageArea * averageValue

    leftReflectionLuminosity = leftReflectionHLS[1]
    rightReflectionLuminosity = rightReflectionHLS[1]

    #leftFluxish = averageArea * leftReflectionLuminosity
    leftFluxish = leftReflectionArea * leftReflectionLuminosity
    print('LEFT FLUXISH :: ' + str(leftFluxish) + ' | AREA :: ' + str(leftReflectionArea) + ' | LUMINOSITY :: ' + str(leftReflectionLuminosity))

    #rightFluxish = averageArea * rightReflectionLuminosity
    rightFluxish = rightReflectionArea * rightReflectionLuminosity
    print('RIGHT FLUXISH :: ' + str(rightFluxish) + ' | AREA :: ' + str(rightReflectionArea) + ' | LUMINOSITY :: ' + str(rightReflectionLuminosity))

    return [averageMedian, leftFluxish, rightFluxish]
