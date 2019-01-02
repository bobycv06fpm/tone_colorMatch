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
    #return cv2.GaussianBlur(img, (15, 15), 0)
    #return cv2.bilateralFilter(img,15,75,75)
    return cv2.medianBlur(img, 9)


def getReflectionBB(mask):
    img = mask.astype('uint8') * 255
    im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    contour = contours[max_index]

    return cv2.boundingRect(contour)

def maskReflection(noFlash, halfFlash, fullFlash):
    noFlashGrey = np.sum(blur(noFlash), axis=2)
    halfFlashGrey = np.sum(blur(halfFlash), axis=2)
    fullFlashGrey = np.sum(blur(fullFlash), axis=2)

    halfFlashGrey = np.clip(halfFlashGrey.astype('int32') - noFlashGrey, 0, (256 * 3))
    fullFlashGrey = np.clip(fullFlashGrey.astype('int32') - noFlashGrey, 0, (256 * 3))

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

    flashMask = np.logical_and(halfFlashMask, fullFlashMask)
    flashMask = np.logical_and(flashMask, np.logical_not(noFlashMask))
    return flashMask


def stretchBW(image):
    #median = np.median(image)
    #sd = np.std(image)
    #lower = median - (3 * sd)
    #lower = lower if lower > 0 else 0
    #upper = median + (3 * sd)
    #upper = upper if upper < 256 else 255

    lower = np.min(image)
    upper = np.max(image)

    #print('MEDIAN :: ' + str(median))
    #print('SD :: ' + str(sd))
    #print('LOWER :: ' + str(lower))
    #print('UPPER :: ' + str(upper))

    #bounds = np.copy(gray)
    #bounds[bounds < lower] = lower
    #bounds[bounds > upper] = upper

    numerator = (image - lower).astype('int32')
    denominator = (upper - lower).astype('int32')
    #stretched = (numerator.astype('int32') / denominator.astype('int32'))
    stretched = (numerator / denominator)
    #stretched = np.clip(stretched * 255, 0, 255).astype('uint8')
    stretched = np.clip(stretched * 255, 0, 255).astype('uint8')
    return stretched

def getEyeWidths(fullFlashCapture, leftEyeOffsets, leftEyeGreyReflectionMask, rightEyeOffsets, rightEyeGreyReflectionMask):
    fullFlashEyeStripCoords = np.array(fullFlashCapture.landmarks.getEyeStripBB())

    eyeStripCoordDiff_left = np.array(fullFlashLeftEyeCoord) - fullFlashEyeStripCoords[0:2]
    eyeStripCoordDiff_right = np.array(fullFlashRightEyeCoord) - fullFlashEyeStripCoords[0:2]

    (x, y, w, h) = fullFlashEyeStripCoords
    fullFlashEyeStripXStart = x
    fullFlashEyeStripXEnd = x + w
    fullFlashEyeStrip = fullFlashCapture.image[y:y+h, x:x+w]

    eyeStripCoordDiff_left += leftEyeOffsets[2]
    eyeStripCoordDiff_right += rightEyeOffsets[2]

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

    leftEyeGreyReflectionMask = maskReflection(noFlashLeftEyeCrop, halfFlashLeftEyeCrop, fullFlashLeftEyeCrop)
    rightEyeGreyReflectionMask = maskReflection(noFlashRightEyeCrop, halfFlashRightEyeCrop, fullFlashRightEyeCrop)

    fullFlashEyeStripCoords = np.array(fullFlashCapture.landmarks.getEyeStripBB())
#
    eyeStripCoordDiff_left = np.array(fullFlashLeftEyeCoord) - fullFlashEyeStripCoords[0:2]
    eyeStripCoordDiff_right = np.array(fullFlashRightEyeCoord) - fullFlashEyeStripCoords[0:2]

    (x, y, w, h) = fullFlashEyeStripCoords
    fullFlashEyeStripXStart = x
    fullFlashEyeStripXEnd = x + w
    fullFlashEyeStrip = fullFlashCapture.image[y:y+h, x:x+w]
    #fullFlashEyeStrip = halfFlashCapture.image[y:y+h, x:x+w]

    #eyeStripCoordDiff_left += leftEyeOffsets[2]
    #eyeStripCoordDiff_right += rightEyeOffsets[2]


    #leftEyeReflectionMask = np.stack((leftEyeGreyReflectionMask, leftEyeGreyReflectionMask, leftEyeGreyReflectionMask), axis=-1)
    #rightEyeReflectionMask = np.stack((rightEyeGreyReflectionMask, rightEyeGreyReflectionMask, rightEyeGreyReflectionMask), axis=-1)

    #leftEye = halfFlashLeftEyeCrop * leftEyeReflectionMask
    #rightEye = halfFlashRightEyeCrop * rightEyeReflectionMask

    #reflections = []


    [x, y, w, h] = getReflectionBB(leftEyeGreyReflectionMask)
    leftReflectionP1 = (x + eyeStripCoordDiff_left[0], y + eyeStripCoordDiff_left[1])
    leftReflectionP2 = (x + w + eyeStripCoordDiff_left[0], y + h + eyeStripCoordDiff_left[1])

    eyeSlitTop = y + eyeStripCoordDiff_left[1]
    eyeSlitBottom = y + h + eyeStripCoordDiff_right[1]


    leftEyeCenter = np.array([x + int(w / 2), y + int(h / 2)])
    print('Left Eye Center! :: ' + str(leftEyeCenter))
    eyeStripCoordDiff_left += leftEyeCenter

    leftEyeX = eyeStripCoordDiff_left[0]

    #cv2.circle(fullFlashEyeStrip, (eyeStripCoordDiff_left[0], eyeStripCoordDiff_left[1]), 5, (0, 255, 0), -1)

    #FOR REFLECITON
    leftEyeReflection = halfFlashLeftEyeCrop[y:y+h, x:x+w]

    leftHighMask = np.max(leftEyeReflection, axis=2) < 253
    leftLowMask = np.min(leftEyeReflection, axis=2) >= 2

    leftEyeMask = np.logical_and(leftHighMask, leftLowMask)
    leftEyePoints = leftEyeReflection[leftEyeMask]
    leftClipRatio = leftEyePoints.shape[0] / (leftEyeMask.shape[0] * leftEyeMask.shape[1])
    print('LEFT CLIP RATIO :: ' + str(leftClipRatio))
    if leftClipRatio < .9:
        print("TOO MUCH CLIPPING!")
        raise NameError('Not enough clean non-clipped pixels in left eye reflections')

    leftReflectionMedian = np.median(leftEyePoints, axis=0) * 2 #Multiply by 2 because we got the value from the half flash
    leftReflectionWidth = w 
    leftReflectionHeight = h
    leftReflectionValue = np.max(leftReflectionMedian)
    #END FOR REFLECITON


    [x, y, w, h] = getReflectionBB(rightEyeGreyReflectionMask)
    rightReflectionP1 = (x + eyeStripCoordDiff_right[0], y + eyeStripCoordDiff_right[1])
    rightReflectionP2 = (x + w + eyeStripCoordDiff_right[0], y + h + eyeStripCoordDiff_right[1])

    eyeSlitTop = y + eyeStripCoordDiff_left[1] if y + eyeStripCoordDiff_left[1] < eyeSlitTop else eyeSlitTop
    eyeSlitBottom = y + h + eyeStripCoordDiff_right[1] if y + h + eyeStripCoordDiff_right[1] > eyeSlitBottom else eyeSlitBottom


    rightEyeCenter = [x + int(w / 2), y + int(h / 2)]
    print('Right Eye Center! :: ' + str(rightEyeCenter))
    eyeStripCoordDiff_right += rightEyeCenter

    #FOR REFLECTION
    rightEyeReflection = halfFlashRightEyeCrop[y:y+h, x:x+w]

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
    rightReflectionWidth = w
    rightReflectionHeight = h
    rightReflectionValue = np.max(rightReflectionMedian)
    #END FOR REFLECTION

    
    rightEyeX = eyeStripCoordDiff_right[0]
    eyeDistance = int((leftEyeX - rightEyeX) / 3)

    print('Eyes left, right :: ' + str(leftEyeX) + ' ' + str(rightEyeX))
    leftEyeLeftEdge = (leftEyeX + eyeDistance) if (leftEyeX + eyeDistance) < fullFlashEyeStrip.shape[1] else fullFlashEyeStrip.shape[1]
    leftEyeRightEdge = (leftEyeX - eyeDistance) if (leftEyeX - eyeDistance) > 0 else 0

    rightEyeLeftEdge = (rightEyeX + eyeDistance) if (rightEyeX + eyeDistance) < fullFlashEyeStrip.shape[1] else fullFlashEyeStrip.shape[1]
    rightEyeRightEdge = (rightEyeX - eyeDistance) if (rightEyeX - eyeDistance) > 0 else 0
    #print('Edges left, right :: ' + str(leftEdge) + ' ' + str(rightEdge))

    #eyeSlitCrop = fullFlashEyeStrip[eyeSlitTop - 50:eyeSlitBottom + 50, rightEdge:leftEdge]

    #getEyeWidths(fullFlashCapture, leftEyeOffsets[2], leftEyeGreyReflectionMask, rightEyeOffsets[2], rightEyeGreyReflectionMask)

    margin = 50
    leftEyeStripCoords = [leftEyeRightEdge, eyeSlitTop - margin]
    leftEyeSlit = np.copy(fullFlashEyeStrip[eyeSlitTop - margin:eyeSlitBottom + margin, leftEyeRightEdge:leftEyeLeftEdge])
    leftEyeSlitMiddle = int(leftEyeSlit.shape[1]/2)
    leftEyeSlit[:, (leftEyeSlitMiddle - int(leftEyeSlitMiddle / 3)):(leftEyeSlitMiddle + int(leftEyeSlitMiddle / 3))] = 0

    rightEyeStripCoords = [rightEyeRightEdge, eyeSlitTop - margin]
    rightEyeSlit = np.copy(fullFlashEyeStrip[eyeSlitTop - margin:eyeSlitBottom + margin, rightEyeRightEdge:rightEyeLeftEdge])
    rightEyeSlitMiddle = int(rightEyeSlit.shape[1]/2)
    rightEyeSlit[:, (rightEyeSlitMiddle - int(rightEyeSlitMiddle / 3)):(rightEyeSlitMiddle + int(rightEyeSlitMiddle / 3))] = 0

    #cv2.imshow('left', leftEyeSlit)
    #cv2.imshow('right', rightEyeSlit)
    #cv2.waitKey(0)


    #eyeSlitCrop = fullFlashEyeStrip[eyeSlitTop - 50:eyeSlitBottom + 50, rightEdge:leftEdge]
    #eyeSlitCrop = fullFlashEyeStrip[eyeSlitTop - 50:eyeSlitBottom + 50, rightEyeX:leftEyeX]


    #eyeSlitCrop = cv2.GaussianBlur(eyeSlitCrop, (51, 51), 0)
    leftEyeSlitHLS = cv2.cvtColor(leftEyeSlit, cv2.COLOR_BGR2HLS_FULL)
    rightEyeSlitHLS = cv2.cvtColor(rightEyeSlit, cv2.COLOR_BGR2HLS_FULL)
    #print('Eye Slit HLS :: ' + str(eyeSlitHLS))
    #eyeSlitH = eyeSlitHLS[:, :, 0]
    #eyeSlitH = cv2.Sobel(eyeSlitH, cv2.CV_64F, 1, 1, ksize=5)
    #eyeSlitH = (eyeSlitH < 26).astype('int32') * 255
    #eyeSlitHOriginal = eyeSlitHLS[:, :, 0]
    #eyeSlitH = np.copy(eyeSlitHOriginal).astype('int32')
    #eyeSlitHigh = np.clip(eyeSlitH - 85, 0, 255) #Shift Hue by 1/3 of range to move red/yellow away from Hue boundary
    #eyeSlitLow = np.clip(eyeSlitH + 170, 0, 255)

    #eyeSlitH[eyeSlitHOriginal >= 85] = eyeSlitHigh[eyeSlitHOriginal >= 85]
    #eyeSlitH[eyeSlitHOriginal < 85] = eyeSlitLow[eyeSlitHOriginal < 85]

    #hueBound = 230
    leftEyeSlitH = (leftEyeSlitHLS[:, :, 0]).astype('int32')
    #leftEyeSlitH[leftEyeSlitH > hueBound] = 0
    #leftEyeSlitH = stretchBW(leftEyeSlitH)
    rightEyeSlitH = (rightEyeSlitHLS[:, :, 0]).astype('int32')
    #rightEyeSlitH[rightEyeSlitH > hueBound] = 0
    #rightEyeSlitH = stretchBW(rightEyeSlitH)



    leftEyeSlitL = (leftEyeSlitHLS[:, :, 1]).astype('int32')
    rightEyeSlitL = (rightEyeSlitHLS[:, :, 1]).astype('int32')
    leftEyeSlitS = (leftEyeSlitHLS[:, :, 2]).astype('int32')
    rightEyeSlitS = (rightEyeSlitHLS[:, :, 2]).astype('int32')

    leftEyeSlitDiff = np.clip(leftEyeSlitS - leftEyeSlitL, 0, 255).astype('uint8')
    rightEyeSlitDiff = np.clip(rightEyeSlitS - rightEyeSlitL, 0, 255).astype('uint8')
    #eyeSlitTest = cv2.GaussianBlur(eyeSlitTest, (51, 51), 0)

    #med = np.median(eyeSlitTest)
    #sd = np.std(eyeSlitTest)

    #lowerbound = med #+ (2 * sd)

    #eyeSlitTest4 = (eyeSlitTest > lowerbound).astype('uint8') * 255


    #eyeSlitTest2 = cv2.Sobel(eyeSlitTest, cv2.CV_16U, 1, 1, ksize=5)
    #eyeSlitTest3 = cv2.GaussianBlur(eyeSlitTest2, (5, 5), 0)

    #cv2.imshow('Eye Slit Crop H', eyeSlitH.astype('uint8'))
    #cv2.imshow('Eye Slit Crop L', eyeSlitL.astype('uint8'))
    #cv2.imshow('Eye Slit Crop S', eyeSlitS.astype('uint8'))
    #cv2.imshow('Eye Slit Crop Test', eyeSlitTest.astype('uint8'))
   # cv2.imshow('Eye Slit Crop Test 2', eyeSlitTest2.astype('uint8'))
   # cv2.imshow('Eye Slit Crop Test 3', eyeSlitTest3.astype('uint8'))
    #cv2.imshow('Eye Slit Crop Test 4', eyeSlitTest4.astype('uint8'))


    #cv2.imshow('left no stretch', leftEyeSlitDiff)
    #cv2.imshow('right no stretch', rightEyeSlitDiff)

    leftEyeSlitDiff1 = stretchBW(leftEyeSlitDiff)
    rightEyeSlitDiff1 = stretchBW(rightEyeSlitDiff)
    
    #margin = 20
    #leftCenter = int(leftEyeSlitDiff.shape[0] / 2)
    #cv2.line(leftEyeSlitDiff, (0, leftCenter), (leftEyeSlitDiff.shape[1], leftCenter), (255, 255, 255))
    #cv2.line(leftEyeSlitDiff, (0, leftCenter + margin), (leftEyeSlitDiff.shape[1], leftCenter + margin), (255, 255, 255))
    #cv2.line(leftEyeSlitDiff, (0, leftCenter - margin), (leftEyeSlitDiff.shape[1], leftCenter - margin), (255, 255, 255))

    #rightCenter = int(rightEyeSlitDiff.shape[0] / 2)
    #cv2.line(rightEyeSlitDiff, (0, rightCenter), (rightEyeSlitDiff.shape[1], rightCenter), (255, 255, 255))
    #cv2.line(rightEyeSlitDiff, (0, rightCenter + margin), (rightEyeSlitDiff.shape[1], rightCenter + margin), (255, 255, 255))
    #cv2.line(rightEyeSlitDiff, (0, rightCenter - margin), (rightEyeSlitDiff.shape[1], rightCenter - margin), (255, 255, 255))

    #cv2.imshow('Left Slit Crop', leftEyeSlitDiff1)
    #cv2.imshow('Right Slit Crop', rightEyeSlitDiff1)

    threshold = 20
    leftEyeSlitDiff2 = (leftEyeSlitDiff1 > threshold).astype('uint8') * 255
    rightEyeSlitDiff2 = (rightEyeSlitDiff1 > threshold).astype('uint8') * 255

    #cv2.imshow('Left Slit Crop mask', leftEyeSlitDiff)
    #cv2.imshow('Right Slit Crop mask', rightEyeSlitDiff)

    #kernel = np.ones((7, 5), np.uint8)
    kernel = np.ones((3, 5), np.uint8)
    #kernel = np.ones((7, 3), np.uint8)
    #algo = cv2.MORPH_CLOSE

    leftEyeSlitDiff3 = leftEyeSlitDiff2
    rightEyeSlitDiff3 = rightEyeSlitDiff2

    leftEyeSlitDiff3 = cv2.morphologyEx(leftEyeSlitDiff3, cv2.MORPH_OPEN, kernel)
    rightEyeSlitDiff3 = cv2.morphologyEx(rightEyeSlitDiff3, cv2.MORPH_OPEN, kernel)

    algo = cv2.MORPH_CROSS
    leftEyeSlitDiff3 = cv2.morphologyEx(leftEyeSlitDiff3, algo, kernel)
    rightEyeSlitDiff3 = cv2.morphologyEx(rightEyeSlitDiff3, algo, kernel)

    margin = 40
    leftCenter = int(leftEyeSlitDiff.shape[0] / 2)
#    cv2.line(leftEyeSlitDiff3, (0, leftCenter), (leftEyeSlitDiff3.shape[1], leftCenter), (255, 255, 255))
#    cv2.line(leftEyeSlitDiff3, (0, leftCenter + margin), (leftEyeSlitDiff3.shape[1], leftCenter + margin), (255, 255, 255))
#    cv2.line(leftEyeSlitDiff3, (0, leftCenter - margin), (leftEyeSlitDiff3.shape[1], leftCenter - margin), (255, 255, 255))
#
    rightCenter = int(rightEyeSlitDiff.shape[0] / 2)
#    cv2.line(rightEyeSlitDiff3, (0, rightCenter), (rightEyeSlitDiff3.shape[1], rightCenter), (255, 255, 255))
#    cv2.line(rightEyeSlitDiff3, (0, rightCenter + margin), (rightEyeSlitDiff3.shape[1], rightCenter + margin), (255, 255, 255))
#    cv2.line(rightEyeSlitDiff3, (0, rightCenter - margin), (rightEyeSlitDiff3.shape[1], rightCenter - margin), (255, 255, 255))

    leftEyeRows = leftEyeSlitDiff3[leftCenter - margin:leftCenter + margin]
    leftEyeRows = leftEyeRows == 0
    rightEyeRows = rightEyeSlitDiff3[rightCenter - margin:rightCenter + margin]
    rightEyeRows = rightEyeRows == 0

    #cv2.imshow('left eye rows', leftEyeRows.astype('uint8')*255)
    #cv2.imshow('right eye rows', rightEyeRows.astype('uint8')*255)


    leftEyeBB = getReflectionBB(leftEyeRows)
    cv2.line(leftEyeSlitDiff1, (leftEyeBB[0], leftEyeBB[1]), (leftEyeBB[0], leftEyeBB[1] + leftEyeBB[3]), (255, 255, 255))
    cv2.line(leftEyeSlitDiff1, (leftEyeBB[0] + leftEyeBB[2], leftEyeBB[1]), (leftEyeBB[0] + leftEyeBB[2], leftEyeBB[1] + leftEyeBB[3]), (255, 255, 255))
    print('left eye bb :: ' + str(leftEyeBB))
    if (leftEyeBB[0] == 0) or ((leftEyeBB[0] + leftEyeBB[2]) == leftEyeRows.shape[1]):
        leftEyeWidth = 0
    else:
        leftEyeWidth = leftEyeBB[2]

    leftRightPoint = np.array(leftEyeStripCoords) + np.array([leftEyeBB[0], leftEyeBB[1] + leftEyeBB[3]])
    leftLeftPoint = np.array(leftEyeStripCoords) + np.array([leftEyeBB[0] + leftEyeBB[2], leftEyeBB[1] + leftEyeBB[3]])

    rightEyeBB = getReflectionBB(rightEyeRows)
    cv2.line(rightEyeSlitDiff1, (rightEyeBB[0], rightEyeBB[1]), (rightEyeBB[0], rightEyeBB[1] + rightEyeBB[3]), (255, 255, 255))
    cv2.line(rightEyeSlitDiff1, (rightEyeBB[0] + rightEyeBB[2], rightEyeBB[1]), (rightEyeBB[0] + rightEyeBB[2], rightEyeBB[1] + rightEyeBB[3]), (255, 255, 255))
    print('right eye bb :: ' + str(rightEyeBB))
    if (rightEyeBB[0] == 0) or ((rightEyeBB[0] + rightEyeBB[2]) == rightEyeRows.shape[1]):
        rightEyeWidth = 0
    else:
        rightEyeWidth = rightEyeBB[2]

    rightRightPoint = np.array(rightEyeStripCoords) + np.array([rightEyeBB[0], rightEyeBB[1] + rightEyeBB[3]])
    rightLeftPoint = np.array(rightEyeStripCoords) + np.array([rightEyeBB[0] + rightEyeBB[2], rightEyeBB[1] + rightEyeBB[3]])


    print('LEFT RIGHT POINT' + str(leftEyeBB[0]))
    print('LEFT LEFT POINT' + str(leftEyeBB[0] + leftEyeBB[2]))

    print('RIGHT RIGHT POINT' + str(rightEyeBB[0]))
    print('RIGHT LEFT POINT' + str(rightEyeBB[0] + rightEyeBB[2]))

    if leftEyeBB[0] != 0:
        cv2.circle(fullFlashEyeStrip, (leftRightPoint[0], leftRightPoint[1]), 5, (0, 255, 0), -1)
    else:
        cv2.circle(fullFlashEyeStrip, (leftRightPoint[0], leftRightPoint[1]), 5, (0, 0, 255), -1)
    
    if (leftEyeBB[0] + leftEyeBB[2]) != leftEyeRows.shape[1]:
        cv2.circle(fullFlashEyeStrip, (leftLeftPoint[0], leftLeftPoint[1]), 5, (0, 255, 0), -1)
    else:
        cv2.circle(fullFlashEyeStrip, (leftLeftPoint[0], leftLeftPoint[1]), 5, (0, 0, 255), -1)


    if rightEyeBB[0] != 0:
        cv2.circle(fullFlashEyeStrip, (rightRightPoint[0], rightRightPoint[1]), 5, (0, 255, 0), -1)
    else:
        cv2.circle(fullFlashEyeStrip, (rightRightPoint[0], rightRightPoint[1]), 5, (0, 0, 255), -1)
    
    if (rightEyeBB[0] + rightEyeBB[2]) != rightEyeRows.shape[1]:
        cv2.circle(fullFlashEyeStrip, (rightLeftPoint[0], rightLeftPoint[1]), 5, (0, 255, 0), -1)
    else:
        cv2.circle(fullFlashEyeStrip, (rightLeftPoint[0], rightLeftPoint[1]), 5, (0, 0, 255), -1)

    cv2.rectangle(fullFlashEyeStrip, leftReflectionP1, leftReflectionP2, (0, 0, 255), 1)
    cv2.rectangle(fullFlashEyeStrip, rightReflectionP1, rightReflectionP2, (0, 0, 255), 1)

    #cv2.imshow('full flash eye strip', fullFlashEyeStrip)
    saveStep.saveReferenceImageBGR(fullFlashEyeStrip, 'eyeStrip')
    #cv2.waitKey(0)


    #NOTE: USING MAX MIGHT BE MORE ACCURATE....
    #averageEyeWidth = int(round((rightEyeWidth + leftEyeWidth) / 2))
    maxEyeWidth = max([rightEyeWidth, leftEyeWidth])

    print('RIGHT EYE WIDTH :: ' + str(rightEyeWidth))
    print('LEFT EYE WIDTH :: ' + str(leftEyeWidth))
    #print('AVERAGE EYE WIDTH :: ' + str(averageEyeWidth))
    print('MAX EYE WIDTH :: ' + str(maxEyeWidth))

    #blur = 5
    #leftEyeSlitDiff = cv2.GaussianBlur(leftEyeSlitDiff, (blur, blur), 0)
    #rightEyeSlitDiff = cv2.GaussianBlur(rightEyeSlitDiff, (blur, blur), 0)

    #threshold = 64
    #leftEyeSlitDiff = (leftEyeSlitDiff > threshold).astype('uint8') * 255
    #rightEyeSlitDiff = (rightEyeSlitDiff > threshold).astype('uint8') * 255

    leftEyeSlitStack = np.vstack((leftEyeSlitL.astype('uint8'), leftEyeSlitS.astype('uint8'), leftEyeSlitDiff1, leftEyeSlitDiff2, leftEyeSlitDiff3))
    rightEyeSlitStack = np.vstack((rightEyeSlitL.astype('uint8'), rightEyeSlitS.astype('uint8'), rightEyeSlitDiff1, rightEyeSlitDiff2, rightEyeSlitDiff3))

    #cv2.imshow('Eye Mask Comparison', np.hstack((rightEyeSlitStack, leftEyeSlitStack)))
    #cv2.waitKey(0)

    valuesDiff = np.abs((rightReflectionMedian - leftReflectionMedian))
    leftReflectionHSV = colorsys.rgb_to_hsv(leftReflectionMedian[2], leftReflectionMedian[1], leftReflectionMedian[0])
    rightReflectionHSV = colorsys.rgb_to_hsv(rightReflectionMedian[2], rightReflectionMedian[1], rightReflectionMedian[0])

    print('rightReflectionMedian :: ' + str(rightReflectionMedian))
    print('right HSV :: ' + str(rightReflectionHSV))
    print('leftReflectionMedian :: ' + str(leftReflectionMedian))
    print('left HSV :: ' + str(leftReflectionHSV))

    hueDiff = np.abs(leftReflectionHSV[0] - rightReflectionHSV[0])
    satDiff = np.abs(leftReflectionHSV[1] - rightReflectionHSV[1])

    print('HUE and SAT diff :: ' + str(hueDiff) + ' | ' + str(satDiff)) 

    print('Values Diff :: ' + str(valuesDiff))


    averageMedian = (leftReflectionMedian + rightReflectionMedian) / 2
    averageValue = (leftReflectionValue + rightReflectionValue) / 2

    leftReflectionArea = (leftReflectionWidth / maxEyeWidth) * (leftReflectionHeight / maxEyeWidth)
    rightReflectionArea = (rightReflectionWidth / maxEyeWidth) * (rightReflectionHeight / maxEyeWidth)
    averageArea = (leftReflectionArea + rightReflectionArea) / 2

    fluxish = averageArea * averageValue

    return [averageMedian, fluxish]


#def getAverageScreenReflectionColor(username, imageName, image, fullFlash_sBGR, imageShape, cameraWB_CIE_xy_coords):
#    leftEye, left_sBGR = getLeftEye(username, imageName, image, fullFlash_sBGR, imageShape)
#    rightEye, right_sBGR = getRightEye(username, imageName, image, fullFlash_sBGR, imageShape)
#
#    #if leftEyeError is not None:
#    eyeWidth = getEyeWidth(username, imageName, image, imageShape)
#        print('Left Num Pixels :: ' + str(leftNumPixels))
#        print('Left reflection Width, Height, Area, Fluxish :: ' + str(leftWidth) + ' ' + str(leftHeight) + ' ' + str(leftWidth * leftHeight) + ' ' + str(leftWidth * leftHeight * max(leftAverageBGR)))
#
#    try:
#        rightEyeReflection = getScreenReflection(username, imageName, rightEye, right_sBGR, 'rightEye', cameraWB_CIE_xy_coords)
#    except:
#        print('Setting Right to None!')
#        rightAverageBGR = None
#        rightFluxish = None
#    else:
#        [rightSumBGR, rightNumPixels, [rightWidth, rightHeight]] = rightEyeReflection
#        rightAverageBGR = rightSumBGR / rightNumPixels
#
#        rightWidth = rightWidth / eyeWidth
#        rightHeight = rightHeight / eyeWidth
#
#        rightFluxish = rightWidth * rightHeight * max(rightAverageBGR)
#        print('Right Num Pixels :: ' + str(rightNumPixels))
#        print('Right reflection Width, Height, Area, Fluxish :: ' + str(rightWidth) + ' ' + str(rightHeight) + ' ' + str(rightWidth * rightHeight) + ' ' + str(rightWidth * rightHeight * max(rightAverageBGR)))
#
#
#
#    return [[leftAverageBGR, leftFluxish, [leftWidth, leftHeight]], [rightAverageBGR, rightFluxish, [rightWidth, rightHeight]]]
