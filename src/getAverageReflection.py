"""Extracts the reflections and sclera from each eye"""
import math
import cv2
import numpy as np
import colorTools
import cropTools
import imageTools

from logger import getLogger
LOGGER = getLogger(__name__, 'app')

def __erode(img):
    """Morphological Erosion Helper Function"""
    kernel = np.ones((5, 5), np.uint16)
    morph = cv2.morphologyEx(img, cv2.MORPH_CROSS, kernel)
    return morph

def __bbToMask(bb, imgShape):
    """Takes a BB and image shape and returns a mask in the dimensions of the image with the BB masked true"""
    img = np.zeros(imgShape)
    img[bb[1]:(bb[1]+bb[3]), bb[0]:(bb[0]+bb[2])] = 1
    return img.astype('bool')

def __getEyeWhiteMask(eyes, reflection_bb, wb, label):
    """Returns a mask for the Sclera of both the left and right eyes"""
    for index, eye in enumerate(eyes):
        if eye.shape[0] * eye.shape[1] == 0:
            raise ValueError('Cannot Find #{} Eye'.format(index))

    eyes = [colorTools.convert_sBGR_to_linearBGR_float_fast(eye) for eye in eyes]
    eyes = [colorTools.whitebalance_from_asShot_to_d65(eye, *wb) for eye in eyes]

    primarySpecularReflectionBB = np.copy(reflection_bb)
    primarySpecularReflectionBB[0:2] -= reflection_bb[2:4]
    primarySpecularReflectionBB[2:4] *= 3
    primarySpecularReflectionMask = bbToMask(primarySpecularReflectionBB, eyes[0][:, :, 0].shape)

    eye_blur = [cv2.blur(eye, (5, 5)) for eye in eyes]
    eyes_hsv = [colorTools.naiveBGRtoHSV(eye) for eye in eye_blur]
    eye_s = []
    for eye in eyes_hsv:
        sat = 1 - eye[:, :, 1]
        val = eye[:, :, 2]

        sat[primarySpecularReflectionMask] = 0
        val[primarySpecularReflectionMask] = 0

        sat = sat * val

        eye[:, :, 0] = sat
        eye[:, :, 1] = sat
        eye[:, :, 2] = sat
        eye_s.append(eye)

    diff = eye_s[0] - eye_s[-1]
    diff = np.clip(diff, 0, 255)
    min_diff = np.min(diff)
    max_diff = np.max(diff)

    scaled_diff = (diff - min_diff) / (max_diff - min_diff)
    scaled_diff = np.clip(scaled_diff * 255, 0, 255).astype('uint8')

    _, thresh = cv2.threshold(scaled_diff[:, :, 0], 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    kernel = np.ones((9, 9), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    totalArea = thresh.shape[0] * thresh.shape[1]
    areaPercents = np.array(areas) / totalArea
    areasMask = areaPercents > 0.01

    possibleContourIndexes = np.arange(len(contours))[areasMask]

    medians = []
    for index in possibleContourIndexes:
        target = np.zeros(thresh.shape, dtype='uint8')
        mask = cv2.drawContours(target, contours, index, 255, cv2.FILLED)
        med = np.median(eye_s[0][mask.astype('bool')])
        medians.append(med)

    max_index = possibleContourIndexes[np.argmax(medians)]

    target = np.zeros(thresh.shape, dtype='uint8')
    sclera_mask = cv2.drawContours(target, contours, max_index, 255, cv2.FILLED)


    masked_scaled_diff = scaled_diff[:, :, 0]
    masked_scaled_diff[np.logical_not(sclera_mask)] = 0
    median = np.median(masked_scaled_diff[sclera_mask.astype('bool')])

    masked_scaled_diff = cv2.GaussianBlur(masked_scaled_diff, (5, 5), 0)
    _, thresh2 = cv2.threshold(masked_scaled_diff, median, 255, cv2.THRESH_BINARY)#+cv2.THRESH_OTSU)

    thresh2 = cv2.morphologyEx(thresh2, cv2.MORPH_OPEN, kernel)
    thresh2 = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel)

    contoursRefined, hierarchy = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areasRefined = [cv2.contourArea(c) for c in contoursRefined]
    maxIndex = np.argmax(areasRefined)

    target = np.zeros(thresh.shape, dtype='uint8')
    maskRefined = cv2.drawContours(target, contoursRefined, maxIndex, 255, cv2.FILLED).astype('bool')

    #cv2.imshow('mask 2 - {}'.format(label), maskRefined)
     #= np.stack((maskRefined, maskRefined, maskRefined), axis=-1)

    #masked_eyes = np.copy(eyes)
    #masked_eyes[:, np.logical_not(maskRefined)] = [0, 0, 0]

    #cv2.imshow('diff', scaled_diff)
    #cv2.imshow('thresh', thresh)
    #cv2.imshow('sclera', sclera_mask)
    #cv2.imshow('sclera {}'.format(label), np.vstack([np.hstack(masked_eyes[0:4]), np.hstack(masked_eyes[4:])]))
    #cv2.waitKey(0)

    return [maskRefined, contoursRefined[maxIndex]]

def __getReflectionBB(eyes, wb):
    """Returns the BB of the device screen specular reflection in the eye"""
    for index, eye in enumerate(eyes):
        if eye.shape[0] * eye.shape[1] == 0:
            raise ValueError('Cannot Find #{} Eye'.format(index))

    eyes = [colorTools.convert_sBGR_to_linearBGR_float_fast(eye) for eye in eyes]
    eyes = [colorTools.whitebalance_from_asShot_to_d65(eye, *wb) for eye in eyes]
    #greyEyes = np.array([np.mean(eye, axis=2) for eye in eyes])
    greyEyes = np.array([np.mean(eye[:, :, 0:2], axis=2) for eye in eyes])


    eyeCropY1 = int(0.15 * greyEyes[0].shape[0])
    eyeCropY2 = int(0.85 * greyEyes[0].shape[0])

    eyeCropX1 = int(0.25 * greyEyes[0].shape[1])
    eyeCropX2 = int(0.75 * greyEyes[0].shape[1])

    croppedGreyEyes = np.array([img[eyeCropY1:eyeCropY2, eyeCropX1:eyeCropX2] for img in greyEyes])

    totalChange = np.sum(croppedGreyEyes[:-1] - croppedGreyEyes[1:], axis=0)
    totalChange = totalChange / np.max(totalChange)

    kernel = np.ones((9, 9), np.uint8)

    totalChangeMask = totalChange > (np.median(totalChange) + np.std(totalChange))
    totalChangeMaskEroded = cv2.erode(totalChangeMask.astype('uint8'), kernel, iterations=1)
    totalChangeMaskOpened = cv2.dilate(totalChangeMaskEroded.astype('uint8'), kernel, iterations=1)

    totalChangeMaskOpenedDilated = cv2.dilate(totalChangeMaskOpened.astype('uint8'), kernel, iterations=1)

    eyeLap = [cv2.Laplacian(imageTools.stretchHistogram(img, [2, 10]), cv2.CV_64F) for img in croppedGreyEyes]
    eyeLap = eyeLap / np.max(eyeLap)

    totalChangeLap = cv2.Laplacian(totalChange, cv2.CV_64F)
    totalChangeLap = totalChangeLap / np.max(totalChangeLap)

    #im2, contours, hierarchy = cv2.findContours(totalChangeMaskOpenedDilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(totalChangeMaskOpenedDilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #areas = [cv2.contourArea(c) for c in contours]

    highScore = 0
    eyeReflectionBB = None
    #gradientMask = None
    for index, contour in enumerate(contours):
        target = np.zeros(totalChangeMaskOpenedDilated.shape, dtype='uint8')
        drawn = cv2.drawContours(target, contours, index, 255, cv2.FILLED)
        drawn = cv2.morphologyEx(drawn, cv2.MORPH_GRADIENT, kernel)

        borderPoints = totalChangeLap[drawn.astype('bool')]

        if len(borderPoints) > 10:
            borderPointsMedian = np.median(np.sort(borderPoints)[-10:])

            if borderPointsMedian > highScore:
                highScore = borderPointsMedian
                eyeReflectionBB = list(cv2.boundingRect(contour))
                gradientMask = drawn.astype('bool')

    if eyeReflectionBB is None:
        raise ValueError('Could Not Find Reflection BB')

    eyeReflectionBB[0] += eyeCropX1
    eyeReflectionBB[1] += eyeCropY1
    return np.array(eyeReflectionBB)

def __getAnnotatedEyeStrip(leftReflectionBB, leftScleraContour, leftEyeCrop, rightReflectionBB, rightScleraContour, rightEyeCrop):
    """For Sanity Checking. Return an image with the left and right eyes of each capture with the sclera and device screen reflection over layed"""

    leftReflectionP1 = leftReflectionBB[0:2]
    leftReflectionP2 = leftReflectionP1 + leftReflectionBB[2:4]
    leftReflectionP1 = tuple(leftReflectionP1)
    leftReflectionP2 = tuple(leftReflectionP2)

    rightReflectionP1 = rightReflectionBB[0:2]
    rightReflectionP2 = rightReflectionP1 + rightReflectionBB[2:4]
    rightReflectionP1 = tuple(rightReflectionP1)
    rightReflectionP2 = tuple(rightReflectionP2)

    leftEyeCropCopy = np.copy(leftEyeCrop)
    rightEyeCropCopy = np.copy(rightEyeCrop)

    cv2.rectangle(leftEyeCropCopy, leftReflectionP1, leftReflectionP2, (0, 0, 255), 1)
    cv2.rectangle(rightEyeCropCopy, rightReflectionP1, rightReflectionP2, (0, 0, 255), 1)
    #mask =  cv2.drawContours(target, contours, index, 255, cv2.FILLED)
    cv2.drawContours(leftEyeCropCopy, [leftScleraContour], 0, (0, 255, 0), 1)
    cv2.drawContours(rightEyeCropCopy, [rightScleraContour], 0, (0, 255, 0), 1)

    canvasShape = np.max([leftEyeCrop.shape, rightEyeCrop.shape], axis=0)

    originLeft_Y_start = math.floor((canvasShape[0] - leftEyeCropCopy.shape[0]) / 2) #Center vertically
    originLeft_Y_end = -1 * math.ceil((canvasShape[0] - leftEyeCropCopy.shape[0]) / 2) #Center vertically
    originLeft_Y_end = originLeft_Y_end if originLeft_Y_end != 0 else leftEyeCropCopy.shape[0]
    originLeft_X = leftEyeCropCopy.shape[1]

    originRight_Y_start = math.floor((canvasShape[0] - rightEyeCropCopy.shape[0]) / 2) #Center vertically
    originRight_Y_end = -1 * math.ceil((canvasShape[0] - rightEyeCropCopy.shape[0]) / 2) #Center vertically
    originRight_Y_end = originRight_Y_end if originRight_Y_end != 0 else rightEyeCropCopy.shape[0]
    originRight_X = canvasShape[1] - rightEyeCropCopy.shape[1]

    leftEyeCanvas = np.zeros(canvasShape, dtype='uint8')
    rightEyeCanvas = np.zeros(canvasShape, dtype='uint8')

    leftEyeCanvas[originLeft_Y_start:originLeft_Y_end, 0:originLeft_X] = leftEyeCropCopy
    rightEyeCanvas[originRight_Y_start:originRight_Y_end, originRight_X:] = rightEyeCropCopy

    eyeStrip = np.hstack([rightEyeCanvas, leftEyeCanvas]) #Backwards because left refers to the user's left eye

    return eyeStrip

#Note: both parent and child offsets should originally be measured to the same origin
#def calculateRelativeOffset(parentOffset, childOffset):
#    return childOffset[0:2] - parentOffset[0:2]

def __calculateRepresentativeReflectionPoint(reflectionPoints):
    """Calculate the point that should be closest to the true refletion color"""
    numPoints = reflectionPoints.shape[0]

    oneTenth = int(numPoints / 10) * -1

    topMedianBlue = np.median(np.array(sorted(reflectionPoints[:, 0]))[oneTenth:])
    topMedianGreen = np.median(np.array(sorted(reflectionPoints[:, 1]))[oneTenth:])
    topMedianRed = np.median(np.array(sorted(reflectionPoints[:, 2]))[oneTenth:])

    newRepValue = [topMedianBlue, topMedianGreen, topMedianRed]
    #print('Old :: {} | New :: {}'.format(old, newRepValue))
    return np.array(newRepValue)

def __extractScleraPoints(eyes, scleraMask):
    """Return the points in the sclera"""
    eyePoints = [eye[scleraMask] for eye in eyes]
    greyEyePoints = [np.mean(eye, axis=1) for eye in eyePoints]
    topTenthIndex = int(len(greyEyePoints[0]) * 0.9)
    brightestThresholds = [math.floor(sorted(points)[topTenthIndex]) for points in greyEyePoints]
    brightestPointsMasks = [greyPoints > threshold for greyPoints, threshold in zip(greyEyePoints, brightestThresholds)]
    brightestMeans = [np.mean(points[brightestMask], axis=0) for points, brightestMask in zip(eyePoints, brightestPointsMasks)]
    #print('eye points :: {}'.format(brightestThresholds))
    print('Means :: {}'.format(brightestMeans))
    return np.array(brightestMeans) / 255

def __extractReflectionPoints(reflectionBB, eyeCrop, eyeMask, ignoreMask):
    """Return the points in the device screen reflection"""

    [x, y, w, h] = reflectionBB

    reflectionCrop = eyeCrop[y:y+h, x:x+w]
    reflectionCrop = colorTools.convert_sBGR_to_linearBGR_float_fast(reflectionCrop)

    #Add together each subpixel mask for each pixel. if the value is greater than 0, one of the subpixels was clipping
    #Just mask = isClipping(Red) or isClipping(Green) or isClipping(Blue)
    clippingMask = np.sum(reflectionCrop == 1.0, axis=2) > 0

    if (reflectionCrop.shape[0] == 0) or (reflectionCrop.shape[1] == 0):
        raise ValueError('Zero width eye reflection')

    cleanPixels = np.sum(np.logical_not(clippingMask).astype('uint8'))
    cleanPixelRatio = cleanPixels / (clippingMask.shape[0] * clippingMask.shape[1])

    if cleanPixelRatio < 0.95:
        raise ValueError('Not enough clean non-clipped pixels in eye reflections')

    threshold = 1/4 #1/5

    stretchedBlueReflectionCrop = (reflectionCrop[:, :, 0] - np.min(reflectionCrop[:, :, 0])) * (1 / (np.max(reflectionCrop[:, :, 0]) - np.min(reflectionCrop[:, :, 0])))
    blueMask = stretchedBlueReflectionCrop > threshold

    stretchedGreenReflectionCrop = (reflectionCrop[:, :, 1] - np.min(reflectionCrop[:, :, 1])) * (1 / (np.max(reflectionCrop[:, :, 1]) - np.min(reflectionCrop[:, :, 1])))
    greenMask = stretchedGreenReflectionCrop > threshold

    stretchedRedReflectionCrop = (reflectionCrop[:, :, 2] - np.min(reflectionCrop[:, :, 2])) * (1 / (np.max(reflectionCrop[:, :, 2]) - np.min(reflectionCrop[:, :, 2])))
    redMask = stretchedRedReflectionCrop > threshold

    reflectionMask = np.logical_not(np.logical_or(np.logical_or(blueMask, greenMask), redMask))
    inv_reflectionMask = np.logical_not(reflectionMask)
    contours, _ = cv2.findContours(inv_reflectionMask.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)

    boundingRectangle = np.array(list(cv2.boundingRect(contours[max_index])))
    boundingRectangle[0] += reflectionBB[0]
    boundingRectangle[1] += reflectionBB[1]

    if ignoreMask:
        reflectionMask.fill(False)

    showMask = np.stack([reflectionMask.astype('uint8'), reflectionMask.astype('uint8'), reflectionMask.astype('uint8')], axis=2) * 255
    maskedReflections = np.copy(reflectionCrop)
    maskedReflections[reflectionMask] = [0, 0, 0]
    stacked = np.vstack([np.clip(reflectionCrop * 255, 0, 255).astype('uint8'), showMask, np.clip(maskedReflections * 255, 0, 255).astype('uint8')])

    reflectionPoints = reflectionCrop[np.logical_not(reflectionMask)]

    representativeReflectionPoint = __calculateRepresentativeReflectionPoint(reflectionPoints)

    return [representativeReflectionPoint, cleanPixelRatio, stacked, boundingRectangle]

def getEyeWidth(capture):
    """Returns the average width of the eye"""
    [leftP1, leftP2] = capture.landmarks.getLeftEyeWidthPoints()
    [rightP1, rightP2] = capture.landmarks.getRightEyeWidthPoints()

    leftEyeWidth = max(leftP1[0], leftP2[0]) - min(leftP1[0], leftP2[0])
    rightEyeWidth = max(rightP1[0], rightP2[0]) - min(rightP1[0], rightP2[0])

    return (leftEyeWidth + rightEyeWidth) / 2

def getAverageScreenReflectionColor(captures, leftEyeOffsets, rightEyeOffsets, state):
    """Returns data retreived from the users eye including Screen reflection color, reflection size, and scelra color and luminance"""
    wb = captures[0].whiteBalance
    isSpecialCase = [capture.isNoFlash for capture in captures]

    leftEyeCrops = [capture.leftEyeImage for capture in captures]
    leftEyeMasks = [capture.leftEyeMask for capture in captures]

    leftEyeCrops, _ = cropTools.cropImagesToOffsets(leftEyeCrops, leftEyeOffsets)
    leftEyeMasks, _ = cropTools.cropImagesToOffsets(leftEyeMasks, leftEyeOffsets)

    rightEyeCrops = [capture.rightEyeImage for capture in captures]
    rightEyeMasks = [capture.rightEyeMask for capture in captures]

    rightEyeCrops, _ = cropTools.cropImagesToOffsets(rightEyeCrops, rightEyeOffsets)
    rightEyeMasks, _ = cropTools.cropImagesToOffsets(rightEyeMasks, rightEyeOffsets)

    leftReflectionBB = __getReflectionBB(leftEyeCrops, wb)
    rightReflectionBB = __getReflectionBB(rightEyeCrops, wb)

    leftEyeWhiteMask, leftEyeWhiteContour = __getEyeWhiteMask(leftEyeCrops, leftReflectionBB, wb, 'left')
    rightEyeWhiteMask, rightEyeWhiteContour = __getEyeWhiteMask(rightEyeCrops, rightReflectionBB, wb, 'right')

    leftEyeScleraPoints = __extractScleraPoints(leftEyeCrops, leftEyeWhiteMask)
    rightEyeScleraPoints = __extractScleraPoints(rightEyeCrops, rightEyeWhiteMask)
    #cv2.waitKey(0)

    #leftEyeCoords[:, 0:2] += leftOffsets
    #rightEyeCoords[:, 0:2] += rightOffsets

    #RESULTS ARE LINEAR
    leftReflectionStats = np.array([__extractReflectionPoints(leftReflectionBB, eyeCrop, eyeMask, ignoreMask) for eyeCrop, eyeMask, ignoreMask in zip(leftEyeCrops, leftEyeMasks, isSpecialCase)])
    rightReflectionStats = np.array([__extractReflectionPoints(rightReflectionBB, eyeCrop, eyeMask, ignoreMask) for eyeCrop, eyeMask, ignoreMask in zip(rightEyeCrops, rightEyeMasks, isSpecialCase)])

    refinedLeftReflectionBBs = np.vstack(leftReflectionStats[:, 3])
    refinedRightReflectionBBs = np.vstack(rightReflectionStats[:, 3])


    TOLERANCE = 0.20
    leftReflectionBBAreas = [r[2] * r[3] for r in refinedLeftReflectionBBs]
    leftReflectionBBAreasMedian = np.median(leftReflectionBBAreas)
    leftMask = np.abs(leftReflectionBBAreas - leftReflectionBBAreasMedian) > (TOLERANCE * leftReflectionBBAreasMedian)

    rightReflectionBBAreas = [r[2] * r[3] for r in refinedRightReflectionBBs]
    rightReflectionBBAreasMedian = np.median(rightReflectionBBAreas)
    rightMask = np.abs(rightReflectionBBAreas - rightReflectionBBAreasMedian) > (TOLERANCE * rightReflectionBBAreasMedian)

    blurryMask = np.logical_and(leftMask, rightMask)


    annotatedEyeStrips = [__getAnnotatedEyeStrip(leftReflectionBBrefined, leftEyeWhiteContour, leftEyeCrop, rightReflectionBBrefined, rightEyeWhiteContour, rightEyeCrop) for leftEyeCrop, rightEyeCrop, leftReflectionBBrefined, rightReflectionBBrefined in zip(leftEyeCrops, rightEyeCrops, refinedLeftReflectionBBs, refinedRightReflectionBBs)]

    stackedAnnotatedEyeStrips = np.vstack(annotatedEyeStrips)
    state.saveReferenceImageBGR(stackedAnnotatedEyeStrips, 'eyeStrips')

    leftReflectionImages = np.hstack(leftReflectionStats[:, 2])
    rightReflectionImages = np.hstack(rightReflectionStats[:, 2])
    state.saveReferenceImageBGR(leftReflectionImages, 'Left Reflections')
    state.saveReferenceImageBGR(rightReflectionImages, 'Right Reflections')

    averageReflections = (leftReflectionStats[:, 0] + rightReflectionStats[:, 0]) / 2
    averageReflections = [(averageReflection if np.all(averageReflection.astype('bool')) else (averageReflection + np.array([1, 1, 1]))) for averageReflection in averageReflections]

    #Whitebalance per flash and eye to get luminance levels... Maybe compare the average reflection values?
    wbLeftReflections = np.vstack(leftReflectionStats[:, 0])
    wbRightReflections = np.vstack(rightReflectionStats[:, 0])

    #GET Luminance in reflection per flash and eye
    leftReflectionLuminances = [colorTools.getRelativeLuminance([leftReflection])[0] for leftReflection in wbLeftReflections]
    rightReflectionLuminances = [colorTools.getRelativeLuminance([rightReflection])[0] for rightReflection in wbRightReflections]

    eyeWidth = getEyeWidth(captures[0])

    if eyeWidth == 0:
        raise ValueError('Zero value Eye Width')

    leftReflectionWidth, leftReflectionHeight = np.mean(refinedLeftReflectionBBs[:, 2:4], axis=0) / eyeWidth
    rightReflectionWidth, rightReflectionHeight = np.mean(refinedRightReflectionBBs[:, 2:4], axis=0) / eyeWidth

    leftReflectionArea = leftReflectionWidth * leftReflectionHeight
    rightReflectionArea = rightReflectionWidth * rightReflectionHeight

    averageReflectionArea = (leftReflectionArea + rightReflectionArea) / 2

    if min(leftReflectionWidth, rightReflectionWidth) == 0:
        raise ValueError('Zero value reflection Width')

    if min(leftReflectionHeight, rightReflectionHeight) == 0:
        raise ValueError('Zero value reflection Height')

    reflectionWidthRatio = max(leftReflectionWidth, rightReflectionWidth) / min(leftReflectionWidth, rightReflectionWidth)
    reflectionHeightRatio = max(leftReflectionHeight, rightReflectionHeight) / min(leftReflectionHeight, rightReflectionHeight)

    if (reflectionWidthRatio > 1.5) or (reflectionHeightRatio > 1.25):
        #raise ValueError('Reflection Sizes are too different!')
        print('Reflection Sizes are too different!')

    middleIndex = math.floor(len(captures) / 2)

    leftHalfReflectionLuminance = leftReflectionLuminances[middleIndex] * 2 #2x because we are using half
    rightHalfReflectionLuminance = rightReflectionLuminances[middleIndex] * 2 #2x because we are using half

    leftFluxish = leftReflectionArea * leftHalfReflectionLuminance
    rightFluxish = rightReflectionArea * rightHalfReflectionLuminance

    LOGGER.info('LEFT FLUXISH :: %s | AREA ::  %s | LUMINOSITY :: %s', leftFluxish, leftReflectionArea, leftHalfReflectionLuminance)
    LOGGER.info('RIGHT FLUXISH :: %s | AREA ::  %s | LUMINOSITY :: %s', rightFluxish, rightReflectionArea, rightHalfReflectionLuminance)

    return [averageReflections[middleIndex], averageReflectionArea, wbLeftReflections, wbRightReflections, leftEyeScleraPoints, rightEyeScleraPoints, blurryMask]
