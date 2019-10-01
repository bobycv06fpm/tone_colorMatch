import utils
import saveStep
import cv2
import numpy as np
import colorTools
import alignImages
import cropTools
import colorsys
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from logger import getLogger
logger = getLogger(__name__)

def erode(img):
    kernel = np.ones((5, 5), np.uint16)
    #kernel = np.ones((9, 9), np.uint16)
    morph = cv2.morphologyEx(img, cv2.MORPH_CROSS, kernel)
    return morph

def stretchHistogram(gray, mask=None):
    upperBound = 1
    lowerBound = 0

    if mask is not None:
        clippedHigh = gray != upperBound
        clippedLow = gray != lowerBound

        mask = np.logical_and(mask, clippedHigh)
        mask = np.logical_and(mask, clippedLow)

        grayPoints = gray[mask]
    else:
        grayPoints = gray.flatten()

    median = np.median(grayPoints)
    sd = np.std(grayPoints)
    lower = median - (2 * sd)
    lower = lower if lower > lowerBound else lowerBound
    upper = median + (10 * sd)
    upper = upper if upper < upperBound else upperBound

    bounds = np.copy(gray)
    bounds[bounds < lower] = lower
    bounds[bounds > upper] = upper

    numerator = bounds - lower
    denominator = upper - lower
    stretched = (numerator / denominator)
    #stretched = np.clip(stretched * 255, 0, 255)
    return stretched


#Stretches each row individially...
def simpleStretch(grey):
    minValue = np.min(grey, axis=1)
    maxValue = np.max(grey, axis=1)
    return (grey - minValue[:, np.newaxis]) / (maxValue - minValue)[:, np.newaxis]

#Extracts a bounding box of all masked objects together
def simpleMaskBB(mask):
    yAxis = (np.arange(mask.shape[0])[:, None] * mask)[mask]
    xAxis = (np.arange(mask.shape[1]) * mask)[mask]

    yStart = np.min(yAxis)
    yEnd = np.max(yAxis)

    xStart = np.min(xAxis)
    xEnd = np.max(xAxis)

    return np.array([xStart, yStart, xEnd - xStart, yEnd - yStart])

def bbToMask(bb, imgShape):
    img = np.zeros(imgShape)
    img[bb[1]:(bb[1]+bb[3]), bb[0]:(bb[0]+bb[2])] = 1
    return img.astype('bool')

def getEyeWhiteMask(eyes, reflection_bb, wb, label):
    for index, eye in enumerate(eyes):
        if eye.shape[0] * eye.shape[1] == 0:
            raise ValueError('Cannot Find #{} Eye'.format(index))

    eyes = [colorTools.convert_sBGR_to_linearBGR_float_fast(eye) for eye in eyes]
    eyes = [colorTools.whitebalance_from_asShot_to_d65(eye, *wb) for eye in eyes]
    ###TESTING

    #print("Reflection BB :: {}".format(reflection_bb))  
    #print("Size :: {}".format(eyes[0][:, :, 0].shape))
    primarySpecularReflectionBB = reflection_bb
    primarySpecularReflectionBB[0:2] -= reflection_bb[2:4]
    primarySpecularReflectionBB[2:4] *= 3
    primarySpecularReflectionMask = bbToMask(primarySpecularReflectionBB, eyes[0][:, :, 0].shape)
    #print('masked :: {}'.format(primarySpecularReflectionMask.astype('uint8') * 255))
    #cv2.imshow('masked', primarySpecularReflectionMask.astype('uint8') * 255)


    #eye_blur = [cv2.medianBlur(eye, 5) for eye in eyes]
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

    #cv2.imshow('s', np.vstack([np.hstack(eye_s[0:4]), np.hstack(eye_s[4:])]))
    #cv2.imshow('brightest {}'.format(label), eye_s[0])
    #cv2.imshow('dimmest {}'.format(label), eye_s[-1])

    diff = eye_s[0] - eye_s[-1]
    diff = np.clip(diff, 0, 255)
    min_diff = np.min(diff)
    max_diff = np.max(diff)

    scaled_diff = (diff - min_diff) / (max_diff - min_diff)
    scaled_diff = np.clip(scaled_diff * 255, 0, 255).astype('uint8')
    #cv2.imshow('scaled_diff {}'.format(label), scaled_diff)

    ret, thresh = cv2.threshold(scaled_diff[:, :, 0], 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #xBlock = int(scaled_diff[:, :, 0].shape[0] / 4)
    #xBlock = xBlock if xBlock % 2 != 0 else xBlock + 1
    #yBlock = int(scaled_diff[:, :, 0].shape[1] / 4)
    #yBlock = yBlock if yBlock % 2 != 0 else yBlock + 1
    #thresh = cv2.adaptiveThreshold(scaled_diff[:, :, 0], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, xBlock, yBlock)
    #cv2.waitKey(0)
    kernel = np.ones((9, 9), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    #cv2.imshow('thresh {}'.format(label), thresh)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    totalArea = thresh.shape[0] * thresh.shape[1]
    areaPercents = np.array(areas) / totalArea
    areasMask = areaPercents > 0.01
    print('Image Area :: {}'.format(totalArea))
    print('Areas :: {}'.format(areas))
    print('Areas % :: {}'.format(areaPercents))
    print('Area Masks :: {}'.format(areasMask))

    possibleContourIndexes = np.arange(len(contours))[areasMask]
    print('PossibleContourIndexes :: {}'.format(possibleContourIndexes))

    medians = []
    for index in possibleContourIndexes:
        target = np.zeros(thresh.shape, dtype='uint8')
        mask =  cv2.drawContours(target, contours, index, 255, cv2.FILLED)
        med = np.median(eye_s[0][mask.astype('bool')])
        medians.append(med)

    print('Median Values :: {}'.format(medians))
    max_index = possibleContourIndexes[np.argmax(medians)]

    #max_index = np.argmax(areas)
    target = np.zeros(thresh.shape, dtype='uint8')
    sclera_mask =  cv2.drawContours(target, contours, max_index, 255, cv2.FILLED)


    masked_scaled_diff = scaled_diff[:, :, 0]
    masked_scaled_diff[np.logical_not(sclera_mask)] = 0
    print(sclera_mask)
    median = np.median(masked_scaled_diff[sclera_mask.astype('bool')])
    print('MEDIAN :: {}'.format(median))

    #cv2.imshow('masked scaled - {}'.format(label), masked_scaled_diff)
    #ret, thresh2 = cv2.threshold(masked_scaled_diff, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    masked_scaled_diff = cv2.GaussianBlur(masked_scaled_diff,(5,5),0)
    ret, thresh2 = cv2.threshold(masked_scaled_diff, median, 255, cv2.THRESH_BINARY)#+cv2.THRESH_OTSU)

    thresh2 = cv2.morphologyEx(thresh2, cv2.MORPH_OPEN, kernel)
    thresh2 = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel)
    #cv2.imshow('thresh 2 - {}'.format(label), thresh2)
    
    contoursRefined, hierarchy = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areasRefined = [cv2.contourArea(c) for c in contoursRefined]
    maxIndex = np.argmax(areasRefined)
    print('MAX INDEX :: {}'.format(maxIndex))
    target = np.zeros(thresh.shape, dtype='uint8')
    maskRefined =  cv2.drawContours(target, contoursRefined, maxIndex, 255, cv2.FILLED)

    #cv2.imshow('mask 2 - {}'.format(label), maskRefined)
     #= np.stack((maskRefined, maskRefined, maskRefined), axis=-1)

    masked_eyes = np.copy(eyes)
    masked_eyes[:, np.logical_not(maskRefined)] = [0, 0, 0]

    #cv2.imshow('diff', scaled_diff)
    #cv2.imshow('thresh', thresh)
    #cv2.imshow('sclera', sclera_mask)
    #cv2.imshow('sclera {}'.format(label), np.vstack([np.hstack(masked_eyes[0:4]), np.hstack(masked_eyes[4:])]))
    #cv2.waitKey(0)

    return [maskRefined, contoursRefined[maxIndex]]


    ##END TESTING

def maskReflectionBB(eyes, wb):
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

    eyeLap = [cv2.Laplacian(stretchHistogram(img), cv2.CV_64F) for img in croppedGreyEyes]
    eyeLap = eyeLap / np.max(eyeLap)

    totalChangeLap = cv2.Laplacian(totalChange, cv2.CV_64F)
    totalChangeLap = totalChangeLap / np.max(totalChangeLap)

    #im2, contours, hierarchy = cv2.findContours(totalChangeMaskOpenedDilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(totalChangeMaskOpenedDilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]

    highScore = 0
    eyeReflectionBB = None
    gradientMask = None
    for index, contour in enumerate(contours):
        target = np.zeros(totalChangeMaskOpenedDilated.shape, dtype='uint8')
        drawn =  cv2.drawContours(target, contours, index, 255, cv2.FILLED)
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


    #NOTE: UNDER EXPERIMENTATION.... Get Tighter Crops/Detect Blur Better
    # Trying to detect and compare each edge of the reflection for sharpness... 
    # Possibly try and generate a global sharpness metric based on the "sharpness" of the reflection edge

    #x, y, w, h = eyeReflectionBB
    #middlePoint = [x + int(0.5 * w), y + int(0.5 * h)]
    #print('Middle Point :: ' + str(middlePoint))

    #halfSampleWidth = 3

    ##columnBB = [middlePoint[0] - halfSampleWidth, 0, 2 * halfSampleWidth, h]
    ##rowBB = [0, middlePoint[1] - halfSampleWidth, w, 2 * halfSampleWidth]

    #topColumnMask = np.zeros(gradientMask.shape, dtype='bool')
    #topColumnMask[:middlePoint[1], (middlePoint[0] - halfSampleWidth):(middlePoint[0] + halfSampleWidth)] = True

    #bottomColumnMask = np.zeros(gradientMask.shape, dtype='bool')
    #bottomColumnMask[middlePoint[1]:, (middlePoint[0] - halfSampleWidth):(middlePoint[0] + halfSampleWidth)] = True

    #leftRowMask = np.zeros(gradientMask.shape, dtype='bool')
    #leftRowMask[(middlePoint[1] - halfSampleWidth):(middlePoint[1] + halfSampleWidth), :middlePoint[0]] = True

    #rightRowMask = np.zeros(gradientMask.shape, dtype='bool')
    #rightRowMask[(middlePoint[1] - halfSampleWidth):(middlePoint[1] + halfSampleWidth), middlePoint[0]:] = True

    ##crossMask = np.logical_or(columnMask, rowMask)
    ##sampleZonesMask = np.logical_and(crossMask, gradientMask)
    #topMask = np.logical_and(topColumnMask, gradientMask)
    #bottomMask = np.logical_and(bottomColumnMask, gradientMask)
    #leftMask = np.logical_and(leftRowMask, gradientMask)
    #rightMask = np.logical_and(rightRowMask, gradientMask)

    #topMaskBB = simpleMaskBB(topMask)
    ##topMaskBB = topMaskBB + [0, int(topMaskBB[3] / 2), 0, 0]
    #topMaskBB = topMaskBB + [0, 0, 0, topMaskBB[3]]
    #topMaskShifted = np.zeros(topMask.shape)
    #topMaskShifted[topMaskBB[1]:topMaskBB[1] + topMaskBB[3], topMaskBB[0]:topMaskBB[0] + topMaskBB[2]] = 255

    #bottomMaskBB = simpleMaskBB(bottomMask)
    ##bottomMaskBB = bottomMaskBB - [0, int(bottomMaskBB[3] / 2), 0, 0]
    #bottomMaskBB = bottomMaskBB - [0, bottomMaskBB[3], 0, (-1) * bottomMaskBB[3]]
    #bottomMaskShifted = np.zeros(bottomMask.shape)
    #bottomMaskShifted[bottomMaskBB[1]:bottomMaskBB[1] + bottomMaskBB[3], bottomMaskBB[0]:bottomMaskBB[0] + bottomMaskBB[2]] = 255

    #leftMaskBB = simpleMaskBB(leftMask)
    ##leftMaskBB = leftMaskBB + [int(leftMaskBB[2] / 2), 0, 0, 0]
    #leftMaskBB = leftMaskBB + [0, 0, leftMaskBB[2], 0]
    #leftMaskShifted = np.zeros(leftMask.shape)
    #leftMaskShifted[leftMaskBB[1]:leftMaskBB[1] + leftMaskBB[3], leftMaskBB[0]:leftMaskBB[0] + leftMaskBB[2]] = 255

    #rightMaskBB = simpleMaskBB(rightMask)
    ##rightMaskBB = rightMaskBB - [int(rightMaskBB[2] / 2), 0, 0, 0]
    #rightMaskBB = rightMaskBB - [rightMaskBB[2], 0, (-1) * rightMaskBB[2], 0]
    #rightMaskShifted = np.zeros(rightMask.shape)
    #rightMaskShifted[rightMaskBB[1]:rightMaskBB[1] + rightMaskBB[3], rightMaskBB[0]:rightMaskBB[0] + rightMaskBB[2]] = 255

    #joinedMask = np.logical_or(topMask, bottomMask)
    #joinedMask = np.logical_or(joinedMask, leftMask)
    #joinedMask = np.logical_or(joinedMask, rightMask)

    #joinedMaskShifted = np.logical_or(topMaskShifted, bottomMaskShifted)
    #joinedMaskShifted = np.logical_or(joinedMaskShifted, leftMaskShifted)
    #joinedMaskShifted = np.logical_or(joinedMaskShifted, rightMaskShifted)

    #test = np.copy(croppedGreyEyes[0])
    #test[np.logical_not(joinedMask)] = 0

    #testShifted = np.copy(croppedGreyEyes[0])
    #testShifted[np.logical_not(joinedMaskShifted)] = 0

    ##cv2.imshow('test', )

    #print('BBs :: {} | {} | {} | {}'.format(topMaskBB, bottomMaskBB, leftMaskBB, rightMaskBB))

    #stack0 = np.vstack([croppedGreyEyes[0], test])
    #stack1 = np.vstack([croppedGreyEyes[0], testShifted])
    #stack2 = np.vstack([gradientMask, joinedMask]).astype('uint8') * 255


    ##cv2.imshow('mask', np.hstack([stack2, stack0, stack1])) cv2.waitKey(0)
    ##cv2.waitKey(0)

    ##refCrops = [eyeCrop[y:y+h, x:x+w] for eyeCrop in croppedGreyEyes]
    #kernel = np.ones((5, 5), np.uint8)
#   # mask = cv2.dilate(mask, kernel, iterations=1).astype('bool') #mask.astype('bool')
#   # invMask = np.logical_not(mask)
#   # eyeCrops = []

    #middlePoint = [math.floor(w / 2), math.floor(h / 2)]
    #sampleHeight = math.floor(w / 2)

    #topColumnSampleBB = topMaskBB#[middlePoint[0] - halfSampleWidth, 0, 2 * halfSampleWidth, sampleHeight]
    #bottomColumnSampleBB = bottomMaskBB#[middlePoint[0] - halfSampleWidth, (h - sampleHeight), 2 * halfSampleWidth, sampleHeight]

    #leftRowSampleBB = leftMaskBB#[0, middlePoint[1] - halfSampleWidth, sampleHeight, 2 * halfSampleWidth]
    #rightRowSampleBB = rightMaskBB#[middlePoint[0], w - sampleHeight, sampleHeight, 2 * halfSampleWidth]

    #topColumns = [simpleStretch(np.rot90(crop[topColumnSampleBB[1]:topColumnSampleBB[1] + topColumnSampleBB[3], topColumnSampleBB[0]:topColumnSampleBB[0] + topColumnSampleBB[2]], 1)) for crop in croppedGreyEyes]

    #bottomColumns = [simpleStretch(np.rot90(crop[bottomColumnSampleBB[1]:bottomColumnSampleBB[1] + bottomColumnSampleBB[3], bottomColumnSampleBB[0]:bottomColumnSampleBB[0] + bottomColumnSampleBB[2]], 3)) for crop in croppedGreyEyes]

    #leftRows = [simpleStretch(crop[leftRowSampleBB[1]:leftRowSampleBB[1] + leftRowSampleBB[3], leftRowSampleBB[0]:leftRowSampleBB[0] + leftRowSampleBB[2]]) for crop in croppedGreyEyes]

    #rightRows = [simpleStretch(np.rot90(crop[rightRowSampleBB[1]:rightRowSampleBB[1] + rightRowSampleBB[3], rightRowSampleBB[0]:rightRowSampleBB[0] + rightRowSampleBB[2]], 2)) for crop in croppedGreyEyes]

    #topColumnStack = np.vstack(topColumns)
    #bottomColumnStack = np.vstack(bottomColumns)
    #leftRowStack = np.vstack(leftRows)
    #rightRowStack = np.vstack(rightRows)

    ##cv2.imshow('AllEdgeSamples', np.hstack([topColumnStack, bottomColumnStack, leftRowStack, rightRowStack]))
    ##cv2.imshow('rows', np.hstack([leftRowStack, rightRowStack]))

    #medianTopColumns = [np.median(topColumn, axis=0) for topColumn in topColumns]
    #medianBottomColumns = [np.median(bottomColumn, axis=0) for bottomColumn in bottomColumns]
    #medianLeftRows = [np.median(leftRow, axis=0) for leftRow in leftRows]
    #medianRightRows = [np.median(rightRow, axis=0) for rightRow in rightRows]

    #medianTopColumnStack = np.vstack(medianTopColumns)
    #medianBottomColumnStack = np.vstack(medianBottomColumns)
    #medianLeftRowStack = np.vstack(medianLeftRows)
    #medianRightRowStack = np.vstack(medianRightRows)
    #NOTE: DONE EXPERIMENTATION

    eyeReflectionBB[0] += eyeCropX1
    eyeReflectionBB[1] += eyeCropY1
    return np.array(eyeReflectionBB)

#def cropToBB(image, bb):
#    [x, y, w, h] = bb
#    return image[y:y+h, x:x+w]

def getAnnotatedEyeStrip2(leftReflectionBB, leftScleraContour, leftEyeCrop, rightReflectionBB, rightScleraContour, rightEyeCrop):

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
    originLeft_Y_end =  -1 * math.ceil((canvasShape[0] - leftEyeCropCopy.shape[0]) / 2) #Center vertically
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

def calculateRepresentativeReflectionPoint(reflectionPoints):
    #return np.median(reflectionPoints, axis=0) # Maybe change to only take median of top 10% of brightnesses?
    #old = np.median(reflectionPoints, axis=0) # Maybe change to only take median of top 10% of brightnesses?
    numPoints = reflectionPoints.shape[0]

    oneTenth = int(numPoints / 10) * -1

    topMedianBlue = np.median(np.array(sorted(reflectionPoints[:, 0]))[oneTenth:])
    topMedianGreen = np.median(np.array(sorted(reflectionPoints[:, 1]))[oneTenth:])
    topMedianRed = np.median(np.array(sorted(reflectionPoints[:, 2]))[oneTenth:])

    newRepValue = [topMedianBlue, topMedianGreen, topMedianRed]
    #print('Old :: {} | New :: {}'.format(old, newRepValue))
    return np.array(newRepValue)

def extractScleraPoints(eyes, scleraMask):
    return []

def extractReflectionPoints(reflectionBB, eyeCrop, eyeMask, ignoreMask):

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
    contours, hierarchy = cv2.findContours(inv_reflectionMask.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
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

    representativeReflectionPoint = calculateRepresentativeReflectionPoint(reflectionPoints)

    return [representativeReflectionPoint, cleanPixelRatio, stacked, boundingRectangle]

def getEyeWidth(capture):
    [leftP1, leftP2] = capture.landmarks.getLeftEyeWidthPoints()
    [rightP1, rightP2] = capture.landmarks.getRightEyeWidthPoints()

    leftEyeWidth = max(leftP1[0], leftP2[0]) - min(leftP1[0], leftP2[0])
    rightEyeWidth = max(rightP1[0], rightP2[0]) - min(rightP1[0], rightP2[0])

    return (leftEyeWidth + rightEyeWidth) / 2

def getEyeWhiteSample(eye, leftPoint, rightPoint):
    return None

def getAverageScreenReflectionColor2(captures, leftEyeOffsets, rightEyeOffsets, saveStep):
    wb = captures[0].getAsShotWhiteBalance()
    isSpecialCase = [capture.isNoFlash for capture in captures]

    leftEyeCrops = [capture.leftEyeImage for capture in captures]
    leftEyeMasks = [capture.leftEyeMask for capture in captures]

    leftEyeCrops, leftOffsets = cropTools.cropImagesToOffsets(leftEyeCrops, leftEyeOffsets)
    leftEyeMasks, offsets = cropTools.cropImagesToOffsets(leftEyeMasks, leftEyeOffsets)

    #cv2.imshow('left eye', np.vstack(leftEyeCrops))
    #cv2.waitKey(0)

    rightEyeCrops = [capture.rightEyeImage for capture in captures]
    rightEyeMasks = [capture.rightEyeMask for capture in captures]

    rightEyeCrops, rightOffsets = cropTools.cropImagesToOffsets(rightEyeCrops, rightEyeOffsets)
    rightEyeMasks, offsets = cropTools.cropImagesToOffsets(rightEyeMasks, rightEyeOffsets)

    leftReflectionBB = maskReflectionBB(leftEyeCrops, wb)
    rightReflectionBB = maskReflectionBB(rightEyeCrops, wb)

    leftEyeWhiteMask, leftEyeWhiteContour  = getEyeWhiteMask(leftEyeCrops, leftReflectionBB, wb, 'left')
    rightEyeWhiteMask, rightEyeWhiteContour = getEyeWhiteMask(rightEyeCrops, rightReflectionBB, wb, 'right')

    extractScleraPoints(leftEyeCrops, leftEyeWhiteMask)
    extractScleraPoints(rightEyeCrops, rightEyeWhiteMask)
    #cv2.waitKey(0)

    #leftEyeCoords[:, 0:2] += leftOffsets
    #rightEyeCoords[:, 0:2] += rightOffsets

    #RESULTS ARE LINEAR
    leftReflectionStats = np.array([extractReflectionPoints(leftReflectionBB, eyeCrop, eyeMask, ignoreMask) for eyeCrop, eyeMask, ignoreMask in zip(leftEyeCrops, leftEyeMasks, isSpecialCase)])
    rightReflectionStats = np.array([extractReflectionPoints(rightReflectionBB, eyeCrop, eyeMask, ignoreMask) for eyeCrop, eyeMask, ignoreMask in zip(rightEyeCrops, rightEyeMasks, isSpecialCase)])

    refinedLeftReflectionBBs = np.vstack(leftReflectionStats[:, 3])
    refinedRightReflectionBBs = np.vstack(rightReflectionStats[:, 3])

    annotatedEyeStrips = [getAnnotatedEyeStrip2(leftReflectionBBrefined, leftEyeWhiteContour, leftEyeCrop, rightReflectionBBrefined, rightEyeWhiteContour, rightEyeCrop) for leftEyeCrop, rightEyeCrop, leftReflectionBBrefined, rightReflectionBBrefined in zip(leftEyeCrops, rightEyeCrops, refinedLeftReflectionBBs, refinedRightReflectionBBs)]

    stackedAnnotatedEyeStrips = np.vstack(annotatedEyeStrips)
    saveStep.saveReferenceImageBGR(stackedAnnotatedEyeStrips, 'eyeStrips')

    leftReflectionImages = np.hstack(leftReflectionStats[:, 2])
    rightReflectionImages = np.hstack(rightReflectionStats[:, 2])
    saveStep.saveReferenceImageSBGR(leftReflectionImages, 'Left Reflections')
    saveStep.saveReferenceImageSBGR(rightReflectionImages, 'Right Reflections')

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
        raise ValueError('Reflection Sizes are too different!')

    middleIndex = math.floor(len(captures) / 2)

    leftHalfReflectionLuminance = leftReflectionLuminances[middleIndex] * 2 #2x because we are using half
    rightHalfReflectionLuminance = rightReflectionLuminances[middleIndex] * 2 #2x because we are using half

    leftFluxish = leftReflectionArea * leftHalfReflectionLuminance
    rightFluxish = rightReflectionArea * rightHalfReflectionLuminance

    logger.info('LEFT FLUXISH :: {} | AREA ::  {} | LUMINOSITY :: {}'.format(leftFluxish, leftReflectionArea, leftHalfReflectionLuminance))
    logger.info('RIGHT FLUXISH :: {} | AREA ::  {} | LUMINOSITY :: {}'.format(rightFluxish, rightReflectionArea, rightHalfReflectionLuminance))

    return [averageReflections[middleIndex], averageReflectionArea, wbLeftReflections, wbRightReflections]

