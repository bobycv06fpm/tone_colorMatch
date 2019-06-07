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

def getLeftEyeCrop(capture):
    (lx, ly, w, h) = capture.landmarks.getLeftEyeBB()
    leftEye = capture.image[ly:ly+h, lx:lx+w]
    leftEyeMask = capture.mask[ly:ly+h, lx:lx+w]
    return np.array([leftEye, leftEyeMask, [lx, ly]])

def getRightEyeCrop(capture):
    (rx, ry, w, h) = capture.landmarks.getRightEyeBB()
    rightEye = capture.image[ry:ry+h, rx:rx+w]
    rightEyeMask = capture.mask[ry:ry+h, rx:rx+w]
    return np.array([rightEye, rightEyeMask, [rx, ry]])

def getRightEyeCoords(capture):
    #return np.array(capture.landmarks.getRightEyeBB())
    #return np.array(capture.landmarks.getRightEyeInnerBBBuffered())
    return np.array(capture.landmarks.getRightEyeBBBuffered())

def getLeftEyeCoords(capture):
    #return np.array(capture.landmarks.getLeftEyeBB())
    #return np.array(capture.landmarks.getLeftEyeInnerBBBuffered())
    return np.array(capture.landmarks.getLeftEyeBBBuffered())

def getMask(capture, coords):
    [x, y, w, h] = coords
    return capture.mask[y:y+h, x:x+w]

def getCrop(capture, coords):
    [x, y, w, h] = coords
    return capture.image[y:y+h, x:x+w]
    #wbImage = capture.getWhiteBalancedImageToD65()
    #return wbImage[y:y+h, x:x+w]

def getEyeCrops(capture):
    leftEyeCrop = getLeftEyeCrop(capture)
    rightEyeCrop = getRightEyeCrop(capture)
    return np.array([leftEyeCrop, rightEyeCrop])

def getEyeCropsInner(capture):
    (lx, ly, w, h) = capture.landmarks.getLeftEyeInnerBB()
    leftEye = capture.image[ly:ly+h, lx:lx+w]
    leftEyeMask = capture.mask[ly:ly+h, lx:lx+w]

    (rx, ry, w, h) = capture.landmarks.getRightEyeInnerBB()
    rightEye = capture.image[ry:ry+h, rx:rx+w]
    rightEyeMask = capture.mask[ry:ry+h, rx:rx+w]

    return np.array([[leftEye, leftEyeMask, [lx, ly]], [rightEye, rightEyeMask, [rx, ry]]])

def blur(img):
    return cv2.GaussianBlur(img, (9, 9), 0)
    #return cv2.bilateralFilter(img,15,75,75)
    #return cv2.medianBlur(img, 9)

def erode(img):
    kernel = np.ones((5, 5), np.uint16)
    #kernel = np.ones((9, 9), np.uint16)
    morph = cv2.morphologyEx(img, cv2.MORPH_CROSS, kernel)
    return morph

def getReflectionBB(maskedImg):
    img = np.clip(maskedImg * 255, 0, 255).astype('uint8')
    im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]

    if not areas:
        print("NO REFLECTION FOUND")
        raise NameError('NO REFLECTION FOUND')

    bbs = [cv2.boundingRect(contour) for contour in contours]
    bb_medians = np.array([np.median(maskedImg[bb[1]:bb[1] + bb[3], bb[0]:bb[0] + bb[2]]) for bb in bbs])

    areaScores = np.array(areas) * (bb_medians ** 2)
    #print('Areas :: {}'.format(areas))
    #print('Medians :: {}'.format(bb_medians))
    #print('Area Scores :: {}'.format(areaScores))
    max_index = np.argmax(areaScores)
    #print('Chosen :: {} | {} | {}'.format(areas[max_index], bb_medians[max_index], areaScores[max_index]))

    return np.array(bbs[max_index])

def maskTopValues(img):
    median = np.median(img)
    std = np.std(img)
    #threshold = median + (3 * std)
    threshold = median + (2.5 * std)
    img[img < threshold] = 0
    return img

def maskBottomValues(img):
    mean = np.mean(img)
    #std = np.std(img)
    #threshold = median + (3 * std)
    threshold = mean
    img[img < threshold] = 0
    return img

def extractBBMask(img, BB):
    x, y, w, h = BB
    mask = np.ones(img.shape).astype('bool')
    mask[y:y+h, x:x+w] = False
    img[mask] = 0
    return img

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


    #stretchedGrey = []
    #for row in grey:
    #    print('row :: ' + str(row))
    #    minValue = np.min(grey)
    #    maxValue = np.max(grey)
    #    stretchedRow = (grey - minValue) / (maxValue - minValue)
    #    stretchedGrey.append(stretchedRow)

#Extracts a bounding box of all masked objects together
def simpleMaskBB(mask):
    yAxis = (np.arange(mask.shape[0])[:, None] * mask)[mask]
    xAxis = (np.arange(mask.shape[1]) * mask)[mask]

    yStart = np.min(yAxis)
    yEnd = np.max(yAxis)

    xStart = np.min(xAxis)
    xEnd = np.max(xAxis)

    return np.array([xStart, yStart, xEnd - xStart, yEnd - yStart])


    #return np.array(stretchedRow)
def maskReflectionBB(eyes, wb):
    for index, eye in enumerate(eyes):
        if eye.shape[0] * eye.shape[1] == 0:
            raise NameError('Cannot Find #{} Eye'.format(index))

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

    im2, contours, hierarchy = cv2.findContours(totalChangeMaskOpenedDilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
        raise NameError('Could Not Find Reflection BB')


    x, y, w, h = eyeReflectionBB
    middlePoint = [x + int(0.5 * w), y + int(0.5 * h)]
    print('Middle Point :: ' + str(middlePoint))

    halfSampleWidth = 3

    #columnBB = [middlePoint[0] - halfSampleWidth, 0, 2 * halfSampleWidth, h]
    #rowBB = [0, middlePoint[1] - halfSampleWidth, w, 2 * halfSampleWidth]

    topColumnMask = np.zeros(gradientMask.shape, dtype='bool')
    topColumnMask[:middlePoint[1], (middlePoint[0] - halfSampleWidth):(middlePoint[0] + halfSampleWidth)] = True

    bottomColumnMask = np.zeros(gradientMask.shape, dtype='bool')
    bottomColumnMask[middlePoint[1]:, (middlePoint[0] - halfSampleWidth):(middlePoint[0] + halfSampleWidth)] = True

    leftRowMask = np.zeros(gradientMask.shape, dtype='bool')
    leftRowMask[(middlePoint[1] - halfSampleWidth):(middlePoint[1] + halfSampleWidth), :middlePoint[0]] = True

    rightRowMask = np.zeros(gradientMask.shape, dtype='bool')
    rightRowMask[(middlePoint[1] - halfSampleWidth):(middlePoint[1] + halfSampleWidth), middlePoint[0]:] = True

    #crossMask = np.logical_or(columnMask, rowMask)
    #sampleZonesMask = np.logical_and(crossMask, gradientMask)
    topMask = np.logical_and(topColumnMask, gradientMask)
    bottomMask = np.logical_and(bottomColumnMask, gradientMask)
    leftMask = np.logical_and(leftRowMask, gradientMask)
    rightMask = np.logical_and(rightRowMask, gradientMask)

    topMaskBB = simpleMaskBB(topMask)
    #topMaskBB = topMaskBB + [0, int(topMaskBB[3] / 2), 0, 0]
    topMaskBB = topMaskBB + [0, 0, 0, topMaskBB[3]]
    topMaskShifted = np.zeros(topMask.shape)
    topMaskShifted[topMaskBB[1]:topMaskBB[1] + topMaskBB[3], topMaskBB[0]:topMaskBB[0] + topMaskBB[2]] = 255

    bottomMaskBB = simpleMaskBB(bottomMask)
    #bottomMaskBB = bottomMaskBB - [0, int(bottomMaskBB[3] / 2), 0, 0]
    bottomMaskBB = bottomMaskBB - [0, bottomMaskBB[3], 0, (-1) * bottomMaskBB[3]]
    bottomMaskShifted = np.zeros(bottomMask.shape)
    bottomMaskShifted[bottomMaskBB[1]:bottomMaskBB[1] + bottomMaskBB[3], bottomMaskBB[0]:bottomMaskBB[0] + bottomMaskBB[2]] = 255

    leftMaskBB = simpleMaskBB(leftMask)
    #leftMaskBB = leftMaskBB + [int(leftMaskBB[2] / 2), 0, 0, 0]
    leftMaskBB = leftMaskBB + [0, 0, leftMaskBB[2], 0]
    leftMaskShifted = np.zeros(leftMask.shape)
    leftMaskShifted[leftMaskBB[1]:leftMaskBB[1] + leftMaskBB[3], leftMaskBB[0]:leftMaskBB[0] + leftMaskBB[2]] = 255

    rightMaskBB = simpleMaskBB(rightMask)
    #rightMaskBB = rightMaskBB - [int(rightMaskBB[2] / 2), 0, 0, 0]
    rightMaskBB = rightMaskBB - [rightMaskBB[2], 0, (-1) * rightMaskBB[2], 0]
    rightMaskShifted = np.zeros(rightMask.shape)
    rightMaskShifted[rightMaskBB[1]:rightMaskBB[1] + rightMaskBB[3], rightMaskBB[0]:rightMaskBB[0] + rightMaskBB[2]] = 255

    joinedMask = np.logical_or(topMask, bottomMask)
    joinedMask = np.logical_or(joinedMask, leftMask)
    joinedMask = np.logical_or(joinedMask, rightMask)

    joinedMaskShifted = np.logical_or(topMaskShifted, bottomMaskShifted)
    joinedMaskShifted = np.logical_or(joinedMaskShifted, leftMaskShifted)
    joinedMaskShifted = np.logical_or(joinedMaskShifted, rightMaskShifted)

    test = np.copy(croppedGreyEyes[0])
    test[np.logical_not(joinedMask)] = 0

    testShifted = np.copy(croppedGreyEyes[0])
    testShifted[np.logical_not(joinedMaskShifted)] = 0

    #cv2.imshow('test', )

    print('BBs :: {} | {} | {} | {}'.format(topMaskBB, bottomMaskBB, leftMaskBB, rightMaskBB))

    stack0 = np.vstack([croppedGreyEyes[0], test])
    stack1 = np.vstack([croppedGreyEyes[0], testShifted])
    stack2 = np.vstack([gradientMask, joinedMask]).astype('uint8') * 255


    cv2.imshow('mask', np.hstack([stack2, stack0, stack1]))
    cv2.waitKey(0)


    #refCrops = [eyeCrop[y:y+h, x:x+w] for eyeCrop in croppedGreyEyes]
    kernel = np.ones((5, 5), np.uint8)
#    mask = cv2.dilate(mask, kernel, iterations=1).astype('bool') #mask.astype('bool')
#    invMask = np.logical_not(mask)
#    eyeCrops = []

    middlePoint = [math.floor(w / 2), math.floor(h / 2)]
    sampleHeight = math.floor(w / 2)

    topColumnSampleBB = topMaskBB#[middlePoint[0] - halfSampleWidth, 0, 2 * halfSampleWidth, sampleHeight]
    bottomColumnSampleBB = bottomMaskBB#[middlePoint[0] - halfSampleWidth, (h - sampleHeight), 2 * halfSampleWidth, sampleHeight]

    leftRowSampleBB = leftMaskBB#[0, middlePoint[1] - halfSampleWidth, sampleHeight, 2 * halfSampleWidth]
    rightRowSampleBB = rightMaskBB#[middlePoint[0], w - sampleHeight, sampleHeight, 2 * halfSampleWidth]

    topColumns = [simpleStretch(np.rot90(crop[topColumnSampleBB[1]:topColumnSampleBB[1] + topColumnSampleBB[3], topColumnSampleBB[0]:topColumnSampleBB[0] + topColumnSampleBB[2]], 1)) for crop in croppedGreyEyes]

    bottomColumns = [simpleStretch(np.rot90(crop[bottomColumnSampleBB[1]:bottomColumnSampleBB[1] + bottomColumnSampleBB[3], bottomColumnSampleBB[0]:bottomColumnSampleBB[0] + bottomColumnSampleBB[2]], 3)) for crop in croppedGreyEyes]

    leftRows = [simpleStretch(crop[leftRowSampleBB[1]:leftRowSampleBB[1] + leftRowSampleBB[3], leftRowSampleBB[0]:leftRowSampleBB[0] + leftRowSampleBB[2]]) for crop in croppedGreyEyes]

    rightRows = [simpleStretch(np.rot90(crop[rightRowSampleBB[1]:rightRowSampleBB[1] + rightRowSampleBB[3], rightRowSampleBB[0]:rightRowSampleBB[0] + rightRowSampleBB[2]], 2)) for crop in croppedGreyEyes]

    topColumnStack = np.vstack(topColumns)
    bottomColumnStack = np.vstack(bottomColumns)
    leftRowStack = np.vstack(leftRows)
    rightRowStack = np.vstack(rightRows)

    cv2.imshow('AllEdgeSamples', np.hstack([topColumnStack, bottomColumnStack, leftRowStack, rightRowStack]))
    #cv2.imshow('rows', np.hstack([leftRowStack, rightRowStack]))

    medianTopColumns = [np.median(topColumn, axis=0) for topColumn in topColumns]
    medianBottomColumns = [np.median(bottomColumn, axis=0) for bottomColumn in bottomColumns]
    medianLeftRows = [np.median(leftRow, axis=0) for leftRow in leftRows]
    medianRightRows = [np.median(rightRow, axis=0) for rightRow in rightRows]

    medianTopColumnStack = np.vstack(medianTopColumns)
    medianBottomColumnStack = np.vstack(medianBottomColumns)
    medianLeftRowStack = np.vstack(medianLeftRows)
    medianRightRowStack = np.vstack(medianRightRows)

    cv2.imshow('AllEdgeSamplesMedians', np.hstack([medianTopColumnStack, medianBottomColumnStack, medianLeftRowStack, medianRightRowStack]))
    cv2.waitKey(0)

    
    for idx in range(0, len(medianTopColumns)):
        plotIndex = 481

        plt.subplot(4, 8, 1 + idx)
        #plt.hist(topColumns[idx].flatten())
        plt.scatter(np.arange(len(medianTopColumns[idx])), medianTopColumns[idx])

        plt.subplot(4, 8, 9 + idx)
        #plt.hist(bottomColumns[idx].flatten())
        plt.scatter(np.arange(len(medianBottomColumns[idx])), medianBottomColumns[idx])

        plt.subplot(4, 8, 17 + idx)
        #plt.hist(leftRows[idx].flatten())
        plt.scatter(np.arange(len(medianLeftRows[idx])), medianLeftRows[idx])

        plt.subplot(4, 8, 25 + idx)
        #plt.hist(rightRows[idx].flatten())
        plt.scatter(np.arange(len(medianRightRows[idx])), medianRightRows[idx])

    plt.show()



    #refinedRefCrops = []
    #refinedGradCrops = []
    #laps = []
    #lapsMasked = []
    #lapsMaskedMean = []
    #gDiffs = []
    #gDiffsMasked = []
    #gDiffsMaskedMean = []

    #for refCrop in refCrops:
    #    minValue = np.min(refCrop)
    #    maxValue = np.max(refCrop)
    #    refCrop = refCrop - minValue
    #    refCrop = refCrop * (1 / (maxValue - minValue))
    #    refCrop = np.clip(refCrop, 0, 1)
    #    refinedRefCrops.append(refCrop)

    #    refCrop2 = np.copy(refCrop)

    #    target = np.median(refCrop2)

    #    gradCropMask = refCrop2 > target
    #    #cv2.imshow('gradCropMask', gradCropMask.astype('uint8') * 255)
    #    gradCropMask = cv2.morphologyEx(gradCropMask.astype('uint8'), cv2.MORPH_GRADIENT, kernel)
    #    refCrop2[np.logical_not(gradCropMask)] = 0
    #    refinedGradCrops.append(refCrop2)

    #    lap = cv2.Laplacian(refCrop, cv2.CV_64F)
    #    laps.append(lap)

    #    lapMasked = np.copy(lap)
    #    lapMaskedMean = np.mean(lapMasked[gradCropMask])
    #    lapMasked[np.logical_not(gradCropMask)] = 0
    #    lapsMasked.append(lapMasked)
    #    lapsMaskedMean.append(lapMaskedMean)

    #    blurred = cv2.GaussianBlur(refCrop, (7, 7), 0)
    #    blurredLite = cv2.GaussianBlur(refCrop, (3, 3), 0)

    #    #gDiff = np.abs(refCrop - blurred)
    #    #gDiff = np.abs(blurredLite - blurred)
    #    gDiff = np.clip(-1 * (blurredLite - blurred), 0, 1)
    #    gDiffs.append(gDiff)

    #    gDiffMasked = np.copy(gDiff)
    #    gDiffPoints = gDiffMasked[gradCropMask]
    #    gDiffMaskedMean = np.mean(gDiffPoints[gDiffPoints > (0)])
    #    #gDiffMaskedMean = np.mean(gDiffMasked)
    #    gDiffMasked[np.logical_not(gradCropMask)] = 0
    #    gDiffsMasked.append(gDiffMasked)
    #    gDiffsMaskedMean.append(gDiffMaskedMean)
    #   
    #    
    #    
    #    #refCrop[invMask] = 0
    #    #eyeCrops.append(eyeCrop)

    #refCropStack = np.vstack(refCrops)
    #refinedRefCropStack = np.vstack(refinedRefCrops)
    #refinedGradCropStack = np.vstack(refinedGradCrops)
    ##cv2.imshow('ref stack', refCropStack)
    ##cv2.imshow('refined stack', refinedRefCropStack)

    ##laps = [cv2.Laplacian(crop, cv2.CV_64F) for crop in refinedRefCrops]
    #lapStack = np.vstack(laps)
    #lapMaskedStack = np.vstack(lapsMasked)
    #gDiffStacked = np.vstack(gDiffs)
    #gDiffMaskedStack = np.vstack(gDiffsMasked)
    #laps_means = lapsMaskedMean#[np.mean(lap) for lap in lapsMaskedPoints]

    #laps_top10 = []

    ##plt.subplot(131)
    ##plt.scatter(hsvPoints[:, 0], hsvPoints[:, 2], 50, colors)
    ##plt.xlabel('Hue')
    ##plt.ylabel('Value')

    ##plotIndexBase = 100 + len(refinedRefCrops) * 10

    ##for idx, ref in enumerate(refinedRefCrops):
    ##    plotIndex = plotIndexBase + idx + 1
    ##    greaterThanZero = ref[ref>0.2]
    ##    plt.subplot(plotIndex)
    ##    plt.hist(greaterThanZero, bins=25, cumulative=False)

    ##plt.show()
    ##for lap in laps:
    ##    greaterThanZero = lap[lap>0]
    ##    margin = -1 * 0.05
    ##    index = int(margin * len(greaterThanZero))
    ##    #bottom5_index = int(margin * len(greaterThanZero))
    ##    #print('Top 10 index :: ' + str(top10_index))
    ##    middle90 = sorted(greaterThanZero)[index:index]
    ##    top10_mean = np.mean(middle90)
    ##    laps_top10.append(top10_mean)
    ##laps_top10 = [np.mean(sorted(lap[lap>0])[-10:]) for lap in laps]
    #print('BLURRIENESS SCORES :: {}'.format(gDiffsMaskedMean))
    ##print('BLURRIENESS SCORES2:: {}'.format(laps_top10))

    ##cv2.imshow('ffts', ffts_shifted[0] / np.max(ffts_shifted))
    ##cv2.waitKey(0)
    ##greyLeftEyeCropsLinearStretchedFFTShiftedMeans = [np.mean(leftEyeFFT) for leftEyeFFT in greyLeftEyeCropsLinearStretchedFFTShifted]

    #lapStack = lapStack / np.max(lapStack)
    #lapMaskedStack = lapMaskedStack / np.max(lapMaskedStack)
    #gDiffStacked = gDiffStacked / np.max(gDiffStacked)
    #gDiffMaskedStack = gDiffMaskedStack / np.max(gDiffMaskedStack)

    #cv2.imshow('stack', np.hstack([refCropStack, refinedRefCropStack, refinedGradCropStack, lapStack, lapMaskedStack, gDiffStacked, gDiffMaskedStack]))
    #cv2.waitKey(0)


    print('Eye Reflection BB :: ' + str(eyeReflectionBB))
    eyeReflectionBB[0] += eyeCropX1
    eyeReflectionBB[1] += eyeCropY1
    return np.array(eyeReflectionBB)

def cropToBB(image, bb):
    [x, y, w, h] = bb
    return image[y:y+h, x:x+w]

def getAnnotatedEyeStrip2(leftReflectionBB, leftEyeCrop, rightReflectionBB, rightEyeCrop):

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

#def getAnnotatedEyeStrip(leftReflectionBB, leftOffsetCoords, rightReflectionBB, rightOffsetCoords, capture):
#    eyeStripBB = np.array(capture.landmarks.getEyeStripBB())
#
#    eyeWidthPoints = np.append(capture.landmarks.getLeftEyeWidthPoints(), capture.landmarks.getRightEyeWidthPoints(), axis=0)
#
#    eyeWidthPoints -= eyeStripBB[0:2]
#    leftOffsetCoords[0:2] -= eyeStripBB[0:2]
#    rightOffsetCoords[0:2] -= eyeStripBB[0:2]
#
#    leftReflectionP1 = leftOffsetCoords[0:2] + leftReflectionBB[0:2]
#    leftReflectionP2 = leftReflectionP1 + leftReflectionBB[2:4]
#    leftReflectionP1 = tuple(leftReflectionP1)
#    leftReflectionP2 = tuple(leftReflectionP2)
#
#    rightReflectionP1 = rightOffsetCoords[0:2] + rightReflectionBB[0:2]
#    rightReflectionP2 = rightReflectionP1 + rightReflectionBB[2:4]
#    rightReflectionP1 = tuple(rightReflectionP1)
#    rightReflectionP2 = tuple(rightReflectionP2)
#
#    eyeStrip = np.copy(cropToBB(capture.image, eyeStripBB))
#
#    for [x, y] in eyeWidthPoints:
#        cv2.circle(eyeStrip, (x, y), 5, (0, 255, 0), -1)
#
#    cv2.rectangle(eyeStrip, leftReflectionP1, leftReflectionP2, (0, 0, 255), 1)
#    cv2.rectangle(eyeStrip, rightReflectionP1, rightReflectionP2, (0, 0, 255), 1)
#
#    return eyeStrip

#Note: both parent and child offsets should originally be measured to the same origin
def calculateRelativeOffset(parentOffset, childOffset):
    return childOffset[0:2] - parentOffset[0:2]

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

def extractReflectionPoints(reflectionBB, eyeCrop, eyeMask, ignoreMask):

    [x, y, w, h] = reflectionBB

    reflectionCrop = eyeCrop[y:y+h, x:x+w]
    reflectionCrop = colorTools.convert_sBGR_to_linearBGR_float_fast(reflectionCrop)
    #reflectionMask = eyeMask[y:y+h, x:x+w]

    #reflectionMask = reflectionMask == 1.0#.fill(False)

    #Add together each subpixel mask for each pixel. if the value is greater than 0, one of the subpixels was clipping
    #Just mask = isClipping(Red) or isClipping(Green) or isClipping(Blue)
    clippingMask = np.sum(reflectionCrop == 1.0, axis=2) > 0

    #reflectionCrop == 1.0
    #print('reflection Crop :: ' + str(reflectionCrop))


    if (reflectionCrop.shape[0] == 0) or (reflectionCrop.shape[1] == 0):
        raise NameError('Zero width eye reflection')

    cleanPixels = np.sum(np.logical_not(clippingMask).astype('uint8'))
    cleanPixelRatio = cleanPixels / (clippingMask.shape[0] * clippingMask.shape[1])

    #print('CLEAN PIXEL RATIO :: ' + str(cleanPixelRatio))

    if cleanPixelRatio < 0.95:
        raise NameError('Not enough clean non-clipped pixels in eye reflections')

    #greyValues = np.sum(reflectionCrop, axis=2).flatten()
    #greyValues = greyValues * (1 / np.max(greyValues))
    #plt.hist(greyValues, bins=50)
    #plt.show()

    #stretchedReflectionCrop = np.copy(reflectionCrop)
    #fig, axs = plt.subplots(3, 1, sharey=False, tight_layout=True)

    stretchedBlueReflectionCrop = (reflectionCrop[:, :, 0] - np.min(reflectionCrop[:, :, 0])) * (1 / (np.max(reflectionCrop[:, :, 0]) - np.min(reflectionCrop[:, :, 0]))) 


    #threshold = 20 / 255
    threshold = 1/4 #1/5

    blueMask = stretchedBlueReflectionCrop > threshold
   # bins = int(round(255 / 4))

   # blueValues = stretchedBlueReflectionCrop.flatten()
   # blueBins = np.clip(round(max(blueValues) * 255) / 3, 0, 255).astype('uint8')
   # axs[0].hist(blueValues, bins=bins)

    stretchedGreenReflectionCrop = (reflectionCrop[:, :, 1] - np.min(reflectionCrop[:, :, 1])) * (1 / (np.max(reflectionCrop[:, :, 1]) - np.min(reflectionCrop[:, :, 1]))) 

    greenMask = stretchedGreenReflectionCrop > threshold
   # greenValues = stretchedGreenReflectionCrop.flatten()
   # greenBins = np.clip(round(max(greenValues) * 255) / 3, 0, 255).astype('uint8')
   # axs[1].hist(greenValues, bins=bins)

    stretchedRedReflectionCrop = (reflectionCrop[:, :, 2] - np.min(reflectionCrop[:, :, 2])) * (1 / (np.max(reflectionCrop[:, :, 2]) - np.min(reflectionCrop[:, :, 2]))) 

    redMask = stretchedRedReflectionCrop > threshold
   # redValues = stretchedRedReflectionCrop.flatten()
   # redBins = np.clip(round(max(redValues) * 255) / 3, 0, 255).astype('uint8')
   # axs[2].hist(redValues, bins=bins)

    reflectionMask = np.logical_not(np.logical_or(np.logical_or(blueMask, greenMask), redMask))


    #stackedChannels = np.hstack([stretchedBlueReflectionCrop, stretchedGreenReflectionCrop, stretchedRedReflectionCrop])
    #cv2.imshow('masked', combined.astype('uint8') * 255)
    #cv2.imshow('stacked channels', stackedChannels)
    #cv2.waitKey(0)

    #plt.show()

    #fullMedianReflection = np.median(reflectionCrop, axis=(0,1))

    #brighter = reflectionCrop[reflectionCrop > fullMedianReflection]

    #medianReflection = np.median(brighter)
    #sdReflection = np.std(brighter)
    #print('MEDIAN REFLECTION :: {}'.format(medianReflection))
    #print('SD REFLECTION :: {}'.format(sdReflection))
    #lowerBoundMask = np.any(reflectionCrop < (medianReflection - sdReflection), axis=2)
    #reflectionMask = np.logical_or(lowerBoundMask, reflectionMask)

    inv_reflectionMask = np.logical_not(reflectionMask)
    im2, contours, hierarchy = cv2.findContours(inv_reflectionMask.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)

    boundingRectangle = np.array(list(cv2.boundingRect(contours[max_index])))
    boundingRectangle[0] += reflectionBB[0]
    boundingRectangle[1] += reflectionBB[1]

    #cv2.imshow('Crop', reflectionCrop)
    #cv2.imshow('Mask', np.stack([reflectionMask.astype('uint8'), reflectionMask.astype('uint8'), reflectionMask.astype('uint8')], axis=2) * 255)
    #cv2.imshow('Reflections', stacked)
    #cv2.waitKey(0)

    if ignoreMask:
        reflectionMask.fill(False)

    showMask = np.stack([reflectionMask.astype('uint8'), reflectionMask.astype('uint8'), reflectionMask.astype('uint8')], axis=2) * 255
    maskedReflections = np.copy(reflectionCrop)
    maskedReflections[reflectionMask] = [0, 0, 0]
    stacked = np.vstack([np.clip(reflectionCrop * 255, 0, 255).astype('uint8'), showMask, np.clip(maskedReflections * 255, 0, 255).astype('uint8')])

    reflectionPoints = reflectionCrop[np.logical_not(reflectionMask)]

    representativeReflectionPoint = calculateRepresentativeReflectionPoint(reflectionPoints)


    #if cleanPixelRatio < 0.8:

    return [representativeReflectionPoint, cleanPixelRatio, stacked, boundingRectangle]

def getEyeWidth(capture):
    [leftP1, leftP2] = capture.landmarks.getLeftEyeWidthPoints()
    [rightP1, rightP2] = capture.landmarks.getRightEyeWidthPoints()

    leftEyeWidth = max(leftP1[0], leftP2[0]) - min(leftP1[0], leftP2[0])
    rightEyeWidth = max(rightP1[0], rightP2[0]) - min(rightP1[0], rightP2[0])

    return (leftEyeWidth + rightEyeWidth) / 2

#def getAverageScreenReflectionColor(captures, leftEyeOffsets, rightEyeOffsets, saveStep):
#    wb = captures[0].getAsShotWhiteBalance()
#    isSpecialCase = [capture.isNoFlash for capture in captures]
#
#    leftEyeCoords = np.array([getLeftEyeCoords(capture) for capture in captures])
#    minLeftWidth = np.min(leftEyeCoords[:, 2])
#    minLeftHeight = np.min(leftEyeCoords[:, 3])
#    leftEyeCoords = np.array([[x, y, minLeftWidth, minLeftHeight] for x, y, w, h, in leftEyeCoords])
#
#    leftEyeCrops = [getCrop(capture, coords) for capture, coords in zip(captures, leftEyeCoords)]
#    leftEyeMasks = [getMask(capture, coords) for capture, coords in zip(captures, leftEyeCoords)]
#
#    leftEyeCrops, leftOffsets = cropTools.cropImagesToOffsets(leftEyeCrops, leftEyeOffsets)
#    leftEyeMasks, offsets = cropTools.cropImagesToOffsets(leftEyeMasks, leftEyeOffsets)
#
#    rightEyeCoords = np.array([getRightEyeCoords(capture) for capture in captures])
#    minRightWidth = np.min(rightEyeCoords[:, 2])
#    minRightHeight = np.min(rightEyeCoords[:, 3])
#    rightEyeCoords = np.array([[x, y, minRightWidth, minRightHeight] for x, y, w, h, in rightEyeCoords])
#
#    rightEyeCrops = [getCrop(capture, coords) for capture, coords in zip(captures, rightEyeCoords)]
#    rightEyeMasks = [getMask(capture, coords) for capture, coords in zip(captures, rightEyeCoords)]
#
#    rightEyeCrops, rightOffsets = cropTools.cropImagesToOffsets(rightEyeCrops, rightEyeOffsets)
#    rightEyeMasks, offsets = cropTools.cropImagesToOffsets(rightEyeMasks, rightEyeOffsets)
#
#    leftReflectionBB = maskReflectionBB(leftEyeCrops, wb)
#    rightReflectionBB = maskReflectionBB(rightEyeCrops, wb)
#
#    leftEyeCoords[:, 0:2] += leftOffsets
#    rightEyeCoords[:, 0:2] += rightOffsets
#
#    annotatedEyeStrips = [getAnnotatedEyeStrip(leftReflectionBB, leftEyeCoord, rightReflectionBB, rightEyeCoord, capture) for leftEyeCoord, rightEyeCoord, capture in zip(leftEyeCoords, rightEyeCoords, captures)]
#
#    minWidth = min([annotatedEyeStrip.shape[1] for annotatedEyeStrip in annotatedEyeStrips])
#    minHeight = min([annotatedEyeStrip.shape[0] for annotatedEyeStrip in annotatedEyeStrips])
#
#    annotatedEyeStrips = [annotatedEyeStrip[0:minHeight, 0:minWidth] for annotatedEyeStrip in annotatedEyeStrips]
#
#    stackedAnnotatedEyeStrips = np.vstack(annotatedEyeStrips)
#    saveStep.saveReferenceImageBGR(stackedAnnotatedEyeStrips, 'eyeStrips')
#
#    #RESULTS ARE LINEAR
#    leftReflectionStats = np.array([extractReflectionPoints(leftReflectionBB, eyeCrop, eyeMask, ignoreMask) for eyeCrop, eyeMask, ignoreMask in zip(leftEyeCrops, leftEyeMasks, isSpecialCase)])
#    rightReflectionStats = np.array([extractReflectionPoints(rightReflectionBB, eyeCrop, eyeMask, ignoreMask) for eyeCrop, eyeMask, ignoreMask in zip(rightEyeCrops, rightEyeMasks, isSpecialCase)])
#
#    leftReflectionImages = np.hstack(leftReflectionStats[:, 2])
#    rightReflectionImages = np.hstack(rightReflectionStats[:, 2])
#    saveStep.saveReferenceImageSBGR(leftReflectionImages, 'Left Reflections')
#    saveStep.saveReferenceImageSBGR(rightReflectionImages, 'Right Reflections')
#    #cv2.imshow('LEFT REFLECTIONS', leftReflectionImages)
#    #cv2.imshow('RIGHT REFLECTIONS', rightReflectionImages)
#    #cv2.waitKey(0)
#
#    averageReflections = (leftReflectionStats[:, 0] + rightReflectionStats[:, 0]) / 2
#
#    averageReflections = [(averageReflection if np.all(averageReflection.astype('bool')) else (averageReflection + np.array([1, 1, 1]))) for averageReflection in averageReflections]
#
#    print('AVERAGE NO, HALF, FULL REFLECTION :: {}'.format(averageReflections))
#
#    #Whitebalance per flash and eye to get luminance levels... Maybe compare the average reflection values?
#    #wbLeftReflections = np.vstack([colorTools.whitebalanceBGRPoints(leftReflection, averageReflection) for leftReflection, averageReflection in zip(leftReflectionStats[:, 0], averageReflections)])
#    print('Left Reflections :: ' + str(leftReflectionStats[:, 0]))
#    wbLeftReflections = np.vstack(leftReflectionStats[:, 0])
#    #wbRightReflections = np.vstack([colorTools.whitebalanceBGRPoints(rightReflection, averageReflection) for rightReflection, averageReflection in zip(rightReflectionStats[:, 0], averageReflections)])
#    print('Right Reflections :: ' + str(rightReflectionStats[:, 0]))
#    wbRightReflections = np.vstack(rightReflectionStats[:, 0])
#
#    #GET Luminance in reflection per flash and eye
#    leftReflectionLuminances = [colorTools.getRelativeLuminance([leftReflection])[0] for leftReflection in wbLeftReflections]
#    rightReflectionLuminances = [colorTools.getRelativeLuminance([rightReflection])[0] for rightReflection in wbRightReflections]
#
#    eyeWidth = getEyeWidth(captures[0])
#
#    if eyeWidth == 0:
#        raise NameError('Zero value Eye Width')
#
#    leftReflectionWidth, leftReflectionHeight = leftReflectionBB[2:4] / eyeWidth
#    rightReflectionWidth, rightReflectionHeight = rightReflectionBB[2:4] / eyeWidth
#
#    leftReflectionArea = leftReflectionWidth * leftReflectionHeight
#    rightReflectionArea = rightReflectionWidth * rightReflectionHeight
#
#    averageReflectionArea = (leftReflectionArea + rightReflectionArea) / 2
#
#    if min(leftReflectionWidth, rightReflectionWidth) == 0:
#        raise NameError('Zero value reflection Width')
#
#    if min(leftReflectionHeight, rightReflectionHeight) == 0:
#        raise NameError('Zero value reflection Height')
#
#    reflectionWidthRatio = max(leftReflectionWidth, rightReflectionWidth) / min(leftReflectionWidth, rightReflectionWidth)
#    reflectionHeightRatio = max(leftReflectionHeight, rightReflectionHeight) / min(leftReflectionHeight, rightReflectionHeight)
#
#    if (reflectionWidthRatio > 1.5) or (reflectionHeightRatio > 1.25):
#        raise NameError('Reflection Sizes are too different!')
#
#    middleIndex = math.floor(len(captures) / 2)
#
#    leftHalfReflectionLuminance = leftReflectionLuminances[middleIndex] * 2 #2x because we are using half
#    rightHalfReflectionLuminance = rightReflectionLuminances[middleIndex] * 2 #2x because we are using half
#
#    leftFluxish = leftReflectionArea * leftHalfReflectionLuminance
#    rightFluxish = rightReflectionArea * rightHalfReflectionLuminance
#
#    print('LEFT FLUXISH :: {} | AREA ::  {} | LUMINOSITY :: {}'.format(leftFluxish, leftReflectionArea, leftHalfReflectionLuminance))
#    print('RIGHT FLUXISH :: {} | AREA ::  {} | LUMINOSITY :: {}'.format(rightFluxish, rightReflectionArea, rightHalfReflectionLuminance))
#
#    return [averageReflections[middleIndex], averageReflectionArea, wbLeftReflections, wbRightReflections]

def getAverageScreenReflectionColor2(captures, leftEyeOffsets, rightEyeOffsets, saveStep):
    wb = captures[0].getAsShotWhiteBalance()
    isSpecialCase = [capture.isNoFlash for capture in captures]

    leftEyeCrops = [capture.leftEyeImage for capture in captures]
    leftEyeMasks = [capture.leftEyeMask for capture in captures]

    leftEyeCrops, leftOffsets = cropTools.cropImagesToOffsets(leftEyeCrops, leftEyeOffsets)
    leftEyeMasks, offsets = cropTools.cropImagesToOffsets(leftEyeMasks, leftEyeOffsets)

    rightEyeCrops = [capture.rightEyeImage for capture in captures]
    rightEyeMasks = [capture.rightEyeMask for capture in captures]

    rightEyeCrops, rightOffsets = cropTools.cropImagesToOffsets(rightEyeCrops, rightEyeOffsets)
    rightEyeMasks, offsets = cropTools.cropImagesToOffsets(rightEyeMasks, rightEyeOffsets)

#    leftEyeStacked = np.vstack(leftEyeCrops)
#    rightEyeStacked = np.vstack(rightEyeCrops)
#    cv2.imshow('Left', leftEyeStacked)
#    cv2.imshow('Right', rightEyeStacked)
#    cv2.waitKey(0)
#

    leftReflectionBB = maskReflectionBB(leftEyeCrops, wb)
    rightReflectionBB = maskReflectionBB(rightEyeCrops, wb)

    #leftEyeCoords[:, 0:2] += leftOffsets
    #rightEyeCoords[:, 0:2] += rightOffsets


    #RESULTS ARE LINEAR
    leftReflectionStats = np.array([extractReflectionPoints(leftReflectionBB, eyeCrop, eyeMask, ignoreMask) for eyeCrop, eyeMask, ignoreMask in zip(leftEyeCrops, leftEyeMasks, isSpecialCase)])
    rightReflectionStats = np.array([extractReflectionPoints(rightReflectionBB, eyeCrop, eyeMask, ignoreMask) for eyeCrop, eyeMask, ignoreMask in zip(rightEyeCrops, rightEyeMasks, isSpecialCase)])

    refinedLeftReflectionBBs = np.vstack(leftReflectionStats[:, 3])
    refinedRightReflectionBBs = np.vstack(rightReflectionStats[:, 3])

    annotatedEyeStrips = [getAnnotatedEyeStrip2(leftReflectionBBrefined, leftEyeCrop, rightReflectionBBrefined, rightEyeCrop) for leftEyeCrop, rightEyeCrop, leftReflectionBBrefined, rightReflectionBBrefined in zip(leftEyeCrops, rightEyeCrops, refinedLeftReflectionBBs, refinedRightReflectionBBs)]

    stackedAnnotatedEyeStrips = np.vstack(annotatedEyeStrips)
    saveStep.saveReferenceImageBGR(stackedAnnotatedEyeStrips, 'eyeStrips')

    leftReflectionImages = np.hstack(leftReflectionStats[:, 2])
    rightReflectionImages = np.hstack(rightReflectionStats[:, 2])
    saveStep.saveReferenceImageSBGR(leftReflectionImages, 'Left Reflections')
    saveStep.saveReferenceImageSBGR(rightReflectionImages, 'Right Reflections')
    #cv2.imshow('LEFT REFLECTIONS', leftReflectionImages)
    #cv2.imshow('RIGHT REFLECTIONS', rightReflectionImages)
    #cv2.waitKey(0)

    averageReflections = (leftReflectionStats[:, 0] + rightReflectionStats[:, 0]) / 2

    averageReflections = [(averageReflection if np.all(averageReflection.astype('bool')) else (averageReflection + np.array([1, 1, 1]))) for averageReflection in averageReflections]

    print('AVERAGE NO, HALF, FULL REFLECTION :: {}'.format(averageReflections))

    #Whitebalance per flash and eye to get luminance levels... Maybe compare the average reflection values?
    #wbLeftReflections = np.vstack([colorTools.whitebalanceBGRPoints(leftReflection, averageReflection) for leftReflection, averageReflection in zip(leftReflectionStats[:, 0], averageReflections)])
    print('Left Reflections :: ' + str(leftReflectionStats[:, 0]))
    wbLeftReflections = np.vstack(leftReflectionStats[:, 0])
    #wbRightReflections = np.vstack([colorTools.whitebalanceBGRPoints(rightReflection, averageReflection) for rightReflection, averageReflection in zip(rightReflectionStats[:, 0], averageReflections)])
    print('Right Reflections :: ' + str(rightReflectionStats[:, 0]))
    wbRightReflections = np.vstack(rightReflectionStats[:, 0])

    #GET Luminance in reflection per flash and eye
    leftReflectionLuminances = [colorTools.getRelativeLuminance([leftReflection])[0] for leftReflection in wbLeftReflections]
    rightReflectionLuminances = [colorTools.getRelativeLuminance([rightReflection])[0] for rightReflection in wbRightReflections]

    eyeWidth = getEyeWidth(captures[0])

    if eyeWidth == 0:
        raise NameError('Zero value Eye Width')

    #leftReflectionWidth, leftReflectionHeight = leftReflectionBB[2:4] / eyeWidth
    leftReflectionWidth, leftReflectionHeight = np.mean(refinedLeftReflectionBBs[:, 2:4], axis=0) / eyeWidth
    #rightReflectionWidth, rightReflectionHeight = rightReflectionBB[2:4] / eyeWidth
    rightReflectionWidth, rightReflectionHeight = np.mean(refinedRightReflectionBBs[:, 2:4], axis=0) / eyeWidth

    print('Left Width, Left Height :: {}, {}'.format(leftReflectionWidth, leftReflectionHeight))
    print('Right Width, Right Height :: {}, {}'.format(rightReflectionWidth, rightReflectionHeight))

    leftReflectionArea = leftReflectionWidth * leftReflectionHeight
    rightReflectionArea = rightReflectionWidth * rightReflectionHeight

    averageReflectionArea = (leftReflectionArea + rightReflectionArea) / 2

    if min(leftReflectionWidth, rightReflectionWidth) == 0:
        raise NameError('Zero value reflection Width')

    if min(leftReflectionHeight, rightReflectionHeight) == 0:
        raise NameError('Zero value reflection Height')

    reflectionWidthRatio = max(leftReflectionWidth, rightReflectionWidth) / min(leftReflectionWidth, rightReflectionWidth)
    reflectionHeightRatio = max(leftReflectionHeight, rightReflectionHeight) / min(leftReflectionHeight, rightReflectionHeight)

    if (reflectionWidthRatio > 1.5) or (reflectionHeightRatio > 1.25):
        raise NameError('Reflection Sizes are too different!')

    middleIndex = math.floor(len(captures) / 2)

    leftHalfReflectionLuminance = leftReflectionLuminances[middleIndex] * 2 #2x because we are using half
    rightHalfReflectionLuminance = rightReflectionLuminances[middleIndex] * 2 #2x because we are using half

    leftFluxish = leftReflectionArea * leftHalfReflectionLuminance
    rightFluxish = rightReflectionArea * rightHalfReflectionLuminance

    print('LEFT FLUXISH :: {} | AREA ::  {} | LUMINOSITY :: {}'.format(leftFluxish, leftReflectionArea, leftHalfReflectionLuminance))
    print('RIGHT FLUXISH :: {} | AREA ::  {} | LUMINOSITY :: {}'.format(rightFluxish, rightReflectionArea, rightHalfReflectionLuminance))

    return [averageReflections[middleIndex], averageReflectionArea, wbLeftReflections, wbRightReflections]

