from loadImages import loadImages
#from detectFace import detectFace
import alignImages
from getAverageReflection import getAverageScreenReflectionColor
from saveStep import Save
from getPolygons import getPolygons, getFullFacePolygon
from extractMask import extractMask, maskPolygons
import colorTools
#import plotTools
#import processPoints
import cv2
import numpy as np
#import dlib
import thresholdMask
import math
#from scipy import ndimage
#import matplotlib.pyplot as plt
#import landmarkPoints
from capture import Capture 

import multiprocessing as mp
#from multiprocessing.sharedctypes import RawValue

#import pympler
#from pympler.tracker import SummaryTracker

#root = '../../'
#root = '/home/dmacewen/Projects/tone/'
#root = os.path.expanduser('~/Projects/tone/')

def getLast(arr):
    return arr[-1]

def getSecond(arr):
    return arr[1]

def getCropWidth(shapeA, shapeB, shapeC, heightMargin, heightMax, widthMargin, widthMax):
    allShapes = np.append(shapeA, shapeB, axis=0)
    allShapes = np.append(allShapes, shapeC, axis=0)
    BB = np.asarray(cv2.boundingRect(allShapes))
    newX = BB[0] - widthMargin if BB[0] - widthMargin > 0 else 0
    newY = BB[1] - heightMargin if BB[1] - heightMargin > 0 else 0
    newWidth = BB[2] + 2*widthMargin if BB[2] + widthMargin < widthMax else widthMax
    newHeight = BB[3] + 2*heightMargin if BB[3] + heightMargin < heightMax else heightMax
    return [newX, newWidth, newY, newHeight]

def extractHistogramValues(username, imageName, image, polygons):
    mask = np.zeros(image.shape[0:2])
    [points, averageFlashContribution] = extractMask(username, image, polygons, mask, imageName)
    values = np.max(points, axis=1)
    return values
    #plt.hist(values, bins=range(0,255))
    #plt.show()

def scaleHSVtoFluxish(hsvValues, fluxish):
    #Calculated From Scatter Plot.... could be wrong...
    #hueFluxishSlope = -10.6641
    #saturationFluxishSlope = -70.53
    #valueFluxishSlope = 163.8858

    hueFluxishSlope = 0
    saturationFluxishSlope = 0
    valueFluxishSlope = 0

    fluxishSlopes = np.array([hueFluxishSlope, saturationFluxishSlope, valueFluxishSlope])

    targetFluxish = .00150

    fluxishDiff = targetFluxish - fluxish

    hsvDiffs = (fluxishSlopes * fluxishDiff)
    hsvValues += hsvDiffs
    return hsvValues

    #hueDiff = hueFluxishSlope * fluxishDiff
    #saturationDiff = saturationFluxishSlope * fluxishDiff
    #valueDiff = valueFluxishSlope * fluxishDiff



    #hsvMedians[0] = hsvMedians[0] + hueDiff
    #hsvMedians[1] = hsvMedians[1] + saturationDiff
    #hsvMedians[2] = hsvMedians[2] + valueDiff

def getLeftEye(image, shape):
    (x, y, w, h) = cv2.boundingRect(np.array([shape[43], shape[44], shape[47], shape[46]]))
    leftEye = image[y:y+h, x:x+w]
    return leftEye

def getRightEye(image, shape):
    (x, y, w, h) = cv2.boundingRect(np.array([shape[37], shape[38], shape[41], shape[40]]))
    rightEye = image[y:y+h, x:x+w]
    return rightEye

def prepEye(image):
    image_gray = cv2.cvtColor(np.clip(image * 255, 0, 255).astype('uint8'), cv2.COLOR_BGR2GRAY)
    #image_gray = cv2.GaussianBlur(image_gray, (11, 11), 0)
    #image_gray = cv2.GaussianBlur(image_gray, (19, 19), 0)


    original = np.copy(image_gray)
    median = np.median(image_gray)
    sd = np.std(image_gray)
    lower = median - (2 * sd)
    upper = median + (2 * sd)
    test = np.copy(image_gray)
    test[test < lower] = lower
    test[test > upper] = upper

    numerator = test - lower
    denominator = upper - lower
    stretched = (numerator / denominator)
    stretched = np.clip(stretched * 255, 0, 255).astype('uint8')


    #stretched = cv2.GaussianBlur(stretched, (19, 19), 0)
    #stretched = cv2.GaussianBlur(stretched, (25, 25), 0)
    stretched = cv2.GaussianBlur(stretched, (5, 5), 0)
    #cv2.imshow('stretched', np.hstack((original, stretched)))
    #cv2.waitKey(0)
    #stackAndShow([[original, stretched]], 'Stretched')

    image_prepped = cv2.Laplacian(stretched, cv2.CV_64F)
    return image_prepped

def alignEye(imageA, imageB):
    (offset, response) = cv2.phaseCorrelate(np.float64(imageA), np.float64(imageB))
    offset = list(offset)
    offset = [round(value) for value in offset]
    print("EYE Offset :: " + str(offset))
    return offset

#Eventually change to stretch histogram?
def trimHistogram(greyImage):
    #median = np.median(greyImage)
    #sd = np.std(greyImage)
    #lower = median - (2 * sd)
    #upper = median + (2 * sd)
    #greyImage[greyImage < lower] = 0
    #greyImage[greyImage > upper] = 0
    return greyImage

def cropToBB(images, imageShapes, start, end):
    points = []
    for imageShape in imageShapes:
        points += list(imageShape[start:end])

    (BB_X, BB_Y, BB_W, BB_H) = cv2.boundingRect(np.array(points))

    croppedImages = []
    for image in images:
        croppedImages.append(image[BB_Y:BB_Y+BB_H, BB_X:BB_X+BB_W])

    return np.array(croppedImages)

def scaleImages(images):
    images = np.copy(images)
    medians = np.array([np.median(np.max(image, axis=2)) for image in images])
    images[0:3] = images[0:3] * (medians[-1] / medians[0:3])
    return images

def stackAndShow(imageSets, name):
    counter = 0
    for imageSet in imageSets:
        imageStack = np.hstack([*imageSet])
        #cv2.imshow('{} :: {}'.format(name, counter), imageStack)
        #print('Displaying Image....(not really)')
        counter += 1

    #cv2.waitKey(0)

#Order Not Maintained
def cropToAxis(images, offset, axis):
    #imageSets = np.dstack((images, offset, np.arange(len(offset))))
    imageSets = []
    for index, image in enumerate(images):
        imageSets.append([np.array(image), offset[index], index])

    imageSets = np.array(sorted(imageSets, key=getSecond))

    if imageSets[0, 1] < 0:
        imageSets[:, 1] += abs(imageSets[0, 1])

    maxCrop = imageSets[-1, 1]

    cropped = []
    for imageSet in imageSets:
        [image, offset, order] = imageSet
        start = maxCrop - offset
        end = image.shape[axis] - offset

        if axis == 0:
            image = image[start:end, :]
            #image = image[:, start:end]
        else:
            image = image[:, start:end]
            #image = image[start:end, :]

        cropped.append([image, order])

    originalOrder = np.array(sorted(cropped, key=getSecond))
    return originalOrder[:, 0]


def cropToOffsets(images, offsets):
    print('Offsets :: ' + str(offsets))
    images = cropToAxis(images, offsets[:, 0], 0)
    images = cropToAxis(images, offsets[:, 1], 1)
    return images
    

def experimentalReflectionDetection(B, BS, BF, BFS, TF, TFS, FF, FFS):
    images = [B, BF, TF, FF]
    shapes = [BS, BFS, TFS, FFS]

    leftEyes = cropToBB(images, shapes, 42, 48)
    rightEyes = cropToBB(images, shapes, 36, 42)
    #stackAndShow([leftEyes, rightEyes], 'Unscaled')

    #Scale Values Left
    leftEyesScaled = scaleImages(leftEyes)
    rightEyesScaled = scaleImages(rightEyes)
    #stackAndShow([leftEyesScaled, rightEyesScaled], 'Scaled')

    #Align left
    print('LEFT')
    leftEyesPrepped = [prepEye(eye) for eye in leftEyesScaled]
    leftEyesTrimmed = [trimHistogram(eye) for eye in leftEyesPrepped]
    leftEyeOffsets = np.array([alignEye(leftEyesTrimmed[3], eye) for eye in leftEyesTrimmed])

    #Align right
    print('RIGHT')
    rightEyesPrepped = [prepEye(eye) for eye in rightEyesScaled]
    rightEyesTrimmed = [trimHistogram(eye) for eye in rightEyesPrepped]
    rightEyeOffsets = np.array([alignEye(rightEyesTrimmed[3], eye) for eye in rightEyesTrimmed])

    #stackAndShow([leftEyesPrepped, rightEyesPrepped], 'Prepped')
    #stackAndShow([leftEyesTrimmed, rightEyesTrimmed], 'Trimmed')

    leftEyesCropped = cropToOffsets(leftEyes, leftEyeOffsets)
    rightEyesCropped = cropToOffsets(rightEyes, rightEyeOffsets)
    #stackAndShow([leftEyesCropped, rightEyesCropped], 'Cropped')

    [BLE, BFLE, TFLE, FFLE] = leftEyesCropped
    leftEyeDiff = np.absolute(FFLE - BFLE - TFLE + BLE)
    leftEyeDiff2 = np.absolute(FFLE - BFLE - TFLE)

    l_t = np.clip((TFLE - BLE) * 255, 0, 255).astype('uint8')
    l_b = np.clip((BFLE - BLE) * 255, 0, 255).astype('uint8')
    l_ff = np.clip((FFLE - BLE) * 255, 0, 255).astype('uint8')
    l_test_ff = np.clip((TFLE + BFLE - BLE) * 255, 0, 255).astype('uint8')

    l_test_t = np.clip((FFLE - BFLE - BLE) * 255, 0, 255).astype('uint8')
    l_test_b = np.clip((FFLE - TFLE - BLE) * 255, 0, 255).astype('uint8')
    #cv2.imshow('Left Eye Diff', (leftEyeDiff * 255 * 10).astype('uint8'))
    #cv2.waitKey(0)

    [BRE, BFRE, TFRE, FFRE] = rightEyesCropped
    rightEyeDiff = np.absolute(FFRE - BFRE - TFRE + BRE)
    rightEyeDiff2 = np.absolute(FFRE - BFRE - TFRE)

    r_t = np.clip((TFRE - BRE) * 255, 0, 255).astype('uint8')
    r_b = np.clip((BFRE - BRE) * 255, 0, 255).astype('uint8')
    r_ff = np.clip((FFRE - BRE) * 255, 0, 255).astype('uint8')
    r_test_ff = np.clip((TFRE + BFRE - BRE) * 255, 0, 255).astype('uint8')

    r_test_t = np.clip((FFRE - BFRE - BRE) * 255, 0, 255).astype('uint8')
    r_test_t = np.clip((FFRE - BFRE - BRE) * 255, 0, 255).astype('uint8')
    r_test_b = np.clip((FFRE - TFRE - BRE) * 255, 0, 255).astype('uint8')

    #This One...
    #stackAndShow([[l_ff, l_b, l_t, l_test_ff, l_test_t, l_test_b], [r_ff, r_b, r_t, r_test_ff, r_test_t, r_test_b]], 'Diffs')

    #stackAndShow([[(leftEyeDiff * 255 * 10).astype('uint8'), (leftEyeDiff2 * 255 * 10).astype('uint8')], [(rightEyeDiff * 255 * 10).astype('uint8'), (rightEyeDiff2 * 255 * 10).astype('uint8')]], 'Diffs')
    #cv2.imshow('Right Eye Diff', (rightEyeDiff * 255 * 10).astype('uint8'))
    #cv2.waitKey(0)

    #test = np.absolute((FF) - (BF + TF))

    #cv2.imshow('TEST... REFLECTION SHOULD BE ZERO', (test * 255).astype('uint8'))
    #cv2.waitKey(0)

    #test2 = np.absolute(test - B)
    #cv2.imshow('JUST THE REFLECTIONS?', (test2 * 255).astype('uint8'))
    #cv2.waitKey(0)

def run(username, imageName, fast=False, saveStats=False):
    #saveStep.resetLogFile(username, imageName)
    saveStep = Save(username, imageName)
    saveStep.resetLogFile()
    images = loadImages(username, imageName)

    [noFlashImage, halfFlashImage, fullFlashImage] = images
    [noFlashMetadata, halfFlashMetadata, fullFlashMetadata] = saveStep.getMetadata()

    noFlashCapture = Capture('No Flash', noFlashImage, noFlashMetadata)
    halfFlashCapture = Capture('Half Flash', halfFlashImage, halfFlashMetadata)
    fullFlashCapture = Capture('Full Flash', fullFlashImage, fullFlashMetadata)

#    noFlashCapture.showImageWithLandmarks()
#    halfFlashCapture.showImageWithLandmarks()
#    fullFlashCapture.showImageWithLandmarks()

    #polygons = getPolygons(noFlashCapture)
    #extractMask(noFlashCapture, polygons, saveStep)

    #polygons = getPolygons(halfFlashCapture)
    #extractMask(halfFlashCapture, polygons, saveStep)

    #polygons = getPolygons(fullFlashCapture)
    #extractMask(fullFlashCapture, polygons, saveStep)



    #detector = dlib.get_frontal_face_detector()
    #predictor = dlib.shape_predictor('/home/dmacewen/Projects/colorMatch/service/predictors/shape_predictor_68_face_landmarks.dat')
    #predictor = dlib.shape_predictor( root + 'tone_colorMatch/predictors/shape_predictor_68_face_landmarks.dat')

    #print('Detecting Base Face')
    #[noFlashImage, noFlashImageShape] = detectFace(originalNoFlashImage, predictor, detector)

    #print('Detecting Half Flash Face')
    #[halfFlashImage, halfFlashImageShape] = detectFace(originalHalfFlashImage, predictor, detector)

    #print('Detecting Full Flash Face')
    #[fullFlashImage, fullFlashImageShape] = detectFace(originalFullFlashImage, predictor, detector)

    #for landmark in noFlashMetadata['faceLandmarks']:
    #    print(landmark, landmark[0] + landmark[1])

    #noFlashLandmarks = landmarkPoints.landmarks(noFlashMetadata['faceLandmarksSource'], noFlashMetadata['faceLandmarks'])
    #halfFlashLandmarks = landmarkPoints.landmarks(halfFlashMetadata['faceLandmarksSource'], halfFlashMetadata['faceLandmarks'])
    #fullFlashLandmarks = landmarkPoints.landmarks(fullFlashMetadata['faceLandmarksSource'], fullFlashMetadata['faceLandmarks'])

    #print("No Flash Landmarks" + str(noFlashLandmarks.landmarks))
    #print("Half Flash Landmarks" + str(halfFlashLandmarks.landmarks))
    #print("Full Flash Landmarks" + str(fullFlashLandmarks.landmarks))

    #print('Trimming Down sRGB Images')
    #margin = .04
    #heightMargin = int(margin * noFlashImage.shape[0]) #Add 5% in both direction on height because crop is a little tight
    #heightMax = noFlashImage.shape[0]
    #widthMargin = int(margin * noFlashImage.shape[1])
    #widthMax = noFlashImage.shape[1]

    #[newX, newWidth, newY, newHeight] = getCropWidth(noFlashImageShape, halfFlashImageShape, fullFlashImageShape, heightMargin, heightMax, widthMargin, widthMax)

    #noFlashImageAligned = noFlashImage[newY:(newY + newHeight), newX:(newX + newWidth)]
    #noFlashImageShape = noFlashImageShape - [newX, newY]

    #halfFlashImageAligned = halfFlashImage[newY:(newY + newHeight), newX:(newX + newWidth)]
    #halfFlashImageShape = halfFlashImageShape - [newX, newY]

    #fullFlashImageAligned = fullFlashImage[newY:(newY + newHeight), newX:(newX + newWidth)]
    #fullFlashImageShape = fullFlashImageShape - [newX, newY]

    #print('Masking No Flash')
    #noFlashImageMask = thresholdMask.getClippedMask(noFlashImageAligned, 5, 5)
    #noFlashImageMask = thresholdMask.getClippedMask(noFlashImage, 1, 1)

    #print('Masking Half Flash')
    #halfFlashImageMask = thresholdMask.getClippedMask(halfFlashImageAligned, 5, 5)
    #halfFlashImageMask = thresholdMask.getClippedMask(halfFlashImage, 1, 1)

    #print('Masking Full Flash')
    #fullFlashImageMask = thresholdMask.getClippedMask(fullFlashImageAligned, 5, 5)
    #fullFlashImageMask = thresholdMask.getClippedMask(fullFlashImage, 1, 1)


    #print('Converting Base to Linear')
    #noFlashImage = colorTools.convert_sBGR_to_linearBGR_float(noFlashImageAligned)

    #print('Converting Top Flash to Linear')
    #halfFlashImage = colorTools.convert_sBGR_to_linearBGR_float(halfFlashImageAligned)

    #print('Converting Full Flash to Linear')
    #fullFlashImage = colorTools.convert_sBGR_to_linearBGR_float(fullFlashImageAligned)

    print('Cropping and Aligning')
    #noFlashCapture = (noFlashImage, noFlashLandmarks.landmarks, noFlashImageMask)
    #halfFlashCapture = (halfFlashImage, halfFlashLandmarks.landmarks, halfFlashImageMask)
    #fullFlashCapture = (fullFlashImage, fullFlashLandmarks.landmarks, fullFlashImageMask)

#    noFlashCapture.show(False)
#    halfFlashCapture.show(False)
#    fullFlashCapture.show()
#
#    print("half flash size :: " + str(halfFlashCapture.image.shape))

    #images = cropAndAlign(noFlashCapture, halfFlashCapture, fullFlashCapture)

    #Needs to be more accurate
    alignImages.cropAndAlign(noFlashCapture, halfFlashCapture, fullFlashCapture)
    print('Done Cropping and aligning')

#    print("half flash size :: " + str(halfFlashCapture.image.shape))
#    noFlashCapture.show(False)
#    halfFlashCapture.show(False)
#    fullFlashCapture.show()

    #(noFlashImage, noFlashImageShape, noFlashImageMask) = images[0]
    #(halfFlashImage, halfFlashImageShape, halfFlashImageMask) = images[1]
    #(fullFlashImage, fullFlashImageShape, fullFlashImageMask) = images[2]

    #crops = np.array(crops)

    #noFlashImage_sBGR = noFlashImageAligned[crops[0, 1, 1]:crops[0, 1, 2], crops[0, 0, 1]:crops[0, 0, 2]]
    #halfFlashImage_sBGR = halfFlashImageAligned[crops[2, 1, 1]:crops[2, 1, 2], crops[2, 0, 1]:crops[2, 0, 2]]
    #fullFlashImage_sBGR = fullFlashImageAligned[crops[1, 1, 1]:crops[1, 1, 2], crops[1, 0, 1]:crops[1, 0, 2]]

    #experimentalReflectionDetection(noFlashImage, noFlashImageShape, bottomFlashImage, bottomFlashImageShape, halfFlashImage, halfFlashImageShape, fullFlashImage, fullFlashImageShape)

    partialMask = np.logical_or(noFlashCapture.mask, halfFlashCapture.mask)
    allPointsMask = np.logical_or(partialMask, fullFlashCapture.mask)
    #clippedPixelsMask = np.copy(allPointsMask)

    # (All Pixels That Are Clipped In Full Flash) AND (All The Pixels That Are NOT Clipped By Base & Top & Bottom Flash)
    # => Pixel Values caused to clip by Full Flash
    #potentiallyRecoverablePixelsMask = np.logical_and(fullFlashImageMask, np.logical_not(partialMask))
    #unrecoverablePixelsMask = np.logical_and(fullFlashImageMask, partialMask)
    #allPointsMask = np.logical_and(allPointsMask, np.logical_not(unrecoverablePixelsMask))


    #unrecoverablePixelsMask = maskPolygons(unrecoverablePixelsMask, polygons)
    #potentiallyRecoverablePixelsMask = maskPolygons(potentiallyRecoverablePixelsMask, polygons)
    #RecoveredFullFlash = ((2 * halfFlashImage) - noFlashImage)

    #fullFlashImage[potentiallyRecoverablePixelsMask] = RecoveredFullFlash[potentiallyRecoverablePixelsMask]

    #NOTE: MIGHT BE NEEDED
    #noFlashImageBlur = cv2.GaussianBlur(noFlashCapture.getClippedImage(), (7, 7), 0)
    #halfFlashImageBlur = cv2.GaussianBlur(halfFlashCapture.getClippedImage(), (7, 7), 0)
    #fullFlashImageBlur = cv2.GaussianBlur(fullFlashCapture.getClippedImage(), (7, 7), 0)
    #END NOTE

    #noFlashImageBlur = cv2.GaussianBlur(noFlashImage, (15, 15), 0)
    #halfFlashImageBlur = cv2.GaussianBlur(halfFlashImage, (15, 15), 0)
    #fullFlashImageBlur = cv2.GaussianBlur(fullFlashImage, (15, 15), 0)

    #TEST STARTING NOW
    print('Beginning Linearity Test')
    #cv2.waitKey(0)

    howLinear = np.abs((2 * halfFlashCapture.image.astype('int32')) - (fullFlashCapture.image.astype('int32') + noFlashCapture.image.astype('int32')))
    #howLinear = np.abs((2 * halfFlashImageBlur.astype('int32')) - (fullFlashImageBlur.astype('int32') + noFlashImageBlur.astype('int32')))
    #cv2.imshow('how linear...', np.clip(np.abs(howLinear), 0, 255).astype('uint8'))
    #cv2.waitKey(0)

    #TODO: Compare Subpixel nonlinearity with full pixel nonlinearity....
    #howLinearSum = np.sum(howLinear, axis=2)
    howLinearMax = np.max(howLinear, axis=2)
    #howLinearMaxBlur = howLinearMax#cv2.GaussianBlur(howLinearMax, (7, 7), 0)

    #mean = np.mean(howLinearMaxBlur)
    #med = np.median(howLinearMaxBlur)
    #sd = np.std(howLinearMaxBlur)

    #print('howLinearMax :: ' + str(howLinearMax))

    #nonLinearMask = howLinearMax > .03 #med
    nonLinearMask = howLinearMax > 6#8 #12 #med
    #cv2.imshow('non linear mask', nonLinearMask.astype('uint8') * 255)
    #howLinearMaxBlurMasked = howLinearMaxBlur + nonLinearMask
    allPointsMask = np.logical_or(allPointsMask, nonLinearMask)

    print('Ending Linearity Test')
    #TEST ENDING NOW

    print('Subtracting Base from Flash')
    diffImage = fullFlashCapture.image.astype('int32') - noFlashCapture.image.astype('int32')
    #diffImage = halfFlashCapture.image.astype('int32') - noFlashCapture.image.astype('int32')

    #print('Diff Image :: ' + str(diffImage))
    diffCapture = Capture('Diff', diffImage, fullFlashCapture.metadata, allPointsMask)


    #image = fullFlashImage - noFlashImage
    #print('Values Histograms')
    #baseValues = extractHistogramValues(username, imageName, noFlashImage_sBGR, polygons)
    #topFlashValues = extractHistogramValues(username, imageName, halfFlashImage_sBGR, polygons)
    #fullFlashValues = extractHistogramValues(username, imageName, fullFlashImage_sBGR, polygons)

    #fig, axs = plt.subplots(2, 2, sharey=True, tight_layout=True)
    #axs[0, 0].hist(baseValues, bins=range(0,260))
    #axs[1, 0].hist(topFlashValues, bins=range(0,260))
    #axs[1, 1].hist(fullFlashValues, bins=range(0,260))
    #saveStep.savePlot(username, imageName, 'originalCaptureValuesHist', plt)
    #plt.show()

    #recoveredMask = np.logical_and(allPointsMask, np.logical_not(potentiallyRecoverablePixelsMask))
    #recoveredMask = unrecoverablePixelsMask
    #cv2.imshow('Recovered Mask', recoveredMask.astype('uint8') * 255)
    #cv2.waitKey(0)

    #diffCapture.show(False)
    #diffCapture.showMasked()

    print('Getting Polygons')
    #imageShape = fullFlashImageShape
    polygons = getPolygons(diffCapture)
    try:
        [points, averageFlashContribution] = extractMask(diffCapture, polygons, saveStep)
    except NameError as err:
        #print('error extracting left side of face')
        #raise NameError('User :: {} | Image :: {} | Error :: {} | Details :: {}'.format(username, imageName, 'Error extracting left side of face', err))
        return 'User :: {} | Image :: {} | Error :: {} | Details :: {}'.format(username, imageName, 'Error extracting Points for Recovered Mask', err)
    else:
        faceValues = np.max(points, axis=1)
        medianFaceValue = np.median(faceValues)

    if not fast:
        print('Saving Step 1')
        #saveStep.saveShapeStep(username, imageName, imageShape, 1)
        saveStep.saveImageStep(diffCapture.image, 1)
        saveStep.saveMaskStep(allPointsMask, 1, 'clippedMask')

    alignImages.alignEyes(noFlashCapture, halfFlashCapture, fullFlashCapture)
    return
    whiteBalance_CIE1931_coord_asShot = saveStep.getAsShotWhiteBalance()
    print('White Balance As Shot :: ' + str(whiteBalance_CIE1931_coord_asShot))

    averageReflectionBGR = getAverageScreenReflectionColor(username, imageName, image, fullFlashImage_sBGR, imageShape, whiteBalance_CIE1931_coord_asShot)

    [[leftAverageReflectionBGR, leftFluxish, leftDimensions], [rightAverageReflectionBGR, rightFluxish, rightDimensions]] = averageReflectionBGR
    print('average left reflection :: ' + str(leftAverageReflectionBGR))
    print('average right reflection :: ' + str(rightAverageReflectionBGR))

    faceMidPoint = np.mean(imageShape[27:30, 0]).astype('uint') #average nose x values
    #Users Right and Users Left
    rightFaceImage = image[:, :faceMidPoint, :]
    leftFaceImage = image[:, faceMidPoint:, :]

    averageMaxReflectionSum = 0
    maxReflectionZones = 0
    testFluxish = 0
    averageReflectionDimensions = np.array([0, 0], dtype=np.float)

    left_hsvMedians = None
    right_hsvMedians = None
    #medianFaceValue = 0

    print('Left Dimensions ({}, {})'.format(*leftDimensions))
    if (leftAverageReflectionBGR is not None) and (leftFluxish > .0007) and ((leftDimensions[0] * leftDimensions[1]) >= .0017):# and (max(leftAverageReflectionBGR) > .2):
        #print('here')
        leftFaceImageWB = colorTools.whitebalanceBGR_float(leftFaceImage, leftAverageReflectionBGR)
        #leftFaceImageWB *= ( / leftFluxish)

        averageMaxReflectionSum = averageMaxReflectionSum + max(leftAverageReflectionBGR)

        #Extract Value from Non WB image using recovered pixels
        valuesMask = recoveredMask[:, faceMidPoint:]
        try:
            #valuesPoints = extractMask(username, leftFaceImage, polygons, valuesMask, imageName)
            valuesPoints = extractMask(leftCapture, polygons, saveStep)

        except NameError as err:
            #print('error extracting left side of face')
            #raise NameError('User :: {} | Image :: {} | Error :: {} | Details :: {}'.format(username, imageName, 'Error extracting left side of face', err))
            return 'User :: {} | Image :: {} | Error :: {} | Details :: {}'.format(username, imageName, 'Error extracting left side of face', err)
        else:
            (valuesPoints, averageFlashContribution) = valuesPoints
            leftFaceValues = np.max(valuesPoints, axis=1)
            medianLeftFaceValue = np.median(leftFaceValues)


        hueSatMask = allPointsMask[:, faceMidPoint:]
        try:
            #maskedLeftPoints = extractMask(username, leftFaceImageWB, polygons, hueSatMask, imageName)
            #maskedLeftPoints = extractMask(username, leftFaceImageWB, polygons, hueSatMask, imageName)
            maskedLeftPoints = extractMask(leftCapture, polygons, saveStep)
        except NameError as err:
            #print('error extracting left side of face')
            #raise NameError('User :: {} | Image :: {} | Error :: {} | Details :: {}'.format(username, imageName, 'Error extracting left side of face', err))
            return 'User :: {} | Image :: {} | Error :: {} | Details :: {}'.format(username, imageName, 'Error extracting left side of face', err)
        else:
            (maskedLeftPoints, left_averageFlashContribution) = maskedLeftPoints
            left_sBGR = colorTools.convert_linearBGR_float_to_sBGR(np.array([maskedLeftPoints]))
            left_hsvPoints = colorTools.convert_linearBGR_float_to_linearHSV_float(left_sBGR / 255)[0]
            left_hsvMedians = np.median(left_hsvPoints, axis=0)
            left_hsvMedians[2] = medianLeftFaceValue
            left_hsvMedians = scaleHSVtoFluxish(left_hsvMedians, leftFluxish)

        
        testFluxish = testFluxish + leftFluxish
        averageReflectionDimensions += np.array(leftDimensions)
        maxReflectionZones = maxReflectionZones + 1
    else:
        allPointsMask[:, faceMidPoint:] = True
        leftFaceImageWB = leftFaceImage #Just dont white balance this part. will get masked out


    print('Right Dimensions ({}, {})'.format(*rightDimensions))
    if (rightAverageReflectionBGR is not None) and (rightFluxish > .0007) and ((rightDimensions[0] * rightDimensions[1]) >= .0017):# and (max(rightAverageReflectionBGR) > .2):
        #print('there')
        rightFaceImageWB = colorTools.whitebalanceBGR_float(rightFaceImage, rightAverageReflectionBGR)
        #rightFaceImageWB *= (150 / rightFluxish)

        averageMaxReflectionSum = averageMaxReflectionSum + max(rightAverageReflectionBGR)

        #Extract Value from Non WB image using recovered pixels
        valuesMask = recoveredMask[:, :faceMidPoint]
        try:
            #[valuesPoints, averageFlashContribution] = extractMask(username, rightFaceImage, polygons, valuesMask, imageName)
            valuesPoints = extractMask(rightCapture, polygons, saveStep)
        except NameError as err:
            #print('error extracting right side of face')
            #raise NameError('User :: {} | Image :: {} | Error :: {} | Details :: {}'.format(username, imageName, 'Error extracting right side of face', err))
            return 'User :: {} | Image :: {} | Error :: {} | Details :: {}'.format(username, imageName, 'Error extracting right side of face', err)
        else:
            (valuesPoints, averageFlashContribution) = valuesPoints
            rightFaceValues = np.max(valuesPoints, axis=1)
            medianRightFaceValue = np.median(rightFaceValues)

        hueSatMask = allPointsMask[:, :faceMidPoint]
        try:
            #[maskedRightPoints, right_averageFlashContribution] = extractMask(username, rightFaceImageWB, polygons, hueSatMask, imageName)
            maskedRightPoints = extractMask(rightCapture, polygons, saveStep)
        except NameError as err:
            #print('error extracting right side of face')
            #raise NameError('User :: {} | Image :: {} | Error :: {} | Details :: {}'.format(username, imageName, 'Error extracting right side of face', err))
            return 'User :: {} | Image :: {} | Error :: {} | Details :: {}'.format(username, imageName, 'Error extracting right side of face', err)
        else:
            (maskedRightPoints, right_averageFlashContribution) = maskedRightPoints
            right_sBGR = colorTools.convert_linearBGR_float_to_sBGR(np.array([maskedRightPoints]))
            right_hsvPoints = colorTools.convert_linearBGR_float_to_linearHSV_float(right_sBGR / 255)[0]
            right_hsvMedians = np.median(right_hsvPoints, axis=0)
            right_hsvMedians[2] = medianRightFaceValue
            right_hsvMedians = scaleHSVtoFluxish(right_hsvMedians, rightFluxish)

        testFluxish = testFluxish + rightFluxish
        averageReflectionDimensions += np.array(rightDimensions)
        maxReflectionZones = maxReflectionZones + 1
    else:
        allPointsMask[:, :faceMidPoint] = True
        rightFaceImageWB = rightFaceImage #Just dont white balance this part. will get masked out

    if maxReflectionZones != 0:
        averageMaxReflection = averageMaxReflectionSum / maxReflectionZones
        testFluxish = testFluxish / maxReflectionZones
        averageReflectionDimensions /= maxReflectionZones
    else:
        #raise NameError('User :: {} | Image :: {} | Error :: {}'.format(username, imageName, 'No valid reflection zones on face'))
        return 'User :: {} | Image :: {} | Error :: {}'.format(username, imageName, 'No valid reflection zones on face')

    imageWB = np.hstack((rightFaceImageWB, leftFaceImageWB))
    image = imageWB

    print('Extracting Mask')
    [points, averageFlashContribution] = extractMask(username, image, polygons, allPointsMask, imageName)

    if not fast:
        print('Saving Step 2')
        saveStep.logMeasurement("Average Flash Contribution", str(averageFlashContribution))
        saveStep.saveReferenceImageBGR(image, 'WhitebalancedImage')

    print('Getting Median')
    sBGR = colorTools.convert_linearBGR_float_to_sBGR(np.array([points]))
    hsvPoints = colorTools.convert_linearBGR_float_to_linearHSV_float(sBGR / 255)[0]

    #plt.clf()
    #plt.hist(hsvPoints[:, 2], bins='auto')
    #plt.hist(faceValues, bins='auto')
    #plt.savefig('maskedLinearValuesHist')
    #saveStep.savePlot(username, imageName,'maskedLinearValuesHist', plt)
    #plt.show()

    print('Median Face Value :: ' + str(medianFaceValue))
    hsvMedians = np.median(hsvPoints, axis=0)
    hsvMedians[2] = medianFaceValue

    fluxishRatio = hsvMedians[2] / testFluxish
    temp = scaleHSVtoFluxish(hsvMedians, testFluxish)
    hsvMedians[2] = temp[2]

    saveStep.savePointsStep([hsvMedians], 4)

    if saveStats:
        saveStep.saveImageStat('reflectionStrength', [averageMaxReflection])
        saveStep.saveImageStat('testFluxish', [testFluxish])
        saveStep.saveImageStat('meanReflectionDimensions', averageReflectionDimensions)
        saveStep.saveImageStat('fluxishRatio', [fluxishRatio])
        saveStep.saveImageStat('medianHSV', hsvMedians)

        if (leftAverageReflectionBGR is not None) and (left_hsvMedians is not None) and (leftFluxish > .0007) and ((leftDimensions[0] * leftDimensions[1]) >= .0017):
            saveStep.saveImageStat('splitReflectionStrength', [max(leftAverageReflectionBGR)])
            saveStep.saveImageStat('splitTestFluxish', [leftFluxish])
            saveStep.saveImageStat('splitMedianHSV', left_hsvMedians)
            saveStep.saveImageStat('splitDimensions', leftDimensions)

        if (rightAverageReflectionBGR is not None) and (right_hsvMedians is not None) and (rightFluxish > .0007) and ((rightDimensions[0] * rightDimensions[1]) >= .0017):
            saveStep.saveImageStat('splitReflectionStrength', [max(rightAverageReflectionBGR)])
            saveStep.saveImageStat('splitTestFluxish', [rightFluxish])
            saveStep.saveImageStat('splitMedianHSV', right_hsvMedians)
            saveStep.saveImageStat('splitDimensions', rightDimensions)

    #print('Done!')
    return None
