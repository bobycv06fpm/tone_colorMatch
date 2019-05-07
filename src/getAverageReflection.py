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

def maskReflectionBB(eyes, wb):
    for index, eye in enumerate(eyes):
        if eye.shape[0] * eye.shape[1] == 0:
            raise NameError('Cannot Find #{} Eye'.format(index))

    eyes = [colorTools.convert_sBGR_to_linearBGR_float_fast(eye) for eye in eyes]
    eyes = [colorTools.whitebalance_from_asShot_to_d65(eye, *wb) for eye in eyes]
    greyEyes = [np.mean(eye, axis=2) for eye in eyes]
    croppedGreyEyes = [img[int(0.25 * img.shape[0]):int(0.75 * img.shape[0]), int(0.33 * img.shape[1]):int(0.66 * img.shape[1])] for img in greyEyes]

    stackedEyes = np.vstack(croppedGreyEyes)

    cv2.imshow('eyes', stackedEyes)

    #eyeDiffs = np.clip((greyEyes[1:] - greyEyes[:-1]) * 255 * 10, 0, 255).astype('uint8')
    eyeLap = [cv2.Laplacian(stretchHistogram(img), cv2.CV_64F) for img in croppedGreyEyes]
    #kernel = np.ones((5, 5), np.uint8)
    #eyeLapFiltered = [cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) for img in eyeLap]
    #eyeLapFiltered = [cv2.dilate(img, kernel, iterations=1) for img in eyeLap]

    #greyEyesFFT = [np.fft.fft2(greyEye) for greyEye in greyEyes]
    #greyEyesFFTShifted = [np.log(np.abs(np.fft.fftshift(greyEyeFFT))) for greyEyeFFT in greyEyesFFT]
    #greyEyesFFTShiftedScaled = [(img / np.max(img)) * 255 for img in greyEyesFFTShifted]

    stackedEyeDiffs = np.vstack(eyeLap)
    cv2.imshow('eyes Laplacian', stackedEyeDiffs * 10)
    cv2.waitKey(0)

    kernel = np.ones((3, 3), np.uint16)
    #Relient on second darkest reflection being AT LEAST 2x brighter than darkest
    secondDarkestImage, darkestImage = [cv2.morphologyEx(img, cv2.MORPH_CROSS, kernel) for img in greyEyes[-2:]]
    externalReflections = np.clip((2 * darkestImage) - secondDarkestImage, 0, 10)

    brightestClean = np.clip(greyEyes[0] - externalReflections, 0, 10)
    brightestTopMask = maskTopValues(brightestClean)

    #cv2.imshow('Clean Masked', np.hstack([greyEyes[0], (brightestTopMask * 255).astype('uint8')]))
    #cv2.waitKey(0)

    reflectionBB = getReflectionBB(brightestTopMask)
    x, y, w, h = getReflectionBB(brightestTopMask)

    print('REFLECTION BB :: ' + str(reflectionBB))

    widthDiff = int(0.5 * w)
    heightDiff = int(0.5 * h)

    cropHeight, cropWidth = brightestTopMask.shape

    new_x = x - widthDiff if x > widthDiff else 0
    new_y = y - heightDiff if y > heightDiff else 0 
    new_w = (x - new_x) + w + widthDiff
    new_h = (y - new_y) + h + heightDiff

    if new_x + new_w > cropWidth:
        new_w = cropWidth - new_x

    if new_y + new_h > cropHeight:
        new_h = cropHeight - new_y

    x, y, w, h = [new_x, new_y, new_w, new_h]

    print('NEW REFLECTION BB :: [{} {} {} {}]'.format(x, y, w, h))

    brightestCleanCrop = greyEyes[0][y:y + h, x:x + w]
    #brightestCleanCrop = brightestClean[y:y + h, x:x + w]
    maskedCropped = maskBottomValues(brightestCleanCrop)
    print('Sizes :: {} | {}'.format(brightestCleanCrop.shape, maskedCropped.shape))
    #cv2.imshow('Cropped Brightest Clean', np.vstack([brightestCleanCrop, (maskedCropped * 255).astype('uint8')]))
    #cv2.waitKey(0)

    reflectionBB = getReflectionBB(maskedCropped)
    reflectionBB[0] += x
    reflectionBB[1] += y

    x, y, w ,h = reflectionBB
    crops = [eye[y:y + h, x:x + w] for eye in eyes]
    #cv2.imshow('eye Crops', np.hstack(crops))
    #cv2.waitKey(0)

    return reflectionBB

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

def getAnnotatedEyeStrip(leftReflectionBB, leftOffsetCoords, rightReflectionBB, rightOffsetCoords, capture):
    eyeStripBB = np.array(capture.landmarks.getEyeStripBB())

    eyeWidthPoints = np.append(capture.landmarks.getLeftEyeWidthPoints(), capture.landmarks.getRightEyeWidthPoints(), axis=0)

    eyeWidthPoints -= eyeStripBB[0:2]
    leftOffsetCoords[0:2] -= eyeStripBB[0:2]
    rightOffsetCoords[0:2] -= eyeStripBB[0:2]

    leftReflectionP1 = leftOffsetCoords[0:2] + leftReflectionBB[0:2]
    leftReflectionP2 = leftReflectionP1 + leftReflectionBB[2:4]
    leftReflectionP1 = tuple(leftReflectionP1)
    leftReflectionP2 = tuple(leftReflectionP2)

    rightReflectionP1 = rightOffsetCoords[0:2] + rightReflectionBB[0:2]
    rightReflectionP2 = rightReflectionP1 + rightReflectionBB[2:4]
    rightReflectionP1 = tuple(rightReflectionP1)
    rightReflectionP2 = tuple(rightReflectionP2)

    eyeStrip = np.copy(cropToBB(capture.image, eyeStripBB))

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

def extractReflectionPoints(reflectionBB, eyeCrop, eyeMask, ignoreMask):

    [x, y, w, h] = reflectionBB

    reflectionCrop = eyeCrop[y:y+h, x:x+w]
    reflectionCrop = colorTools.convert_sBGR_to_linearBGR_float_fast(reflectionCrop)
    reflectionMask = eyeMask[y:y+h, x:x+w]

    reflectionMask.fill(False)

    if (reflectionMask.shape[0] == 0) or (reflectionMask.shape[1] == 0):
        raise NameError('Zero width eye reflection')

    cleanPixels = np.sum(np.logical_not(reflectionMask).astype('uint8'))
    cleanPixelRatio = cleanPixels / (reflectionMask.shape[0] * reflectionMask.shape[1])

    print('CLEAN PIXEL RATIO :: ' + str(cleanPixelRatio))

    if cleanPixelRatio < 0.8:
        raise NameError('Not enough clean non-clipped pixels in eye reflections')

    medianReflection = np.median(reflectionCrop, axis=(0,1))
    sdReflection = np.std(reflectionCrop, axis=(0,1))
    print('MEDIAN REFLECTION :: {}'.format(medianReflection))
    print('SD REFLECTION :: {}'.format(sdReflection))
    lowerBoundMask = np.any(reflectionCrop < (medianReflection - sdReflection), axis=2)
    reflectionMask = np.logical_or(lowerBoundMask, reflectionMask)

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

    return [representativeReflectionPoint, cleanPixelRatio, stacked]

def getEyeWidth(capture):
    [leftP1, leftP2] = capture.landmarks.getLeftEyeWidthPoints()
    [rightP1, rightP2] = capture.landmarks.getRightEyeWidthPoints()

    leftEyeWidth = max(leftP1[0], leftP2[0]) - min(leftP1[0], leftP2[0])
    rightEyeWidth = max(rightP1[0], rightP2[0]) - min(rightP1[0], rightP2[0])

    return (leftEyeWidth + rightEyeWidth) / 2

def getAverageScreenReflectionColor(captures, leftEyeOffsets, rightEyeOffsets, saveStep):
    wb = captures[0].getAsShotWhiteBalance()
    isSpecialCase = [capture.isNoFlash for capture in captures]

    leftEyeCoords = np.array([getLeftEyeCoords(capture) for capture in captures])
    minLeftWidth = np.min(leftEyeCoords[:, 2])
    minLeftHeight = np.min(leftEyeCoords[:, 3])
    leftEyeCoords = np.array([[x, y, minLeftWidth, minLeftHeight] for x, y, w, h, in leftEyeCoords])

    leftEyeCrops = [getCrop(capture, coords) for capture, coords in zip(captures, leftEyeCoords)]
    leftEyeMasks = [getMask(capture, coords) for capture, coords in zip(captures, leftEyeCoords)]

    leftEyeCrops, leftOffsets = cropTools.cropImagesToOffsets(leftEyeCrops, leftEyeOffsets)
    leftEyeMasks, offsets = cropTools.cropImagesToOffsets(leftEyeMasks, leftEyeOffsets)

    rightEyeCoords = np.array([getRightEyeCoords(capture) for capture in captures])
    minRightWidth = np.min(rightEyeCoords[:, 2])
    minRightHeight = np.min(rightEyeCoords[:, 3])
    rightEyeCoords = np.array([[x, y, minRightWidth, minRightHeight] for x, y, w, h, in rightEyeCoords])

    rightEyeCrops = [getCrop(capture, coords) for capture, coords in zip(captures, rightEyeCoords)]
    rightEyeMasks = [getMask(capture, coords) for capture, coords in zip(captures, rightEyeCoords)]

    rightEyeCrops, rightOffsets = cropTools.cropImagesToOffsets(rightEyeCrops, rightEyeOffsets)
    rightEyeMasks, offsets = cropTools.cropImagesToOffsets(rightEyeMasks, rightEyeOffsets)

    leftReflectionBB = maskReflectionBB(leftEyeCrops, wb)
    rightReflectionBB = maskReflectionBB(rightEyeCrops, wb)

    leftEyeCoords[:, 0:2] += leftOffsets
    rightEyeCoords[:, 0:2] += rightOffsets

    annotatedEyeStrips = [getAnnotatedEyeStrip(leftReflectionBB, leftEyeCoord, rightReflectionBB, rightEyeCoord, capture) for leftEyeCoord, rightEyeCoord, capture in zip(leftEyeCoords, rightEyeCoords, captures)]

    minWidth = min([annotatedEyeStrip.shape[1] for annotatedEyeStrip in annotatedEyeStrips])
    minHeight = min([annotatedEyeStrip.shape[0] for annotatedEyeStrip in annotatedEyeStrips])

    annotatedEyeStrips = [annotatedEyeStrip[0:minHeight, 0:minWidth] for annotatedEyeStrip in annotatedEyeStrips]

    stackedAnnotatedEyeStrips = np.vstack(annotatedEyeStrips)
    saveStep.saveReferenceImageBGR(stackedAnnotatedEyeStrips, 'eyeStrips')

    #RESULTS ARE LINEAR
    leftReflectionStats = np.array([extractReflectionPoints(leftReflectionBB, eyeCrop, eyeMask, ignoreMask) for eyeCrop, eyeMask, ignoreMask in zip(leftEyeCrops, leftEyeMasks, isSpecialCase)])
    rightReflectionStats = np.array([extractReflectionPoints(rightReflectionBB, eyeCrop, eyeMask, ignoreMask) for eyeCrop, eyeMask, ignoreMask in zip(rightEyeCrops, rightEyeMasks, isSpecialCase)])

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

    leftReflectionWidth, leftReflectionHeight = leftReflectionBB[2:4] / eyeWidth
    rightReflectionWidth, rightReflectionHeight = rightReflectionBB[2:4] / eyeWidth

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

    annotatedEyeStrips = [getAnnotatedEyeStrip2(leftReflectionBB, leftEyeCrop, rightReflectionBB, rightEyeCrop) for leftEyeCrop, rightEyeCrop in zip(leftEyeCrops, rightEyeCrops)]

    stackedAnnotatedEyeStrips = np.vstack(annotatedEyeStrips)
    saveStep.saveReferenceImageBGR(stackedAnnotatedEyeStrips, 'eyeStrips')

    #RESULTS ARE LINEAR
    leftReflectionStats = np.array([extractReflectionPoints(leftReflectionBB, eyeCrop, eyeMask, ignoreMask) for eyeCrop, eyeMask, ignoreMask in zip(leftEyeCrops, leftEyeMasks, isSpecialCase)])
    rightReflectionStats = np.array([extractReflectionPoints(rightReflectionBB, eyeCrop, eyeMask, ignoreMask) for eyeCrop, eyeMask, ignoreMask in zip(rightEyeCrops, rightEyeMasks, isSpecialCase)])

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

    leftReflectionWidth, leftReflectionHeight = leftReflectionBB[2:4] / eyeWidth
    rightReflectionWidth, rightReflectionHeight = rightReflectionBB[2:4] / eyeWidth

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

