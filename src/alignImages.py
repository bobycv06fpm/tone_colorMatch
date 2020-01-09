import utils
import numpy as np
import cv2
import colorTools
import math
import cropTools
import logger
logger = logger.getLogger(__name__, 'app')

#TAKES A FLOAT
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
    lower = median - (3 * sd)
    lower = lower if lower > lowerBound else lowerBound
    upper = median + (3 * sd)
    upper = upper if upper < upperBound else upperBound

    bounds = np.copy(gray)
    bounds[bounds < lower] = lower
    bounds[bounds > upper] = upper

    numerator = bounds - lower
    denominator = upper - lower
    stretched = (numerator / denominator)
    #stretched = np.clip(stretched * 255, 0, 255)
    return stretched

#def calculateOffset(preparedNoFlashImage, preparedFlashImage):
def calculateOffset(offsetImage, targetImage):
    #(offset, response) = cv2.phaseCorrelate(offsetImage, targetImage)
    (offset, response) = cv2.phaseCorrelate(targetImage, offsetImage)
    offset = list(offset)
    offset = [round(value) for value in offset]
    #print("Offset :: " + str(offset))
    return np.array(offset)

def getPreparedEye(gray):
    #gray = cv2.bilateralFilter(np.clip(gray * 255, 0, 255).astype('uint8'),30,150,150)
    gray = cv2.bilateralFilter(np.clip(gray * 255, 0, 255).astype('uint8'),5,50,50)
    #gray = cv2.GaussianBlur(np.clip(gray * 255, 0, 255).astype('uint8'),(31,31), 0)
    prepped = cv2.Sobel(gray, cv2.CV_16S, 1, 1, ksize=5)
    return np.float32(prepped)

def getEyeOffsets(eyes, sharpestIndex, wb=None):
    eyes = [colorTools.convert_sBGR_to_linearBGR_float_fast(eye) for eye in eyes]
    if wb is not None:
        eyes = [colorTools.whitebalance_from_asShot_to_d65(eye, *wb) for eye in eyes]

    greyEyes = [np.min(eye, axis=2) for eye in eyes] #Sort of counter intuitive, but using min we basically isolate white values/reflections
    stretchedEyes = [stretchHistogram(greyEye) for greyEye in greyEyes]
    preparedEyes = [getPreparedEye(stretchedEye) for stretchedEye in stretchedEyes]

    relativeEyeOffsets = [calculateOffset(preparedEye, preparedEyes[sharpestIndex]) for index, preparedEye in enumerate(preparedEyes)]

    eyeOffsets = relativeEyeOffsets

    return np.array(eyeOffsets)

def getCaptureEyeOffsets(captures):
    wb = captures[0].getAsShotWhiteBalance()
    sharpestMask = np.array([capture.isSharpest for capture in captures])
    sharpestIndex = np.arange(len(sharpestMask))[sharpestMask][0]

    leftEyeCrops = [capture.leftEyeImage for capture in captures]
    rightEyeCrops = [capture.rightEyeImage for capture in captures]

    if (not leftEyeCrops) or (not rightEyeCrops):
        raise ValueError('Eye Capture Missing')

    leftEyeOffsets = getEyeOffsets(leftEyeCrops, sharpestIndex, wb)
    rightEyeOffsets = getEyeOffsets(rightEyeCrops, sharpestIndex, wb)

    leftEyeBBs = np.array([capture.landmarks.getLeftEyeBB() for capture in captures])
    leftEyeBBPoints = np.array([[bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]] for bb in leftEyeBBs])
    leftJointBB = [np.min(leftEyeBBPoints[:, 0]), np.min(leftEyeBBPoints[:, 1]), np.max(leftEyeBBPoints[:, 2]), np.max(leftEyeBBPoints[:, 3])]
    leftEyeJointCrops = [capture.faceImage[leftJointBB[1]:leftJointBB[3], leftJointBB[0]:leftJointBB[2]] for capture in captures]
    leftEyeJointOffsets = getEyeOffsets(leftEyeJointCrops, sharpestIndex, wb)
    #cv2.imshow('left', np.vstack(leftEyeJointCrops))


    rightEyeBBs = np.array([capture.landmarks.getRightEyeBB() for capture in captures])
    rightEyeBBPoints = np.array([[bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]] for bb in rightEyeBBs])
    rightJointBB = [np.min(rightEyeBBPoints[:, 0]), np.min(rightEyeBBPoints[:, 1]), np.max(rightEyeBBPoints[:, 2]), np.max(rightEyeBBPoints[:, 3])]
    rightEyeJointCrops = [capture.faceImage[rightJointBB[1]:rightJointBB[3], rightJointBB[0]:rightJointBB[2]] for capture in captures]
    rightEyeJointOffsets = getEyeOffsets(rightEyeJointCrops, sharpestIndex, wb)
    #cv2.imshow('right', np.vstack(rightEyeJointCrops))

    print('L {} | R {}'.format(leftEyeBBs, rightEyeBBs))
    print('L Points {} | R Points {}'.format(leftEyeBBPoints, rightEyeBBPoints))
    print('L Joint {} | R Joint {}'.format(leftJointBB, rightJointBB))
    print('L Offsets {} | R Offsets {}'.format(leftEyeJointOffsets, rightEyeJointOffsets))

    averageJointOffset = np.round((leftEyeJointOffsets + rightEyeJointOffsets) / 2).astype('int32')
    print('Average {}'.format(averageJointOffset))

    #cv2.waitKey(0)

    leftEyeBBOrigins = np.array([capture.leftEyeBB[0] for capture in captures])
    rightEyeBBOrigins = np.array([capture.rightEyeBB[0] for capture in captures])

    scaleRatio = captures[0].scaleRatio

    scaledLeftEyeOffsets = leftEyeOffsets * scaleRatio
    scaledRightEyeOffsets = rightEyeOffsets * scaleRatio

    scaledAverageOffsets = averageJointOffset#np.round(np.mean([scaledLeftEyeOffsets, scaledRightEyeOffsets], axis=0)).astype('int32')

    alignedCoordLeft = leftEyeBBOrigins + scaledLeftEyeOffsets
    alignedCoordRight = rightEyeBBOrigins + scaledRightEyeOffsets

    leftFaceOffsets = alignedCoordLeft - alignedCoordLeft[0]
    rightFaceOffsets = alignedCoordRight - alignedCoordRight[0]

    averageFaceLandmarksOffsets = np.round(np.mean([leftFaceOffsets, rightFaceOffsets], axis=0)).astype('int32')

    logger.info('L Eye Offset ::\n{}'.format(leftEyeOffsets))
    logger.info('R Eye Offset ::\n{}'.format(rightEyeOffsets))
    logger.info('L/R Eye Offset Averages ::\n{}'.format(averageFaceLandmarksOffsets))
    logger.info('L/R TEST Eye Offset Averages ::\n{}'.format(scaledAverageOffsets))

    print('L Eye Offset ::\n{}'.format(leftEyeOffsets))
    print('R Eye Offset ::\n{}'.format(rightEyeOffsets))
    print('L/R Eye Offset Averages ::\n{}'.format(averageFaceLandmarksOffsets))
    print('L/R TEST Eye Offset Averages ::\n{}'.format(scaledAverageOffsets))

    return [leftEyeOffsets, rightEyeOffsets, averageFaceLandmarksOffsets, scaledAverageOffsets]

