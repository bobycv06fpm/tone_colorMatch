"""Contains a set of functions used to align images"""
import numpy as np
import cv2
import colorTools
import imageTools
import logger

LOGGER = logger.getLogger(__name__, 'app')

def __calculateOffset(offsetImage, targetImage):
    offset, _ = cv2.phaseCorrelate(targetImage, offsetImage)
    offset = list(offset)
    offset = [round(value) for value in offset]
    return np.array(offset)

def __getPreparedEye(gray):
    gray = cv2.bilateralFilter(np.clip(gray * 255, 0, 255).astype('uint8'),5,50,50)
    prepped = cv2.Sobel(gray, cv2.CV_16S, 1, 1, ksize=5)
    return np.float32(prepped)

def __getEyeOffsets(eyes, sharpestIndex, wb=None):
    eyes = [colorTools.convert_sBGR_to_linearBGR_float_fast(eye) for eye in eyes]
    if wb is not None:
        eyes = [colorTools.whitebalance_from_asShot_to_d65(eye, *wb) for eye in eyes]

    greyEyes = [np.min(eye, axis=2) for eye in eyes] #Sort of counter intuitive, but using min we basically isolate white values/reflections
    stretchedEyes = [imageTools.stretchHistogram(greyEye, [3, 3]) for greyEye in greyEyes]
    preparedEyes = [__getPreparedEye(stretchedEye) for stretchedEye in stretchedEyes]

    relativeEyeOffsets = [__calculateOffset(preparedEye, preparedEyes[sharpestIndex]) for index, preparedEye in enumerate(preparedEyes)]

    eyeOffsets = relativeEyeOffsets

    return np.array(eyeOffsets)

def getCaptureEyeOffsets(captures):
    """
    Returns the image offsets between captures for the left eye, right eye, full face, and an average of left eye and right eye
        Left eye and Right eye are calculated using phase correlation - should be reasonably accurate
        Full face simply uses facial landmarks to calculate the offset - Expect to be less accurate
    """
    wb = captures[0].whiteBalance
    sharpestMask = np.array([capture.isSharpest for capture in captures])
    sharpestIndex = np.arange(len(sharpestMask))[sharpestMask][0]

    leftEyeCrops = [capture.leftEyeImage for capture in captures]
    rightEyeCrops = [capture.rightEyeImage for capture in captures]

    if (not leftEyeCrops) or (not rightEyeCrops):
        raise ValueError('Eye Capture Missing')

    leftEyeOffsets = __getEyeOffsets(leftEyeCrops, sharpestIndex, wb)
    rightEyeOffsets = __getEyeOffsets(rightEyeCrops, sharpestIndex, wb)

    leftEyeBBs = np.array([capture.landmarks.getLeftEyeBB() for capture in captures])
    leftEyeBBPoints = np.array([[bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]] for bb in leftEyeBBs])
    leftJointBB = [np.min(leftEyeBBPoints[:, 0]), np.min(leftEyeBBPoints[:, 1]), np.max(leftEyeBBPoints[:, 2]), np.max(leftEyeBBPoints[:, 3])]
    leftEyeJointCrops = [capture.faceImage[leftJointBB[1]:leftJointBB[3], leftJointBB[0]:leftJointBB[2]] for capture in captures]
    leftEyeJointOffsets = __getEyeOffsets(leftEyeJointCrops, sharpestIndex, wb)

    rightEyeBBs = np.array([capture.landmarks.getRightEyeBB() for capture in captures])
    rightEyeBBPoints = np.array([[bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]] for bb in rightEyeBBs])
    rightJointBB = [np.min(rightEyeBBPoints[:, 0]), np.min(rightEyeBBPoints[:, 1]), np.max(rightEyeBBPoints[:, 2]), np.max(rightEyeBBPoints[:, 3])]
    rightEyeJointCrops = [capture.faceImage[rightJointBB[1]:rightJointBB[3], rightJointBB[0]:rightJointBB[2]] for capture in captures]
    rightEyeJointOffsets = __getEyeOffsets(rightEyeJointCrops, sharpestIndex, wb)

    averageJointOffset = np.round((leftEyeJointOffsets + rightEyeJointOffsets) / 2).astype('int32')

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

    LOGGER.info('L Eye Offset ::\n%s', leftEyeOffsets)
    LOGGER.info('R Eye Offset ::\n%s', rightEyeOffsets)
    LOGGER.info('L/R Eye Offset Averages ::\n%s', averageFaceLandmarksOffsets)
    LOGGER.info('L/R TEST Eye Offset Averages ::\n%s', scaledAverageOffsets)

    return [leftEyeOffsets, rightEyeOffsets, averageFaceLandmarksOffsets, scaledAverageOffsets]
