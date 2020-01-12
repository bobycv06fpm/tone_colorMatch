"""Contains a set of functions used to align images"""
import numpy as np
import cv2
import colorTools
import imageTools
import cropTools
import logger

LOGGER = logger.getLogger(__name__, 'app')

def __calculateOffset(offsetImage, targetImage):
    offset, _ = cv2.phaseCorrelate(targetImage, offsetImage)
    offset = list(offset)
    offset = [round(value) for value in offset]
    return np.array(offset)

def __getPreparedEye(gray):
    gray = cv2.bilateralFilter(np.clip(gray * 255, 0, 255).astype('uint8'), 5, 50, 50)
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


def getCapturesOffsets(captures):
    """
    Returns the image offsets between captures for the left eye crop, right eye crop, and face crop
    """
    wb = captures[0].whiteBalance
    sharpestMask = np.array([capture.isSharpest for capture in captures])
    sharpestIndex = np.arange(len(sharpestMask))[sharpestMask][0]

    leftEyeCrops = [capture.leftEyeImage for capture in captures]
    rightEyeCrops = [capture.rightEyeImage for capture in captures]

    if (not leftEyeCrops) or (not rightEyeCrops):
        raise ValueError('Eye Capture Missing')

    #Offsets from using the left and right eyes from the face crop
    faceCropLeftEyes = cropTools.cropImagesToParentBB([capture.faceImage for capture in captures], [capture.landmarks.getLeftEyeBB() for capture in captures])
    faceCropLeftEyeOffsets = __getEyeOffsets(faceCropLeftEyes, sharpestIndex, wb)

    faceCropRightEyes = cropTools.cropImagesToParentBB([capture.faceImage for capture in captures], [capture.landmarks.getRightEyeBB() for capture in captures])
    faceCropRightEyeOffsets = __getEyeOffsets(faceCropRightEyes, sharpestIndex, wb)

    faceCropOffsets = np.round((faceCropLeftEyeOffsets + faceCropRightEyeOffsets) / 2).astype('int32')

    #Offsets from using the full resolution left and right eye crops
    eyeCropLeftEyeOffsets = __getEyeOffsets(leftEyeCrops, sharpestIndex, wb)
    eyeCropRightEyeOffsets = __getEyeOffsets(rightEyeCrops, sharpestIndex, wb)

    LOGGER.info('Face Offsets ::\n%s', faceCropOffsets)

    return [eyeCropLeftEyeOffsets, eyeCropRightEyeOffsets, faceCropOffsets]
