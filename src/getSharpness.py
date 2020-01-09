import utils
import numpy as np
import cv2
import colorTools
import math
from logger import getLogger
logger = getLogger(__name__, 'app')

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


def labelSharpestCaptures(captures):
    leftEyeCrops = [capture.leftEyeImage for capture in captures]
    rightEyeCrops = [capture.rightEyeImage for capture in captures]

    leftEyeCropsLinear = [colorTools.convert_sBGR_to_linearBGR_float_fast(leftEyeCrop) for leftEyeCrop in leftEyeCrops]
    greyLeftEyeCropsLinear = [np.mean(linearLeftEyeCrop, axis=2) for linearLeftEyeCrop in leftEyeCropsLinear]
    greyLeftEyeCropsLinearStretched = [stretchHistogram(greyLeftEyeCropLinear) for greyLeftEyeCropLinear in greyLeftEyeCropsLinear]
    greyLeftEyeCropsLinearStretchedFFT = [np.fft.fft2(greyLeftEyeCropLinearStretched) for greyLeftEyeCropLinearStretched in greyLeftEyeCropsLinearStretched]
    greyLeftEyeCropsLinearStretchedFFTShifted = [np.abs(np.fft.fftshift(greyLeftEyeCropLinearStretchedFFT)) for greyLeftEyeCropLinearStretchedFFT in greyLeftEyeCropsLinearStretchedFFT]
    greyLeftEyeCropsLinearStretchedFFTShiftedMeans = [np.mean(leftEyeFFT) for leftEyeFFT in greyLeftEyeCropsLinearStretchedFFTShifted]
    #logger.info('Left Eye Sharpness Scores :: {}'.format(greyLeftEyeCropsLinearStretchedFFTShiftedMeans))

    rightEyeCropsLinear = [colorTools.convert_sBGR_to_linearBGR_float_fast(rightEyeCrop) for rightEyeCrop in rightEyeCrops]
    greyRightEyeCropsLinear = [np.mean(linearRightEyeCrop, axis=2) for linearRightEyeCrop in rightEyeCropsLinear]
    greyRightEyeCropsLinearStretched = [stretchHistogram(greyRightEyeCropLinear) for greyRightEyeCropLinear in greyRightEyeCropsLinear]
    greyRightEyeCropsLinearStretchedFFT = [np.fft.fft2(greyRightEyeCropLinearStretched) for greyRightEyeCropLinearStretched in greyRightEyeCropsLinearStretched]
    greyRightEyeCropsLinearStretchedFFTShifted = [np.abs(np.fft.fftshift(greyRightEyeCropLinearStretchedFFT)) for greyRightEyeCropLinearStretchedFFT in greyRightEyeCropsLinearStretchedFFT]
    greyRightEyeCropsLinearStretchedFFTShiftedMeans = [np.mean(rightEyeFFT) for rightEyeFFT in greyRightEyeCropsLinearStretchedFFTShifted]
    #logger.info('Right Eye Sharpness Scores :: {}'.format(greyRightEyeCropsLinearStretchedFFTShiftedMeans))

    scores = [(left + right) / 2 for (left, right) in zip(greyLeftEyeCropsLinearStretchedFFTShiftedMeans, greyRightEyeCropsLinearStretchedFFTShiftedMeans)]
    sortedScores = sorted(scores)

    logger.info('Eye Sharpness Scores :: {}'.format(scores))

    for score, capture in zip(scores, captures):
        if (score == sortedScores[0]) or (score == sortedScores[1]):
            capture.isBlurry = True
            logger.info('Blurry Capture :: {}'.format(capture.name))

        if (score == sortedScores[-1]):
            capture.isSharpest = True
            logger.info('Sharpest Capture :: {}'.format(capture.name))

