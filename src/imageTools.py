"""Functions for working with images and captures"""
import numpy as np
import colorTools

from logger import getLogger
LOGGER = getLogger(__name__, 'app')

#TAKES A FLOAT
def __stretchHistogram(gray, mask=None):
    """Stretches grey image to fill full range"""
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
    """Compares all of the captures an labels the sharpest"""
    leftEyeCrops = [capture.leftEyeImage for capture in captures]
    rightEyeCrops = [capture.rightEyeImage for capture in captures]

    leftEyeCropsLinear = [colorTools.convert_sBGR_to_linearBGR_float_fast(leftEyeCrop) for leftEyeCrop in leftEyeCrops]
    greyLeftEyeCropsLinear = [np.mean(linearLeftEyeCrop, axis=2) for linearLeftEyeCrop in leftEyeCropsLinear]
    greyLeftEyeCropsLinearStretched = [__stretchHistogram(greyLeftEyeCropLinear) for greyLeftEyeCropLinear in greyLeftEyeCropsLinear]
    greyLeftEyeCropsLinearStretchedFFT = [np.fft.fft2(greyLeftEyeCropLinearStretched) for greyLeftEyeCropLinearStretched in greyLeftEyeCropsLinearStretched]
    greyLeftEyeCropsLinearStretchedFFTShifted = [np.abs(np.fft.fftshift(greyLeftEyeCropLinearStretchedFFT)) for greyLeftEyeCropLinearStretchedFFT in greyLeftEyeCropsLinearStretchedFFT]
    greyLeftEyeCropsLinearStretchedFFTShiftedMeans = [np.mean(leftEyeFFT) for leftEyeFFT in greyLeftEyeCropsLinearStretchedFFTShifted]

    rightEyeCropsLinear = [colorTools.convert_sBGR_to_linearBGR_float_fast(rightEyeCrop) for rightEyeCrop in rightEyeCrops]
    greyRightEyeCropsLinear = [np.mean(linearRightEyeCrop, axis=2) for linearRightEyeCrop in rightEyeCropsLinear]
    greyRightEyeCropsLinearStretched = [__stretchHistogram(greyRightEyeCropLinear) for greyRightEyeCropLinear in greyRightEyeCropsLinear]
    greyRightEyeCropsLinearStretchedFFT = [np.fft.fft2(greyRightEyeCropLinearStretched) for greyRightEyeCropLinearStretched in greyRightEyeCropsLinearStretched]
    greyRightEyeCropsLinearStretchedFFTShifted = [np.abs(np.fft.fftshift(greyRightEyeCropLinearStretchedFFT)) for greyRightEyeCropLinearStretchedFFT in greyRightEyeCropsLinearStretchedFFT]
    greyRightEyeCropsLinearStretchedFFTShiftedMeans = [np.mean(rightEyeFFT) for rightEyeFFT in greyRightEyeCropsLinearStretchedFFTShifted]

    scores = [(left + right) / 2 for (left, right) in zip(greyLeftEyeCropsLinearStretchedFFTShiftedMeans, greyRightEyeCropsLinearStretchedFFTShiftedMeans)]
    sortedScores = sorted(scores)

    LOGGER.info('Eye Sharpness Scores :: %s', scores)

    for score, capture in zip(scores, captures):
        #Label blurriest 2 captures as such
        if score in sortedScores[:2]:
            capture.isBlurry = True
            LOGGER.info('Blurry Capture :: %s', capture.name)

        if score == sortedScores[-1]:
            capture.isSharpest = True
            LOGGER.info('Sharpest Capture :: %s', capture.name)

def getClippedMask(img, shadowPixels=1):
    """Returns a mask covering pixels that are blown out or too small"""
    highlightPixels = np.iinfo(img.dtype).max - 1 #Blown Out Highlights

    isSmallSubPixelMask = img < shadowPixels
    isLargeSubPixelMask = img > highlightPixels

    isClippedSubPixelMask = np.logical_or(isSmallSubPixelMask, isLargeSubPixelMask)
    isClippedPixelMask = np.any(isClippedSubPixelMask, axis=2)

    return isClippedPixelMask
