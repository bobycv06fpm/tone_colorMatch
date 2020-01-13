"""Functions for working with images and captures"""
import numpy as np
import cv2
import colorTools
import cropTools
import extractMask

from logger import getLogger
LOGGER = getLogger(__name__, 'app')

#TAKES A FLOAT
def stretchHistogram(gray, clipValues=None, mask=None):
    """
    Stretches grey image to fill full range.
        clipValues takes a set of two values, [lowerSDMult, upperSDMult]
        Lower standard deviation multiplier (everything less than (lowerSDMult * SD) gets clipped to (lowerSDMult * SD). vice-versa for upperSDMult.
    """
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

    if clipValues is None:
        lower = np.min(grayPoints)
        upper = np.max(grayPoints)
    else:
        lowerSDMult, upperSDMult = clipValues

        median = np.median(grayPoints)
        sd = np.std(grayPoints)
        lower = median - (lowerSDMult * sd)
        lower = lower if lower > lowerBound else lowerBound
        upper = median + (upperSDMult * sd)
        upper = upper if upper < upperBound else upperBound


    bounds = np.copy(gray)
    bounds[bounds < lower] = lower
    bounds[bounds > upper] = upper

    numerator = bounds - lower
    denominator = upper - lower
    stretched = (numerator / denominator)
    #stretched = np.clip(stretched * 255, 0, 255)
    return stretched

def simpleStretchHistogram(gray):
    """Very simple stretch histogram. Will return all values with a range (0-1)"""
    return (gray - np.min(gray)) * (1 / (np.max(gray) - np.min(gray)))


def __getImageSetFFTMeans(images):
    """Takes a set of sBGR images and returns the mean FFT for each image"""
    linearImages = [colorTools.convert_sBGR_to_linearBGR_float(image) for image in images]
    greyLinearImages = [np.mean(linearImage, axis=2) for linearImage in linearImages] #Convert to greyscale by averaging BGR channels
    stretchedGreyLinearImages = [stretchHistogram(greyLinearImage, [2, 10]) for greyLinearImage in greyLinearImages] #Stretching helps normalize contrast across different exposures

    fftImages = [np.fft.fft2(stretchedGreyLinearImage) for stretchedGreyLinearImage in stretchedGreyLinearImages]
    shiftedFFTImages = [np.abs(np.fft.fftshift(fftImage)) for fftImage in fftImages]
    meanFFTs = [np.mean(shiftedFFTImage) for shiftedFFTImage in shiftedFFTImages]
    return meanFFTs

def labelSharpestCaptures(captures):
    """
    Compares all of the captures an labels the single sharpest and the two blurriest
        Modifies some of the passed in captures with blurriest or sharpest labels
    """
    leftEyeCrops = [capture.leftEyeImage for capture in captures]
    rightEyeCrops = [capture.rightEyeImage for capture in captures]

    #Use Mean FFT as an approximate for sharpness. Get for each eye independently
    leftEyeCropsMeanFFT = __getImageSetFFTMeans(leftEyeCrops)
    rightEyeCropsMeanFFT = __getImageSetFFTMeans(rightEyeCrops)

    #Average L and R meanFFT to generate a sharpness score
    scores = [(left + right) / 2 for (left, right) in zip(leftEyeCropsMeanFFT, rightEyeCropsMeanFFT)]
    sortedScores = sorted(scores)

    LOGGER.info('Eye Sharpness Scores :: %s', scores)

    for score, capture in zip(scores, captures):
        #Label blurriest 2 captures as such
        if score in sortedScores[:2]:
            capture.isBlurry = True
            LOGGER.info('Blurry Capture :: %s', capture.name)

        #Label sharpest capture
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

def __calculateImageOffset(offsetImage, targetImage):
    """Takes a prepared image and a prepared target image to find the offset from"""
    offset, _ = cv2.phaseCorrelate(targetImage, offsetImage)
    offset = list(offset)
    offset = [round(value) for value in offset]
    return np.array(offset)

def __getPreparedEyeImage(gray):
    """Takes a grayscale image and returns an image ready to be aligned"""
    gray = cv2.bilateralFilter(np.clip(gray * 255, 0, 255).astype('uint8'), 5, 50, 50)
    prepped = cv2.Sobel(gray, cv2.CV_16S, 1, 1, ksize=5)
    return np.float32(prepped)

def __getEyeOffsets(eyes, sharpestIndex, wb=None):
    """Takes a set of eye images, the index of the sharpest eye, and returns a set of offsets for each image"""
    eyes = [colorTools.convert_sBGR_to_linearBGR_float(eye) for eye in eyes]
    if wb is not None:
        eyes = [colorTools.whitebalance_from_asShot_to_d65(eye, *wb) for eye in eyes]

    greyEyes = [np.min(eye, axis=2) for eye in eyes] #Sort of counter intuitive, but using min we basically isolate white values/reflections
    stretchedEyes = [stretchHistogram(greyEye, [3, 3]) for greyEye in greyEyes]
    preparedEyes = [__getPreparedEyeImage(stretchedEye) for stretchedEye in stretchedEyes]

    relativeEyeOffsets = [__calculateImageOffset(preparedEye, preparedEyes[sharpestIndex]) for index, preparedEye in enumerate(preparedEyes)]

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

def getReflectionColor(reflectionPoints):
    """
    Converts a set of points to a single BGR point representative of that color, without value information
       Only Use for WB!
    """
    reflectionHSV = colorTools.naiveBGRtoHSV(np.array([reflectionPoints]))[0]
    medianHSV = np.median(reflectionHSV, axis=0)
    hue, sat, _ = medianHSV

    proportionalBGR = colorTools.hueSatToBGRRatio(hue, sat)
    return np.asarray(proportionalBGR)

def extractSkinReflectionMask(brightestCapture, dimmestCapture, wb_ratios):
    """Returns a mask based on the face regions and a hueristic to avoid surface specular reflection on the skin"""
    brightest = colorTools.convert_sBGR_to_linearBGR_float(brightestCapture.faceImage)
    dimmest = colorTools.convert_sBGR_to_linearBGR_float(dimmestCapture.faceImage)

    brightest_wb = brightest / wb_ratios
    dimmest_wb = dimmest / wb_ratios

    brightest = cv2.GaussianBlur(brightest_wb, (5, 5), 0)
    dimmest = cv2.GaussianBlur(dimmest_wb, (5, 5), 0)

    diff = brightest - dimmest
    subZero = diff < 0
    subzeroMask = np.sum(subZero.astype('uint8'), axis=2) > 0
    diff[subzeroMask] = [0, 0, 0]

    leftCheekPolygon = brightestCapture.landmarks.getLeftCheekPoints()
    rightCheekPolygon = brightestCapture.landmarks.getRightCheekPoints()
    chinPolygon = brightestCapture.landmarks.getChinPoints()
    foreheadPolygon = brightestCapture.landmarks.getForeheadPoints()

    hsv = colorTools.naiveBGRtoHSV(diff)
    brightv2 = np.mean(diff, axis=2)
    hsv[:, :, 2] = brightv2

    masked_hsv = extractMask.getMaskedImage(hsv, brightestCapture.faceMask, [leftCheekPolygon, rightCheekPolygon, chinPolygon, foreheadPolygon])
    masked_region_rough = masked_hsv[:, :, 1] != 0

    rotateHue = masked_hsv[:, :, 0] + 0.25
    rotateHue[rotateHue > 1] -= 1

    masked_hsv[masked_region_rough, 0] = rotateHue[masked_region_rough]

    hue_median = np.median(masked_hsv[masked_region_rough, 0])
    hue_std = np.std(masked_hsv[masked_region_rough, 0])

    sat_median = np.median(masked_hsv[masked_region_rough, 1])
    sat_std = np.std(masked_hsv[masked_region_rough, 1])

    mask_template = np.zeros(masked_hsv.shape[0:2], dtype='bool')

    hue_lower_mask = np.copy(mask_template)
    hue_upper_mask = np.copy(mask_template)
    sat_lower_mask = np.copy(mask_template)
    sat_upper_mask = np.copy(mask_template)

    multiplier = 0.5
    hue_range = multiplier * hue_std
    hue_lower_mask[masked_region_rough] = masked_hsv[masked_region_rough, 0] > (hue_median - hue_range)
    hue_upper_mask[masked_region_rough] = masked_hsv[masked_region_rough, 0] < (hue_median + hue_range)
    hue_mask = np.logical_and(hue_lower_mask, hue_upper_mask)

    sat_range = multiplier * sat_std
    sat_lower_mask[masked_region_rough] = masked_hsv[masked_region_rough, 1] > (sat_median - sat_range)
    sat_upper_mask[masked_region_rough] = masked_hsv[masked_region_rough, 1] < (sat_median + sat_range)
    sat_mask = np.logical_and(sat_lower_mask, sat_upper_mask)

    points_mask = np.logical_and(sat_mask, hue_mask)

    return points_mask


def synthesis(captures):
    """Create a visual approximate for what the color measuring algorithm is doing. Helpful for spot and santiy checks"""
    #Just Temp ... figure out robust way to do this?
    scale = 10

    images = [capture.faceImage for capture in captures]
    linearImages = np.asarray([colorTools.convert_sBGR_to_linearBGR_float(image) for image in images], dtype='float32')
    #linearImagesBlur = np.array([cv2.GaussianBlur(img, (3, 3), 0) for img in linearImages])
    #linearImagesDiffsOld = linearImages[:-1] - linearImages[1:]

    count = np.arange(len(captures))
    oddMask = (np.arange(len(captures)) % 2).astype('bool')
    evenMask = np.logical_not(oddMask)

    oddMaskBrighter = np.copy(oddMask)
    oddMaskBrighter[max(count[oddMaskBrighter])] = False
    oddMaskDarker = np.copy(oddMask)
    oddMaskDarker[min(count[oddMaskBrighter])] = False

    evenMaskBrighter = np.copy(evenMask)
    evenMaskBrighter[max(count[evenMaskBrighter])] = False
    evenMaskDarker = np.copy(evenMask)
    evenMaskDarker[min(count[evenMaskBrighter])] = False

    linearImages[oddMaskBrighter] -= linearImages[oddMaskDarker]
    linearImages[evenMaskBrighter] -= linearImages[evenMaskDarker]

    linearImageSynth = np.median(linearImages[:-2], axis=0)
    linearImageSynthMedianBlur = cv2.medianBlur(linearImageSynth, 5)

    linearImageSynthTransform = np.clip(np.round(linearImageSynth * scale * 255), 0, 255).astype('uint8')
    linearImageSynthTransformMedianBlur = cv2.medianBlur(linearImageSynthTransform, 5)

    show = np.hstack([linearImageSynthTransform, linearImageSynthTransformMedianBlur])
    showSBGR = colorTools.convert_linearBGR_float_to_sBGR(show / 255).astype('uint8')

    shows = np.vstack([show, showSBGR])

    return [linearImageSynthMedianBlur, shows]
