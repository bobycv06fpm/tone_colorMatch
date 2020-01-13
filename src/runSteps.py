"""
Skin tone measuring image processing pipeline
"""
import cv2
import numpy as np
from getAverageReflection import getAverageScreenReflectionColor
from state import State
import colorTools
import plotTools
import cropTools
import imageTools
from dataFormatTools import getFailureResponse, getSuccessfulResponse
import extractMask
from capture import Capture
from faceRegions import FaceRegions
from logger import getLogger

LOGGER = getLogger(__name__, 'app')

def scoreLinearFit(linearFitObject):
    """Generates a rough score for the linear fit using the residuals scaled by the rise of the line"""
    residuals = linearFitObject[:, 1]
    linearFit = np.vstack(linearFitObject[:, 0])

    minValue = linearFit[:, 0] * 0.5 + linearFit[:, 1]
    maxValue = linearFit[:, 0] * 1.0 + linearFit[:, 1]
    valueRange = maxValue - minValue

    valueRange = valueRange * valueRange

    score = residuals / valueRange
    return score

def getLinearFits(leftEyeReflections, rightEyeReflections, leftSclera, rightSclera, faceRegions, blurryMask):
    """Get the linear fit for each region of the face, each eye inner sclera, and each eye specular reflection"""
    flashRatios = np.array([regions.capture.flashRatio for regions in faceRegions])

    filteredFlashRatios = flashRatios[np.logical_not(blurryMask)]

    filteredLeftEyeReflections = leftEyeReflections[np.logical_not(blurryMask)]
    filteredRightEyeReflections = rightEyeReflections[np.logical_not(blurryMask)]

    filteredLeftSclera = leftSclera[np.logical_not(blurryMask)]
    filteredRightSclera = rightSclera[np.logical_not(blurryMask)]


    leftEyeLinearFitFull = np.array([plotTools.fitLine(filteredFlashRatios, filteredLeftEyeReflections[:, subPixel]) for subPixel in range(0, 3)])

    leftEyeLinearFit = np.vstack(leftEyeLinearFitFull[:, 0])
    leftEyeScores = scoreLinearFit(leftEyeLinearFitFull)

    rightEyeLinearFitFull = np.array([plotTools.fitLine(filteredFlashRatios, filteredRightEyeReflections[:, subPixel]) for subPixel in range(0, 3)])

    rightEyeLinearFit = np.vstack(rightEyeLinearFitFull[:, 0])
    rightEyeScores = scoreLinearFit(rightEyeLinearFitFull)

    leftScleraLinearFitFull = np.array([plotTools.fitLine(filteredFlashRatios, filteredLeftSclera[:, subPixel]) for subPixel in range(0, 3)])
    leftScleraLinearFit = np.vstack(leftScleraLinearFitFull[:, 0])

    rightScleraLinearFitFull = np.array([plotTools.fitLine(filteredFlashRatios, filteredRightSclera[:, subPixel]) for subPixel in range(0, 3)])
    rightScleraLinearFit = np.vstack(rightScleraLinearFitFull[:, 0])

    captureFaceRegions = np.array([regions.getRegionMeans() for regions in faceRegions])
    filteredCaptureFaceRegions = captureFaceRegions[np.logical_not(blurryMask)]

    captureFaceRegionsLinearFit = []
    captureFaceRegionsScores = []

    for regionIndex in range(0, captureFaceRegions.shape[1]):
        linearFitFull = np.array([plotTools.fitLine(filteredFlashRatios, filteredCaptureFaceRegions[:, regionIndex, subPixel]) for subPixel in range(0, 3)])

        linearFit = np.vstack(linearFitFull[:, 0])
        scores = scoreLinearFit(linearFitFull)

        captureFaceRegionsLinearFit.append(linearFit)
        captureFaceRegionsScores.append(scores)

    captureFaceRegionsLinearFit = np.array(captureFaceRegionsLinearFit)
    captureFaceRegionsScores = np.array(captureFaceRegionsScores)
    reflectionScores = np.array([leftEyeScores, rightEyeScores])

    maxReflectionScores = np.stack(np.max(reflectionScores, axis=0)).flatten()
    maxFaceRegionScores = np.stack(np.max(captureFaceRegionsScores, axis=0)).flatten()

    linearFits = {}
    linearFits["reflections"] = {}
    linearFits["reflections"]["left"] = list(leftEyeLinearFit[:, 0])
    linearFits["reflections"]["right"] = list(rightEyeLinearFit[:, 0])
    linearFits["reflections"]["linearityScore"] = list(maxReflectionScores)

    linearFits["regions"] = {}
    linearFits["regions"]["left"] = list(captureFaceRegionsLinearFit[0, :, 0])
    linearFits["regions"]["right"] = list(captureFaceRegionsLinearFit[1, :, 0])
    linearFits["regions"]["chin"] = list(captureFaceRegionsLinearFit[2, :, 0])
    linearFits["regions"]["forehead"] = list(captureFaceRegionsLinearFit[3, :, 0])
    linearFits["regions"]["linearityScore"] = list(maxFaceRegionScores)

    linearFits["sclera"] = {}
    linearFits["sclera"]["left"] = list(leftScleraLinearFit[:, 0])
    linearFits["sclera"]["right"] = list(rightScleraLinearFit[:, 0])

    return linearFits

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
    """Return a mask of the specular reflections on the surface of the skin. Only works on ambient light reflections, not specular reflections caused by device screen"""
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

def getBestGuess(faceRegions, leftEyeReflections, rightEyeReflections):
    """
    Best Guess is an alternative to linear fit
    Do not think there is any promise in this, probably better just to use the median slopes
    """
    LOGGER.info('PLOTTING: Region Scaled Linearity')
    captureFaceRegions = np.array([regions.getRegionMeans() for regions in faceRegions])
    numberOfRegions = captureFaceRegions.shape[1]

    scaledCaptureFaceRegions = []

    for regionIndex in range(0, numberOfRegions):
        diff = plotTools.getDiffs(captureFaceRegions[3:-1, regionIndex, :])
        scaledCaptureFaceRegion = diff 
        scaledCaptureFaceRegions.append(scaledCaptureFaceRegion)

    scaledCaptureFaceRegions = np.vstack(scaledCaptureFaceRegions)

    leftEyeDiffs = plotTools.getDiffs(leftEyeReflections[3:-1])
    rightEyeDiffs = plotTools.getDiffs(rightEyeReflections[3:-1])
    leftEyeDiffs[:, 2][leftEyeDiffs[:, 2] == 0] = 0.001
    rightEyeDiffs[:, 2][rightEyeDiffs[:, 2] == 0] = 0.001
    scaledLeftEyeReflections = leftEyeDiffs #/ (np.ones(3) * np.reshape(leftEyeDiffs[:, 2], (leftEyeDiffs.shape[0], 1)))
    scaledRightEyeReflections = rightEyeDiffs #/ (np.ones(3) * np.reshape(rightEyeDiffs[:, 2], (rightEyeDiffs.shape[0], 1)))

    scaledDiffReflections = np.vstack((scaledLeftEyeReflections, scaledRightEyeReflections))

    medianScaledDiffFace = list(np.median(scaledCaptureFaceRegions, axis=0))
    medianScaledDiffReflections = list(np.median(scaledDiffReflections, axis=0))
    return [medianScaledDiffReflections, medianScaledDiffFace]


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

def run(user_id, capture_id=None, isProduction=False):
    """Run the color measuring pipeline"""
    failOnError = True
    #failOnError = False
    LOGGER.info('BEGINNING COLOR MATCH PROCESSING FOR USER %s CAPTURE %s', user_id, capture_id if capture_id is not None else '-1')
    LOGGER.info('IS PRODUCTION :: %s', isProduction)

    state = State(user_id, capture_id, isProduction)

    try:
        images = state.loadImages()
    except ValueError as err:
        LOGGER.error('User :: %s | Image :: %s | Error :: %s | Details ::\n%s', state.user_id, state.imageName(), 'Error Loading Images', err)
        state.errorProccessing()
        if failOnError:
            raise
        return getFailureResponse(state.imageName())

    state.saveExposurePointImage('exposurePoints', images)

    try:
        metadata = state.getValidatedMetadata()
    except ValueError as err:
        LOGGER.error('User :: %s | Image :: %s | Error :: %s', state.user_id, state.imageName(), 'Metadata does not Match')
        state.errorProccessing()
        if failOnError:
            raise
        return getFailureResponse(state.imageName())

    captures = [Capture(image, meta) for image, meta in zip(images, metadata)]
    imageTools.labelSharpestCaptures(captures)

    try:
        leftEyeCropOffsets, rightEyeCropOffsets, faceCropOffsets = imageTools.getCapturesOffsets(captures)
    except ValueError as err:
        LOGGER.error('User :: %s | Image :: %s | Error :: %s | Details ::\n%s', state.user_id, state.imageName(), 'Error Cropping and Aligning Images', err)
        state.errorProccessing()
        if failOnError:
            raise
        return getFailureResponse(state.imageName())

    cropTools.cropCapturesToFaceOffsets(captures, faceCropOffsets)

    #All offsets are relative to capture[0]
    for capture in captures:
        capture.landmarks = captures[0].landmarks

    _, displaySynth = synthesis(captures)
    state.saveReferenceImageBGR(displaySynth, captures[0].name + '_syntheticSanityCheck')

    try:
        leftEye, rightEye, averageReflectionArea, blurryMask = getAverageScreenReflectionColor(captures, leftEyeCropOffsets, rightEyeCropOffsets, state)
    except ValueError as err:
        LOGGER.error('User :: %s | Image :: %s | Error :: %s | Details ::\n%s', state.user_id, state.imageName(), 'Error Extracting Reflection', err)
        state.errorProccessing()
        if failOnError:
            raise
        return getFailureResponse(state.imageName())

    leftEyeReflections, leftSclera = leftEye
    rightEyeReflections, rightSclera = rightEye

    leftEyeProportionalBGR = getReflectionColor(leftEyeReflections)
    rightEyeProportionalBGR = getReflectionColor(rightEyeReflections)

    propBGR = (leftEyeProportionalBGR + rightEyeProportionalBGR) / 2

    mask = extractSkinReflectionMask(captures[0], captures[-1], propBGR)

    try:
        faceRegions = np.array([FaceRegions(capture, mask) for capture in captures])
    except ValueError as err:
        LOGGER.error('User :: %s | Image :: %s | Error :: %s | Details ::\n%s', state.user_id, state.imageName(), 'Error extracting Points for Recovered Mask', err)
        state.errorProccessing()
        if failOnError:
            raise
        return getFailureResponse(state.imageName())

    LOGGER.info('Finished Image Processing - Beginning Analysis')
    state.saveReferenceImageBGR(faceRegions[0].getMaskedImage(), faceRegions[0].capture.name + '_masked')

    #Used for iterating and tweaking
    plotTools.plotPerRegionLinearity(faceRegions, leftEyeReflections, rightEyeReflections, leftSclera, rightSclera, blurryMask, state)
    plotTools.plotPerRegionLinearityAlt(faceRegions, leftEyeReflections, rightEyeReflections, blurryMask, state)
    plotTools.plotPerRegionScaledLinearity(faceRegions, leftEyeReflections, rightEyeReflections, state)
    plotTools.plotPerRegionDiffs(faceRegions, leftEyeReflections, rightEyeReflections, state)

    captureSets = zip(faceRegions, leftEyeReflections, rightEyeReflections)

    linearFitSets = getLinearFits(leftEyeReflections, rightEyeReflections, leftSclera, rightSclera, faceRegions, blurryMask)

    averageSkinSlope = (np.array(linearFitSets["regions"]["left"]) + np.array(linearFitSets["regions"]["right"])
                        + np.array(linearFitSets["regions"]["chin"]) + np.array(linearFitSets["regions"]["forehead"])) / 4
    averageCheekSlope = (np.array(linearFitSets["regions"]["left"]) + np.array(linearFitSets["regions"]["right"])) / 2
    averageScleraSlope = (np.array(linearFitSets["sclera"]["left"]) + np.array(linearFitSets["sclera"]["right"])) / 2

    averageSkinSlopeWB = averageSkinSlope / propBGR
    averageCheekSlopeWB = averageCheekSlope  / propBGR
    averageScleraSlopeWB = averageScleraSlope / propBGR

    faceRatio = averageCheekSlopeWB[2] / averageScleraSlopeWB[2]

    channelRatio = averageSkinSlopeWB

    hue = 60 * ((channelRatio[1] - channelRatio[0]) / (channelRatio[2])) % 6
    sat = (max(channelRatio) - min(channelRatio)) / max(channelRatio)
    val = faceRatio

    bestGuess = getBestGuess(faceRegions, leftEyeReflections, rightEyeReflections)

    LOGGER.info('Done Analysis - Generating Results')

    calibrated_skin_color = [hue, sat, val]#[0.0, 0.0, 0.0]
    matched_skin_color_id = 0
    state.saveCaptureResults(calibrated_skin_color, matched_skin_color_id)

    response = getSuccessfulResponse(state.imageName(), captureSets, linearFitSets, bestGuess, averageReflectionArea)

    LOGGER.info('Done - Returing Results')
    return response
