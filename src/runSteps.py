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

def getBestGuess(faceRegions, leftEyeReflections, rightEyeReflections):
    """
    Best Guess is an alternative to linear fit
    Do not think there is any promise in this, probably better just to use the median slopes
    """
    LOGGER.info('PLOTTING: Region Scaled Linearity')
    captureFaceRegions = np.array([regions.getRegionMedians() for regions in faceRegions])
    numberOfRegions = captureFaceRegions.shape[1]

    #averageEyeReflections = (leftEyeReflections + rightEyeReflections) / 2

    scaledCaptureFaceRegions = []

    for regionIndex in range(0, numberOfRegions):
        diff = plotTools.getDiffs(captureFaceRegions[3:-1, regionIndex, :])
        scaledCaptureFaceRegion = diff #/ (np.ones(3) * np.reshape(diff[:, 2], (diff.shape[0], 1)))
        scaledCaptureFaceRegions.append(scaledCaptureFaceRegion)

    scaledCaptureFaceRegions = np.vstack(scaledCaptureFaceRegions)
    #LOGGER.info('SCALED DIFFS CAPTURE FACE REGIONS :: ' + str(scaledCaptureFaceRegions))

    leftEyeDiffs = plotTools.getDiffs(leftEyeReflections[3:-1])
    #leftEyeDiffs = getDiffs(leftEyeReflections[-4:-1])
    rightEyeDiffs = plotTools.getDiffs(rightEyeReflections[3:-1])
    #rightEyeDiffs = getDiffs(rightEyeReflections[-4:-1])
    leftEyeDiffs[:, 2][leftEyeDiffs[:, 2] == 0] = 0.001
    rightEyeDiffs[:, 2][rightEyeDiffs[:, 2] == 0] = 0.001
    scaledLeftEyeReflections = leftEyeDiffs #/ (np.ones(3) * np.reshape(leftEyeDiffs[:, 2], (leftEyeDiffs.shape[0], 1)))
    scaledRightEyeReflections = rightEyeDiffs #/ (np.ones(3) * np.reshape(rightEyeDiffs[:, 2], (rightEyeDiffs.shape[0], 1)))

    scaledDiffReflections = np.vstack((scaledLeftEyeReflections, scaledRightEyeReflections))

    medianScaledDiffFace = list(np.median(scaledCaptureFaceRegions, axis=0))
    medianScaledDiffReflections = list(np.median(scaledDiffReflections, axis=0))
    return [medianScaledDiffReflections, medianScaledDiffFace]

def scoreLinearFit(linearFitObject):
    """Generates a rough score for the linear fit using the residuals scaled by the rise of the line"""
    residuals = linearFitObject[:, 1]
    linearFit = np.vstack(linearFitObject[:, 0])

    minValue = linearFit[:, 0] * 0.5 + linearFit[:, 1]
    maxValue = linearFit[:, 0] * 1.0 + linearFit[:, 1]
    valueRange = maxValue - minValue

    valueRange = valueRange * valueRange

    score = residuals / valueRange
    #print('\nLinear Fit Residuals for range {} :: {}\nSCORE :: {}\n'.format(valueRange, residuals, score))
    return score

def getLinearFits(leftEyeReflections, rightEyeReflections, leftSclera, rightSclera, faceRegions, blurryMask):
    """Get the linear fit for each region of the face, each eye inner sclera, and each eye specular reflection"""
    flashRatios = np.array([regions.capture.flashRatio for regions in faceRegions])
    print('Flash Ratios :: {}'.format(flashRatios))

    filteredFlashRatios = flashRatios[np.logical_not(blurryMask)]

    filteredLeftEyeReflections = leftEyeReflections[np.logical_not(blurryMask)]
    filteredRightEyeReflections = rightEyeReflections[np.logical_not(blurryMask)]

    filteredLeftSclera = leftSclera[np.logical_not(blurryMask)]
    filteredRightSclera = rightSclera[np.logical_not(blurryMask)]


    leftEyeLinearFitFull = np.array([plotTools.fitLine(filteredFlashRatios, filteredLeftEyeReflections[:, subPixel]) for subPixel in range(0, 3)])
    #leftEyeLinearFitFullTest = np.array([fitLine2(filteredFlashRatios, filteredLeftEyeReflections[:, subPixel]) for subPixel in range(0, 3)])

    print('Left Eye Linear Fit Full :: {}'.format(leftEyeLinearFitFull))
    #print('Left Eye Linear Fit Full Test :: {}'.format(leftEyeLinearFitFullTest))

    leftEyeLinearFit = np.vstack(leftEyeLinearFitFull[:, 0])
    leftEyeScores = scoreLinearFit(leftEyeLinearFitFull)

    rightEyeLinearFitFull = np.array([plotTools.fitLine(filteredFlashRatios, filteredRightEyeReflections[:, subPixel]) for subPixel in range(0, 3)])

    rightEyeLinearFit = np.vstack(rightEyeLinearFitFull[:, 0])
    rightEyeScores = scoreLinearFit(rightEyeLinearFitFull)

    #TODO: HERE IS WHERE THE BUG WAS FIXED. RightScelra -> LeftScelra
    leftScleraLinearFitFull = np.array([plotTools.fitLine(filteredFlashRatios, filteredLeftSclera[:, subPixel]) for subPixel in range(0, 3)])
    leftScleraLinearFit = np.vstack(leftScleraLinearFitFull[:, 0])

    rightScleraLinearFitFull = np.array([plotTools.fitLine(filteredFlashRatios, filteredRightSclera[:, subPixel]) for subPixel in range(0, 3)])
    rightScleraLinearFit = np.vstack(rightScleraLinearFitFull[:, 0])

    captureFaceRegions = np.array([regions.getRegionMedians() for regions in faceRegions])
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
    #LOGGER.info('CAPTURE FACE REGIONS LINEAR FIT :: {}'.format(captureFaceRegionsLinearFit))
    captureFaceRegionsScores = np.array(captureFaceRegionsScores)
    reflectionScores = np.array([leftEyeScores, rightEyeScores])

    #meanReflectionScores = np.stack(np.mean(reflectionScores, axis=0)).flatten()
    maxReflectionScores = np.stack(np.max(reflectionScores, axis=0)).flatten()
    #meanFaceRegionScores = np.stack(np.mean(captureFaceRegionsScores, axis=0)).flatten()
    maxFaceRegionScores = np.stack(np.max(captureFaceRegionsScores, axis=0)).flatten()

    #leftEyeLinearFitHSV = colorTools.bgr_to_hsv(leftEyeLinearFit)
    #rightEyeLinearFitHSV = colorTools.bgr_to_hsv(rightEyeLinearFit)
    #captureFaceRegionsLinearFitHSV  = [colorTools.bgr_to_hsv(point) for point in  captureFaceRegionsLinearFit]

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

#TODO: Not sure this really makes sense...
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

    #wb = np.hstack([dimmest_wb, brightest_wb])
    #no_wb = np.hstack([dimmest, brightest])
    #comp = np.vstack([no_wb, wb])
    #cv2.imshow('WB', comp)
    #cv2.waitKey(0)

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

    print('MIN HUE :: {}'.format(np.min(masked_hsv[masked_region_rough, 0])))
    print('MAX HUE :: {}'.format(np.max(masked_hsv[masked_region_rough, 0])))
    print('MIN SAT :: {}'.format(np.min(masked_hsv[masked_region_rough, 1])))
    print('MAX SAT :: {}'.format(np.max(masked_hsv[masked_region_rough, 1])))
    print('---')
    rotateHue = masked_hsv[:, :, 0] + 0.25
    rotateHue[rotateHue > 1] -= 1

    masked_hsv[masked_region_rough, 0] = rotateHue[masked_region_rough]

    print('MIN HUE :: {}'.format(np.min(masked_hsv[masked_region_rough, 0])))
    print('MAX HUE :: {}'.format(np.max(masked_hsv[masked_region_rough, 0])))
    print('MIN SAT :: {}'.format(np.min(masked_hsv[masked_region_rough, 1])))
    print('MAX SAT :: {}'.format(np.max(masked_hsv[masked_region_rough, 1])))

    print('Hue Len :: {}'.format(len(masked_hsv[masked_region_rough, 0].ravel())))
    print('Sat Len :: {}'.format(len(masked_hsv[masked_region_rough, 1].ravel())))

    hue_median = np.median(masked_hsv[masked_region_rough, 0])
    hue_std = np.std(masked_hsv[masked_region_rough, 0])

    sat_median = np.median(masked_hsv[masked_region_rough, 1])
    sat_std = np.std(masked_hsv[masked_region_rough, 1])

    print('Hue - Median :: {} | STD :: {}'.format(hue_median, hue_std))
    print('Sat - Median :: {} | STD :: {}'.format(sat_median, sat_std))

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

    synth, displaySynth = synthesis(captures)
    state.saveReferenceImageBGR(displaySynth, captures[0].name + '_syntheticSanityCheck')

    try:
        averageReflection, averageReflectionArea, leftEyeReflections, rightEyeReflections, leftSclera, rightSclera, blurryMask = getAverageScreenReflectionColor(captures, leftEyeCropOffsets, rightEyeCropOffsets, state)
    except ValueError as err:
        LOGGER.error('User :: %s | Image :: %s | Error :: %s | Details ::\n%s', state.user_id, state.imageName(), 'Error Extracting Reflection', err)
        state.errorProccessing()
        if failOnError:
            raise
        return getFailureResponse(state.imageName())

    leftEyeProportionalBGR = getReflectionColor(leftEyeReflections)
    rightEyeProportionalBGR = getReflectionColor(rightEyeReflections)

    propBGR = (leftEyeProportionalBGR + rightEyeProportionalBGR) / 2
    print('Average Reflection PROP BGR :: {}'.format(propBGR))

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

    #--TEMP FOR DEBUG?---
    plotTools.plotPerRegionLinearity(faceRegions, leftEyeReflections, rightEyeReflections, leftSclera, rightSclera, blurryMask, state)
    plotTools.plotPerRegionLinearityAlt(faceRegions, leftEyeReflections, rightEyeReflections, blurryMask, state)
    plotTools.plotPerRegionScaledLinearity(faceRegions, leftEyeReflections, rightEyeReflections, state)
    plotTools.plotPerRegionDiffs(faceRegions, leftEyeReflections, rightEyeReflections, state)
    #--END TEMP FOR DEBUG?---

    captureSets = zip(faceRegions, leftEyeReflections, rightEyeReflections)

    print('Left Eye Reflections :: ' + str(leftEyeReflections))
    linearFitSets = getLinearFits(leftEyeReflections, rightEyeReflections, leftSclera, rightSclera, faceRegions, blurryMask)
    #linearFitExperimental = getLinearFitExperimental(leftEyeReflections, rightEyeReflections, faceRegions, blurryMask)

    print('Left Eye Reflection Slope :: ' + str(linearFitSets["reflections"]["left"]))
    #averageReflectionSlope = (np.array(linearFitSets["reflections"]["left"]) + np.array(linearFitSets["reflections"]["right"])) / 2
    averageSkinSlope = (np.array(linearFitSets["regions"]["left"]) + np.array(linearFitSets["regions"]["right"])
                        + np.array(linearFitSets["regions"]["chin"]) + np.array(linearFitSets["regions"]["forehead"])) / 4
    averageCheekSlope = (np.array(linearFitSets["regions"]["left"]) + np.array(linearFitSets["regions"]["right"])) / 2
    averageScleraSlope = (np.array(linearFitSets["sclera"]["left"]) + np.array(linearFitSets["sclera"]["right"])) / 2

    print('Reflection Proprtional BGR :: {}'.format(propBGR))

    print('Average Skin Slope :: {}'.format(averageSkinSlope))
    print('Average Cheek Slope :: {}'.format(averageCheekSlope))
    print('AverageScleraSlope :: {}'.format(averageScleraSlope))

    averageSkinSlopeWB = averageSkinSlope / propBGR
    averageCheekSlopeWB = averageCheekSlope  / propBGR
    averageScleraSlopeWB = averageScleraSlope / propBGR
    #approxScleraValue = max(averageScleraSlopeWB) * np.array([1.0, 1.0, 1.0])

    print('Average Skin Slope WB:: {}'.format(averageSkinSlopeWB))
    print('Average Cheek Slope WB:: {}'.format(averageCheekSlopeWB))
    print('AverageScleraSlope WB:: {}'.format(averageScleraSlopeWB))

    faceRatio = averageCheekSlopeWB[2] / averageScleraSlopeWB[2]

    print('Ratio :: {}'.format(faceRatio))

    channelRatio = averageSkinSlopeWB
    print('\n----\nChannel Ratio :: {}'.format(channelRatio))
    print('B - G Channel Ratio :: {}\n---\n'.format(channelRatio[0] / channelRatio[1]))
    hue = 60 * ((channelRatio[1] - channelRatio[0]) / (channelRatio[2])) % 6
    sat = (max(channelRatio) - min(channelRatio)) / max(channelRatio)
    val = faceRatio

    bestGuess = getBestGuess(faceRegions, leftEyeReflections, rightEyeReflections)

    LOGGER.info('Done Analysis - Generating Results')

    calibrated_skin_color = [hue, sat, val]#[0.0, 0.0, 0.0]
    matched_skin_color_id = 0
    state.saveCaptureResults(calibrated_skin_color, matched_skin_color_id)

    response = getSuccessfulResponse(state.imageName(), captureSets, linearFitSets, bestGuess, averageReflectionArea)
    #print(json.dumps(response))
    LOGGER.info('Done - Returing Results')
    return response
