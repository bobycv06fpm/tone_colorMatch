import alignImages
from getAverageReflection import getAverageScreenReflectionColor2
from state import State
from getPolygons import getPolygons, getFullFacePolygon
import colorTools
import plotTools
import cv2
import numpy as np
import thresholdMask
import math
import matplotlib.pyplot as plt
import cropTools
import sharpness
import extractMask
from capture import Capture 
from faceRegions import FaceRegions
from logger import getLogger

import json
import colorsys

logger = getLogger(__name__, 'app')

def getReflectionMap(leftReflection, rightReflection):
    value = {}
    value['left'] = [float(value) for value in leftReflection]
    value['right'] = [float(value) for value in rightReflection]

    return value

def getResponse(imageName, successful, captureSets=None, linearFits=None, bestGuess=None, averageReflectionArea=None):
    response = {}
    response['name'] = imageName
    response['successful'] = successful
    response['captures'] = {}
    response['linearFits'] = linearFits
    response['bestGuess'] = bestGuess
    response['reflectionArea'] = averageReflectionArea

    if not successful:
        return response

    for captureSet in captureSets:
        faceRegions, leftEyeReflection, rightEyeReflection = captureSet
        key = faceRegions.capture.name
        response['captures'][key] = {}
        response['captures'][key]['regions'] = faceRegions.getRegionMapValue()
        response['captures'][key]['reflections'] = getReflectionMap(leftEyeReflection, rightEyeReflection)

    return response

#Best Guess is an alternative to linear fit. Just uses the median slopes
# Do not think there is any promise in this
def getBestGuess(faceRegions, leftEyeReflections, rightEyeReflections):
    logger.info('PLOTTING: Region Scaled Linearity')
    captureFaceRegions = np.array([regions.getRegionMedians() for regions in faceRegions])
    flashRatios = np.array([regions.capture.flashRatio for regions in faceRegions])
    numberOfRegions = captureFaceRegions.shape[1]
    numberOfCaptures = captureFaceRegions.shape[0]

    #averageEyeReflections = (leftEyeReflections + rightEyeReflections) / 2

    scaledCaptureFaceRegions = []

    for regionIndex in range(0, numberOfRegions):
        diff = plotTools.getDiffs(captureFaceRegions[3:-1, regionIndex, :])
        scaledCaptureFaceRegion = diff #/ (np.ones(3) * np.reshape(diff[:, 2], (diff.shape[0], 1)))
        scaledCaptureFaceRegions.append(scaledCaptureFaceRegion)

    scaledCaptureFaceRegions = np.vstack(scaledCaptureFaceRegions)
    #logger.info('SCALED DIFFS CAPTURE FACE REGIONS :: ' + str(scaledCaptureFaceRegions))

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

def isMetadataValid(metadata):
    expectedISO = metadata[0]["iso"]
    expectedExposure = metadata[0]["exposureTime"]

    if not 'faceImageTransforms' in metadata[0]:
        logger.warning('No Face Image Transforms')
        return False

    expectedWB = metadata[0]["whiteBalance"]

    for captureMetadata in metadata:
        iso = captureMetadata["iso"]
        exposure = captureMetadata["exposureTime"]
        wb = captureMetadata["whiteBalance"]

        if (iso != expectedISO) or (exposure != expectedExposure) or (wb['x'] != expectedWB['x']) or (wb['y'] != expectedWB['y']):
            print("White Balance Does Not Match")
            print("Expected :: {} | Received :: {}".format(expectedWB, wb))
            return False
        
    return True

def getMedianDiff(points):
    diffs = []
    for index in range(1, len(points)):
        diffs.append(np.abs(points[index - 1] - points[index]))

    return np.median(np.array(diffs), axis=0)

def scoreLinearFit(linearFit):
    residuals = linearFit[:, 1]
    linearFit = np.vstack(linearFit[:, 0])

    minValue = linearFit[:, 0] * 0.5 + linearFit[:, 1]
    maxValue = linearFit[:, 0] * 1.0 + linearFit[:, 1]
    valueRange = maxValue - minValue

    valueRange = valueRange * valueRange

    score = residuals / valueRange
    #print('\nLinear Fit Residuals for range {} :: {}\nSCORE :: {}\n'.format(valueRange, residuals, score))
    return score

def getLinearFits(leftEyeReflections, rightEyeReflections, leftSclera, rightSclera, faceRegions, blurryMask):
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

    leftScleraLinearFitFull = np.array([plotTools.fitLine(filteredFlashRatios, filteredRightSclera[:, subPixel]) for subPixel in range(0, 3)])
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
    #logger.info('CAPTURE FACE REGIONS LINEAR FIT :: {}'.format(captureFaceRegionsLinearFit))
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

def getMedianDiffs(leftEyeReflections, rightEyeReflections, faceRegions):
    leftEyeDiffReflectionMedian = getMedianDiff(leftEyeReflections)
    leftEyeDiffReflectionMedianHSV = colorTools.bgr_to_hsv(leftEyeDiffReflectionMedian)

    rightEyeDiffReflectionMedian = getMedianDiff(rightEyeReflections)
    rightEyeDiffReflectionMedianHSV = colorTools.bgr_to_hsv(rightEyeDiffReflectionMedian)

    faceRegionMedians = np.vstack([[region.getRegionMedians() for region in faceRegions]])

    #Take half of the face diffs for better accuracy... Maybe
    #print('Regions before :: ' + str(faceRegionMedians))
    #faceRegionMedians = np.array([faceRegionMedian for i, faceRegionMedian in enumerate(faceRegionMedians) if i % 2 == 0])
    #print('Regions after :: ' + str(faceRegionMedians))

    faceRegionDiffMedians = [getMedianDiff(faceRegionMedians[:, idx]) for idx in range(0, faceRegionMedians.shape[1])]
    faceRegionDiffMediansHSV  = [colorTools.bgr_to_hsv(point) for point in faceRegionDiffMedians]

    medianDiffs = {}
    medianDiffs["reflections"] = {}
    medianDiffs["reflections"]["left"] = list(leftEyeDiffReflectionMedian)
    medianDiffs["reflections"]["right"] = list(rightEyeDiffReflectionMedian)

    medianDiffs["regions"] = {}
    medianDiffs["regions"]["left"] = list(faceRegionDiffMedians[0])
    medianDiffs["regions"]["right"] = list(faceRegionDiffMedians[1])
    medianDiffs["regions"]["chin"] = list(faceRegionDiffMedians[2])
    medianDiffs["regions"]["forehead"] = list(faceRegionDiffMedians[3])

    return medianDiffs

def getReflectionColor(reflectionPoints):
    reflectionHSV = colorTools.naiveBGRtoHSV(np.array([reflectionPoints]))[0]
    medianHSV = np.median(reflectionHSV, axis=0)
    hue, sat, val = medianHSV

    proportionalBGR = colorTools.hueSatToProportionalBGR(hue, sat)
    return np.asarray(proportionalBGR)

def extractSkinReflectionMask(brightestCapture, dimmestCapture, wb_ratios):
    brightest = colorTools.convert_sBGR_to_linearBGR_float_fast(brightestCapture.faceImage)
    dimmest = colorTools.convert_sBGR_to_linearBGR_float_fast(dimmestCapture.faceImage)

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

def showGroup(images):
    show = np.hstack(images)
    cv2.imshow('imgs', 20*show)
    cv2.waitKey(0)

def multilayerMedianBlur(alignedImages, size):
    alignedImages = np.asarray(alignedImages)
    if size%2 != 1:
        print('Median Blur must be odd')
        return None

    segment = np.floor(size / 2).astype('uint8')

    imgCount, height, width, chan = alignedImages.shape
    print('Aligned Images shape :: {}'.format(alignedImages.shape))

    output = np.copy(alignedImages[0])
    output[:, :] = [0, 0, 0]

    for h in range(segment, height-segment):
        print('{} / {}'.format(h, height-segment))
        for w in range(segment, width-segment):
            medians = np.median(alignedImages[:, h-segment:h+segment, w-segment:w+segment], axis=[0,1,2])
            output[h+segment, w+segment] = medians

    #cv2.imshow('multi', output)
    #cv2.waitKey(0)
    return output

def synthesis(captures):
    #Just Temp ... figure out robust way to do this?
    scale = 10
    scaleOld = 20
    scaleTest = 7

    images = [capture.faceImage for capture in captures]
    linearImages = np.asarray([colorTools.convert_sBGR_to_linearBGR_float_fast(image) for image in images], dtype='float32')
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
    failOnError = True
    #failOnError = False
    logger.info('BEGINNING COLOR MATCH PROCESSING FOR USER {} CAPTURE {}'.format(user_id, capture_id if capture_id is not None else '-1'))
    state = State(user_id, capture_id, isProduction)
    logger.info('IS PRODUCTION :: {}'.format(isProduction))

    try:
        images = state.loadImages()
    except Exception as err:
        logger.error('User :: {} | Image :: {} | Error :: {} | Details ::\n{}'.format(state.user_id, state.imageName(), 'Error Loading Images', err))
        state.errorProccessing()
        if failOnError: raise
        return getResponse(state.imageName(), False)

    state.saveExposurePointImage('exposurePoints', images)

    metadata = state.getMetadata()

    if not isMetadataValid(metadata):
        logger.error('User :: {} | Image :: {} | Error :: {}'.format(state.user_id, state.imageName(), 'Metadata does not Match'))
        state.errorProccessing()
        if failOnError: raise ValueError('Metadata does not Match')
        return getResponse(state.imageName(), False)

    captures = [Capture(image, meta) for image, meta in zip(images, metadata)]
    sharpness.labelSharpestCaptures(captures)

    try:
        leftEyeCropOffsets, rightEyeCropOffsets, faceLandmarkCropOffsets, faceCropOffsets = alignImages.getCaptureEyeOffsets(captures)
    except Exception as err:
        logger.error('User :: {} | Image :: {} | Error :: {} | Details ::\n{}'.format(state.user_id, state.imageName(), 'Error Cropping and Aligning Images', err))
        state.errorProccessing()
        if failOnError: raise
        return getResponse(state.imageName(), False)

    print('Left \n {} \n Right \n {} \n Face \n {}'.format(leftEyeCropOffsets, rightEyeCropOffsets, faceCropOffsets))
    print('average \n {}'.format((leftEyeCropOffsets + rightEyeCropOffsets) / 2))
    updatedAverageOffset = cropTools.cropCapturesToOffsets(captures, faceCropOffsets)
    #All offsets are relative to capture[0]
    for capture in captures:
        capture.landmarks = captures[0].landmarks

    synth, displaySynth = synthesis(captures)
    #state.saveReferenceImageBGR(displaySynth, captures[0].name + '_synth')

    try:
        averageReflection, averageReflectionArea, leftEyeReflections, rightEyeReflections, leftSclera, rightSclera, blurryMask = getAverageScreenReflectionColor2(captures, leftEyeCropOffsets, rightEyeCropOffsets, state)
    except Exception as err:
        logger.error('User :: {} | Image :: {} | Error :: {} | Details ::\n{}'.format(state.user_id, state.imageName(), 'Error Extracting Reflection', err))
        state.errorProccessing()
        if failOnError: raise
        return getResponse(state.imageName(), False)

    leftEyeProportionalBGR = getReflectionColor(leftEyeReflections)
    rightEyeProportionalBGR = getReflectionColor(rightEyeReflections)

    propBGR = (leftEyeProportionalBGR + rightEyeProportionalBGR) / 2
    print('Average Reflection PROP BGR :: {}'.format(propBGR))

    mask = extractSkinReflectionMask(captures[0], captures[-1], propBGR)

    try:
        faceRegions = np.array([FaceRegions(capture, mask) for capture in captures])
    except Exception as err:
        logger.error('User :: {} | Image :: {} | Error :: {} | Details ::\n{}'.format(state.user_id, state.imageName(), 'Error extracting Points for Recovered Mask', err))
        state.errorProccessing()
        if failOnError: raise
        return getResponse(state.imageName(), False)

    logger.info('Finished Image Processing - Beginning Analysis')
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
    averageSkinSlope = (np.array(linearFitSets["regions"]["left"]) + np.array(linearFitSets["regions"]["right"]) + np.array(linearFitSets["regions"]["chin"]) + np.array(linearFitSets["regions"]["forehead"])) / 4
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

    channelRatio = averageSkinSlopeWB #averageSkinSlope / averageReflectionSlope
    print('\n----\nChannel Ratio :: {}'.format(channelRatio))
    print('B - G Channel Ratio :: {}\n---\n'.format(channelRatio[0] / channelRatio[1]))
    hue = 60 * ((channelRatio[1] - channelRatio[0]) / (channelRatio[2])) % 6
    sat = (max(channelRatio) - min(channelRatio)) / max(channelRatio) 
    val = faceRatio#colorTools.getRelativeLuminance([channelRatio])[0]
    #val = sum(channelRatio) / 3

    #reflectionBestGuess, faceBestGuess = getBestGuess(faceRegions, leftEyeReflections, rightEyeReflections)
    bestGuess = getBestGuess(faceRegions, leftEyeReflections, rightEyeReflections)

    print('Average Sclera Sclope WB :: {}'.format(averageScleraSlopeWB))
    targetSlope = 0.80
    multiplier = targetSlope / np.max(averageScleraSlopeWB)

    synthWB = synth / propBGR

    synthComp = np.hstack([synth * multiplier, synthWB * multiplier])
    synthWB *= multiplier

    maxVal = np.max(synthComp)
    if maxVal > 1.0:
        synthComp /= maxVal

    maxVal = np.max(synthWB)
    if maxVal > 1.0:
        synthWB /= maxVal

    synthCompGamma = colorTools.convert_linearBGR_to_sBGR_float_fast(np.copy(synthComp))
    synthCompGamma = np.clip(np.round(synthCompGamma * 255), 0, 255).astype('uint8')

    synthWBGamma = colorTools.convert_linearBGR_to_sBGR_float_fast(np.copy(synthWB))
    synthWBGamma = np.clip(np.round(synthWBGamma * 255), 0, 255).astype('uint8')

    displaySynth = np.vstack([displaySynth, synthCompGamma])

    state.saveReferenceImageBGR(displaySynth, captures[0].name + '_synthComp')
    state.saveReferenceImageBGR(synthWBGamma, captures[0].name + '_synth')
    #cv2.imshow('synth', synthWBGamma)
    #cv2.imshow('synth Gamma', displaySynth)
    #cv2.waitKey(0)

    logger.info('Done Analysis - Generating Results')

    calibrated_skin_color = [hue, sat, val]#[0.0, 0.0, 0.0]
    matched_skin_color_id = 0
    state.saveCaptureResults(calibrated_skin_color, matched_skin_color_id)

    response = getResponse(state.imageName(), True, captureSets, linearFitSets, bestGuess, averageReflectionArea)
    #print(json.dumps(response))
    logger.info('Done - Returing Results')
    return response
