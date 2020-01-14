"""
Skin tone measuring image processing pipeline
"""
import numpy as np
from getAverageReflection import getAverageScreenReflectionColor
from state import State
import plotTools
import cropTools
import imageTools
import dataInterpretation
from dataFormatTools import getFailureResponse, getSuccessfulResponse
from capture import Capture
from faceRegions import FaceRegions
from logger import getLogger

LOGGER = getLogger(__name__, 'app')

def run(user_id, capture_id=None, isProduction=False):
    """Run the color measuring pipeline"""
    failOnError = True

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

    #Helpful for understanding why an image was exposed the way it was
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

    _, displaySynth = imageTools.synthesis(captures)
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

    leftEyeProportionalBGR = imageTools.getReflectionColor(leftEyeReflections)
    rightEyeProportionalBGR = imageTools.getReflectionColor(rightEyeReflections)

    propBGR = (leftEyeProportionalBGR + rightEyeProportionalBGR) / 2

    mask = imageTools.extractSkinReflectionMask(captures[0], captures[-1], propBGR)

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

    linearFitSets = dataInterpretation.getLinearFits(leftEyeReflections, rightEyeReflections, leftSclera, rightSclera, faceRegions, blurryMask)

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

    bestGuess = dataInterpretation.getBestGuess(faceRegions, leftEyeReflections, rightEyeReflections)

    LOGGER.info('Done Analysis - Generating Results')

    calibrated_skin_color = [hue, sat, val]
    matched_skin_color_id = 0
    state.saveCaptureResults(calibrated_skin_color, matched_skin_color_id)

    response = getSuccessfulResponse(state.imageName(), captureSets, linearFitSets, bestGuess, averageReflectionArea)

    LOGGER.info('Done - Returing Results')
    return response
