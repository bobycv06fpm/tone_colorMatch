"""Set of tools to interpret the data into color information"""
import numpy as np
import plotTools
from logger import getLogger

LOGGER = getLogger(__name__, 'app')

def __scoreLinearFit(linearFitObject):
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
    leftEyeScores = __scoreLinearFit(leftEyeLinearFitFull)

    rightEyeLinearFitFull = np.array([plotTools.fitLine(filteredFlashRatios, filteredRightEyeReflections[:, subPixel]) for subPixel in range(0, 3)])

    rightEyeLinearFit = np.vstack(rightEyeLinearFitFull[:, 0])
    rightEyeScores = __scoreLinearFit(rightEyeLinearFitFull)

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
        scores = __scoreLinearFit(linearFitFull)

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
        scaledCaptureFaceRegion = plotTools.getDiffs(captureFaceRegions[3:-1, regionIndex, :])
        scaledCaptureFaceRegions.append(scaledCaptureFaceRegion)

    scaledCaptureFaceRegions = np.vstack(scaledCaptureFaceRegions)

    leftEyeDiffs = plotTools.getDiffs(leftEyeReflections[3:-1])
    rightEyeDiffs = plotTools.getDiffs(rightEyeReflections[3:-1])
    leftEyeDiffs[:, 2][leftEyeDiffs[:, 2] == 0] = 0.001
    rightEyeDiffs[:, 2][rightEyeDiffs[:, 2] == 0] = 0.001
    scaledLeftEyeReflections = leftEyeDiffs
    scaledRightEyeReflections = rightEyeDiffs

    scaledDiffReflections = np.vstack((scaledLeftEyeReflections, scaledRightEyeReflections))

    medianScaledDiffFace = list(np.median(scaledCaptureFaceRegions, axis=0))
    medianScaledDiffReflections = list(np.median(scaledDiffReflections, axis=0))
    return [medianScaledDiffReflections, medianScaledDiffFace]
