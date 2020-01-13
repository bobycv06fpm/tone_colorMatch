"""A set of functions to help with plotting"""
import matplotlib.pyplot as plt
import numpy as np
from logger import getLogger

LOGGER = getLogger(__name__, 'app')

def getDiffs(points):
    """Returns the difference between sequntial points in a list"""
    diffs = []
    for index in range(1, len(points)):
        diffs.append(points[index - 1] - points[index])

    return np.array(diffs)

def fitLine(X, Y):
    """Fits a line between X and Y values, w/ x,y representing a point"""
    X_prepped = np.vstack([X, np.ones(len(X))]).T
    return np.linalg.lstsq(X_prepped, Y, rcond=None)

def __samplePoints(pointsA, pointsB):
    sampleSize = 1000
    if len(pointsA) > sampleSize:
        sample = np.random.choice(len(pointsA), sampleSize)
        return [np.take(pointsA, sample, axis=0), np.take(pointsB, sample, axis=0)]

    return [list(pointsA), list(pointsB)]

def __plotBGR(axs, color, size, x, y, blurryMask, pointRange=None, fit=True):
    x_sample, y_sample = __samplePoints(x, y)

    start_x = 0
    end_x = max(x_sample)

    colorList = np.repeat([list(color)], len(x_sample), axis=0).astype('float32')
    colorList[blurryMask] = [1, 0.4, 0.7] #Highlight the blurry captures w/ pink

    if fit:
        axs.scatter(x_sample, y_sample, size, colorList)
    else:
        axs.plot(x_sample, y_sample, 'ro-')

    x_sample = np.array(x_sample)
    y_sample = np.array(y_sample)

    x_sampleFiltered = x_sample[np.logical_not(blurryMask)]
    y_sampleFiltered = y_sample[np.logical_not(blurryMask)]

    if fit:
        if pointRange is not None:
            m, c = fitLine(x_sampleFiltered[pointRange[0]:pointRange[1]], y_sampleFiltered[pointRange[0]:pointRange[1]])[0]
        else:
            m, c = fitLine(x_sampleFiltered, y_sampleFiltered)[0]

        axs.plot([start_x, end_x], [(m * start_x + c), (m * end_x + c)], color=color)

def plotPerRegionDiffs(faceRegions, leftEyeReflections, rightEyeReflections, state):
    """
    Plots the change between points for each region of the face
        This gives perspective on how the amount of illumination is changing as more light is added, by channel
    """
    captureFaceRegions = np.array([regions.getRegionMeans() for regions in faceRegions])
    flashRatios = np.array([regions.capture.flashRatio for regions in faceRegions])
    numberOfRegions = captureFaceRegions.shape[1]
    numberOfCaptures = captureFaceRegions.shape[0]

    captureFaceRegionsDiffs = []
    for region in range(0, captureFaceRegions.shape[1]):
        diff = getDiffs(captureFaceRegions[:, region, :]) * (numberOfCaptures - 1)
        captureFaceRegionsDiffs.append(diff)

    leftEyeDiffs = getDiffs(leftEyeReflections) * (numberOfCaptures - 1)
    rightEyeDiffs = getDiffs(rightEyeReflections) * (numberOfCaptures - 1)

    captureFaceRegionsDiffs = np.array(captureFaceRegionsDiffs)

    LOGGER.info('PLOTTING: Region Diffs')

    size = 1
    colors = [(1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1)]
    _, axs = plt.subplots(2, 3, sharex=True, sharey=True, tight_layout=True)

    for regionIndex in range(0, numberOfRegions):
        axs[0, 0].plot(flashRatios[1:], captureFaceRegionsDiffs[regionIndex, :, 2], color=colors[regionIndex])
        axs[0, 1].plot(flashRatios[1:], captureFaceRegionsDiffs[regionIndex, :, 1], color=colors[regionIndex])
        axs[0, 2].plot(flashRatios[1:], captureFaceRegionsDiffs[regionIndex, :, 0], color=colors[regionIndex])

    axs[1, 0].plot(flashRatios[1:], rightEyeDiffs[:, 2], color=colors[0])
    axs[1, 0].plot(flashRatios[1:], leftEyeDiffs[:, 2], color=colors[2])

    axs[1, 1].plot(flashRatios[1:], rightEyeDiffs[:, 1], color=colors[0])
    axs[1, 1].plot(flashRatios[1:], leftEyeDiffs[:, 1], color=colors[2])

    axs[1, 2].plot(flashRatios[1:], rightEyeDiffs[:, 0], color=colors[0])
    axs[1, 2].plot(flashRatios[1:], leftEyeDiffs[:, 0], color=colors[2])

    axs[0, 0].set_title('Red')
    axs[0, 1].set_title('Green')
    axs[0, 2].set_title('Blue')

    axs[0, 0].set_xlabel('Screen Flash Ratio')
    axs[0, 0].set_ylabel('Channel Slope Mag')

    axs[1, 0].set_xlabel('Screen Flash Ratio')
    axs[1, 0].set_ylabel('Measured Reflection Slope Mag')
    state.savePlot('RegionDiffs', plt)

def plotPerRegionScaledLinearity(faceRegions, leftEyeReflections, rightEyeReflections, state):
    """
    Plots the Diffs for each region, scaled to the red channel diff
        This gives a little perspective on if the RATIO of RGB is changing as more light is added
    """
    LOGGER.info('PLOTTING: Region Scaled Linearity')
    captureFaceRegions = np.array([regions.getRegionMeans() for regions in faceRegions])
    flashRatios = np.array([regions.capture.flashRatio for regions in faceRegions])
    numberOfRegions = captureFaceRegions.shape[1]

    size = 1
    colors = [(1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1)]

    _, axs = plt.subplots(2, 3, sharex=False, sharey=False, tight_layout=True)

    for regionIndex in range(0, numberOfRegions):
        diff = getDiffs(captureFaceRegions[:, regionIndex, :])
        diff[diff == 0] = 0.0001 #Not so great work around for divide by 0. Still makes the divide by zero stand out on the chart
        scaledCaptureFaceRegion = diff / (np.ones(3) * np.reshape(diff[:, 2], (diff.shape[0], 1))) #Scale to the red channel

        axs[0, 0].plot(flashRatios[1:], scaledCaptureFaceRegion[:, 2], color=colors[regionIndex])
        axs[0, 1].plot(flashRatios[1:], scaledCaptureFaceRegion[:, 1], color=colors[regionIndex])
        axs[0, 2].plot(flashRatios[1:], scaledCaptureFaceRegion[:, 0], color=colors[regionIndex])

    leftEyeDiffs = getDiffs(leftEyeReflections)
    rightEyeDiffs = getDiffs(rightEyeReflections)
    leftEyeDiffs[:, 2][leftEyeDiffs[:, 2] == 0] = 0.001 #Not so great work around for divide by 0. Still makes the divide by zero stand out on the chart
    rightEyeDiffs[:, 2][rightEyeDiffs[:, 2] == 0] = 0.001 #Not so great work around for divide by 0. Still makes the divide by zero stand out on the chart

    scaledLeftEyeReflections = leftEyeDiffs / (np.ones(3) * np.reshape(leftEyeDiffs[:, 2], (leftEyeDiffs.shape[0], 1))) #Scale to the red channel
    scaledRightEyeReflections = rightEyeDiffs / (np.ones(3) * np.reshape(rightEyeDiffs[:, 2], (rightEyeDiffs.shape[0], 1))) #Scale to the red channel

    axs[1, 0].plot(flashRatios[1:], scaledRightEyeReflections[:, 2], color=colors[0])
    axs[1, 0].plot(flashRatios[1:], scaledLeftEyeReflections[:, 2], color=colors[2])

    axs[1, 1].plot(flashRatios[1:], scaledRightEyeReflections[:, 1], color=colors[0])
    axs[1, 1].plot(flashRatios[1:], scaledLeftEyeReflections[:, 1], color=colors[2])

    axs[1, 2].plot(flashRatios[1:], scaledRightEyeReflections[:, 0], color=colors[0])
    axs[1, 2].plot(flashRatios[1:], scaledLeftEyeReflections[:, 0], color=colors[2])

    axs[0, 0].set_title('Red')
    axs[0, 1].set_title('Green')
    axs[0, 2].set_title('Blue')

    axs[0, 0].set_xlabel('Screen Flash Ratio')
    axs[0, 0].set_ylabel('Scaled to Red Channel Mag')

    axs[1, 0].set_xlabel('Screen Flash Ratio')
    axs[1, 0].set_ylabel('Scaled to Red Reflection Mag')
    state.savePlot('ScaledRegionLinearity', plt)

def plotPerRegionLinearity(faceRegions, leftEyeReflections, rightEyeReflections, leftSclera, rightSclera, blurryMask, state):
    """
    Plots each point and fits a line
        This is helpful to check that the brightness is increasing linearly
    """
    LOGGER.info('PLOTTING: Region Linearity')
    #blurryMask = [False for isBlurry in blurryMask]
    captureFaceRegions = np.array([regions.getRegionMeans() for regions in faceRegions])
    flashRatios = np.array([regions.capture.flashRatio for regions in faceRegions])
    numberOfRegions = captureFaceRegions.shape[1]

    size = 1
    colors = [(1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1)]

    _, axs = plt.subplots(3, 3, sharex=False, sharey=False, tight_layout=True)

    for regionIndex in range(0, numberOfRegions):
        __plotBGR(axs[0, 0], colors[regionIndex], size, flashRatios, captureFaceRegions[:, regionIndex, 2], blurryMask, None, False)
        __plotBGR(axs[0, 1], colors[regionIndex], size, flashRatios, captureFaceRegions[:, regionIndex, 1], blurryMask, None, False)
        __plotBGR(axs[0, 2], colors[regionIndex], size, flashRatios, captureFaceRegions[:, regionIndex, 0], blurryMask, None, False)

    # ---- SCLERA -----
    __plotBGR(axs[1, 0], colors[0], 1, flashRatios, rightSclera[:, 2], blurryMask)
    __plotBGR(axs[1, 0], colors[2], 1, flashRatios, leftSclera[:, 2], blurryMask)

    __plotBGR(axs[1, 1], colors[0], 1, flashRatios, rightSclera[:, 1], blurryMask)
    __plotBGR(axs[1, 1], colors[2], 1, flashRatios, leftSclera[:, 1], blurryMask)

    __plotBGR(axs[1, 2], colors[0], 1, flashRatios, rightSclera[:, 0], blurryMask)
    __plotBGR(axs[1, 2], colors[2], 1, flashRatios, leftSclera[:, 0], blurryMask)

    # ---- REFLECTIONS -----
    __plotBGR(axs[2, 0], colors[0], 1, flashRatios, rightEyeReflections[:, 2], blurryMask)
    __plotBGR(axs[2, 0], colors[2], 1, flashRatios, leftEyeReflections[:, 2], blurryMask)

    __plotBGR(axs[2, 1], colors[0], 1, flashRatios, rightEyeReflections[:, 1], blurryMask)
    __plotBGR(axs[2, 1], colors[2], 1, flashRatios, leftEyeReflections[:, 1], blurryMask)
    #__plotBGR(axs[2, 1], colors[3], 1, flashRatios, averageEyeReflections[:, 1])

    __plotBGR(axs[2, 2], colors[0], 1, flashRatios, rightEyeReflections[:, 0], blurryMask)
    __plotBGR(axs[2, 2], colors[2], 1, flashRatios, leftEyeReflections[:, 0], blurryMask)

    axs[0, 0].set_title('Red')
    axs[0, 1].set_title('Green')
    axs[0, 2].set_title('Blue')

    axs[0, 0].set_xlabel('Screen Flash Ratio')
    axs[0, 0].set_ylabel('Channel Mag')

    axs[1, 0].set_ylabel('Sclera Mag')

    axs[2, 0].set_xlabel('Screen Flash Ratio')
    axs[2, 0].set_ylabel('Reflection Mag')
    #plt.show()
    state.savePlot('RegionLinearity', plt)

def plotPerRegionLinearityAlt(faceRegions, leftEyeReflections, rightEyeReflections, blurryMask, state):
    """Plots each face region magnitude against eye reflection magnitude"""
    LOGGER.info('PLOTTING: Region Linearity')
    captureFaceRegions = np.array([regions.getRegionMeans() for regions in faceRegions])

    numberOfRegions = captureFaceRegions.shape[1]

    averageEyeReflections = (leftEyeReflections + rightEyeReflections) / 2
    print('Average Eye Reflections :: {}'.format(averageEyeReflections))

    blurryMask = np.zeros(len(blurryMask)).astype('bool')

    size = 1
    colors = [(1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1)]

    _, axs = plt.subplots(1, 3, sharex=False, sharey=False, tight_layout=True)

    for regionIndex in range(0, numberOfRegions):
        __plotBGR(axs[0], colors[regionIndex], size, averageEyeReflections[:, 2], captureFaceRegions[:, regionIndex, 2], blurryMask)
        __plotBGR(axs[1], colors[regionIndex], size, averageEyeReflections[:, 1], captureFaceRegions[:, regionIndex, 1], blurryMask)
        __plotBGR(axs[2], colors[regionIndex], size, averageEyeReflections[:, 0], captureFaceRegions[:, regionIndex, 0], blurryMask)

    axs[0].set_title('Red')
    axs[1].set_title('Green')
    axs[2].set_title('Blue')

    axs[0].set_xlabel('Eye Reflection Mag')
    axs[0].set_ylabel('Face Mag')

    state.savePlot('RegionLinearityAlt', plt)
