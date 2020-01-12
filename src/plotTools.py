import matplotlib.pyplot as plt
import numpy as np
from logger import getLogger

logger = getLogger(__name__, 'app')

def getDiffs(points):
    diffs = []
    for index in range(1, len(points)):
        diffs.append(points[index - 1] - points[index])

    return np.array(diffs)


def plotPoints(pixels, markers=[[]]):
    step = 100
    print("Plotting RGB Points")
    print("Printing " + str(len(pixels)) + " pixels in steps of " + str(step))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')

    scaled = []
    unscaled_x = []
    unscaled_y = []
    unscaled_z = []

    #for pixel in pixels:
    for index in range(0, len(pixels), step):
        pixel = pixels[index]
        scaled.append((pixel[2]/255, pixel[1]/255, pixel[0]/255))
        unscaled_x.append(pixel[2])
        unscaled_y.append(pixel[1])
        unscaled_z.append(pixel[0])

    for index, marker in enumerate(markers):
        for point in marker:
            if index != 0:
                scaled.append((0, 0, 1))
            else:
                scaled.append((0, 1, 0))
            unscaled_x.append(point[2])
            unscaled_y.append(point[1])
            unscaled_z.append(point[0])


    ax.scatter(unscaled_x, unscaled_y, unscaled_z, 'z', 20, scaled, True)
    plt.show()

def plotHSV(hsvValues, rgb):
    hue = []
    sat = []
    val = []
    scaled = []
    step = 100

    print("Plotting Points")
    print("Printing " + str(len(hsvValues)) + " pixels in steps of " + str(step))

    for i in range(0, len(hsvValues), step):
        hsv = hsvValues[i]
        (h, s, v) = hsv
        hue.append(h)
        sat.append(s)
        val.append(v)

        scaled.append([rgb[i][0], rgb[i][1], rgb[i][2]])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Hue')
    ax.set_ylabel('Saturation')
    ax.set_zlabel('Value')

    base_size = 20

    ax.scatter(hue, sat, val, 'z', base_size, scaled, False)
    plt.show()

def hsvMultiplot(sets):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Hue')
    ax.set_ylabel('Saturation')
    ax.set_zlabel('Value')

    for series in sets:
        ((h, s, v), scaled, size) = series
        ax.scatter(h, s, v, 'z', size, scaled, False)

    plt.show()

def scaleRGBValues(rgb_values):
    rgb_scaled = []
    for rgb in rgb_values:
        (r, g, b) = rgb
        r = r / 255
        g = g / 255
        b = b / 255

        rgb_scaled.append((r, g, b))

    return rgb_scaled

def fitLine(A, B):
    A_prepped = np.vstack([A, np.ones(len(A))]).T
    return np.linalg.lstsq(A_prepped, B, rcond=None)

#def fitLine2(A, B):
#    return np.polyfit(A, B, 1)
#    #A_prepped = np.vstack([A, np.ones(len(A))]).T
#    #return np.linalg.lstsq(A_prepped, B, rcond=None)

def samplePoints(pointsA, pointsB):
    sampleSize = 1000
    if len(pointsA) > sampleSize:
        sample = np.random.choice(len(pointsA), sampleSize)
        return [np.take(pointsA, sample, axis=0), np.take(pointsB, sample, axis=0)]

    return [list(pointsA), list(pointsB)]

#def plotPerRegionDistribution(faceRegionsSets, state):
#    logger.info('PLOTTING: Per Region Distribution')
#    faceRegionsSetsLuminance = np.array([faceRegionSet.getRegionLuminance() for faceRegionSet in faceRegionsSets])
#    faceRegionsSetsHSV = np.array([faceRegionSet.getRegionHSV() for faceRegionSet in faceRegionsSets])
#
#    numCaptures = len(faceRegionsSets)
#    numRegions = len(faceRegionsSets[0].getRegionMedians())
#
#    size = 1
#    color = (1, 0, 0)
#    fig, axs = plt.subplots(numRegions + 1, 3, sharey=False, tight_layout=True) #Extra region for cumulative region
#
#    #Luminance VS Saturation
#    chartRow = 0
#    allRegionsX = []
#    allRegionsY = []
#    for region in range(0, numRegions):
#        for capture in range(0, numCaptures):
#            xValues = faceRegionsSetsLuminance[capture, region][:]
#            yValues = faceRegionsSetsHSV[capture, region][:, 1]
#
#            xValues, yValues = samplePoints(xValues, yValues)
#
#            axs[region, chartRow].scatter(xValues, yValues, size, color)
#
#            allRegionsX.append(xValues)
#            allRegionsY.append(yValues)
#
#    axs[numRegions, chartRow].scatter(allRegionsX, allRegionsY, size, color)
#
#    #Luminance VS Hue
#    chartRow = 1
#    allRegionsX = []
#    allRegionsY = []
#    for region in range(0, numRegions):
#        for capture in range(0, numCaptures):
#            xValues = faceRegionsSetsLuminance[capture, region][:]
#            yValues = faceRegionsSetsHSV[capture, region][:, 0]
#
#            xValues, yValues = samplePoints(xValues, yValues)
#            yValues = colorTools.rotateHue(yValues)
#
#            axs[region, chartRow].scatter(xValues, yValues, size, color)
#
#            allRegionsX.append(xValues)
#            allRegionsY.append(yValues)
#
#    axs[numRegions, chartRow].scatter(allRegionsX, allRegionsY, size, color)
#
#    #Hue VS Saturation
#    chartRow = 2
#    allRegionsX = []
#    allRegionsY = []
#    for region in range(0, numRegions):
#        for capture in range(0, numCaptures):
#            xValues = faceRegionsSetsHSV[capture, region][:, 0]
#            yValues = faceRegionsSetsHSV[capture, region][:, 1]
#
#            xValues, yValues = samplePoints(xValues, yValues)
#            xValues = colorTools.rotateHue(xValues)
#
#            axs[region, chartRow].scatter(xValues, yValues, size, color)
#
#            allRegionsX.append(xValues)
#            allRegionsY.append(yValues)
#
#    axs[numRegions, chartRow].scatter(allRegionsX, allRegionsY, size, color)
#
#    state.savePlot('Regions_Scatter', plt)

def plotBGR(axs, color, size, x, y, blurryMask, pointRange=None, fit=True):
    
    x_sample, y_sample = samplePoints(x, y)

    start_x = 0#min(x_sample)
    end_x = max(x_sample)

    colorList = np.repeat([list(color)], len(x_sample), axis=0).astype('float32')
    colorList[blurryMask] = [1, 0.4, 0.7] #pink... idk why... why not

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
    captureFaceRegions = np.array([regions.getRegionMedians() for regions in faceRegions])
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

    logger.info('PLOTTING: Region Diffs')

    ##averageEyeReflections = (leftEyeReflections + rightEyeReflections) / 2

    size=1
    colors = [(1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1)]
    fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, tight_layout=True)

    #logger.info('Flash Ratio vs Face Region Diff :: ' + str(flashRatios) + ' ' + str(captureFaceRegionsDiffs[:, 0, 0]))

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
    logger.info('PLOTTING: Region Scaled Linearity')
    captureFaceRegions = np.array([regions.getRegionMedians() for regions in faceRegions])
    flashRatios = np.array([regions.capture.flashRatio for regions in faceRegions])
    numberOfRegions = captureFaceRegions.shape[1]
    numberOfCaptures = captureFaceRegions.shape[0]

    size=1
    colors = [(1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1)]

    #fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, tight_layout=True)
    fig, axs = plt.subplots(2, 3, sharex=False, sharey=False, tight_layout=True)

    for regionIndex in range(0, numberOfRegions):
        diff = getDiffs(captureFaceRegions[:, regionIndex, :])
        diff[diff == 0] = 0.0001 #Kinda shitty work around for divide by 0. Still makes the divide by zero stand out on the chart
        scaledCaptureFaceRegion = diff / (np.ones(3) * np.reshape(diff[:, 2], (diff.shape[0], 1)))

        axs[0, 0].plot(flashRatios[1:], scaledCaptureFaceRegion[:, 2], color=colors[regionIndex])
        axs[0, 1].plot(flashRatios[1:], scaledCaptureFaceRegion[:, 1], color=colors[regionIndex])
        axs[0, 2].plot(flashRatios[1:], scaledCaptureFaceRegion[:, 0], color=colors[regionIndex])


    #logger.info('LEFT EYE REFLECTIONS :: ' + str(leftEyeReflections[:, 2]))
    leftEyeDiffs = getDiffs(leftEyeReflections)
    rightEyeDiffs = getDiffs(rightEyeReflections)
    leftEyeDiffs[:, 2][leftEyeDiffs[:, 2] == 0] = 0.001
    rightEyeDiffs[:, 2][rightEyeDiffs[:, 2] == 0] = 0.001
    scaledLeftEyeReflections = leftEyeDiffs / (np.ones(3) * np.reshape(leftEyeDiffs[:, 2], (leftEyeDiffs.shape[0], 1)))
    scaledRightEyeReflections = rightEyeDiffs / (np.ones(3) * np.reshape(rightEyeDiffs[:, 2], (rightEyeDiffs.shape[0], 1)))

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
    logger.info('PLOTTING: Region Linearity')
    #blurryMask = [False for isBlurry in blurryMask]
    captureFaceRegions = np.array([regions.getRegionMedians() for regions in faceRegions])
    flashRatios = np.array([regions.capture.flashRatio for regions in faceRegions])
    numberOfRegions = captureFaceRegions.shape[1]
    numberOfCaptures = captureFaceRegions.shape[0]

    #averageEyeReflections = (leftEyeReflections + rightEyeReflections) / 2

    size=1
    colors = [(1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1)]
    isBlurryColor = (0, 0, 0)

    #fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, tight_layout=True)
    fig, axs = plt.subplots(3, 3, sharex=False, sharey=False, tight_layout=True)

    for regionIndex in range(0, numberOfRegions):
        plotBGR(axs[0, 0], colors[regionIndex], size, flashRatios, captureFaceRegions[:, regionIndex, 2], blurryMask, None, False)
        plotBGR(axs[0, 1], colors[regionIndex], size, flashRatios, captureFaceRegions[:, regionIndex, 1], blurryMask, None, False)
        plotBGR(axs[0, 2], colors[regionIndex], size, flashRatios, captureFaceRegions[:, regionIndex, 0], blurryMask, None, False)

    # ---- SCLERA -----
    plotBGR(axs[1, 0], colors[0], 1, flashRatios, rightSclera[:, 2], blurryMask)
    plotBGR(axs[1, 0], colors[2], 1, flashRatios, leftSclera[:, 2], blurryMask)

    plotBGR(axs[1, 1], colors[0], 1, flashRatios, rightSclera[:, 1], blurryMask)
    plotBGR(axs[1, 1], colors[2], 1, flashRatios, leftSclera[:, 1], blurryMask)

    plotBGR(axs[1, 2], colors[0], 1, flashRatios, rightSclera[:, 0], blurryMask)
    plotBGR(axs[1, 2], colors[2], 1, flashRatios, leftSclera[:, 0], blurryMask)

    # ---- REFLECTIONS -----
    plotBGR(axs[2, 0], colors[0], 1, flashRatios, rightEyeReflections[:, 2], blurryMask)
    plotBGR(axs[2, 0], colors[2], 1, flashRatios, leftEyeReflections[:, 2], blurryMask)
    #plotBGR(axs[2, 0], colors[3], 1, flashRatios, averageEyeReflections[:, 2])

    plotBGR(axs[2, 1], colors[0], 1, flashRatios, rightEyeReflections[:, 1], blurryMask)
    plotBGR(axs[2, 1], colors[2], 1, flashRatios, leftEyeReflections[:, 1], blurryMask)
    #plotBGR(axs[2, 1], colors[3], 1, flashRatios, averageEyeReflections[:, 1])

    plotBGR(axs[2, 2], colors[0], 1, flashRatios, rightEyeReflections[:, 0], blurryMask)
    plotBGR(axs[2, 2], colors[2], 1, flashRatios, leftEyeReflections[:, 0], blurryMask)
    #plotBGR(axs[2, 2], colors[3], 1, flashRatios, averageEyeReflections[:, 0])

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
    logger.info('PLOTTING: Region Linearity')
    captureFaceRegions = np.array([regions.getRegionMedians() for regions in faceRegions])
    flashRatios = np.array([regions.capture.flashRatio for regions in faceRegions])

    numberOfRegions = captureFaceRegions.shape[1]

    averageEyeReflections = (leftEyeReflections + rightEyeReflections) / 2
    print('Average Eye Reflections :: {}'.format(averageEyeReflections))

    blurryMask = np.zeros(len(blurryMask)).astype('bool')

    size=1
    colors = [(1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1)]
    isBlurryColor = (0, 0, 0)

    #fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, tight_layout=True)
    fig, axs = plt.subplots(1, 3, sharex=False, sharey=False, tight_layout=True)

    for regionIndex in range(0, numberOfRegions):
        plotBGR(axs[0], colors[regionIndex], size, averageEyeReflections[:, 2], captureFaceRegions[:, regionIndex, 2], blurryMask)
        plotBGR(axs[1], colors[regionIndex], size, averageEyeReflections[:, 1], captureFaceRegions[:, regionIndex, 1], blurryMask)
        plotBGR(axs[2], colors[regionIndex], size, averageEyeReflections[:, 0], captureFaceRegions[:, regionIndex, 0], blurryMask)

    axs[0].set_title('Red')
    axs[1].set_title('Green')
    axs[2].set_title('Blue')

    axs[0].set_xlabel('Screen Flash Ratio')
    axs[0].set_ylabel('Channel Mag')

    state.savePlot('RegionLinearityAlt', plt)
