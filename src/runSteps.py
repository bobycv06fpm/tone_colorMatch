from loadImages import loadImages
#from detectFace import detectFace
import alignImages
from getAverageReflection import getAverageScreenReflectionColor2
from saveStep import Save
from getPolygons import getPolygons, getFullFacePolygon
#from extractMask import extractMask
import colorTools
import plotTools
#import processPoints
import cv2
import numpy as np
#import dlib
import thresholdMask
import math
#from scipy import ndimage
import matplotlib.pyplot as plt
import cropTools
import getSharpness
#import landmarkPoints
from capture import Capture 
from faceRegions import FaceRegions

import colorsys

def fitLine(A, B):
    A_prepped = np.vstack([A, np.ones(len(A))]).T
    return np.linalg.lstsq(A_prepped, B, rcond=None)[0]

def samplePoints(pointsA, pointsB):
    sampleSize = 1000
    if len(pointsA) > sampleSize:
        sample = np.random.choice(len(pointsA), sampleSize)
        return [np.take(pointsA, sample, axis=0), np.take(pointsB, sample, axis=0)]

    return [list(pointsA), list(pointsB)]

def plotPerRegionDistribution(faceRegionsSets, saveStep):
    print('PLOTTING: Per Region Distribution')
    faceRegionsSetsLuminance = np.array([faceRegionSet.getRegionLuminance() for faceRegionSet in faceRegionsSets])
    faceRegionsSetsHSV = np.array([faceRegionSet.getRegionHSV() for faceRegionSet in faceRegionsSets])

    numCaptures = len(faceRegionsSets)
    numRegions = len(faceRegionsSets[0].getRegionMedians())

    size = 1
    color = (1, 0, 0)
    fig, axs = plt.subplots(numRegions + 1, 3, sharey=False, tight_layout=True) #Extra region for cumulative region

    #Luminance VS Saturation
    chartRow = 0
    allRegionsX = []
    allRegionsY = []
    for region in range(0, numRegions):
        for capture in range(0, numCaptures):
            xValues = faceRegionsSetsLuminance[capture, region][:]
            yValues = faceRegionsSetsHSV[capture, region][:, 1]

            xValues, yValues = samplePoints(xValues, yValues)

            axs[region, chartRow].scatter(xValues, yValues, size, color)

            allRegionsX.append(xValues)
            allRegionsY.append(yValues)

    axs[numRegions, chartRow].scatter(allRegionsX, allRegionsY, size, color)

    #Luminance VS Hue
    chartRow = 1
    allRegionsX = []
    allRegionsY = []
    for region in range(0, numRegions):
        for capture in range(0, numCaptures):
            xValues = faceRegionsSetsLuminance[capture, region][:]
            yValues = faceRegionsSetsHSV[capture, region][:, 0]

            xValues, yValues = samplePoints(xValues, yValues)
            yValues = colorTools.rotateHue(yValues)

            axs[region, chartRow].scatter(xValues, yValues, size, color)

            allRegionsX.append(xValues)
            allRegionsY.append(yValues)

    axs[numRegions, chartRow].scatter(allRegionsX, allRegionsY, size, color)

    #Hue VS Saturation
    chartRow = 2
    allRegionsX = []
    allRegionsY = []
    for region in range(0, numRegions):
        for capture in range(0, numCaptures):
            xValues = faceRegionsSetsHSV[capture, region][:, 0]
            yValues = faceRegionsSetsHSV[capture, region][:, 1]

            xValues, yValues = samplePoints(xValues, yValues)
            xValues = colorTools.rotateHue(xValues)

            axs[region, chartRow].scatter(xValues, yValues, size, color)

            allRegionsX.append(xValues)
            allRegionsY.append(yValues)

    axs[numRegions, chartRow].scatter(allRegionsX, allRegionsY, size, color)

    saveStep.savePlot('Regions_Scatter', plt)

def plotBGR(axs, color, size, x, y, blurryMask, pointRange=None):
    
    x_sample, y_sample = samplePoints(x, y)

    start_x = 0#min(x_sample)
    end_x = max(x_sample)

    colorList = np.repeat([list(color)], len(x_sample), axis=0).astype('float32')
    colorList[blurryMask] = [1, 0.4, 0.7] #pink... idk why... why not

    axs.scatter(x_sample, y_sample, size, colorList)

    x_sample = np.array(x_sample)
    y_sample = np.array(y_sample)

    x_sampleFiltered = x_sample[np.logical_not(blurryMask)]
    y_sampleFiltered = y_sample[np.logical_not(blurryMask)]

    if pointRange is not None:
        m, c = fitLine(x_sampleFiltered[pointRange[0]:pointRange[1]], y_sampleFiltered[pointRange[0]:pointRange[1]])
    else:
        m, c = fitLine(x_sampleFiltered, y_sampleFiltered)

    axs.plot([start_x, end_x], [(m * start_x + c), (m * end_x + c)], color=color)

def plotPerEyeReflectionBrightness(faceRegions, leftEyeReflections, rightEyeReflections, saveStep):
    print('PLOTTING: Per Eye Reflection Brightness')
    size = 25
    numCaptures = len(leftEyeReflections)
    expectedBrightness = np.array([regions.capture.flashRatio for regions in faceRegions])

    leftEyeReflectionsLuminance = colorTools.getRelativeLuminance(leftEyeReflections)
    rightEyeReflectionsLuminance = colorTools.getRelativeLuminance(rightEyeReflections)

    plt.scatter(expectedBrightness, leftEyeReflectionsLuminance, size, (1, 0, 0))
    m, c = fitLine(expectedBrightness, leftEyeReflectionsLuminance)
    plt.plot([0, 1], [c, (m + c)], color=(1, 0, 0))

    plt.scatter(expectedBrightness, rightEyeReflectionsLuminance, size, (0, 0, 1))
    m, c = fitLine(expectedBrightness, rightEyeReflectionsLuminance)
    plt.plot([0, 1], [c, (m + c)], color=(1, 0, 0))

    plt.title('Measured Reflection Brightness vs Expected Reflection Brightness')
    plt.xlabel('Expected Reflection Brightness')
    plt.ylabel('Measured Reflectoin Brightness')

    saveStep.savePlot('Measured_vs_Expected_Reflection', plt)

def getRegionMapBGR(leftCheek, rightCheek, chin, forehead):
    value = {}
    value['left'] = list(leftCheek)
    value['right'] = list(rightCheek)
    value['chin'] = list(chin)
    value['forehead'] = list(forehead)

    return value

def getReflectionMap(leftReflection, rightReflection):
    value = {}
    value['left'] = [float(value) for value in leftReflection]
    value['right'] = [float(value) for value in rightReflection]

    return value

def getResponse(imageName, successful, captureSets=None, medianDiffSets=None, bestGuess=None, averageReflectionArea=None):
    response = {}
    response['name'] = imageName
    response['successful'] = successful
    response['captures'] = {}
    response['medianDiffs'] = medianDiffSets
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

    #response['noFlashValues'] = getRegionMapBGR(*noFlashValues)
    #response['halfFlashValues'] = getRegionMapBGR(*halfFlashValues)
    #response['fullFlashValues'] = getRegionMapBGR(*fullFlashValues)
    #response['linearity'] = getRegionMapValue(*linearity)
    #response['cleanRatio'] = getRegionMapValue(*cleanRatio)
    #response['reflectionValues'] = getReflectionMap(*reflectionValues)
    #response['fluxishValues'] = getRegionMapValue(*fluxishValues)

    return response

def getNonLinearityMask(flashStepDiff, fullFlashRangeDiff):
    blurDiffs = np.abs(cv2.blur(flashStepDiff, (5, 5)))#.astype('uint8')

    percentError = blurDiffs / fullFlashRangeDiff
    perSubPixelMaxError = np.mean(percentError, axis=2)

    maxFullDiffImage = np.max(fullFlashRangeDiff, axis=2)
    lowValueMask = maxFullDiffImage < (10 / 255)
    medLowValueMask = maxFullDiffImage < (25 / 255)
    medHighValueMask = maxFullDiffImage < (100 / 255)
    medHigherValueMask = maxFullDiffImage < (180 / 255)

    nonLinearMaskHigh = perSubPixelMaxError > .04 #All Values less than 255
    nonLinearMaskMedHigher = perSubPixelMaxError > .06 #All Values less than 180
    nonLinearMaskMedHigh = perSubPixelMaxError > .09 #All Values less than 100
    nonLinearMaskMedLow = perSubPixelMaxError > .12 #All Values less than 25
    nonLinearMaskLow = perSubPixelMaxError > .25 #All Values less than 10
    
    nonLinearMask = nonLinearMaskHigh
    nonLinearMask[medHigherValueMask] = nonLinearMaskMedHigher[medHigherValueMask]
    nonLinearMask[medHighValueMask] = nonLinearMaskMedHigh[medHighValueMask]
    nonLinearMask[medLowValueMask] = nonLinearMaskMedLow[medLowValueMask]
    nonLinearMask[lowValueMask] = nonLinearMaskLow[lowValueMask]

    return nonLinearMask

def getDiffs(points):
    diffs = []
    for index in range(1, len(points)):
        diffs.append(points[index - 1] - points[index])

    return np.array(diffs)

def plotPerRegionDiffs(faceRegions, leftEyeReflections, rightEyeReflections, saveStep):
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

    print('PLOTTING: Region Diffs')

    ##averageEyeReflections = (leftEyeReflections + rightEyeReflections) / 2

    size=1
    colors = [(1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1)]
    fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, tight_layout=True)

    print('Flash Ratio vs Face Region Diff :: ' + str(flashRatios) + ' ' + str(captureFaceRegionsDiffs[:, 0, 0]))

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
    saveStep.savePlot('RegionDiffs', plt)

def getBestGuess(faceRegions, leftEyeReflections, rightEyeReflections):
    print('PLOTTING: Region Scaled Linearity')
    captureFaceRegions = np.array([regions.getRegionMedians() for regions in faceRegions])
    flashRatios = np.array([regions.capture.flashRatio for regions in faceRegions])
    numberOfRegions = captureFaceRegions.shape[1]
    numberOfCaptures = captureFaceRegions.shape[0]

    #averageEyeReflections = (leftEyeReflections + rightEyeReflections) / 2

    scaledCaptureFaceRegions = []

    for regionIndex in range(0, numberOfRegions):
        #print('Regions :: ' + str(captureFaceRegions[:, regionIndex]))
        #diff = getDiffs(captureFaceRegions[1:-1, regionIndex, :])
        diff = getDiffs(captureFaceRegions[3:-1, regionIndex, :])
        #captureFaceRegion = captureFaceRegions[:, regionIndex, :]
        #print('FACE REGION :: ' + str(captureFaceRegion[:, 2]))
        scaledCaptureFaceRegion = diff #/ (np.ones(3) * np.reshape(diff[:, 2], (diff.shape[0], 1)))
        #print('SCALED FACE REGION :: ' + str(scaledCaptureFaceRegion))
        scaledCaptureFaceRegions.append(scaledCaptureFaceRegion)

    scaledCaptureFaceRegions = np.vstack(scaledCaptureFaceRegions)
    print('SCALED DIFFS CAPTURE FACE REGIONS :: ' + str(scaledCaptureFaceRegions))

    leftEyeDiffs = getDiffs(leftEyeReflections[3:-1])
    #leftEyeDiffs = getDiffs(leftEyeReflections[-4:-1])
    rightEyeDiffs = getDiffs(rightEyeReflections[3:-1])
    #rightEyeDiffs = getDiffs(rightEyeReflections[-4:-1])
    leftEyeDiffs[:, 2][leftEyeDiffs[:, 2] == 0] = 0.001
    rightEyeDiffs[:, 2][rightEyeDiffs[:, 2] == 0] = 0.001
    scaledLeftEyeReflections = leftEyeDiffs #/ (np.ones(3) * np.reshape(leftEyeDiffs[:, 2], (leftEyeDiffs.shape[0], 1)))
    scaledRightEyeReflections = rightEyeDiffs #/ (np.ones(3) * np.reshape(rightEyeDiffs[:, 2], (rightEyeDiffs.shape[0], 1)))

    scaledDiffReflections = np.vstack((scaledLeftEyeReflections, scaledRightEyeReflections))
    print('SCALED DIFFS REFLECTIONS :: ' + str(scaledDiffReflections))
    #print('SCALED DIFFS LEFT REFLECTIONS :: ' + str(scaledLeftEyeReflections))
    #print('SCALED DIFFS RIGHT REFLECTIONS:: ' + str(scaledRightEyeReflections))

    medianScaledDiffFace = list(np.median(scaledCaptureFaceRegions, axis=0))
    medianScaledDiffReflections = list(np.median(scaledDiffReflections, axis=0))
    return [medianScaledDiffReflections, medianScaledDiffFace]


    

def plotPerRegionScaledLinearity(faceRegions, leftEyeReflections, rightEyeReflections, saveStep):
    print('PLOTTING: Region Scaled Linearity')
    captureFaceRegions = np.array([regions.getRegionMedians() for regions in faceRegions])
    flashRatios = np.array([regions.capture.flashRatio for regions in faceRegions])
    numberOfRegions = captureFaceRegions.shape[1]
    numberOfCaptures = captureFaceRegions.shape[0]

    size=1
    colors = [(1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1)]

    #fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, tight_layout=True)
    fig, axs = plt.subplots(2, 3, sharex=False, sharey=False, tight_layout=True)

    for regionIndex in range(0, numberOfRegions):
        #print('Regions :: ' + str(captureFaceRegions[:, regionIndex]))
        diff = getDiffs(captureFaceRegions[:, regionIndex, :])
        print("Diffs :: {}".format(diff))
        #captureFaceRegion = captureFaceRegions[:, regionIndex, :]
        #print('FACE REGION :: ' + str(captureFaceRegion[:, 2]))
        diff[diff == 0] = 0.0001 #Kinda shitty work around for divide by 0. Still makes the divide by zero stand out on the chart
        scaledCaptureFaceRegion = diff / (np.ones(3) * np.reshape(diff[:, 2], (diff.shape[0], 1)))
        #print('SCALED FACE REGION :: ' + str(scaledCaptureFaceRegion))

        #plotBGR(axs[0, 0], colors[regionIndex], size, flashRatios, scaledCaptureFaceRegion[:, 2])
        #plotBGR(axs[0, 1], colors[regionIndex], size, flashRatios, scaledCaptureFaceRegion[:, 1])
        #plotBGR(axs[0, 2], colors[regionIndex], size, flashRatios, scaledCaptureFaceRegion[:, 0])
        axs[0, 0].plot(flashRatios[1:], scaledCaptureFaceRegion[:, 2], color=colors[regionIndex])
        axs[0, 1].plot(flashRatios[1:], scaledCaptureFaceRegion[:, 1], color=colors[regionIndex])
        axs[0, 2].plot(flashRatios[1:], scaledCaptureFaceRegion[:, 0], color=colors[regionIndex])


    print('LEFT EYE REFLECTIONS :: ' + str(leftEyeReflections[:, 2]))
    leftEyeDiffs = getDiffs(leftEyeReflections)
    rightEyeDiffs = getDiffs(rightEyeReflections)
    leftEyeDiffs[:, 2][leftEyeDiffs[:, 2] == 0] = 0.001
    rightEyeDiffs[:, 2][rightEyeDiffs[:, 2] == 0] = 0.001
    scaledLeftEyeReflections = leftEyeDiffs / (np.ones(3) * np.reshape(leftEyeDiffs[:, 2], (leftEyeDiffs.shape[0], 1)))
    scaledRightEyeReflections = rightEyeDiffs / (np.ones(3) * np.reshape(rightEyeDiffs[:, 2], (rightEyeDiffs.shape[0], 1)))

    #plotBGR(axs[1, 0], colors[0], 1, flashRatios, scaledRightEyeReflections[:, 2])
    axs[1, 0].plot(flashRatios[1:], scaledRightEyeReflections[:, 2], color=colors[0])
    #plotBGR(axs[1, 0], colors[2], 1, flashRatios, scaledLeftEyeReflections[:, 2])
    axs[1, 0].plot(flashRatios[1:], scaledLeftEyeReflections[:, 2], color=colors[2])

    #plotBGR(axs[1, 1], colors[0], 1, flashRatios, scaledRightEyeReflections[:, 1])
    axs[1, 1].plot(flashRatios[1:], scaledRightEyeReflections[:, 1], color=colors[0])
    #plotBGR(axs[1, 1], colors[2], 1, flashRatios, scaledLeftEyeReflections[:, 1])
    axs[1, 1].plot(flashRatios[1:], scaledLeftEyeReflections[:, 1], color=colors[2])

    #plotBGR(axs[1, 2], colors[0], 1, flashRatios, scaledRightEyeReflections[:, 0])
    axs[1, 2].plot(flashRatios[1:], scaledRightEyeReflections[:, 0], color=colors[0])
    #plotBGR(axs[1, 2], colors[2], 1, flashRatios, scaledLeftEyeReflections[:, 0])
    axs[1, 2].plot(flashRatios[1:], scaledLeftEyeReflections[:, 0], color=colors[2])

    axs[0, 0].set_title('Red')
    axs[0, 1].set_title('Green')
    axs[0, 2].set_title('Blue')

    axs[0, 0].set_xlabel('Screen Flash Ratio')
    axs[0, 0].set_ylabel('Scaled to Red Channel Mag')

    axs[1, 0].set_xlabel('Screen Flash Ratio')
    axs[1, 0].set_ylabel('Scaled to Red Reflection Mag')
    saveStep.savePlot('ScaledRegionLinearity', plt)

def plotPerRegionLinearity(faceRegions, leftEyeReflections, rightEyeReflections, blurryMask, saveStep):
    print('PLOTTING: Region Linearity')
    captureFaceRegions = np.array([regions.getRegionMedians() for regions in faceRegions])
    flashRatios = np.array([regions.capture.flashRatio for regions in faceRegions])
    numberOfRegions = captureFaceRegions.shape[1]
    numberOfCaptures = captureFaceRegions.shape[0]

    #averageEyeReflections = (leftEyeReflections + rightEyeReflections) / 2

    size=1
    colors = [(1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1)]
    isBlurryColor = (0, 0, 0)

    #fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, tight_layout=True)
    fig, axs = plt.subplots(2, 3, sharex=False, sharey=False, tight_layout=True)

    for regionIndex in range(0, numberOfRegions):
        #print('Regions :: ' + str(captureFaceRegions[:, regionIndex]))
        plotBGR(axs[0, 0], colors[regionIndex], size, flashRatios, captureFaceRegions[:, regionIndex, 2], blurryMask)
        plotBGR(axs[0, 1], colors[regionIndex], size, flashRatios, captureFaceRegions[:, regionIndex, 1], blurryMask)
        plotBGR(axs[0, 2], colors[regionIndex], size, flashRatios, captureFaceRegions[:, regionIndex, 0], blurryMask)

    plotBGR(axs[1, 0], colors[0], 1, flashRatios, rightEyeReflections[:, 2], blurryMask)
    plotBGR(axs[1, 0], colors[2], 1, flashRatios, leftEyeReflections[:, 2], blurryMask)
    #plotBGR(axs[1, 0], colors[3], 1, flashRatios, averageEyeReflections[:, 2])

    plotBGR(axs[1, 1], colors[0], 1, flashRatios, rightEyeReflections[:, 1], blurryMask)
    plotBGR(axs[1, 1], colors[2], 1, flashRatios, leftEyeReflections[:, 1], blurryMask)
    #plotBGR(axs[1, 1], colors[3], 1, flashRatios, averageEyeReflections[:, 1])

    plotBGR(axs[1, 2], colors[0], 1, flashRatios, rightEyeReflections[:, 0], blurryMask)
    plotBGR(axs[1, 2], colors[2], 1, flashRatios, leftEyeReflections[:, 0], blurryMask)
    #plotBGR(axs[1, 2], colors[3], 1, flashRatios, averageEyeReflections[:, 0])

    axs[0, 0].set_title('Red')
    axs[0, 1].set_title('Green')
    axs[0, 2].set_title('Blue')

    axs[0, 0].set_xlabel('Screen Flash Ratio')
    axs[0, 0].set_ylabel('Channel Mag')

    axs[1, 0].set_xlabel('Screen Flash Ratio')
    axs[1, 0].set_ylabel('Measured Reflection Mag')
    saveStep.savePlot('RegionLinearity', plt)

#def plotPerRegionPoints(faceRegionsSets, saveStep):
#    print('PLOTTING: Per Region Points')
#    size=1
#    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)]
#
#    numRegions = faceRegionsSets[0].getNumberOfRegions()
#    fig, axs = plt.subplots(3, numRegions, sharex=True, sharey=True, tight_layout=True)
#
#    #Red vs Green
#    for region in range(0, numRegions):
#        for i, faceRegionsSet in enumerate(faceRegionsSets):
#            plotBGR(axs[0, region], colors[i], size, faceRegionsSet.getRegionPoints()[region][:, 2], faceRegionsSet.getRegionPoints()[region][:, 1])
#
#    axs[0, 0].set_title('Red vs Green')
#
#    #Red vs Blue
#    for region in range(0, numRegions):
#        for i, faceRegionsSet in enumerate(faceRegionsSets):
#            plotBGR(axs[1, region], colors[i], size, faceRegionsSet.getRegionPoints()[region][:, 2], faceRegionsSet.getRegionPoints()[region][:, 0])
#
#    axs[1, 0].set_title('Red vs Blue')
#
#    #Green vs Blue
#    for region in range(0, numRegions):
#        for i, faceRegionsSet in enumerate(faceRegionsSets):
#            plotBGR(axs[2, region], colors[i], size, faceRegionsSet.getRegionPoints()[region][:, 1], faceRegionsSet.getRegionPoints()[region][:, 0])
#
#    axs[2, 0].set_title('Green vs Blue')
#
#    saveStep.savePlot('BGR_Scatter', plt)

def isMetadataValid(metadata):
    expectedISO = metadata[0]["iso"]
    expectedExposure = metadata[0]["exposureTime"]
    expectedWB = metadata[0]["whiteBalance"]

    for captureMetadata in metadata:
        iso = captureMetadata["iso"]
        exposure = captureMetadata["exposureTime"]
        wb = captureMetadata["whiteBalance"]

        if (iso != expectedISO) or (exposure != expectedExposure) or (wb['x'] != expectedWB['x']) or (wb['y'] != expectedWB['y']):
            return False
        
    return True

def getMedianDiff(points):
    diffs = []
    for index in range(1, len(points)):
        diffs.append(np.abs(points[index - 1] - points[index]))

    return np.median(np.array(diffs), axis=0)
    #return np.median(np.array(diffs)[-6:-2], axis=0)
    #return np.mean(np.array(diffs)[-6:-2], axis=0)

def getLinearFits(leftEyeReflections, rightEyeReflections, faceRegions, blurryMask):
    #start = 2
    #end = -2


    flashRatios = np.array([regions.capture.flashRatio for regions in faceRegions])

    filteredFlashRatios = flashRatios[np.logical_not(blurryMask)]
    filteredLeftEyeReflections = leftEyeReflections[np.logical_not(blurryMask)]
    filteredRightEyeReflections = rightEyeReflections[np.logical_not(blurryMask)]

    leftEyeLinearFit = np.array([fitLine(filteredFlashRatios, filteredLeftEyeReflections[:, subPixel])[0] for subPixel in range(0, 3)])
    #print('Left Eye Linear Fit :: ' + str(leftEyeLinearFit))
    #leftEyeLinearFit /= (np.ones(3) * leftEyeLinearFit[2])
    #print('Left Eye Linear Fit Scaled :: ' + str(leftEyeLinearFit))

    rightEyeLinearFit = np.array([fitLine(filteredFlashRatios, filteredRightEyeReflections[:, subPixel])[0] for subPixel in range(0, 3)])
    #print('Right Eye Linear Fit :: ' + str(rightEyeLinearFit))
    #rightEyeLinearFit /= (np.ones(3) * rightEyeLinearFit[2])
    #print('Right Eye Linear Fit Scaled :: ' + str(rightEyeLinearFit))


    #for faceRegion in faceRegions:
    #    print('Pixels per region {}'.format(faceRegion.getNumPixelsPerRegion()))

    captureFaceRegions = np.array([regions.getRegionMedians() for regions in faceRegions])
    filteredCaptureFaceRegions = captureFaceRegions[np.logical_not(blurryMask)]
    captureFaceRegionsLinearFit = []

    for regionIndex in range(0, captureFaceRegions.shape[1]):
        #linearFit = [fitLine(flashRatios[start:end], captureFaceRegions[start:end, regionIndex, subPixel])[0] for subPixel in range(0, 3)]
        linearFit = [fitLine(filteredFlashRatios, filteredCaptureFaceRegions[:, regionIndex, subPixel])[0] for subPixel in range(0, 3)]
        #linearFit /= (np.ones(3) * linearFit[2])
        captureFaceRegionsLinearFit.append(linearFit)


    print('Left Eye Linear Fit :: ' + str(leftEyeLinearFit))
    print('Right Eye Linear Fit :: ' + str(rightEyeLinearFit))
    print('Face Regions Linear Fit :: ' + str(captureFaceRegionsLinearFit))

    #leftEyeLinearFitHSV = colorTools.bgr_to_hsv(leftEyeLinearFit)
    #rightEyeLinearFitHSV = colorTools.bgr_to_hsv(rightEyeLinearFit)
    #captureFaceRegionsLinearFitHSV  = [colorTools.bgr_to_hsv(point) for point in  captureFaceRegionsLinearFit]

    linearFits = {}
    linearFits["reflections"] = {}
    linearFits["reflections"]["left"] = list(leftEyeLinearFit)
    linearFits["reflections"]["right"] = list(rightEyeLinearFit)

    linearFits["regions"] = {}
    linearFits["regions"]["left"] = list(captureFaceRegionsLinearFit[0])
    linearFits["regions"]["right"] = list(captureFaceRegionsLinearFit[1])
    linearFits["regions"]["chin"] = list(captureFaceRegionsLinearFit[2])
    linearFits["regions"]["forehead"] = list(captureFaceRegionsLinearFit[3])

    formatString = '\nLINEAR FITS :: {}\n\tEYES \n\t\tLEFT \t\t{}\n\t\tRIGHT \t\t{}\n\tFACE\n\t\tLEFT \t\t{}\n\t\tRIGHT \t\t{}\n\t\tCHIN \t\t{}\n\t\tFOREHEAD \t{}\n'
    formatted = formatString.format('BGR', leftEyeLinearFit, rightEyeLinearFit, *captureFaceRegionsLinearFit)
    print(formatted)

    #formattedHSV = formatString.format('HSV', leftEyeLinearFitHSV, rightEyeLinearFitHSV, *captureFaceRegionsLinearFitHSV)
    #print(formattedHSV)

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

    formatString = '\nMEDIAN DIFFS :: {}\n\tEYES \n\t\tLEFT \t\t{}\n\t\tRIGHT \t\t{}\n\tFACE\n\t\tLEFT \t\t{}\n\t\tRIGHT \t\t{}\n\t\tCHIN \t\t{}\n\t\tFOREHEAD \t{}\n'
    formatted = formatString.format('BGR', leftEyeDiffReflectionMedian, rightEyeDiffReflectionMedian, *faceRegionDiffMedians)
    formattedHSV = formatString.format('HSV', leftEyeDiffReflectionMedianHSV, rightEyeDiffReflectionMedianHSV, *faceRegionDiffMediansHSV)
    print(formatted)
    print(formattedHSV)

    return medianDiffs

def run(username, imageName, fast=False, saveStats=False, failOnError=False):
    saveStep = Save(username, imageName)
    saveStep.resetLogFile()
    saveStep.deleteReference()
    images = loadImages(username, imageName)
    print('shape :: ' + str(images.shape))

    metadata = saveStep.getMetadata()

    if not isMetadataValid(metadata):
        print('User :: {} | Image :: {} | Error :: {}'.format(username, imageName, 'Metadata does not Match'))
        return getResponse(imageName, False)

    #numImages = len(images)
    captures = [Capture(image, meta) for image, meta in zip(images, metadata)]
    #captures[0].showImageWithLandmarks()
    #Brightest is index 0, dimmest is last

    getSharpness.labelSharpestCaptures(captures)

    blurryMask = [capture.isBlurry for capture in captures]
    print('BLURRY MASK :: {}'.format(blurryMask))

    try:
        leftEyeCropOffsets, rightEyeCropOffsets, faceCropOffsets = alignImages.getCaptureEyeOffsets2(captures)
    except Exception as err:
        if failOnError:
            raise NameError('User :: {} | Image :: {} | Error :: {} | Details :: {}'.format(username, imageName, 'Error Cropping and Aligning Images', err))
        else:
            print('User :: {} | Image :: {} | Error :: {} | Details :: {}'.format(username, imageName, 'Error Cropping and Aligning Images', err))
            return getResponse(imageName, False)

    #leftEyeOffsets -= averageOffsets #Need offsets relative to averageOffset now that we are aligned
    #rightEyeOffsets -= averageOffsets #Need offsets relative to averageOffset

    updatedAverageOffset = cropTools.cropCapturesToOffsets(captures, faceCropOffsets)
    #All offsets are relative to capture[0]
    for capture in captures:
        capture.landmarks = captures[0].landmarks

    try:
        averageReflection, averageReflectionArea, leftEyeReflections, rightEyeReflections = getAverageScreenReflectionColor2(captures, leftEyeCropOffsets, rightEyeCropOffsets, saveStep)
    except Exception as err:
        if failOnError:
            raise NameError('User :: {} | Image :: {} | Error :: {} | Details :: {}'.format(username, imageName, 'Error Extracting Reflection', err))
        else:
            print('User :: {} | Image :: {} | Error :: {} | Details :: {}'.format(username, imageName, 'Error Extracting Reflection', err))
            return getResponse(imageName, False)

    try:
        faceRegions = np.array([FaceRegions(capture) for capture in captures])
    except Exception as err:
        if failOnError:
            raise NameError('User :: {} | Image :: {} | Error :: {} | Details :: {}'.format(username, imageName, 'Error extracting Points for Recovered Mask', err))
        else:
            print('User :: {} | Image :: {} | Error :: {} | Details :: {}'.format(username, imageName, 'Error extracting Points for Recovered Mask', err))
            return getResponse(imageName, False)

    saveStep.saveReferenceImageBGR(faceRegions[0].getMaskedImage(), faceRegions[0].capture.name + '_masked')

    plotPerRegionLinearity(faceRegions, leftEyeReflections, rightEyeReflections, blurryMask, saveStep)
    plotPerRegionScaledLinearity(faceRegions, leftEyeReflections, rightEyeReflections, saveStep)
    #plotPerEyeReflectionBrightness(faceRegions, leftEyeReflections, rightEyeReflections, saveStep)
    plotPerRegionDiffs(faceRegions, leftEyeReflections, rightEyeReflections, saveStep)

    #plotPerRegionPoints(faceRegions, saveStep)
    #plotPerRegionDistribution(faceRegions, saveStep)

    captureSets = zip(faceRegions, leftEyeReflections, rightEyeReflections)
    #medianDiffSets = getMedianDiffs(leftEyeReflections, rightEyeReflections, faceRegions)
    #print('Median Diff Sets :: ' + str(medianDiffSets))
    linearFitSets = getLinearFits(leftEyeReflections, rightEyeReflections, faceRegions, blurryMask)
    print('Linear Fits :: ' + str(linearFitSets))

    #reflectionBestGuess, faceBestGuess = getBestGuess(faceRegions, leftEyeReflections, rightEyeReflections)
    bestGuess = getBestGuess(faceRegions, leftEyeReflections, rightEyeReflections)

    #x = metadata[0]["whiteBalance"]["x"]
    #y = metadata[0]["whiteBalance"]["y"]
    #asShotBGR = colorTools.convert_CIE_xy_to_unscaledBGR(x, y)
    #targetBGR = colorTools.convert_CIE_xy_to_unscaledBGR(0.31271, 0.32902) #Illuminant D65
    ##bgrMultiplier = asShotBGR / targetBGR / asShotBGR
    #bgrMultiplier = targetBGR / asShotBGR
    #scaledBGR = bgrMultiplier / bgrMultiplier[2]
    #print('WB BGR vs Reflection :: ' + str(scaledBGR) + ' ' + str(bestGuess[0]))
    #print('BEST GUESS -> REFLECTION :: {} | FACE :: {}'.format(bestGuess[0], bestGuess[1]))
    #bestGuess[0] = list(scaledBGR)

    response = getResponse(imageName, True, captureSets, linearFitSets, bestGuess, averageReflectionArea)
    return response

