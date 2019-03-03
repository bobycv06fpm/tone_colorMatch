from loadImages import loadImages
#from detectFace import detectFace
import alignImages
from getAverageReflection import getAverageScreenReflectionColor
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
#import landmarkPoints
from capture import Capture 
from faceRegions import FaceRegions

import multiprocessing as mp
import colorsys

def fitLine(A, B):
    A_prepped = np.vstack([A, np.ones(len(A))]).T
    return np.linalg.lstsq(A_prepped, B, rcond=None)[0]

def rotateHue(hue):
    hue = hue.copy()
    shiftMask = hue <= 2/3
    hue[shiftMask] += 1/3
    hue[np.logical_not(shiftMask)] -= 2/3
    return hue

def unRotateHue(hue):
    hue = hue.copy()
    shiftMask = hue >= 1/3
    hue[shiftMask] -= 1/3
    hue[np.logical_not(shiftMask)] += 2/3
    return hue

def samplePoints(pointsA, pointsB):
    sampleSize = 1000
    if len(pointsA) > sampleSize:
        sample = np.random.choice(len(pointsA), sampleSize)
        return [np.take(pointsA, sample, axis=0), np.take(pointsB, sample, axis=0)]

    return [list(pointsA), list(pointsB)]

def plotPerRegionDistribution(faceRegionsSets, saveStep):
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
            yValues = rotateHue(yValues)

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
            xValues = rotateHue(xValues)

            axs[region, chartRow].scatter(xValues, yValues, size, color)

            allRegionsX.append(xValues)
            allRegionsY.append(yValues)

    axs[numRegions, chartRow].scatter(allRegionsX, allRegionsY, size, color)

    saveStep.savePlot('Regions_Scatter', plt)

def plotBGR(axs, color, size, x, y):
    
    x_sample, y_sample = samplePoints(x, y)

    start_x = 0#min(x_sample)
    end_x = max(x_sample)

    axs.scatter(x_sample, y_sample, size, [list(color)])

    m, c = fitLine(x_sample, y_sample)
    axs.plot([start_x, end_x], [(m * start_x + c), (m * end_x + c)], color=color)

def plotPerEyeReflectionBrightness(faceRegions, leftEyeReflections, rightEyeReflections, saveStep):
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

def getResponse(imageName, successful, captureSets=None):
    response = {}
    response['name'] = imageName
    response['successful'] = successful
    response['captures'] = {}

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
    lowValueMask = maxFullDiffImage < 10
    medLowValueMask = maxFullDiffImage < 25
    medHighValueMask = maxFullDiffImage < 100
    medHigherValueMask = maxFullDiffImage < 180

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


def plotPerRegionLinearity(faceRegions, leftEyeReflections, rightEyeReflections, saveStep):
    captureFaceRegions = np.array([regions.getRegionMedians() for regions in faceRegions])
    flashRatios = np.array([regions.capture.flashRatio for regions in faceRegions])
    numberOfRegions = captureFaceRegions.shape[1]
    numberOfCaptures = captureFaceRegions.shape[0]

    size=25
    colors = [(1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1)]

    fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, tight_layout=True)

    for regionIndex in range(0, numberOfRegions):
        print('Regions :: ' + str(captureFaceRegions[:, regionIndex]))
        plotBGR(axs[0, 0], colors[regionIndex], size, flashRatios, captureFaceRegions[:, regionIndex, 2])
        plotBGR(axs[0, 1], colors[regionIndex], size, flashRatios, captureFaceRegions[:, regionIndex, 1])
        plotBGR(axs[0, 2], colors[regionIndex], size, flashRatios, captureFaceRegions[:, regionIndex, 0])

    plotBGR(axs[1, 0], colors[0], size, flashRatios, rightEyeReflections[:, 2])
    plotBGR(axs[1, 0], colors[2], size, flashRatios, leftEyeReflections[:, 2])

    plotBGR(axs[1, 1], colors[0], size, flashRatios, rightEyeReflections[:, 1])
    plotBGR(axs[1, 1], colors[2], size, flashRatios, leftEyeReflections[:, 1])

    plotBGR(axs[1, 2], colors[0], size, flashRatios, rightEyeReflections[:, 0])
    plotBGR(axs[1, 2], colors[2], size, flashRatios, leftEyeReflections[:, 0])

    axs[0, 0].set_title('Red')
    axs[0, 1].set_title('Green')
    axs[0, 2].set_title('Blue')

    axs[0, 0].set_xlabel('Screen Flash Ratio')
    axs[0, 0].set_ylabel('Channel Mag')

    axs[1, 0].set_xlabel('Screen Flash Ratio')
    axs[1, 0].set_ylabel('Measured Reflection Mag')
    saveStep.savePlot('RegionLinearity', plt)

def plotPerRegionPoints(faceRegionsSets, saveStep):
    size=1
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)]

    numRegions = faceRegionsSets[0].getNumberOfRegions()
    fig, axs = plt.subplots(3, numRegions, sharex=True, sharey=True, tight_layout=True)

    #Red vs Green
    for region in range(0, numRegions):
        for i, faceRegionsSet in enumerate(faceRegionsSets):
            plotBGR(axs[0, region], colors[i], size, faceRegionsSet.getRegionPoints()[region][:, 2], faceRegionsSet.getRegionPoints()[region][:, 1])

    axs[0, 0].set_title('Red vs Green')

    #Red vs Blue
    for region in range(0, numRegions):
        for i, faceRegionsSet in enumerate(faceRegionsSets):
            plotBGR(axs[1, region], colors[i], size, faceRegionsSet.getRegionPoints()[region][:, 2], faceRegionsSet.getRegionPoints()[region][:, 0])

    axs[1, 0].set_title('Red vs Blue')

    #Green vs Blue
    for region in range(0, numRegions):
        for i, faceRegionsSet in enumerate(faceRegionsSets):
            plotBGR(axs[2, region], colors[i], size, faceRegionsSet.getRegionPoints()[region][:, 1], faceRegionsSet.getRegionPoints()[region][:, 0])

    axs[2, 0].set_title('Green vs Blue')

    saveStep.savePlot('BGR_Scatter', plt)

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



def run(username, imageName, fast=False, saveStats=False, failOnError=False):
    saveStep = Save(username, imageName)
    saveStep.resetLogFile()
    saveStep.deleteReference()
    images = loadImages(username, imageName)

    metadata = saveStep.getMetadata()

    if not isMetadataValid(metadata):
        print('User :: {} | Image :: {} | Error :: {}'.format(username, imageName, 'Metadata does not Match'))
        return getResponse(imageName, False)

    numImages = len(images)
    captures = [Capture(image, meta) for image, meta in zip(images, metadata)]
    #Brightest is index 0, dimmest is last

    print('Cropping and Aligning')
    try:
        alignImages.cropAndAlignCaptures(captures)
    except Exception as err:
        if failOnError:
            raise NameError('User :: {} | Image :: {} | Error :: {} | Details :: {}'.format(username, imageName, 'Error Cropping and Aligning Images', err))
        else:
            print('User :: {} | Image :: {} | Error :: {} | Details :: {}'.format(username, imageName, 'Error Cropping and Aligning Images', err))
            return getResponse(imageName, False)
        
    #Now that they are aligned, use the same landmarks for all of them... (Maybe use the average?)
    # -> Simplifies things down the line...
    for capture in captures:
        capture.landmarks = captures[0].landmarks
    
    print('Done Cropping and aligning')

    allPointsMask = captures[0].mask
    for capture in captures:
        allPointsMask = np.logical_or(allPointsMask, capture.mask)

    for capture in captures:
        capture.whiteBalanceImageToD65()

    maxValue = np.max([np.max(capture.image) for capture in captures])
    print('MAX VALUE :: ' + str(maxValue))

    for capture in captures:
        capture.scaleToValue(maxValue)

    print('Subtracting Base from Flash')

    fullFlashRangeDiff = captures[0].image.astype('int32') - captures[-1].image.astype('int32')
    fullFlashRangeDiff[fullFlashRangeDiff == 0] = 1 #We will be dividing by this later

    flashSteps = np.array([captures[index].image.astype('int32') - captures[index - 1].image.astype('int32') for index in range(1, len(captures))])

    meanFlashStep = np.mean(flashSteps, axis=0)
    flashStepDiffs = flashSteps - meanFlashStep
    nonLinearityMasks = [getNonLinearityMask(flashStepDiff, fullFlashRangeDiff) for flashStepDiff in flashStepDiffs]

    for nonLinearityMask in nonLinearityMasks:
        allPointsMask = np.logical_or(allPointsMask, nonLinearityMask)


    #smallShowImg = cv2.resize(allPointsMask.astype('uint8') * 255, (0, 0), fx=1/2, fy=1/2)
    #cv2.imshow('All Points Mask', smallShowImg.astype('uint8'))
    #cv2.waitKey(0)

    if not fast:
        print('Saving Step 1')
        saveStep.saveImageStep(np.clip(fullFlashRangeDiff, 0, 255).astype('uint8'), 1)
        saveStep.saveMaskStep(allPointsMask, 1, 'clippedMask')


    try:
        averageReflection, averageReflectionArea, leftEyeReflections, rightEyeReflections = getAverageScreenReflectionColor(captures, saveStep)
    except Exception as err:
        if failOnError:
            raise NameError('User :: {} | Image :: {} | Error :: {} | Details :: {}'.format(username, imageName, 'Error Extracting Reflection', err))
        else:
            print('User :: {} | Image :: {} | Error :: {} | Details :: {}'.format(username, imageName, 'Error Extracting Reflection', err))
            return getResponse(imageName, False)

    #Want to reassign mask after getting screen reflection color. Masks get used in that step

    print('Left Eye Reflections :: {}'.format(leftEyeReflections))
    print('Right Eye Reflections :: {}'.format(rightEyeReflections))

    for capture in captures:
        colorTools.whitebalanceBGR(capture, averageReflection)
        capture.mask = allPointsMask

    maxValue = np.max([np.max(capture.image) for capture in captures])
    print('MAX VALUE :: ' + str(maxValue))

    for capture in captures:
        capture.scaleToValue(maxValue)

    try:
        faceRegions = np.array([FaceRegions(capture) for capture in captures])
    except Exception as err:
        if failOnError:
            raise NameError('User :: {} | Image :: {} | Error :: {} | Details :: {}'.format(username, imageName, 'Error extracting Points for Recovered Mask', err))
        else:
            print('User :: {} | Image :: {} | Error :: {} | Details :: {}'.format(username, imageName, 'Error extracting Points for Recovered Mask', err))
            return getResponse(imageName, False)
    else:
        saveStep.saveReferenceImageBGR(faceRegions[0].getMaskedImage(), faceRegions[0].capture.name + '_masked')

        plotPerRegionLinearity(faceRegions, leftEyeReflections, rightEyeReflections, saveStep)
        plotPerRegionPoints(faceRegions, saveStep)
        plotPerRegionDistribution(faceRegions, saveStep)
        plotPerEyeReflectionBrightness(faceRegions, leftEyeReflections, rightEyeReflections, saveStep)

        leftEyeDiffReflectionMedian = getMedianDiff(leftEyeReflections)
        leftEyeDiffReflectionMedianHSV = colorTools.bgr_to_hsv(leftEyeDiffReflectionMedian)

        rightEyeDiffReflectionMedian = getMedianDiff(rightEyeReflections)
        rightEyeDiffReflectionMedianHSV = colorTools.bgr_to_hsv(rightEyeDiffReflectionMedian)

        faceRegionMedians = np.vstack([[region.getRegionMedians() for region in faceRegions]])
        faceRegionDiffMedians = [getMedianDiff(faceRegionMedians[:, idx]) for idx in range(0, faceRegionMedians.shape[1])]
        faceRegionDiffMediansHSV  = [colorTools.bgr_to_hsv(point) for point in faceRegionDiffMedians]

        formatString = '\nMEDIAN DIFFS :: {}\n\tEYES \n\t\tLEFT \t\t{}\n\t\tRIGHT \t\t{}\n\tFACE\n\t\tLEFT \t\t{}\n\t\tRIGHT \t\t{}\n\t\tCHIN \t\t{}\n\t\tFOREHEAD \t{}\n'
        formatted = formatString.format('BGR', leftEyeDiffReflectionMedian, rightEyeDiffReflectionMedian, *faceRegionDiffMedians)
        formattedHSV = formatString.format('HSV', leftEyeDiffReflectionMedianHSV, rightEyeDiffReflectionMedianHSV, *faceRegionDiffMediansHSV)
        print(formatted)
        print(formattedHSV)






        #NEW RULES: COLORS ARE RETURNED IN BGR
        #           FIELDS are Left Cheek, Right Cheek, Chin Forehead
        #noFlashValues = [noFlashPointsLeftCheekMedian, noFlashPointsRightCheekMedian, noFlashPointsChinMedian, noFlashPointsForeheadMedian]
        #halfFlashValues = [leftCheekMedianHalfBGR, rightCheekMedianHalfBGR, chinMedianHalfBGR, foreheadMedianHalfBGR]
        #fullFlashValues = [leftCheekMedianFullBGR, rightCheekMedianFullBGR, chinMedianFullBGR, foreheadMedianFullBGR]
        #linearity = [leftCheekLinearityError, rightCheekLinearityError, chinLinearityError, foreheadLinearityError]
        #cleanRatio = [leftCheekClippingRatio, rightCheekClippingRatio, chinClippingRatio, foreheadClippingRatio]
        #reflectionValues = [leftReflectionValues, rightReflectionValues]
        #fluxishValues = [scaledLeftFluxish, scaledRightFluxish, scaledAverageFluxish, scaledAverageFluxish]
        captureSets = zip(faceRegions, leftEyeReflections, rightEyeReflections)

        response = getResponse(imageName, True, captureSets)
        return response



