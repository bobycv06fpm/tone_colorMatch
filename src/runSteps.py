from loadImages import loadImages
#from detectFace import detectFace
import alignImages
from getAverageReflection import getAverageScreenReflectionColor
from saveStep import Save
from getPolygons import getPolygons, getFullFacePolygon
from extractMask import extractMask, maskPolygons
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

import multiprocessing as mp
import colorsys

chartSampleSize = 1000

def scalePointstoFluxish(capture, fluxish):
    #targetFluxish = 100000
    targetFluxish = 50000
    fluxishMultiplier = targetFluxish / fluxish
    capture.image *= fluxishMultiplier#).astype('int32')
    #return scaledPoints

def correctHLS(hls, luminance, fluxish):
    print('------')
    print('Old HLS :: ' + str(hls))
    #targetFluxish = 0.8
    targetFluxish = 0.25
    #slope = 0.145696 
    #slope = 0.2731371365514631 
    luminanceSlope = 121.092803122
    hueSlope = 0.018374
    #lightnessDiff = (0.106485 * (targetFluxish - fluxish))
    lightnessDiff = (slope * (targetFluxish - fluxish))
    hls[1] += lightnessDiff

    hls[0] = (hueSlope * hls[1]) + .059859

    print('Corrected HLS :: ' + str(hls))
    print('-----')

    return hls

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

def cullPoints(points):
    median = np.median(points, axis=0)
    sd = np.std(points, axis=0)

    deviations = 2
    boundsLow = median - (deviations * sd)
    boundsHigh = median + (deviations * sd)

    hueMask = np.logical_and(points[:, 0] > boundsLow[0], points[:, 0] < boundsHigh[0])
    saturationMask = np.logical_and(points[:, 1] > boundsLow[1], points[:, 1] < boundsHigh[1])
    valueMask = np.logical_and(points[:, 2] > boundsLow[2], points[:, 2] < boundsHigh[2])

    fullMask = np.logical_and(hueMask, saturationMask)
    fullMask = np.logical_and(fullMask, valueMask)
    culled = points[fullMask]
    #print('Shape Before :: ' + str(points.shape))
    #print('Shape After :: ' + str(culled.shape))
    return culled


def plotZones(leftCheek, rightCheek, chin, forehead, saveStep, tag=''):
    [leftCheek_hsv, leftCheekLuminance] = leftCheek
    [rightCheek_hsv, rightCheekLuminance] = rightCheek
    [chin_hsv, chinLuminance] = chin
    [forehead_hsv, foreheadLuminance] = forehead

    if len(leftCheek_hsv) > chartSampleSize:
        sample = np.random.choice(len(leftCheek_hsv), chartSampleSize)
        leftCheek_hsv_sample = np.take(leftCheek_hsv, sample, axis=0)
        leftCheekLuminance_sample = np.take(leftCheekLuminance, sample, axis=0)
    else:
        leftCheek_hsv_sample = leftCheek_hsv
        leftCheekLuminance_sample = leftCheekLuminance

    if len(rightCheek_hsv) > chartSampleSize:
        sample = np.random.choice(len(rightCheek_hsv), chartSampleSize)
        rightCheek_hsv_sample = np.take(rightCheek_hsv, sample, axis=0)
        rightCheekLuminance_sample = np.take(rightCheekLuminance, sample, axis=0)
    else:
        rightCheek_hsv_sample = rightCheek_hsv
        rightCheekLuminance_sample = rightCheekLuminance

    if len(chin_hsv) > chartSampleSize:
        sample = np.random.choice(len(chin_hsv), chartSampleSize)
        chin_hsv_sample = np.take(chin_hsv, sample, axis=0)
        chinLuminance_sample = np.take(chinLuminance, sample, axis=0)
    else:
        chin_hsv_sample = chin_hsv
        chinLuminance_sample = chinLuminance

    if len(forehead_hsv) > chartSampleSize:
        sample = np.random.choice(len(forehead_hsv), chartSampleSize)
        forehead_hsv_sample = np.take(forehead_hsv, sample, axis=0)
        foreheadLuminance_sample = np.take(foreheadLuminance, sample, axis=0)
    else:
        forehead_hsv_sample = forehead_hsv
        foreheadLuminance_sample = foreheadLuminance


    fig, axs = plt.subplots(5, 3, sharey=False, tight_layout=True)
    size = 1

    #Luminance_sample
    axs[0, 0].scatter(leftCheekLuminance_sample, leftCheek_hsv_sample[:, 1], size, (1, 0, 0))
    leftCheekLine = fitLine(leftCheekLuminance_sample, leftCheek_hsv_sample[:, 1])

    axs[1, 0].scatter(rightCheekLuminance_sample, rightCheek_hsv_sample[:, 1], size, (1, 0, 0))
    rightCheekLine = fitLine(rightCheekLuminance_sample, rightCheek_hsv_sample[:, 1])

    axs[2, 0].scatter(chinLuminance_sample, chin_hsv_sample[:, 1], size, (1, 0, 0))
    chinLine = fitLine(chinLuminance_sample, chin_hsv_sample[:, 1])
    axs[2, 0].plot([min(chinLuminance_sample), max(chinLuminance_sample)], [min(chinLuminance_sample) * chinLine[0] + chinLine[1], max(chinLuminance_sample) * chinLine[0] + chinLine[1]])

    axs[3, 0].scatter(foreheadLuminance_sample, forehead_hsv_sample[:, 1], size, (1, 0, 0))
    foreheadLine = fitLine(foreheadLuminance_sample, forehead_hsv_sample[:, 1])
    axs[3, 0].plot([min(foreheadLuminance_sample), max(foreheadLuminance_sample)], [min(foreheadLuminance_sample) * foreheadLine[0] + foreheadLine[1], max(foreheadLuminance_sample) * foreheadLine[0] + foreheadLine[1]])

    axs[4, 0].scatter(leftCheekLuminance_sample, leftCheek_hsv_sample[:, 1], size, (1, 0, 0))
    axs[4, 0].scatter(rightCheekLuminance_sample, rightCheek_hsv_sample[:, 1], size, (1, 0, 0))
    axs[4, 0].scatter(chinLuminance_sample, chin_hsv_sample[:, 1], size, (1, 0, 0))
    axs[4, 0].scatter(foreheadLuminance_sample, forehead_hsv_sample[:, 1], size, (1, 0, 0))

    print('Lines :: ' + str(leftCheekLine) + ' | ' + str(rightCheekLine) + ' | ' + str(chinLine) + ' | ' + str(foreheadLine))

    #Value
    #axs[0, 0].scatter(leftCheek_hsv_sample[:, 2], leftCheek_hsv_sample[:, 1], size, (1, 0, 0))
    #axs[1, 0].scatter(rightCheek_hsv_sample[:, 2], rightCheek_hsv_sample[:, 1], size, (1, 0, 0))
    #axs[2, 0].scatter(chin_hsv_sample[:, 2], chin_hsv_sample[:, 1], size, (1, 0, 0))
    #axs[3, 0].scatter(forehead_hsv_sample[:, 2], forehead_hsv_sample[:, 1], size, (1, 0, 0))

    #Intensity
    #axs[0, 0].scatter(leftCheekIntensity, leftCheek_hsv_sample[:, 1], size, (1, 0, 0))
    #axs[1, 0].scatter(rightCheekIntensity, rightCheek_hsv_sample[:, 1], size, (1, 0, 0))
    #axs[2, 0].scatter(chinIntensity, chin_hsv_sample[:, 1], size, (1, 0, 0))
    #axs[3, 0].scatter(foreheadIntensity, forehead_hsv_sample[:, 1], size, (1, 0, 0))

    #Luminance_sample
    axs[0, 1].scatter(leftCheekLuminance_sample, rotateHue(leftCheek_hsv_sample[:, 0]), size, (1, 0, 0))
    axs[1, 1].scatter(rightCheekLuminance_sample, rotateHue(rightCheek_hsv_sample[:, 0]), size, (1, 0, 0))
    axs[2, 1].scatter(chinLuminance_sample, rotateHue(chin_hsv_sample[:, 0]), size, (1, 0, 0))
    axs[3, 1].scatter(foreheadLuminance_sample, rotateHue(forehead_hsv_sample[:, 0]), size, (1, 0, 0))

    axs[4, 1].scatter(leftCheekLuminance_sample, rotateHue(leftCheek_hsv_sample[:, 0]), size, (1, 0, 0))
    axs[4, 1].scatter(rightCheekLuminance_sample, rotateHue(rightCheek_hsv_sample[:, 0]), size, (1, 0, 0))
    axs[4, 1].scatter(chinLuminance_sample, rotateHue(chin_hsv_sample[:, 0]), size, (1, 0, 0))
    axs[4, 1].scatter(foreheadLuminance_sample, rotateHue(forehead_hsv_sample[:, 0]), size, (1, 0, 0))

    #Value
    #axs[0, 1].scatter(leftCheek_hsv_sample[:, 2], np.clip(leftCheek_hsv_sample[:, 0], 0, 0.1), size, (1, 0, 0))
    #axs[1, 1].scatter(rightCheek_hsv_sample[:, 2], np.clip(rightCheek_hsv_sample[:, 0], 0, 0.1), size, (1, 0, 0))
    #axs[2, 1].scatter(chin_hsv_sample[:, 2], np.clip(chin_hsv_sample[:, 0], 0, 0.1), size, (1, 0, 0))
    #axs[3, 1].scatter(forehead_hsv_sample[:, 2], np.clip(forehead_hsv_sample[:, 0], 0, 0.1), size, (1, 0, 0))

    #Intensity
    #axs[0, 1].scatter(leftCheekIntensity, np.clip(leftCheek_hsv_sample[:, 0], 0, 0.1), size, (1, 0, 0))
    #axs[1, 1].scatter(rightCheekIntensity, np.clip(rightCheek_hsv_sample[:, 0], 0, 0.1), size, (1, 0, 0))
    #axs[2, 1].scatter(chinIntensity, np.clip(chin_hsv_sample[:, 0], 0, 0.1), size, (1, 0, 0))
    #axs[3, 1].scatter(foreheadIntensity, np.clip(forehead_hsv_sample[:, 0], 0, 0.1), size, (1, 0, 0))

    minH = min(rotateHue(chin_hsv_sample[:, 0]))
    maxH = max(rotateHue(chin_hsv_sample[:, 0]))
    A = np.vstack([rotateHue(chin_hsv_sample[:, 0]), np.ones(len(chin_hsv_sample))]).T
    m, c = np.linalg.lstsq(A, chin_hsv_sample[:, 1], rcond=None)[0]
    axs[2, 2].plot([minH, maxH], [(m * minH + c), (m * maxH + c)])

    minH = min(rotateHue(forehead_hsv_sample[:, 0]))
    maxH = max(rotateHue(forehead_hsv_sample[:, 0]))
    A = np.vstack([rotateHue(forehead_hsv_sample[:, 0]), np.ones(len(forehead_hsv_sample))]).T
    m, c = np.linalg.lstsq(A, forehead_hsv_sample[:, 1], rcond=None)[0]
    axs[3, 2].plot([minH, maxH], [(m * minH + c), (m * maxH + c)])

    axs[0, 2].scatter(rotateHue(leftCheek_hsv_sample[:, 0]), leftCheek_hsv_sample[:, 1], size, (1, 0, 0))
    axs[1, 2].scatter(rotateHue(rightCheek_hsv_sample[:, 0]), rightCheek_hsv_sample[:, 1], size, (1, 0, 0))
    axs[2, 2].scatter(rotateHue(chin_hsv_sample[:, 0]), chin_hsv_sample[:, 1], size, (1, 0, 0))
    axs[3, 2].scatter(rotateHue(forehead_hsv_sample[:, 0]), forehead_hsv_sample[:, 1], size, (1, 0, 0))

    axs[4, 2].scatter(rotateHue(leftCheek_hsv_sample[:, 0]), leftCheek_hsv_sample[:, 1], size, (1, 0, 0))
    axs[4, 2].scatter(rotateHue(rightCheek_hsv_sample[:, 0]), rightCheek_hsv_sample[:, 1], size, (1, 0, 0))
    axs[4, 2].scatter(rotateHue(chin_hsv_sample[:, 0]), chin_hsv_sample[:, 1], size, (1, 0, 0))
    axs[4, 2].scatter(rotateHue(forehead_hsv_sample[:, 0]), forehead_hsv_sample[:, 1], size, (1, 0, 0))

    #plt.show()
    saveStep.savePlot('Luminance_sample_Hue_Saturation_Scatter' + tag, plt)
    #saveStep.savePlot('Value_Hue_Saturation_Scatter', plt)
    #saveStep.savePlot('Intensity_Hue_Saturation_Scatter', plt)

    bins = 50
    fig, axs = plt.subplots(5, 3, sharey=False, tight_layout=True)
    axs[0, 0].hist(leftCheekLuminance_sample, bins=bins)
    axs[1, 0].hist(rightCheekLuminance_sample, bins=bins)
    axs[2, 0].hist(chinLuminance_sample, bins=bins)
    axs[3, 0].hist(foreheadLuminance_sample, bins=bins)
    axs[4, 0].hist([list(foreheadLuminance_sample) + list(chinLuminance_sample) + list(rightCheekLuminance_sample) + list(leftCheekLuminance_sample)], bins=bins)

    axs[0, 1].hist(leftCheek_hsv_sample[:, 1], bins=bins)
    axs[1, 1].hist(rightCheek_hsv_sample[:, 1], bins=bins)
    axs[2, 1].hist(chin_hsv_sample[:, 1], bins=bins)
    axs[3, 1].hist(forehead_hsv_sample[:, 1], bins=bins)
    axs[4, 1].hist([list(forehead_hsv_sample[:, 1]) + list(chin_hsv_sample[:, 1]) + list(rightCheek_hsv_sample[:, 1]) + list(leftCheek_hsv_sample[:, 1])], bins=bins)

    axs[0, 2].hist(rotateHue(leftCheek_hsv_sample[:, 0]), bins=bins)   #Watch for clipping...
    axs[1, 2].hist(rotateHue(rightCheek_hsv_sample[:, 0]), bins=bins)
    axs[2, 2].hist(rotateHue(chin_hsv_sample[:, 0]), bins=bins)   #Watch for clipping...
    axs[3, 2].hist(rotateHue(forehead_hsv_sample[:, 0]), bins=bins)
    axs[4, 2].hist([list(rotateHue(forehead_hsv_sample[:, 0])) + list(rotateHue(chin_hsv_sample[:, 0])) + list(rotateHue(rightCheek_hsv_sample[:, 0])) + list(rotateHue(leftCheek_hsv_sample[:, 0]))], bins=bins)
    #plt.show()
    saveStep.savePlot('Luminance_Hue_Saturation_hist' + tag, plt)

    return [leftCheekLine, rightCheekLine, chinLine, foreheadLine]

def convertPoints(points):
    luminance = colorTools.getRelativeLuminance(points)
    #intensity = np.mean(points, axis=1)
    medianLuminance = np.median(luminance)

    RGB = np.flip(points, axis=1) / 255
    HSV = np.array([list(colorsys.rgb_to_hsv(r, g, b)) for [r, g, b] in RGB])

    medianHSV = np.median(HSV, axis=0)
    medianBGR = np.median(points, axis=0)

    return [HSV, medianHSV, medianBGR, luminance, medianLuminance]

def applyLinearAdjustment(A, B):
    [m, x] = np.polyfit(B, A, 1)
    print('Slope :: ' + str(m) + ' | Const :: ' + str(x))

    medB = np.median(B)

    diffB = B - medB
    print('Diff B :: ' + str(diffB))
    diffA = diffB * m
    print('Diff A :: ' + str(diffA))
    return A - diffA

#def adjustSatToHue(sat, hue):
#    hue = rotateHue(hue)
#    sat = applyLinearAdjustment(sat, hue)
#    return sat

def adjustSatToHue(chinHSV, foreheadHSV):
    chinHue = rotateHue(chinHSV[:, 0])
    foreheadHue = rotateHue(foreheadHSV[:, 0])

    chinSlope, chinConst = fitLine(chinHue, chinHSV[:, 1])
    foreheadSlope, foreheadConst = fitLine(foreheadHue, foreheadHSV[:, 1])

    averageSlope = (chinSlope + foreheadSlope) / 2

    medianHue = np.median(list(chinHue) + list(foreheadHue))

    diffChinHue = chinHue - medianHue
    diffForeheadHue = foreheadHue - medianHue

    diffChinSat = diffChinHue * averageSlope
    diffForeheadSat = diffForeheadHue * averageSlope

    chinHSV[:, 1] -= diffChinSat
    foreheadHSV[:, 1] -= diffForeheadSat

    return [chinHSV, foreheadHSV]

def plotBGR(axs, color, x, y):
    if len(x) > chartSampleSize:
        sample = np.random.choice(len(x), chartSampleSize)
        x_sample = np.take(x, sample, axis=0)
        y_sample = np.take(y, sample, axis=0)
    else:
        x_sample = x
        y_sample = y

    size = 1
    start_x = min(x_sample)
    end_x = max(x_sample)

    axs.scatter(x_sample, y_sample, size, color)

    m, c = fitLine(x_sample, y_sample)
    axs.plot([start_x, end_x], [(m * start_x + c), (m * end_x + c)])

def getRegionMapBGR(leftCheek, rightCheek, chin, forehead):
    value = {}
    print('Left Cheek BGR :: ' + str(leftCheek))
    value['left'] = list(leftCheek)
    value['right'] = list(rightCheek)
    value['chin'] = list(chin)
    value['forehead'] = list(forehead)

    return value

def getRegionMapValue(leftCheek, rightCheek, chin, forehead):
    value = {}
    print('Left Cheek Value :: ' + str(leftCheek))
    value['left'] = leftCheek
    value['right'] = rightCheek
    value['chin'] = chin
    value['forehead'] = forehead

    return value

def getReflectionMap(leftReflection, rightReflection):
    value = {}
    print('Left Reflection :: ' + str(leftReflection))
    value['left'] = [list(reflection) for reflection in leftReflection]
    value['right'] = [list(reflection) for reflection in rightReflection]

    return value

def getResponse(imageName, successful, noFlashValues=None, halfFlashValues=None, fullFlashValues=None, linearity=None, cleanRatio=None, reflectionValues=None, fluxishValues=None):
    response = {}
    response['name'] = imageName
    response['successful'] = successful

    if not successful:
        return response

    response['noFlashValues'] = getRegionMapBGR(*noFlashValues)
    response['halfFlashValues'] = getRegionMapBGR(*halfFlashValues)
    response['fullFlashValues'] = getRegionMapBGR(*fullFlashValues)
    response['linearity'] = getRegionMapValue(*linearity)
    response['cleanRatio'] = getRegionMapValue(*cleanRatio)
    response['reflectionValues'] = getReflectionMap(*reflectionValues)
    response['fluxishValues'] = getRegionMapValue(*fluxishValues)

    return response

def run(username, imageName, fast=False, saveStats=False, failOnError=False):
    #saveStep.resetLogFile(username, imageName)
    saveStep = Save(username, imageName)
    saveStep.resetLogFile()
    saveStep.deleteReference()
    images = loadImages(username, imageName)

    [noFlashImage, halfFlashImage, fullFlashImage] = images
    [noFlashMetadata, halfFlashMetadata, fullFlashMetadata] = saveStep.getMetadata()

    noFlashCapture = Capture('No Flash', noFlashImage, noFlashMetadata)
    noFlashCapture.whiteBalanceImageToD65()

    halfFlashCapture = Capture('Half Flash', halfFlashImage, halfFlashMetadata)
    halfFlashCapture.whiteBalanceImageToD65()

    fullFlashCapture = Capture('Full Flash', fullFlashImage, fullFlashMetadata)
    fullFlashCapture.whiteBalanceImageToD65()

    #noFlashCapture.showImageWithLandmarks()
    #halfFlashCapture.showImageWithLandmarks()
    #fullFlashCapture.showImageWithLandmarks()

    print('Cropping and Aligning')
    try:
        alignImages.cropAndAlign(noFlashCapture, halfFlashCapture, fullFlashCapture)
    except Exception as err:
        if failOnError:
            raise NameError('User :: {} | Image :: {} | Error :: {} | Details :: {}'.format(username, imageName, 'Error Cropping and Aligning Images', err))
        else:
            print('User :: {} | Image :: {} | Error :: {} | Details :: {}'.format(username, imageName, 'Error Cropping and Aligning Images', err))
            return getResponse(imageName, False)
        
    print('Done Cropping and aligning')

    partialMask = np.logical_or(noFlashCapture.mask, halfFlashCapture.mask)
    allPointsMask = np.logical_or(partialMask, fullFlashCapture.mask)

    # (All Pixels That Are Clipped In Full Flash) AND (All The Pixels That Are NOT Clipped By Base & Top & Bottom Flash)
    # => Pixel Values caused to clip by Full Flash
    #potentiallyRecoverablePixelsMask = np.logical_and(fullFlashImageMask, np.logical_not(partialMask))
    #unrecoverablePixelsMask = np.logical_and(fullFlashImageMask, partialMask)
    #allPointsMask = np.logical_and(allPointsMask, np.logical_not(unrecoverablePixelsMask))


    #unrecoverablePixelsMask = maskPolygons(unrecoverablePixelsMask, polygons)
    #potentiallyRecoverablePixelsMask = maskPolygons(potentiallyRecoverablePixelsMask, polygons)
    #RecoveredFullFlash = ((2 * halfFlashImage) - noFlashImage)

    #fullFlashImage[potentiallyRecoverablePixelsMask] = RecoveredFullFlash[potentiallyRecoverablePixelsMask]

    #NOTE: MIGHT BE NEEDED
    #noFlashImageBlur = cv2.GaussianBlur(noFlashCapture.getClippedImage(), (7, 7), 0)
    #halfFlashImageBlur = cv2.GaussianBlur(halfFlashCapture.getClippedImage(), (7, 7), 0)
    #fullFlashImageBlur = cv2.GaussianBlur(fullFlashCapture.getClippedImage(), (7, 7), 0)
    #END NOTE

    print('Testing Linearity')
    #howLinear = np.abs((2 * halfFlashCapture.image) - (fullFlashCapture.image + noFlashCapture.image))
    print('Subtracting Base from Flash')
    #halfDiffImage = halfFlashCapture.image.astype('int32') - noFlashCapture.image.astype('int32')
    halfDiffImage = halfFlashCapture.blurredImage().astype('int32') - noFlashCapture.blurredImage().astype('int32')

    #halfDiffImageBlur = cv2.GaussianBlur(halfDiffImage, (11, 11), 0)
    #halfDiffImageBlur = cv2.medianBlur(halfDiffImage, 9)

    #fullDiffImage = fullFlashCapture.image.astype('int32') - noFlashCapture.image.astype('int32')
    fullDiffImage = fullFlashCapture.blurredImage().astype('int32') - noFlashCapture.blurredImage().astype('int32')

    #fullDiffImageBlur = cv2.GaussianBlur(fullDiffImage, (25, 25), 0)
    #fullDiffImageBlur = cv2.medianBlur(fullDiffImage, 9)

    howLinear = np.abs((2 * halfDiffImage) - fullDiffImage)
    fullDiffImage[fullDiffImage == 0] = 1
    percentError = howLinear / fullDiffImage
    perSubPixelMaxError = np.mean(percentError, axis=2)
    #perSubPixelMaxError = np.max(percentError, axis=2)
    #nonLinearMask = perSubPixelMaxError > .10
    #perChannelNonLinearMask = perSubPixelMaxError > .02
    nonLinearMask = perSubPixelMaxError > .05

    #cv2.imshow('All Points mask', allPointsMask.astype('uint8') * 255)
    #cv2.imshow('Non Linear Mask', nonLinearMask.astype('uint8') * 255)
    #cv2.waitKey(0)

    #TODO: Compare Subpixel nonlinearity with full pixel nonlinearity....
    #howLinearSum = np.sum(howLinear, axis=2)
    #nonLinearMask = howLinearMax > 6#8 #12

    allPointsMask = np.logical_or(allPointsMask, nonLinearMask)

    #fullDiffImage = fullFlashCapture.image - noFlashCapture.image
    #halfDiffImage = halfFlashCapture.image - noFlashCapture.image

    #print('Diff Image :: ' + str(diffImage))
    #noMask = np.zeros(allPointsMask.shape).astype('bool')
    #fullDiffCapture = Capture('Diff', fullDiffImage, fullFlashCapture.metadata, noMask)

    fullDiffCapture = Capture('Diff', fullDiffImage, fullFlashCapture.metadata, allPointsMask)
    #halfDiffCapture = Capture('Diff', halfDiffImage, halfFlashCapture.metadata, noMask)
    halfDiffCapture = Capture('Diff', halfDiffImage, halfFlashCapture.metadata, allPointsMask)

    noFlashCapture.mask = allPointsMask

    #print('Getting Polygons')
    #polygons = fullDiffCapture.landmarks.getFacePolygons()
    #print('POLYGONS :: ' + str(polygons))

    if not fast:
        print('Saving Step 1')
        #saveStep.saveShapeStep(username, imageName, imageShape, 1)
        saveStep.saveImageStep(fullDiffCapture.getClippedImage(), 1)
        saveStep.saveMaskStep(allPointsMask, 1, 'clippedMask')

    #alignImages.alignEyes(noFlashCapture, halfFlashCapture, fullFlashCapture)
    #whiteBalance_CIE1931_coord_asShot = saveStep.getAsShotWhiteBalance()
    #print('White Balance As Shot :: ' + str(whiteBalance_CIE1931_coord_asShot))

    noFlashCapture.landmarks = halfFlashCapture.landmarks
    fullFlashCapture.landmarks = halfFlashCapture.landmarks

    try:
        [reflectionValue, leftLuminance, leftFluxish, rightLuminance, rightFluxish, leftReflectionValues, rightReflectionValues] = getAverageScreenReflectionColor(noFlashCapture, halfFlashCapture, fullFlashCapture, saveStep)
    except Exception as err:
        if failOnError:
            raise NameError('User :: {} | Image :: {} | Error :: {} | Details :: {}'.format(username, imageName, 'Error Extracting Reflection', err))
        else:
            print('User :: {} | Image :: {} | Error :: {} | Details :: {}'.format(username, imageName, 'Error Extracting Reflection', err))
            return getResponse(imageName, False)

    averageFluxish = (leftFluxish + rightFluxish) / 2
    print("Reflection Value:: " + str(reflectionValue))
    print("Fluxish :: " + str(averageFluxish))
    #diffCapture.show()

    saveStep.saveReferenceImageBGR(fullDiffCapture.getClippedImage(), 'full_noWhitebalancedImage')
    saveStep.saveReferenceImageBGR(halfDiffCapture.getClippedImage(), 'half_noWhitebalancedImage')

    colorTools.whitebalanceBGR(fullDiffCapture, reflectionValue)
    colorTools.whitebalanceBGR(halfDiffCapture, reflectionValue)

    saveStep.saveReferenceImageBGR(fullDiffCapture.getClippedImage(), 'full_WhitebalancedImage')
    saveStep.saveReferenceImageBGR(halfDiffCapture.getClippedImage(), 'half_WhitebalancedImage')

    try:
        [fullPoints, fullPointsLeftCheek, fullPointsRightCheek, fullPointsChin, fullPointsForehead] = extractMask(fullDiffCapture, percentError, saveStep)
        [halfPoints, halfPointsLeftCheek, halfPointsRightCheek, halfPointsChin, halfPointsForehead] = extractMask(halfDiffCapture, percentError, saveStep)
        [noFlashPoints, noFlashPointsLeftCheek, noFlashPointsRightCheek, noFlashPointsChin, noFlashPointsForehead] = extractMask(noFlashCapture, percentError, saveStep)
    except Exception as err:
        if failOnError:
            raise NameError('User :: {} | Image :: {} | Error :: {} | Details :: {}'.format(username, imageName, 'Error extracting Points for Recovered Mask', err))
        else:
            print('User :: {} | Image :: {} | Error :: {} | Details :: {}'.format(username, imageName, 'Error extracting Points for Recovered Mask', err))
            return getResponse(imageName, False)
    else:
        noFlashPointsLeftCheek, leftCheekLinearityError, leftCheekClippingRatio = noFlashPointsLeftCheek
        noFlashPointsRightCheek, rightCheekLinearityError, rightCheekClippingRatio = noFlashPointsRightCheek
        noFlashPointsChin, chinLinearityError, chinClippingRatio = noFlashPointsChin
        noFlashPointsForehead, foreheadLinearityError, foreheadClippingRatio = noFlashPointsForehead

        noFlashPointsLeftCheekMedian = np.median(noFlashPointsLeftCheek, axis=0)
        noFlashPointsRightCheekMedian = np.median(noFlashPointsRightCheek, axis=0)
        noFlashPointsChinMedian = np.median(noFlashPointsChin, axis=0)
        noFlashPointsForeheadMedian = np.median(noFlashPointsForehead, axis=0)

        fullPointsLeftCheek, leftCheekLinearityError, leftCheekClippingRatio = fullPointsLeftCheek
        fullPointsRightCheek, rightCheekLinearityError, rightCheekClippingRatio = fullPointsRightCheek
        fullPointsChin, chinLinearityError, chinClippingRatio = fullPointsChin
        fullPointsForehead, foreheadLinearityError, foreheadClippingRatio = fullPointsForehead

        halfPointsLeftCheek, leftCheekLinearityError, leftCheekClippingRatio = halfPointsLeftCheek
        halfPointsRightCheek, rightCheekLinearityError, rightCheekClippingRatio = halfPointsRightCheek
        halfPointsChin, chinLinearityError, chinClippingRatio = halfPointsChin
        halfPointsForehead, foreheadLinearityError, foreheadClippingRatio = halfPointsForehead

        largestValue = np.max(fullPoints)
        print('LARGEST VALUE :: ' + str(largestValue))

        scaleDivisor = largestValue / 255
        scaledLeftFluxish = leftFluxish / scaleDivisor
        scaledRightFluxish = rightFluxish / scaleDivisor
        scaledAverageFluxish = averageFluxish / scaleDivisor

        scaledFullPointsLeftCheek = fullPointsLeftCheek / scaleDivisor
        scaledFullPointsRightCheek = fullPointsRightCheek / scaleDivisor
        scaledFullPointsChin = fullPointsChin / scaleDivisor
        scaledFullPointsForehead = fullPointsForehead / scaleDivisor

        scaledHalfPointsLeftCheek = halfPointsLeftCheek / scaleDivisor
        scaledHalfPointsRightCheek = halfPointsRightCheek / scaleDivisor
        scaledHalfPointsChin = halfPointsChin / scaleDivisor
        scaledHalfPointsForehead = halfPointsForehead / scaleDivisor

        print('Unscaled :: ' + str(fullPointsChin))
        print('Scaled Full :: ' + str(scaledFullPointsChin))
        print('Scaled Half :: ' + str(scaledHalfPointsChin))


        #scaledFullPointsLeftCheek = cullPoints(scaledFullPointsLeftCheek)
        #scaledFullPointsRightCheek = cullPoints(scaledFullPointsRightCheek)
        #scaledFullPointsChin = cullPoints(scaledFullPointsChin)
        #scaledFullPointsForehead = cullPoints(scaledFullPointsForehead)

        #scaledHalfPointsLeftCheek = cullPoints(scaledHalfPointsLeftCheek)
        #scaledHalfPointsRightCheek = cullPoints(scaledHalfPointsRightCheek)
        #scaledHalfPointsChin = cullPoints(scaledHalfPointsChin)
        #scaledHalfPointsForehead = cullPoints(scaledHalfPointsForehead)

        #BGR PLOT
        size=1
        fig, axs = plt.subplots(3, 3, sharex=False, sharey=False, tight_layout=True)
        scaledHalfPointsCheek = np.array(list(scaledHalfPointsLeftCheek) + list(scaledHalfPointsRightCheek))
        scaledFullPointsCheek = np.array(list(scaledFullPointsLeftCheek) + list(scaledFullPointsRightCheek))

        # --- (0, 0) ---
        plotBGR(axs[0, 0], (1, 0, 0), scaledHalfPointsCheek[:, 2], scaledHalfPointsCheek[:, 1])
        plotBGR(axs[0, 0], (0, 1, 0), scaledFullPointsCheek[:, 2], scaledFullPointsCheek[:, 1])

        axs[0, 0].set_xlabel('Red')
        axs[0, 0].set_ylabel('Green')

        # --- (0, 1) ---
        plotBGR(axs[0, 1], (1, 0, 0), scaledHalfPointsChin[:, 2], scaledHalfPointsChin[:, 1])
        plotBGR(axs[0, 1], (0, 1, 0), scaledFullPointsChin[:, 2], scaledFullPointsChin[:, 1])

        axs[0, 1].set_xlabel('Red')
        axs[0, 1].set_ylabel('Green')

        # --- (0, 2) ---
        plotBGR(axs[0, 2], (1, 0, 0), scaledHalfPointsForehead[:, 2], scaledHalfPointsForehead[:, 1])
        plotBGR(axs[0, 2], (0, 1, 0), scaledFullPointsForehead[:, 2], scaledFullPointsForehead[:, 1])

        axs[0, 2].set_xlabel('Red')
        axs[0, 2].set_ylabel('Green')

        # --- (1, 0) ---
        plotBGR(axs[1, 0], (1, 0, 0), scaledHalfPointsCheek[:, 2], scaledHalfPointsCheek[:, 0])
        plotBGR(axs[1, 0], (0, 1, 0), scaledFullPointsCheek[:, 2], scaledFullPointsCheek[:, 0])

        axs[1, 0].set_xlabel('Red')
        axs[1, 0].set_ylabel('Blue')

        # --- (1, 1) ---
        plotBGR(axs[1, 1], (1, 0, 0), scaledHalfPointsChin[:, 2], scaledHalfPointsChin[:, 0])
        plotBGR(axs[1, 1], (0, 1, 0), scaledFullPointsChin[:, 2], scaledFullPointsChin[:, 0])

        axs[1, 1].set_xlabel('Red')
        axs[1, 1].set_ylabel('Blue')

        # --- (1, 2) ---
        plotBGR(axs[1, 2], (1, 0, 0), scaledHalfPointsForehead[:, 2], scaledHalfPointsForehead[:, 0])
        plotBGR(axs[1, 2], (0, 1, 0), scaledFullPointsForehead[:, 2], scaledFullPointsForehead[:, 0])

        axs[1, 2].set_xlabel('Red')
        axs[1, 2].set_ylabel('Blue')

        # --- (2, 0) ---
        plotBGR(axs[2, 0], (1, 0, 0), scaledHalfPointsCheek[:, 1], scaledHalfPointsCheek[:, 0])
        plotBGR(axs[2, 0], (0, 1, 0), scaledFullPointsCheek[:, 1], scaledFullPointsCheek[:, 0])

        axs[2, 0].set_xlabel('Green')
        axs[2, 0].set_ylabel('Blue')

        # --- (2, 1) ---
        plotBGR(axs[2, 1], (1, 0, 0), scaledHalfPointsChin[:, 1], scaledHalfPointsChin[:, 0])
        plotBGR(axs[2, 1], (0, 1, 0), scaledFullPointsChin[:, 1], scaledFullPointsChin[:, 0])

        axs[2, 1].set_xlabel('Green')
        axs[2, 1].set_ylabel('Blue')

        # --- (2, 2) ---
        plotBGR(axs[2, 2], (1, 0, 0), scaledHalfPointsForehead[:, 1], scaledHalfPointsForehead[:, 0])
        plotBGR(axs[2, 2], (0, 1, 0), scaledFullPointsForehead[:, 1], scaledFullPointsForehead[:, 0])

        axs[2, 2].set_xlabel('Green')
        axs[2, 2].set_ylabel('Blue')

        #plt.show()
        saveStep.savePlot('BGR', plt)

        #CALCULATE IN LINEAR

        [leftCheekFullHSV, leftCheekMedianFullHSV, leftCheekMedianFullBGR, leftCheekFullLuminance, leftCheekMedianFullLuminance] = convertPoints(scaledFullPointsLeftCheek)
        [rightCheekFullHSV, rightCheekMedianFullHSV, rightCheekMedianFullBGR, rightCheekFullLuminance, rightCheekMedianFullLuminance] = convertPoints(scaledFullPointsRightCheek)
        [chinFullHSV, chinMedianFullHSV, chinMedianFullBGR, chinFullLuminance, chinMedianFullLuminance] = convertPoints(scaledFullPointsChin)
        [foreheadFullHSV, foreheadMedianFullHSV, foreheadMedianFullBGR, foreheadFullLuminance, foreheadMedianFullLuminance] = convertPoints(scaledFullPointsForehead)

        [leftCheekHalfHSV, leftCheekMedianHalfHSV, leftCheekMedianHalfBGR, leftCheekHalfLuminance, leftCheekMedianHalfLuminance] = convertPoints(scaledHalfPointsLeftCheek)
        [rightCheekHalfHSV, rightCheekMedianHalfHSV, rightCheekMedianHalfBGR, rightCheekHalfLuminance, rightCheekMedianHalfLuminance] = convertPoints(scaledHalfPointsRightCheek)
        [chinHalfHSV, chinMedianHalfHSV, chinMedianHalfBGR, chinHalfLuminance, chinMedianHalfLuminance] = convertPoints(scaledHalfPointsChin)
        [foreheadHalfHSV, foreheadMedianHalfHSV, foreheadMedianHalfBGR, foreheadHalfLuminance, foreheadMedianHalfLuminance] = convertPoints(scaledHalfPointsForehead)

        #leftCheekRatio = (leftCheekMedianFullLuminance - leftCheekMedianHalfLuminance) / (0.5 * scaledLeftFluxish)
        #rightCheekRatio = (rightCheekMedianFullLuminance - rightCheekMedianHalfLuminance) / (0.5 * scaledRightFluxish)
        #chinRatio = (chinMedianFullLuminance - chinMedianHalfLuminance) / (0.5 * scaledAverageFluxish)
        #foreheadRatio = (foreheadMedianFullLuminance - foreheadMedianHalfLuminance) / (0.5 * scaledAverageFluxish)

        print('---------------------')
        print('LEFT FLUXISH :: ' + str(scaledLeftFluxish))
        print('MEDIAN HSV LEFT Full Points :: ' + str(leftCheekMedianFullHSV))
        print('MEDIAN LEFT FULL LUMINANCE :: ' + str(leftCheekMedianFullLuminance))
        print('~~~')
        print('RIGHT FLUXISH :: ' + str(scaledRightFluxish))
        print('MEDIAN FullHSV RIGHT Full Points :: ' + str(rightCheekMedianFullHSV))
        print('MEDIAN RIGHT FULL LUMINANCE :: ' + str(rightCheekMedianFullLuminance))
        print('~~~')
        print('MEDIAN FullHSV CHIN Full Points :: ' + str(chinMedianFullHSV))
        print('MEDIAN CHIN FULLLUMINANCE :: ' + str(chinMedianFullLuminance))
        print('~~~')
        print('MEDIAN FullHSV RIGHT Full Points :: ' + str(foreheadMedianFullHSV))
        print('MEDIAN FOREHEAD FULLLUMINANCE :: ' + str(foreheadMedianFullLuminance))
        print('---------------------')

        #chinFullHSV[:, 1] = applyLinearAdjustment(chinFullHSV[:, 1], chinFullHSV[:, 0])
        #chinFullHSV[:, 1] = adjustSatToHue(chinFullHSV[:, 1], chinFullHSV[:, 0])
        #FULL FLASH
        #[chinFullHSV, foreheadFullHSV] = adjustSatToHue(chinFullHSV, foreheadFullHSV)

        leftCheekFull = [leftCheekFullHSV, leftCheekFullLuminance]
        rightCheekFull = [rightCheekFullHSV, rightCheekFullLuminance]
        chinFull = [chinFullHSV, chinFullLuminance]
        foreheadFull = [foreheadFullHSV, foreheadFullLuminance]

        [leftCheekLineFull, rightCheekLineFull, chinLineFull, foreheadLineFull] = plotZones(leftCheekFull, rightCheekFull, chinFull, foreheadFull, saveStep, '_full_linear')

        #HALF FLASH
        #[chinHalfHSV, foreheadHalfHSV] = adjustSatToHue(chinHalfHSV, foreheadHalfHSV)

        leftCheekHalf = [leftCheekHalfHSV, leftCheekHalfLuminance]
        rightCheekHalf = [rightCheekHalfHSV, rightCheekHalfLuminance]
        chinHalf = [chinHalfHSV, chinHalfLuminance]
        foreheadHalf = [foreheadHalfHSV, foreheadHalfLuminance]

        [leftCheekLineHalf, rightCheekLineHalf, chinLineHalf, foreheadLineHalf] = plotZones(leftCheekHalf, rightCheekHalf, chinHalf, foreheadHalf, saveStep, '_half_linear')

        #TEST ALL POINTS

        leftCheekHalf = [np.append(leftCheekHalfHSV, leftCheekFullHSV, axis=0), np.append(leftCheekHalfLuminance, leftCheekFullLuminance, axis=0)]
        rightCheekHalf = [np.append(rightCheekHalfHSV, rightCheekFullHSV, axis=0), np.append(rightCheekHalfLuminance, rightCheekFullLuminance, axis=0)]
        chinHalf = [np.append(chinHalfHSV, chinFullHSV, axis=0), np.append(chinHalfLuminance, chinFullLuminance, axis=0)]
        foreheadHalf = [np.append(foreheadHalfHSV, foreheadFullHSV, axis=0), np.append(foreheadHalfLuminance, foreheadFullLuminance, axis=0)]

        [leftCheekLineHalf, rightCheekLineHalf, chinLineHalf, foreheadLineHalf] = plotZones(leftCheekHalf, rightCheekHalf, chinHalf, foreheadHalf, saveStep, '_all_linear')

        #RECALCULATE IN FULL in sBGR

        #print('Scaled Points Left Cheek :: ' + str(scaledFullPointsLeftCheek))

        #scaledFullPointsLeftCheek_sBGR = colorTools.convert_linearBGR_float_to_sBGR(scaledFullPointsLeftCheek / 255)
        #scaledFullPointsRightCheek_sBGR = colorTools.convert_linearBGR_float_to_sBGR(scaledFullPointsRightCheek / 255)
        #scaledFullPointsChin_sBGR = colorTools.convert_linearBGR_float_to_sBGR(scaledFullPointsChin / 255)
        #scaledFullPointsForehead_sBGR = colorTools.convert_linearBGR_float_to_sBGR(scaledFullPointsForehead / 255)

        #print('Scaled Points Left Cheek sBGR :: ' + str(scaledFullPointsLeftCheek_sBGR))

        #[leftCheekHSV_sBGR, leftCheekMedianHSV_sBGR, leftCheekLuminance_sBGR, leftCheekMedianLuminance_sBGR] = convertPoints(scaledFullPointsLeftCheek_sBGR)
        #[rightCheekHSV_sBGR, rightCheekMedianHSV_sBGR, rightCheekLuminance_sBGR, rightCheekMedianLuminance_sBGR] = convertPoints(scaledFullPointsRightCheek_sBGR)
        #[chinHSV_sBGR, chinMedianHSV_sBGR, chinLuminance_sBGR, chinMedianLuminance_sBGR] = convertPoints(scaledFullPointsChin_sBGR)
        #[foreheadHSV_sBGR, foreheadMedianHSV_sBGR, foreheadLuminance_sBGR, foreheadMedianLuminance_sBGR] = convertPoints(scaledFullPointsForehead_sBGR)

        #leftCheek_sBGR = [leftCheekHSV_sBGR, leftCheekLuminance_sBGR]
        #rightCheek_sBGR = [rightCheekHSV_sBGR, rightCheekLuminance_sBGR]
        #chin_sBGR = [chinHSV_sBGR, chinLuminance_sBGR]
        #forehead_sBGR = [foreheadHSV_sBGR, foreheadLuminance_sBGR]

        #[leftCheekLine_sBGR, rightCheekLine_sBGR, chinLine_sBGR, foreheadLine_sBGR] = plotZones(leftCheek_sBGR, rightCheek_sBGR, chin_sBGR, forehead_sBGR, saveStep, '_sBGR')

        #PREP RETURN SBGR
        #leftCheekValues = [scaledLeftFluxish, leftCheekMedianLuminance_sBGR, list(leftCheekMedianHSV_sBGR), list(leftCheekLine_sBGR)]
        #rightCheekValues = [scaledRightFluxish, rightCheekMedianLuminance_sBGR, list(rightCheekMedianHSV_sBGR), list(rightCheekLine_sBGR)]
        #chinValues = [scaledAverageFluxish, chinMedianLuminance_sBGR, list(chinMedianHSV_sBGR), list(chinLine_sBGR)]
        #foreheadValues = [scaledAverageFluxish, foreheadMedianLuminance_sBGR, list(foreheadMedianHSV_sBGR), list(foreheadLine_sBGR)]

        #PREP FULL RETURN
        #leftCheekValuesFull = [scaledLeftFluxish, leftCheekMedianFullLuminance, list(leftCheekMedianFullHSV), list(leftCheekMedianFullBGR), list(leftCheekLineFull)]
        #rightCheekValuesFull = [scaledRightFluxish, rightCheekMedianFullLuminance, list(rightCheekMedianFullHSV), list(rightCheekMedianFullBGR), list(rightCheekLineFull)]
        #chinValuesFull = [scaledAverageFluxish, chinMedianFullLuminance, list(chinMedianFullHSV), list(chinMedianFullBGR), list(chinLineFull)]
        #foreheadValuesFull = [scaledAverageFluxish, foreheadMedianFullLuminance, list(foreheadMedianFullHSV), list(foreheadMedianFullBGR), list(foreheadLineFull)]

        #PREP HALF RETURN
        #leftCheekValuesHalf = [scaledLeftFluxish / 2, leftCheekMedianHalfLuminance, list(leftCheekMedianHalfHSV), list(leftCheekMedianHalfBGR), list(leftCheekLineHalf)]
        #rightCheekValuesHalf = [scaledRightFluxish / 2, rightCheekMedianHalfLuminance, list(rightCheekMedianHalfHSV), list(rightCheekMedianHalfBGR), list(rightCheekLineHalf)]
        #chinValuesHalf = [scaledAverageFluxish / 2, chinMedianHalfLuminance, list(chinMedianHalfHSV), list(chinMedianHalfBGR), list(chinLineHalf)]
        #foreheadValuesHalf = [scaledAverageFluxish / 2, foreheadMedianHalfLuminance, list(foreheadMedianHalfHSV), list(foreheadMedianHalfBGR), list(foreheadLineHalf)]

        #return [imageName, True, [leftCheekValuesHalf, rightCheekValuesHalf, chinValuesHalf, foreheadValuesHalf], [leftCheekValuesFull, rightCheekValuesFull, chinValuesFull, foreheadValuesFull], [leftCheekLinearityError, rightCheekLinearityError, chinLinearityError, foreheadLinearityError], [leftCheekClippingRatio, rightCheekClippingRatio, chinClippingRatio, foreheadClippingRatio], [list(noFlashPointsLeftCheekMedian), list(noFlashPointsRightCheekMedian), list(noFlashPointsChinMedian), list(noFlashPointsForeheadMedian)], [list(leftReflectionValues), list(rightReflectionValues)]]

        #NEW RULES: COLORS ARE RETURNED IN BGR
        #           FIELDS are Left Cheek, Right Cheek, Chin Forehead
        noFlashValues = [noFlashPointsLeftCheekMedian, noFlashPointsRightCheekMedian, noFlashPointsChinMedian, noFlashPointsForeheadMedian]
        halfFlashValues = [leftCheekMedianHalfBGR, rightCheekMedianHalfBGR, chinMedianHalfBGR, foreheadMedianHalfBGR]
        fullFlashValues = [leftCheekMedianFullBGR, rightCheekMedianFullBGR, chinMedianFullBGR, foreheadMedianFullBGR]
        linearity = [leftCheekLinearityError, rightCheekLinearityError, chinLinearityError, foreheadLinearityError]
        cleanRatio = [leftCheekClippingRatio, rightCheekClippingRatio, chinClippingRatio, foreheadClippingRatio]
        reflectionValues = [leftReflectionValues, rightReflectionValues]
        fluxishValues = [scaledLeftFluxish, scaledRightFluxish, scaledAverageFluxish, scaledAverageFluxish]


        return getResponse(imageName, True, noFlashValues, halfFlashValues, fullFlashValues, linearity, cleanRatio, reflectionValues, fluxishValues)


        #return [0, 1, 2, 3, 4, 5, 6, 7]

