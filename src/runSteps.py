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
    shiftMask = hue < 2/3
    hue[shiftMask] += 1/3
    hue[np.logical_not(shiftMask)] -= 2/3
    return hue

def plotZones(leftCheek, rightCheek, chin, forehead, saveStep, tag=''):
    [leftCheek_hsv, leftCheekLuminance] = leftCheek
    [rightCheek_hsv, rightCheekLuminance] = rightCheek
    [chin_hsv, chinLuminance] = chin
    [forehead_hsv, foreheadLuminance] = forehead

    fig, axs = plt.subplots(4, 3, sharey=False, tight_layout=True)
    size = 5

    #Luminance
    axs[0, 0].scatter(leftCheekLuminance, leftCheek_hsv[:, 1], size, (1, 0, 0))
    leftCheekLine = fitLine(leftCheekLuminance, leftCheek_hsv[:, 1])

    axs[1, 0].scatter(rightCheekLuminance, rightCheek_hsv[:, 1], size, (1, 0, 0))
    rightCheekLine = fitLine(rightCheekLuminance, rightCheek_hsv[:, 1])

    axs[2, 0].scatter(chinLuminance, chin_hsv[:, 1], size, (1, 0, 0))
    chinLine = fitLine(chinLuminance, chin_hsv[:, 1])

    axs[3, 0].scatter(foreheadLuminance, forehead_hsv[:, 1], size, (1, 0, 0))
    foreheadLine = fitLine(foreheadLuminance, forehead_hsv[:, 1])

    print('Lines :: ' + str(leftCheekLine) + ' | ' + str(rightCheekLine) + ' | ' + str(chinLine) + ' | ' + str(foreheadLine))

    #Value
    #axs[0, 0].scatter(leftCheek_hsv[:, 2], leftCheek_hsv[:, 1], size, (1, 0, 0))
    #axs[1, 0].scatter(rightCheek_hsv[:, 2], rightCheek_hsv[:, 1], size, (1, 0, 0))
    #axs[2, 0].scatter(chin_hsv[:, 2], chin_hsv[:, 1], size, (1, 0, 0))
    #axs[3, 0].scatter(forehead_hsv[:, 2], forehead_hsv[:, 1], size, (1, 0, 0))

    #Intensity
    #axs[0, 0].scatter(leftCheekIntensity, leftCheek_hsv[:, 1], size, (1, 0, 0))
    #axs[1, 0].scatter(rightCheekIntensity, rightCheek_hsv[:, 1], size, (1, 0, 0))
    #axs[2, 0].scatter(chinIntensity, chin_hsv[:, 1], size, (1, 0, 0))
    #axs[3, 0].scatter(foreheadIntensity, forehead_hsv[:, 1], size, (1, 0, 0))

    #Luminance
    axs[0, 1].scatter(leftCheekLuminance, rotateHue(leftCheek_hsv[:, 0]), size, (1, 0, 0))
    axs[1, 1].scatter(rightCheekLuminance, rotateHue(rightCheek_hsv[:, 0]), size, (1, 0, 0))
    axs[2, 1].scatter(chinLuminance, rotateHue(chin_hsv[:, 0]), size, (1, 0, 0))
    axs[3, 1].scatter(foreheadLuminance, rotateHue(forehead_hsv[:, 0]), size, (1, 0, 0))

    #Value
    #axs[0, 1].scatter(leftCheek_hsv[:, 2], np.clip(leftCheek_hsv[:, 0], 0, 0.1), size, (1, 0, 0))
    #axs[1, 1].scatter(rightCheek_hsv[:, 2], np.clip(rightCheek_hsv[:, 0], 0, 0.1), size, (1, 0, 0))
    #axs[2, 1].scatter(chin_hsv[:, 2], np.clip(chin_hsv[:, 0], 0, 0.1), size, (1, 0, 0))
    #axs[3, 1].scatter(forehead_hsv[:, 2], np.clip(forehead_hsv[:, 0], 0, 0.1), size, (1, 0, 0))

    #Intensity
    #axs[0, 1].scatter(leftCheekIntensity, np.clip(leftCheek_hsv[:, 0], 0, 0.1), size, (1, 0, 0))
    #axs[1, 1].scatter(rightCheekIntensity, np.clip(rightCheek_hsv[:, 0], 0, 0.1), size, (1, 0, 0))
    #axs[2, 1].scatter(chinIntensity, np.clip(chin_hsv[:, 0], 0, 0.1), size, (1, 0, 0))
    #axs[3, 1].scatter(foreheadIntensity, np.clip(forehead_hsv[:, 0], 0, 0.1), size, (1, 0, 0))

    axs[0, 2].scatter(rotateHue(leftCheek_hsv[:, 0]), leftCheek_hsv[:, 1], size, (1, 0, 0))
    axs[1, 2].scatter(rotateHue(rightCheek_hsv[:, 0]), rightCheek_hsv[:, 1], size, (1, 0, 0))
    axs[2, 2].scatter(rotateHue(chin_hsv[:, 0]), chin_hsv[:, 1], size, (1, 0, 0))
    axs[3, 2].scatter(rotateHue(forehead_hsv[:, 0]), forehead_hsv[:, 1], size, (1, 0, 0))

    #plt.show()
    saveStep.savePlot('Luminance_Hue_Saturation_Scatter' + tag, plt)
    #saveStep.savePlot('Value_Hue_Saturation_Scatter', plt)
    #saveStep.savePlot('Intensity_Hue_Saturation_Scatter', plt)

    bins = 50
    fig, axs = plt.subplots(4, 3, sharey=False, tight_layout=True)
    axs[0, 0].hist(leftCheekLuminance, bins=bins)
    axs[1, 0].hist(rightCheekLuminance, bins=bins)
    axs[2, 0].hist(chinLuminance, bins=bins)
    axs[3, 0].hist(foreheadLuminance, bins=bins)

    axs[0, 2].hist(rotateHue(leftCheek_hsv[:, 0]), bins=bins)   #Watch for clipping...
    axs[1, 2].hist(rotateHue(rightCheek_hsv[:, 0]), bins=bins)
    axs[2, 2].hist(rotateHue(chin_hsv[:, 0]), bins=bins)   #Watch for clipping...
    axs[3, 2].hist(rotateHue(forehead_hsv[:, 0]), bins=bins)

    axs[0, 1].hist(leftCheek_hsv[:, 1], bins=bins)
    axs[1, 1].hist(rightCheek_hsv[:, 1], bins=bins)
    axs[2, 1].hist(chin_hsv[:, 1], bins=bins)
    axs[3, 1].hist(forehead_hsv[:, 1], bins=bins)
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

    return [HSV, medianHSV, luminance, medianLuminance]


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
            return [imageName, False]
        
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
    halfDiffImage = halfFlashCapture.image.astype('int32') - noFlashCapture.image.astype('int32')
    #halfDiffImageBlur = cv2.GaussianBlur(halfDiffImage, (25, 25), 0)
    #halfDiffImageBlur = cv2.medianBlur(halfDiffImage, 9)

    fullDiffImage = fullFlashCapture.image.astype('int32') - noFlashCapture.image.astype('int32')
    #fullDiffImageBlur = cv2.GaussianBlur(fullDiffImage, (25, 25), 0)
    #fullDiffImageBlur = cv2.medianBlur(fullDiffImage, 9)

    howLinear = np.abs((2 * halfDiffImage) - fullDiffImage)
    fullDiffImage[fullDiffImage == 0] = 1
    percentError = howLinear / fullDiffImage
    perSubPixelMaxError = np.max(percentError, axis=2)
    nonLinearMask = perSubPixelMaxError > .10
    #nonLinearMask = perSubPixelMaxError > .05

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
        [reflectionValue, leftFluxish, rightFluxish] = getAverageScreenReflectionColor(noFlashCapture, halfFlashCapture, fullFlashCapture, saveStep)
    except Exception as err:
        if failOnError:
            raise NameError('User :: {} | Image :: {} | Error :: {} | Details :: {}'.format(username, imageName, 'Error Extracting Reflection', err))
        else:
            print('User :: {} | Image :: {} | Error :: {} | Details :: {}'.format(username, imageName, 'Error Extracting Reflection', err))
            return [imageName, False]

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
        [fullPoints, fullPointsLeftCheek, fullPointsRightCheek, fullPointsChin, fullPointsForehead] = extractMask(fullDiffCapture, saveStep)
        [halfPoints, halfPointsLeftCheek, halfPointsRightCheek, halfPointsChin, halfPointsForehead] = extractMask(halfDiffCapture, saveStep)
    except Exception as err:
        if failOnError:
            raise NameError('User :: {} | Image :: {} | Error :: {} | Details :: {}'.format(username, imageName, 'Error extracting Points for Recovered Mask', err))
        else:
            print('User :: {} | Image :: {} | Error :: {} | Details :: {}'.format(username, imageName, 'Error extracting Points for Recovered Mask', err))
            return [imageName, False]
    else:

        largestValue = np.max(fullPoints)
        print('LARGEST VALUE :: ' + str(largestValue))

        scaleDivisor = largestValue / 255
        scaledLeftFluxish = leftFluxish / scaleDivisor
        scaledRightFluxish = rightFluxish / scaleDivisor
        scaledAverageFluxish = averageFluxish / scaleDivisor

        scaledPointsLeftCheek = fullPointsLeftCheek / scaleDivisor
        scaledPointsRightCheek = fullPointsRightCheek / scaleDivisor
        scaledPointsChin = fullPointsChin / scaleDivisor
        scaledPointsForehead = fullPointsForehead / scaleDivisor

        print('Unscaled :: ' + str(fullPointsLeftCheek))
        print('Scaled :: ' + str(scaledPointsLeftCheek))


        #CALCULATE IN LINEAR

        [leftCheekHSV, leftCheekMedianHSV, leftCheekLuminance, leftCheekMedianLuminance] = convertPoints(scaledPointsLeftCheek)
        [rightCheekHSV, rightCheekMedianHSV, rightCheekLuminance, rightCheekMedianLuminance] = convertPoints(scaledPointsRightCheek)
        [chinHSV, chinMedianHSV, chinLuminance, chinMedianLuminance] = convertPoints(scaledPointsChin)
        [foreheadHSV, foreheadMedianHSV, foreheadLuminance, foreheadMedianLuminance] = convertPoints(scaledPointsForehead)

        print('---------------------')
        print('LEFT FLUXISH :: ' + str(scaledLeftFluxish))
        print('MEDIAN HSV LEFT Full Points :: ' + str(leftCheekMedianHSV))
        print('MEDIAN LEFT LUMINANCE :: ' + str(leftCheekMedianLuminance))
        print('~~~')
        print('RIGHT FLUXISH :: ' + str(scaledRightFluxish))
        print('MEDIAN HSV RIGHT Full Points :: ' + str(rightCheekMedianHSV))
        print('MEDIAN RIGHT LUMINANCE :: ' + str(rightCheekMedianLuminance))
        print('~~~')
        print('MEDIAN HSV CHIN Full Points :: ' + str(chinMedianHSV))
        print('MEDIAN CHIN LUMINANCE :: ' + str(chinMedianLuminance))
        print('~~~')
        print('MEDIAN HSV RIGHT Full Points :: ' + str(foreheadMedianHSV))
        print('MEDIAN FOREHEAD LUMINANCE :: ' + str(foreheadMedianLuminance))
        print('---------------------')

        leftCheek = [leftCheekHSV, leftCheekLuminance]
        rightCheek = [rightCheekHSV, rightCheekLuminance]
        chin = [chinHSV, chinLuminance]
        forehead = [foreheadHSV, foreheadLuminance]

        [leftCheekLine, rightCheekLine, chinLine, foreheadLine] = plotZones(leftCheek, rightCheek, chin, forehead, saveStep, '_linear')

        #RECALCULATE IN sBGR

        print('Scaled Points Left Cheek :: ' + str(scaledPointsLeftCheek))

        scaledPointsLeftCheek_sBGR = colorTools.convert_linearBGR_float_to_sBGR(scaledPointsLeftCheek / 255)
        scaledPointsRightCheek_sBGR = colorTools.convert_linearBGR_float_to_sBGR(scaledPointsRightCheek / 255)
        scaledPointsChin_sBGR = colorTools.convert_linearBGR_float_to_sBGR(scaledPointsChin / 255)
        scaledPointsForehead_sBGR = colorTools.convert_linearBGR_float_to_sBGR(scaledPointsForehead / 255)

        print('Scaled Points Left Cheek sBGR :: ' + str(scaledPointsLeftCheek_sBGR))

        [leftCheekHSV_sBGR, leftCheekMedianHSV_sBGR, leftCheekLuminance_sBGR, leftCheekMedianLuminance_sBGR] = convertPoints(scaledPointsLeftCheek_sBGR)
        [rightCheekHSV_sBGR, rightCheekMedianHSV_sBGR, rightCheekLuminance_sBGR, rightCheekMedianLuminance_sBGR] = convertPoints(scaledPointsRightCheek_sBGR)
        [chinHSV_sBGR, chinMedianHSV, chinLuminance_sBGR, chinMedianLuminance_sBGR] = convertPoints(scaledPointsChin_sBGR)
        [foreheadHSV_sBGR, foreheadMedianHSV_sBGR, foreheadLuminance_sBGR, foreheadMedianLuminance_sBGR] = convertPoints(scaledPointsForehead_sBGR)

        leftCheek_sBGR = [leftCheekHSV_sBGR, leftCheekLuminance_sBGR]
        rightCheek_sBGR = [rightCheekHSV_sBGR, rightCheekLuminance_sBGR]
        chin_sBGR = [chinHSV_sBGR, chinLuminance_sBGR]
        forehead_sBGR = [foreheadHSV_sBGR, foreheadLuminance_sBGR]

        [leftCheekLine_sBGR, rightCheekLine_sBGR, chinLine_sBGR, foreheadLine_sBGR] = plotZones(leftCheek_sBGR, rightCheek_sBGR, chin_sBGR, forehead_sBGR, saveStep, '_sBGR')

        #PREP RETURN
        leftCheekValues = [scaledLeftFluxish, leftCheekMedianLuminance, list(leftCheekMedianHSV), list(leftCheekLine)]
        rightCheekValues = [scaledRightFluxish, rightCheekMedianLuminance, list(rightCheekMedianHSV), list(rightCheekLine)]
        chinValues = [scaledAverageFluxish, chinMedianLuminance, list(chinMedianHSV), list(chinLine)]
        foreheadValues = [scaledAverageFluxish, foreheadMedianLuminance, list(foreheadMedianHSV), list(foreheadLine)]

        return [imageName, True, [leftCheekValues, rightCheekValues, chinValues, foreheadValues]]
