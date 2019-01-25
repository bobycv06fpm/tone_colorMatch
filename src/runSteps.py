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

def run(username, imageName, fast=False, saveStats=False, failOnError=False):
    #saveStep.resetLogFile(username, imageName)
    saveStep = Save(username, imageName)
    saveStep.resetLogFile()
    #saveStep.deleteReference()
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
        #fullPointsLeftCheek = fullPointsLeftCheek / leftFluxish
        fullPointsLeftCheekLuminance = colorTools.getRelativeLuminance(fullPointsLeftCheek)
        fullPointsLeftCheekIntensity = np.mean(fullPointsLeftCheek, axis=1)
        fullPointsLeftCheekMedianLuminance = np.median(fullPointsLeftCheekLuminance)

        fullPointsLeftCheek_RGB = np.flip(fullPointsLeftCheek, axis=1)
        fullPointsLeftCheek_RGB = fullPointsLeftCheek_RGB / 255
        #fullPointsLeftCheek_hls = np.array([list(colorsys.rgb_to_hls(r, g, b)) for [r, g, b] in fullPointsLeftCheek_RGB])
        fullPointsLeftCheek_hsv = np.array([list(colorsys.rgb_to_hsv(r, g, b)) for [r, g, b] in fullPointsLeftCheek_RGB])

        fullPointsLeftCheekMedian_hsv = np.median(fullPointsLeftCheek_hsv, axis=0)
        print('---------------------')
        print('LEFT FLUXISH :: ' + str(leftFluxish))
        print('MEDIAN HSV LEFT Full Points :: ' + str(fullPointsLeftCheekMedian_hsv))
        print('MEDIAN LEFT LUMINANCE :: ' + str(fullPointsLeftCheekMedianLuminance))

        print('~~~')

        #fullPointsRightCheek = fullPointsRightCheek / leftFluxish
        fullPointsRightCheekLuminance = colorTools.getRelativeLuminance(fullPointsRightCheek)
        fullPointsRightCheekIntensity = np.mean(fullPointsRightCheek, axis=1)
        fullPointsRightCheekMedianLuminance = np.median(fullPointsRightCheekLuminance)

        fullPointsRightCheek_RGB = np.flip(fullPointsRightCheek, axis=1)
        fullPointsRightCheek_RGB = fullPointsRightCheek_RGB / 255
        #fullPointsRightCheek_hls = np.array([list(colorsys.rgb_to_hls(r, g, b)) for [r, g, b] in fullPointsRightCheek_RGB])
        fullPointsRightCheek_hsv = np.array([list(colorsys.rgb_to_hsv(r, g, b)) for [r, g, b] in fullPointsRightCheek_RGB])

        fullPointsRightCheekMedian_hsv = np.median(fullPointsRightCheek_hsv, axis=0)
        print('RIGHT FLUXISH :: ' + str(rightFluxish))
        print('MEDIAN HSV RIGHT Full Points :: ' + str(fullPointsRightCheekMedian_hsv))
        print('MEDIAN RIGHT LUMINANCE :: ' + str(fullPointsRightCheekMedianLuminance))

        print('~~~')

        #fullPointsChin = fullPointsChin / averageFluxish
        fullPointsChinLuminance = colorTools.getRelativeLuminance(fullPointsChin)
        fullPointsChinIntensity = np.mean(fullPointsChin, axis=1)
        fullPointsChinMedianLuminance = np.median(fullPointsChinLuminance)

        fullPointsChin_RGB = np.flip(fullPointsChin, axis=1)
        fullPointsChin_RGB = fullPointsChin_RGB / 255
        #fullPointsChin_hls = np.array([list(colorsys.rgb_to_hls(r, g, b)) for [r, g, b] in fullPointsChin_RGB])
        fullPointsChin_hsv = np.array([list(colorsys.rgb_to_hsv(r, g, b)) for [r, g, b] in fullPointsChin_RGB])

        fullPointsChinMedian_hsv = np.median(fullPointsChin_hsv, axis=0)
        print('MEDIAN HSV CHIN Full Points :: ' + str(fullPointsChinMedian_hsv))
        print('MEDIAN CHIN LUMINANCE :: ' + str(fullPointsChinMedianLuminance))

        print('~~~')

        #fullPointsForehead = fullPointsForehead / averageFluxish
        fullPointsForeheadLuminance = colorTools.getRelativeLuminance(fullPointsForehead)
        fullPointsForeheadIntensity = np.mean(fullPointsForehead, axis=1)
        fullPointsForeheadMedianLuminance = np.median(fullPointsForeheadLuminance)

        fullPointsForehead_RGB = np.flip(fullPointsForehead, axis=1)
        fullPointsForehead_RGB = fullPointsForehead_RGB / 255
        #fullPointsForehead_hls = np.array([list(colorsys.rgb_to_hls(r, g, b)) for [r, g, b] in fullPointsForehead_RGB])
        fullPointsForehead_hsv = np.array([list(colorsys.rgb_to_hsv(r, g, b)) for [r, g, b] in fullPointsForehead_RGB])


        fullPointsForeheadMedian_hsv = np.median(fullPointsForehead_hsv, axis=0)
        print('MEDIAN HSV RIGHT Full Points :: ' + str(fullPointsForeheadMedian_hsv))
        print('MEDIAN FOREHEAD LUMINANCE :: ' + str(fullPointsForeheadMedianLuminance))
        print('---------------------')

        fig, axs = plt.subplots(4, 3, sharey=False, tight_layout=True)
        size = 5

        #Luminance
        axs[0, 0].scatter(fullPointsLeftCheekLuminance, fullPointsLeftCheek_hsv[:, 1], size, (1, 0, 0))
        leftCheekLine = fitLine(fullPointsLeftCheekLuminance, fullPointsLeftCheek_hsv[:, 1])

        axs[1, 0].scatter(fullPointsRightCheekLuminance, fullPointsRightCheek_hsv[:, 1], size, (1, 0, 0))
        rightCheekLine = fitLine(fullPointsRightCheekLuminance, fullPointsRightCheek_hsv[:, 1])

        axs[2, 0].scatter(fullPointsChinLuminance, fullPointsChin_hsv[:, 1], size, (1, 0, 0))
        chinLine = fitLine(fullPointsChinLuminance, fullPointsChin_hsv[:, 1])

        axs[3, 0].scatter(fullPointsForeheadLuminance, fullPointsForehead_hsv[:, 1], size, (1, 0, 0))
        foreheadLine = fitLine(fullPointsForeheadLuminance, fullPointsForehead_hsv[:, 1])

        print('Lines :: ' + str(leftCheekLine) + ' | ' + str(rightCheekLine) + ' | ' + str(chinLine) + ' | ' + str(foreheadLine))

        #Value
        #axs[0, 0].scatter(fullPointsLeftCheek_hsv[:, 2], fullPointsLeftCheek_hsv[:, 1], size, (1, 0, 0))
        #axs[1, 0].scatter(fullPointsRightCheek_hsv[:, 2], fullPointsRightCheek_hsv[:, 1], size, (1, 0, 0))
        #axs[2, 0].scatter(fullPointsChin_hsv[:, 2], fullPointsChin_hsv[:, 1], size, (1, 0, 0))
        #axs[3, 0].scatter(fullPointsForehead_hsv[:, 2], fullPointsForehead_hsv[:, 1], size, (1, 0, 0))

        #Intensity
        #axs[0, 0].scatter(fullPointsLeftCheekIntensity, fullPointsLeftCheek_hsv[:, 1], size, (1, 0, 0))
        #axs[1, 0].scatter(fullPointsRightCheekIntensity, fullPointsRightCheek_hsv[:, 1], size, (1, 0, 0))
        #axs[2, 0].scatter(fullPointsChinIntensity, fullPointsChin_hsv[:, 1], size, (1, 0, 0))
        #axs[3, 0].scatter(fullPointsForeheadIntensity, fullPointsForehead_hsv[:, 1], size, (1, 0, 0))

        #Luminance
        axs[0, 1].scatter(fullPointsLeftCheekLuminance, rotateHue(fullPointsLeftCheek_hsv[:, 0]), size, (1, 0, 0))
        axs[1, 1].scatter(fullPointsRightCheekLuminance, rotateHue(fullPointsRightCheek_hsv[:, 0]), size, (1, 0, 0))
        axs[2, 1].scatter(fullPointsChinLuminance, rotateHue(fullPointsChin_hsv[:, 0]), size, (1, 0, 0))
        axs[3, 1].scatter(fullPointsForeheadLuminance, rotateHue(fullPointsForehead_hsv[:, 0]), size, (1, 0, 0))

        #Value
        #axs[0, 1].scatter(fullPointsLeftCheek_hsv[:, 2], np.clip(fullPointsLeftCheek_hsv[:, 0], 0, 0.1), size, (1, 0, 0))
        #axs[1, 1].scatter(fullPointsRightCheek_hsv[:, 2], np.clip(fullPointsRightCheek_hsv[:, 0], 0, 0.1), size, (1, 0, 0))
        #axs[2, 1].scatter(fullPointsChin_hsv[:, 2], np.clip(fullPointsChin_hsv[:, 0], 0, 0.1), size, (1, 0, 0))
        #axs[3, 1].scatter(fullPointsForehead_hsv[:, 2], np.clip(fullPointsForehead_hsv[:, 0], 0, 0.1), size, (1, 0, 0))

        #Intensity
        #axs[0, 1].scatter(fullPointsLeftCheekIntensity, np.clip(fullPointsLeftCheek_hsv[:, 0], 0, 0.1), size, (1, 0, 0))
        #axs[1, 1].scatter(fullPointsRightCheekIntensity, np.clip(fullPointsRightCheek_hsv[:, 0], 0, 0.1), size, (1, 0, 0))
        #axs[2, 1].scatter(fullPointsChinIntensity, np.clip(fullPointsChin_hsv[:, 0], 0, 0.1), size, (1, 0, 0))
        #axs[3, 1].scatter(fullPointsForeheadIntensity, np.clip(fullPointsForehead_hsv[:, 0], 0, 0.1), size, (1, 0, 0))

        axs[0, 2].scatter(rotateHue(fullPointsLeftCheek_hsv[:, 0]), fullPointsLeftCheek_hsv[:, 1], size, (1, 0, 0))
        axs[1, 2].scatter(rotateHue(fullPointsRightCheek_hsv[:, 0]), fullPointsRightCheek_hsv[:, 1], size, (1, 0, 0))
        axs[2, 2].scatter(rotateHue(fullPointsChin_hsv[:, 0]), fullPointsChin_hsv[:, 1], size, (1, 0, 0))
        axs[3, 2].scatter(rotateHue(fullPointsForehead_hsv[:, 0]), fullPointsForehead_hsv[:, 1], size, (1, 0, 0))

        #plt.show()
        saveStep.savePlot('Luminance_Hue_Saturation_Scatter', plt)
        #saveStep.savePlot('Value_Hue_Saturation_Scatter', plt)
        #saveStep.savePlot('Intensity_Hue_Saturation_Scatter', plt)

        bins = 50
        fig, axs = plt.subplots(4, 3, sharey=False, tight_layout=True)
        axs[0, 0].hist(fullPointsLeftCheekLuminance, bins=bins)
        axs[1, 0].hist(fullPointsRightCheekLuminance, bins=bins)
        axs[2, 0].hist(fullPointsChinLuminance, bins=bins)
        axs[3, 0].hist(fullPointsForeheadLuminance, bins=bins)

        axs[0, 2].hist(rotateHue(fullPointsLeftCheek_hsv[:, 0]), bins=bins)   #Watch for clipping...
        axs[1, 2].hist(rotateHue(fullPointsRightCheek_hsv[:, 0]), bins=bins)
        axs[2, 2].hist(rotateHue(fullPointsChin_hsv[:, 0]), bins=bins)   #Watch for clipping...
        axs[3, 2].hist(rotateHue(fullPointsForehead_hsv[:, 0]), bins=bins)

        axs[0, 1].hist(fullPointsLeftCheek_hsv[:, 1], bins=bins)
        axs[1, 1].hist(fullPointsRightCheek_hsv[:, 1], bins=bins)
        axs[2, 1].hist(fullPointsChin_hsv[:, 1], bins=bins)
        axs[3, 1].hist(fullPointsForehead_hsv[:, 1], bins=bins)
        #plt.show()
        saveStep.savePlot('Luminance_Hue_Saturation_hist', plt)

        leftCheekValues = [leftFluxish, fullPointsLeftCheekMedianLuminance, list(fullPointsLeftCheekMedian_hsv), list(leftCheekLine)]
        rightCheekValues = [rightFluxish, fullPointsRightCheekMedianLuminance, list(fullPointsRightCheekMedian_hsv), list(rightCheekLine)]
        chinValues = [averageFluxish, fullPointsChinMedianLuminance, list(fullPointsChinMedian_hsv), list(chinLine)]
        foreheadValues = [averageFluxish, fullPointsForeheadMedianLuminance, list(fullPointsForeheadMedian_hsv), list(foreheadLine)]

        return [imageName, True, [leftCheekValues, rightCheekValues, chinValues, foreheadValues]]



