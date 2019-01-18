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

def run(username, imageName, fast=False, saveStats=False, failOnError=False):
    #saveStep.resetLogFile(username, imageName)
    saveStep = Save(username, imageName)
    saveStep.resetLogFile()
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

    fluxish = (leftFluxish + rightFluxish) / 2
    print("Reflection Value:: " + str(reflectionValue))
    print("Fluxish :: " + str(fluxish))
    #diffCapture.show()

    saveStep.saveReferenceImageBGR(fullDiffCapture.getClippedImage(), 'full_noWhitebalancedImage')
    saveStep.saveReferenceImageBGR(halfDiffCapture.getClippedImage(), 'half_noWhitebalancedImage')

    colorTools.whitebalanceBGR(fullDiffCapture, reflectionValue)
    colorTools.whitebalanceBGR(halfDiffCapture, reflectionValue)

    saveStep.saveReferenceImageBGR(fullDiffCapture.getClippedImage(), 'full_WhitebalancedImage')
    saveStep.saveReferenceImageBGR(halfDiffCapture.getClippedImage(), 'half_WhitebalancedImage')

    try:
        [fullPoints, fullPointsLeftCheek, fullPointsRightCheek] = extractMask(fullDiffCapture, saveStep)
        [halfPoints, halfPointsLeftCheek, halfPointsRightCheek] = extractMask(halfDiffCapture, saveStep)
    except Exception as err:
        if failOnError:
            raise NameError('User :: {} | Image :: {} | Error :: {} | Details :: {}'.format(username, imageName, 'Error extracting Points for Recovered Mask', err))
        else:
            print('User :: {} | Image :: {} | Error :: {} | Details :: {}'.format(username, imageName, 'Error extracting Points for Recovered Mask', err))
            return [imageName, False]
    else:
        #fullFaceMedian = np.median(fullPoints, axis=0)
        #halfFaceMedian = np.median(halfPoints, axis=0)
        #print('fullPointsLeftCheek :: ' + str(fullPointsLeftCheek))
        #print('fullPointsRightCheek :: ' + str(fullPointsRightCheek))

        #leftFluxish = leftFluxish / 100
        #rightFluxish = rightFluxish / 100

        fullPointsLeftCheekLuminance = colorTools.getRelativeLuminance(fullPointsLeftCheek)
        fullPointsLeftCheekMedianLuminance = np.median(fullPointsLeftCheekLuminance)

        fullPointsLeftCheek_RGB = np.flip(fullPointsLeftCheek, axis=1)
        fullPointsLeftCheek_RGB = fullPointsLeftCheek_RGB / 255
        fullPointsLeftCheek_hls = np.array([list(colorsys.rgb_to_hls(r, g, b)) for [r, g, b] in fullPointsLeftCheek_RGB])
        fullPointsLeftCheekMedian_hls = np.median(fullPointsLeftCheek_hls, axis=0)
        print('---------------------')
        print('LEFT FLUXISH :: ' + str(leftFluxish))
        print('MEDIAN HLS LEFT Full Points :: ' + str(fullPointsLeftCheekMedian_hls))
        print('MEDIAN LEFT LUMINANCE :: ' + str(fullPointsLeftCheekMedianLuminance))

        #colorsys.rgb_to_hls(fullPointsLeftCheek)
        #fullPointsLeftCheekMedianLuminance = np.max(fullPointsLeftCheekLuminance)

        #fullPointsLeftCheekMedian = np.median(fullPointsLeftCheek, axis=0)
        #fullPointsLeftCheekMedianHLS = colorsys.rgb_to_hls(fullPointsLeftCheekMedian[2] / 255, fullPointsLeftCheekMedian[1] / 255, fullPointsLeftCheekMedian[0] / 255)
        #leftLightness = fullPointsLeftCheekMedianHLS[1]
        #correctedFullPointsLeftCheekHLS = correctHLS(np.copy(fullPointsLeftCheekMedianHLS), leftFluxish)

        fullPointsRightCheekLuminance = colorTools.getRelativeLuminance(fullPointsRightCheek)
        fullPointsRightCheekMedianLuminance = np.median(fullPointsRightCheekLuminance)

        fullPointsRightCheek_RGB = np.flip(fullPointsRightCheek, axis=1)
        fullPointsRightCheek_RGB = fullPointsRightCheek_RGB / 255
        fullPointsRightCheek_hls = np.array([list(colorsys.rgb_to_hls(r, g, b)) for [r, g, b] in fullPointsRightCheek_RGB])
        fullPointsRightCheekMedian_hls = np.median(fullPointsRightCheek_hls, axis=0)
        print('~~~')
        print('RIGHT FLUXISH :: ' + str(rightFluxish))
        print('MEDIAN HLS RIGHT Full Points :: ' + str(fullPointsRightCheekMedian_hls))
        print('MEDIAN RIGHT LUMINANCE :: ' + str(fullPointsRightCheekMedianLuminance))
        print('---------------------')

        bins = 50
        fig, axs = plt.subplots(3, 2, sharey=True, tight_layout=True)
        axs[0, 0].hist(fullPointsLeftCheekLuminance, bins=bins)
        axs[0, 1].hist(fullPointsRightCheekLuminance, bins=bins)
        axs[1, 0].hist(np.clip(fullPointsLeftCheek_hls[:, 0], 0, 0.1), bins=bins)   #Watch for clipping...
        axs[1, 1].hist(np.clip(fullPointsRightCheek_hls[:, 0], 0, 0.1), bins=bins)
        axs[2, 0].hist(fullPointsLeftCheek_hls[:, 2], bins=bins)
        axs[2, 1].hist(fullPointsRightCheek_hls[:, 2], bins=bins)
        plt.show()

        #fullPointsRightCheekMedianLuminance = np.max(fullPointsRightCheekLuminance)

        #fullPointsRightCheekMedian = np.median(fullPointsRightCheek, axis=0)
        #fullPointsRightCheekMedianHLS = colorsys.rgb_to_hls(fullPointsRightCheekMedian[2] / 255, fullPointsRightCheekMedian[1] / 255, fullPointsRightCheekMedian[0] / 255)
        #rightLightness = fullPointsRightCheekMedianHLS[1]
        #correctedFullPointsRightCheekHLS = correctHLS(np.copy(fullPointsRightCheekMedianHLS), rightFluxish)


        #halfPointsLeftCheekLuminance = colorTools.getRelativeLuminance(halfPointsLeftCheek)
        #halfPointsLeftCheekMedianLuminance = np.median(halfPointsLeftCheekLuminance) * 2

        #halfPointsLeftCheekMedian = np.median(halfPointsLeftCheek, axis=0)
        #halfPointsLeftCheekMedianHLS = colorsys.rgb_to_hls(halfPointsLeftCheekMedian[2] / 255, halfPointsLeftCheekMedian[1] / 255, halfPointsLeftCheekMedian[0] / 255)
        #correctedHalfPointsLeftCheekHLS = correctHLS(np.copy(halfPointsLeftCheekMedianHLS), leftFluxish)

        #halfPointsRightCheekLuminance = colorTools.getRelativeLuminance(halfPointsRightCheek)
        #halfPointsRightCheekMedianLuminance = np.median(halfPointsRightCheekLuminance) * 2
        #halfPointsRightCheekMedian = np.median(halfPointsRightCheek, axis=0)
        #halfPointsRightCheekMedianHLS = colorsys.rgb_to_hls(halfPointsRightCheekMedian[2] / 255, halfPointsRightCheekMedian[1] / 255, halfPointsRightCheekMedian[0] / 255)
        #correctedHalfPointsRightCheekHLS = correctHLS(np.copy(halfPointsRightCheekMedianHLS), rightFluxish)

        #print('As Sampled Fluxish L Left vs Right  :: ' + str(leftFluxish) + " | " + str(rightFluxish))
        #print('As Sampled Full Cheek L Left vs Right  :: ' + str(fullPointsLeftCheekMedianLuminance) + " | " + str(fullPointsRightCheekMedianLuminance))
        #print('As Sampled Half Cheek L Left vs Right  :: ' + str(halfPointsLeftCheekMedianLuminance) + " | " + str(halfPointsRightCheekMedianLuminance))
        #print('Corrected Full Cheek L Left vs Right  :: ' + str(correctedFullPointsLeftCheekHLS[1]) + " | " + str(correctedFullPointsRightCheekHLS[1]))

        #saveStep.saveReferenceImageBGR(fullDiffCapture.image, 'full_noWhitebalancedImage')
        #saveStep.saveReferenceImageBGR(halfDiffCapture.image, 'half_noWhitebalancedImage')
        #fullPointsWB = colorTools.whitebalanceBGRPoints(fullPoints, reflectionValue)
        #halfPointsWB = colorTools.whitebalanceBGRPoints(halfPoints, reflectionValue)

        #fullFaceMedian = np.median(fullPointsWB, axis=0)
        #halfFaceMedian = np.median(halfPointsWB, axis=0)

        #fullPointsLeftCheekMedian = np.median(fullPointsLeftCheek)
        #fullFaceMedian = np.median(fullPoints, axis=0)
        #halfFaceMedian = np.median(halfPoints, axis=0)
        #saveStep.saveReferenceImageBGR(fullDiffCapture.image, 'full_WhitebalancedImage')
        #saveStep.saveReferenceImageBGR(halfDiffCapture.image, 'half_WhitebalancedImage')

        #fullPointsLeftCheekMedian = colorTools.whitebalanceBGRValue(fullPointsLeftCheekMedian, reflectionValue)
        #fullPointsRightCheekMedian = colorTools.whitebalanceBGRValue(fullPointsRightCheekMedian, reflectionValue)
        #halfPointsLeftCheekMedian = colorTools.whitebalanceBGRValue(halfPointsLeftCheekMedian, reflectionValue)
        #halfPointsRightCheekMedian = colorTools.whitebalanceBGRValue(halfPointsRightCheekMedian, reflectionValue)

        #plotTools.plotPoints(fullPoints)
        #plotTools.plotPoints(halfPoints)
        #both = np.array(list(fullPoints) + list(halfPoints))

        #r = both[:, 2]
        #g = both[:, 1]
        #b = both[:, 0]
        #max_b = max(b)

        #b_A = np.vstack([b, np.ones(len(b))]).T

        #gb = g/b
        #gb_median = np.median(gb)
        ##g_A = np.vstack([g, np.ones(len(g))]).T
        #g_m, g_c = np.linalg.lstsq(b_A, g, rcond=None)[0]


        #rb = r/b
        #rb_median = np.median(rb)
        ##r_A = np.vstack([r, np.ones(len(r))]).T
        #r_m, r_c = np.linalg.lstsq(b_A, r, rcond=None)[0]

        #print('\tNaive Ratios (G/B, R/B) :: ' + str((gb_median, rb_median)))
        #print('\tLSTSQR Coeficents (G/B, R/B) :: ' + str(((g_m, g_c), (r_m, r_c))))

        #green = np.arange(max_b) * gb_median
        #green_fit = (np.arange(max_b) * g_m) + g_c

        #red = np.arange(max_b) * rb_median
        #red_fit = (np.arange(max_b) * r_m) + r_c

        #blue = np.arange(max_b)

        #line = np.stack((blue, green, red), axis=-1)
        #line_fit = np.stack((blue, green_fit, red_fit), axis=-1)
        ##plotTools.plotPoints(both, [fullFaceMedian, halfFaceMedian])
        #plotTools.plotPoints(both, [line, line_fit])

        #print('full median face :: ' + str(fullFaceMedian))
        #print('half median face :: ' + str(halfFaceMedian))

        #sBGR_fullMedian = colorTools.convertSingle_linearValue_to_sValue(fullFaceMedian)
        #sBGR_halfMedian = colorTools.convertSingle_linearValue_to_sValue(halfFaceMedian)
        #print('full median face sBGR:: ' + str(sBGR_fullMedian))
        #print('half median face sBGR:: ' + str(sBGR_halfMedian))

        #fullMedianFacesHSV = list(colorsys.rgb_to_hsv(sBGR_fullMedian[2], sBGR_fullMedian[1], sBGR_fullMedian[0]))
        #fullMedianFacesHLS = list(colorsys.rgb_to_hls(sBGR_fullMedian[2] / 255, sBGR_fullMedian[1] / 255, sBGR_fullMedian[0] / 255))
        #halfMedianFacesHSV = list(colorsys.rgb_to_hsv(sBGR_halfMedian[2], sBGR_halfMedian[1], sBGR_halfMedian[0]))
        #halfMedianFacesHLS = list(colorsys.rgb_to_hls(sBGR_halfMedian[2]/ 255, sBGR_halfMedian[1] / 255, sBGR_halfMedian[0] / 255))

        #print('full median face sHSV :: ' + str(fullMedianFacesHSV))
        #print('full median face sHLS :: ' + str((np.array(fullMedianFacesHLS) * [360, 100, 100]).astype('int32')))
        #print('half median face sHSV :: ' + str(halfMedianFacesHSV))
        #print('half median face sHLS :: ' + str((np.array(halfMedianFacesHLS) * [360, 100, 100]).astype('int32')))

        #fullMedianFacesHSV = [float(value) for value in fullMedianFacesHSV]
        #fullMedianFacesHLS = [float(value) for value in fullMedianFacesHLS]
        #halfMedianFacesHSV = [float(value) for value in halfMedianFacesHSV]
        #halfMedianFacesHLS = [float(value) for value in halfMedianFacesHLS]

        #return [fullMedianFacesHSV, halfMedianFacesHSV, fluxish]
        #correctedHLS = fullMedianFacesHLS#correctHLS(np.copy(fullMedianFacesHLS), fluxish)

        #[r, g, b] = colorsys.hls_to_rgb(hls)
        #colorTools.getRelativeLuminance(fullPointsLeftCheek)

        #return [fullMedianFacesHLS, halfMedianFacesHLS, list(correctedHLS), fluxish, [leftFluxish, halfPointsLeftCheekMedianLuminance], [rightFluxish, halfPointsRightCheekMedianLuminance]]
        return [imageName, True, [[leftFluxish, fullPointsLeftCheekMedianLuminance, list(fullPointsLeftCheekMedian_hls)], [rightFluxish, fullPointsRightCheekMedianLuminance, list(fullPointsRightCheekMedian_hls)]]]
