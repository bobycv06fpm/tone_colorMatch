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
#import matplotlib.pyplot as plt
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

def correctHLS(hls, fluxish):
    print('------')
    print('Old HLS :: ' + str(hls))
    targetFluxish = 0.8
    #slope = 0.145696 
    slope = 0.2731371365514631 
    #lightnessDiff = (0.106485 * (targetFluxish - fluxish))
    lightnessDiff = (slope * (targetFluxish - fluxish))
    hls[1] += lightnessDiff
    hls[0] = (0.018374 * hls[1]) + .059859

    print('Corrected HLS :: ' + str(hls))
    print('-----')

    return hls


def run(username, imageName, fast=False, saveStats=False):
    #saveStep.resetLogFile(username, imageName)
    saveStep = Save(username, imageName)
    saveStep.resetLogFile()
    images = loadImages(username, imageName)

    [noFlashImage, halfFlashImage, fullFlashImage] = images
    [noFlashMetadata, halfFlashMetadata, fullFlashMetadata] = saveStep.getMetadata()

    noFlashCapture = Capture('No Flash', noFlashImage, noFlashMetadata)
    halfFlashCapture = Capture('Half Flash', halfFlashImage, halfFlashMetadata)
    fullFlashCapture = Capture('Full Flash', fullFlashImage, fullFlashMetadata)

    print('Cropping and Aligning')
    alignImages.cropAndAlign(noFlashCapture, halfFlashCapture, fullFlashCapture)
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
    howLinear = np.abs((2 * halfFlashCapture.image.astype('int32')) - (fullFlashCapture.image.astype('int32') + noFlashCapture.image.astype('int32')))
    #TODO: Compare Subpixel nonlinearity with full pixel nonlinearity....
    #howLinearSum = np.sum(howLinear, axis=2)
    howLinearMax = np.max(howLinear, axis=2)
    nonLinearMask = howLinearMax > 6#8 #12

    allPointsMask = np.logical_or(allPointsMask, nonLinearMask)

    print('Subtracting Base from Flash')
    fullDiffImage = fullFlashCapture.image.astype('int32') - noFlashCapture.image.astype('int32')
    halfDiffImage = halfFlashCapture.image.astype('int32') - noFlashCapture.image.astype('int32')

    #print('Diff Image :: ' + str(diffImage))
    fullDiffCapture = Capture('Diff', fullDiffImage, fullFlashCapture.metadata, allPointsMask)
    halfDiffCapture = Capture('Diff', halfDiffImage, halfFlashCapture.metadata, allPointsMask)

    print('Getting Polygons')
    polygons = fullDiffCapture.landmarks.getFacePolygons()
    print('POLYGONS :: ' + str(polygons))

    if not fast:
        print('Saving Step 1')
        #saveStep.saveShapeStep(username, imageName, imageShape, 1)
        saveStep.saveImageStep(fullDiffCapture.image, 1)
        saveStep.saveMaskStep(allPointsMask, 1, 'clippedMask')

    #alignImages.alignEyes(noFlashCapture, halfFlashCapture, fullFlashCapture)
    whiteBalance_CIE1931_coord_asShot = saveStep.getAsShotWhiteBalance()
    print('White Balance As Shot :: ' + str(whiteBalance_CIE1931_coord_asShot))

    noFlashCapture.landmarks = halfFlashCapture.landmarks
    fullFlashCapture.landmarks = halfFlashCapture.landmarks

    [reflectionValue, fluxish] = getAverageScreenReflectionColor(noFlashCapture, halfFlashCapture, fullFlashCapture, saveStep)
    print("Reflection Value:: " + str(reflectionValue))
    print("Fluxish :: " + str(fluxish))
    #diffCapture.show()
    saveStep.saveReferenceImageBGR(fullDiffCapture.image, 'full_noWhitebalancedImage')
    saveStep.saveReferenceImageBGR(halfDiffCapture.image, 'half_noWhitebalancedImage')
    colorTools.whitebalanceBGR(fullDiffCapture, reflectionValue)
    colorTools.whitebalanceBGR(halfDiffCapture, reflectionValue)
    saveStep.saveReferenceImageBGR(fullDiffCapture.image, 'full_WhitebalancedImage')
    saveStep.saveReferenceImageBGR(halfDiffCapture.image, 'half_WhitebalancedImage')

    medianFaceValue = None

    try:
        [fullPoints, averageFlashContribution] = extractMask(fullDiffCapture, polygons, saveStep)
        [halfPoints, averageFlashContribution] = extractMask(halfDiffCapture, polygons, saveStep)
    except NameError as err:
        #print('error extracting left side of face')
        #raise NameError('User :: {} | Image :: {} | Error :: {} | Details :: {}'.format(username, imageName, 'Error extracting left side of face', err))
        raise NameError('User :: {} | Image :: {} | Error :: {} | Details :: {}'.format(username, imageName, 'Error extracting Points for Recovered Mask', err))
    else:
        fullFaceMedian = np.median(fullPoints, axis=0)
        halfFaceMedian = np.median(halfPoints, axis=0)


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

        print('full median face :: ' + str(fullFaceMedian))
        print('half median face :: ' + str(halfFaceMedian))

        sBGR_fullMedian = colorTools.convertSingle_linearValue_to_sValue(fullFaceMedian)
        sBGR_halfMedian = colorTools.convertSingle_linearValue_to_sValue(halfFaceMedian)
        print('full median face sBGR:: ' + str(sBGR_fullMedian))
        print('half median face sBGR:: ' + str(sBGR_halfMedian))

        #fullMedianFacesHSV = list(colorsys.rgb_to_hsv(sBGR_fullMedian[2], sBGR_fullMedian[1], sBGR_fullMedian[0]))
        fullMedianFacesHLS = list(colorsys.rgb_to_hls(sBGR_fullMedian[2] / 255, sBGR_fullMedian[1] / 255, sBGR_fullMedian[0] / 255))
        #halfMedianFacesHSV = list(colorsys.rgb_to_hsv(sBGR_halfMedian[2], sBGR_halfMedian[1], sBGR_halfMedian[0]))
        halfMedianFacesHLS = list(colorsys.rgb_to_hls(sBGR_halfMedian[2]/ 255, sBGR_halfMedian[1] / 255, sBGR_halfMedian[0] / 255))

        #print('full median face sHSV :: ' + str(fullMedianFacesHSV))
        print('full median face sHLS :: ' + str((np.array(fullMedianFacesHLS) * [360, 100, 100]).astype('int32')))
        #print('half median face sHSV :: ' + str(halfMedianFacesHSV))
        print('half median face sHLS :: ' + str((np.array(halfMedianFacesHLS) * [360, 100, 100]).astype('int32')))

        #fullMedianFacesHSV = [float(value) for value in fullMedianFacesHSV]
        fullMedianFacesHLS = [float(value) for value in fullMedianFacesHLS]
        #halfMedianFacesHSV = [float(value) for value in halfMedianFacesHSV]
        halfMedianFacesHLS = [float(value) for value in halfMedianFacesHLS]

        #return [fullMedianFacesHSV, halfMedianFacesHSV, fluxish]
        correctedHLS = correctHLS(np.copy(fullMedianFacesHLS), fluxish)
        return [fullMedianFacesHLS, halfMedianFacesHLS, list(correctedHLS), fluxish]
