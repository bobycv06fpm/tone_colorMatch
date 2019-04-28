import utils
import numpy as np
import cv2
import colorTools
import math

#TAKES A FLOAT
def stretchHistogram(gray, mask=None):
    upperBound = 1
    lowerBound = 0

    if mask is not None:
        clippedHigh = gray != upperBound
        clippedLow = gray != lowerBound

        mask = np.logical_and(mask, clippedHigh)
        mask = np.logical_and(mask, clippedLow)

        grayPoints = gray[mask]
    else:
        grayPoints = gray.flatten()

    median = np.median(grayPoints)
    sd = np.std(grayPoints)
    lower = median - (3 * sd)
    lower = lower if lower > lowerBound else lowerBound
    upper = median + (3 * sd)
    upper = upper if upper < upperBound else upperBound

    bounds = np.copy(gray)
    bounds[bounds < lower] = lower
    bounds[bounds > upper] = upper

    numerator = bounds - lower
    denominator = upper - lower
    stretched = (numerator / denominator)
    #stretched = np.clip(stretched * 255, 0, 255)
    return stretched

def getSharpnessScore(gray):
    return 0.0

def labelSharpestCaptures(captures):
    leftEyeCrops = [capture.leftEyeImage for capture in captures]
    rightEyeCrops = [capture.rightEyeImage for capture in captures]

    greyLeftEyeCrops = [np.mean(leftEyeCrop) for leftEyeCrop in leftEyeCrops]
    leftEyes = np.hstack(greyLeftEyeCrops)
    cv2.imshow('left eyes', leftEyes)
    cv2.waitKey(0)

