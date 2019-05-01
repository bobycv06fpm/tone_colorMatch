import sys
sys.path.append('../src/')

import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy
import colorTools
import math

#Point to String
def pts(point):
    return '({:.3}, {:.3}, {:.3})'.format(*[float(value) for value in point])

def plot3d(points, xLabel, yLabel, zLabel):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.set_zlabel(zLabel)
    plt.show()

def plot2d(points, xLabel, yLabel):
    plt.scatter(points[:, 0], points[:, 1])
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.show()

def plotHist(values):
    plt.hist(values, 20)
    plt.show()

def getSaturation(point):
    [name, bestGuesses, hsv, bgr, bestGuessReflectionHSV, fluxish] = point
    #return bestGuesses[1]
    return fluxish

#EXPECTS RED TO BE LARGEST VALUE
def convertRatiosToHueSatValue(bgrRatios):
    bgrRatios = np.array(bgrRatios)
    print('BGR RATIOS :: ' + str(bgrRatios))
    v = max(bgrRatios)

    bgrRatios = bgrRatios / max(bgrRatios)
    delta = max(bgrRatios) - min(bgrRatios)

    s = delta / max(bgrRatios)

    if max(bgrRatios) == bgrRatios[0]:
        h = (1/6) * (((bgrRatios[2] - bgrRatios[1]) / delta) + 4)
    elif max(bgrRatios) == bgrRatios[1]:
        h = (1/6) * (((bgrRatios[0] - bgrRatios[2]) / delta) + 2)
    else:
        h = (1/6) * (((bgrRatios[1] - bgrRatios[0]) / delta) % 6)

    return [h, s, v]


with open('faceColors.json', 'r') as f:
    facesData = f.read()
    facesData = json.loads(facesData)

size = 10


medianDiffs = []

for faceData in facesData:

    if not faceData['successful']:
        continue

    #for key in faceData['captures']:
        #print('CAPTURES {} -> {}'.format(key, faceData['captures'][key]))

    medianDiffs.append([faceData['name'], faceData['medianDiffs'], faceData['bestGuess'], faceData['reflectionArea']])
   # for key in faceData['medianDiffs']:
   #     print('MEDIAN DIFFS {} -> {}'.format(key, faceData['medianDiffs'][key]))


if len(medianDiffs) == 0:
    print('No Results :(')
else:
    wbBGR = []
    wbHSV = []

    medianBGRs = []
    medianHSVs = []
    bestGuesses = []
    printPoints = []
    reflections = []
    fluxishes = []

    for medianDiff in medianDiffs:
        #print('Median Diff :: ' + str(medianDiff))
        name, medianDiff, bestGuess, reflectionArea = medianDiff
        leftReflection = np.array(medianDiff['reflections']['left'])
        rightReflection = np.array(medianDiff['reflections']['right'])
        averageReflection = (leftReflection + rightReflection) / 2

        bestGuessReflection, bestGuessFace = bestGuess
        bestGuessReflectionHSV = convertRatiosToHueSatValue(bestGuessReflection)
        averageReflectionHSV = convertRatiosToHueSatValue(averageReflection)

        print('Best Guess vs Average [Hue, Sat] :: {} vs {}'.format(bestGuessReflectionHSV, averageReflectionHSV))

        #print('L :: {} | R :: {} | A :: {}'.format(leftReflection, rightReflection, averageReflection))
        
        leftPoint = np.array(medianDiff['regions']['left'])
        rightPoint = np.array(medianDiff['regions']['right'])
        chinPoint = np.array(medianDiff['regions']['chin'])
        foreheadPoint = np.array(medianDiff['regions']['forehead'])

        #leftPointWB = leftPoint
        #rightPointWB = rightPoint
        #chinPointWB = chinPoint
        #foreheadPointWB = foreheadPoint
        #medianBGR = np.median(np.array([leftPoint, rightPoint, chinPoint, foreheadPoint]), axis=0)

        leftPointWB = colorTools.whitebalanceBGRPoints(leftPoint, averageReflection)
        rightPointWB = colorTools.whitebalanceBGRPoints(rightPoint, averageReflection)
        chinPointWB = colorTools.whitebalanceBGRPoints(chinPoint, averageReflection)
        foreheadPointWB = colorTools.whitebalanceBGRPoints(foreheadPoint, averageReflection)
        medianBGR = np.median(np.array([leftPointWB, rightPointWB, chinPointWB, foreheadPointWB]), axis=0)
        #medianBGR = chinPointWB

        leftPointHSV = colorTools.bgr_to_hsv(leftPointWB)
        rightPointHSV = colorTools.bgr_to_hsv(rightPointWB)
        chinPointHSV = colorTools.bgr_to_hsv(chinPointWB)
        foreheadPointHSV = colorTools.bgr_to_hsv(foreheadPointWB)
        medianHSV = np.median(np.array([leftPointHSV, rightPointHSV, chinPointHSV, foreheadPointHSV]), axis=0)
        #medianHSV = chinPointHSV

        bestGuessFaceWB = colorTools.whitebalanceBGRPoints(np.array(bestGuessFace), np.array(bestGuessReflection))
        bestGuessReflectionWB = colorTools.whitebalanceBGRPoints(np.array(bestGuessReflection), np.array(bestGuessReflection))

        fluxish = bestGuessReflectionWB[0] * reflectionArea
        fluxishes.append(fluxish)
        bestGuessFaceWBScaled = bestGuessFaceWB #/ fluxish

        #bestGuessFaceWB = bestGuessFace
        bestGuessHue, bestGuessSat, bestGuessValue = convertRatiosToHueSatValue(bestGuessFaceWBScaled)
        bestGuesses.append([bestGuessHue, bestGuessSat, bestGuessValue])

        wbBGR.append(leftPointWB)
        wbBGR.append(rightPointWB)
        wbBGR.append(chinPointWB)
        wbBGR.append(foreheadPointWB)
        medianBGRs.append(medianBGR)

        wbHSV.append(leftPointHSV)
        wbHSV.append(rightPointHSV)
        wbHSV.append(chinPointHSV)
        wbHSV.append(foreheadPointHSV)
        medianHSVs.append(medianHSV)

        #print('{}'.format(name))
        #print('\tBGR -> Median :: {} || Left :: {} | Right :: {} | Chin :: {} | Forehead :: {}'.format(pts(medianBGR), pts(leftPointWB), pts(rightPointWB), pts(chinPointWB), pts(foreheadPointWB)))
        #print('\tHSV -> Median :: {} || Left :: {} | Right :: {} | Chin :: {} | Forehead :: {}'.format(pts(medianHSV), pts(leftPointHSV), pts(rightPointHSV), pts(chinPointHSV), pts(foreheadPointHSV)))
        printPoints.append([name, [bestGuessHue, bestGuessSat, bestGuessValue], medianHSV, medianBGR, bestGuessReflectionHSV, fluxish])
        #print('\t{} - HSV -> Median :: {}'.format(name, pts(medianHSV)))

    printPoints.sort(key=getSaturation)
    #print('\t{} - HSV -> Median :: {}'.format(name, pts(medianHSV)))
    for index, printPoint in enumerate(printPoints):
        print('{}\t{} - MEDIANS -> BG :: {} | HSV :: {} | BGR :: {} | BG R :: {} | Fluxish :: {}'.format(index, *printPoint))

    wbBGR = np.array(wbBGR)
    wbHSV = np.array(wbHSV)
    wbHSV[:, 0] = colorTools.rotateHue(wbHSV[:, 0])

    medianBGRs = np.array(medianBGRs)
    medianHSVs = np.array(medianHSVs)
    bestGuesses = np.array(bestGuesses)
    medianHSVs[:, 0] = colorTools.rotateHue(medianHSVs[:, 0])

    #plot3d(wbBGR, 'Blue', 'Green', 'Red')
    #plot3d(wbHSV, 'Hue', 'Saturation', 'Value')
    #plot3d(medianBGRs, 'Blue', 'Green', 'Red')
    #plot3d(medianHSVs, 'Hue', 'Saturation', 'Value')
    #plotHist(wbHSV[:, 1])
    valueVsFluxish = np.stack([bestGuesses[:, 2], fluxishes], axis=1)
    print('Values vs Fluxishes :: ' + str(valueVsFluxish))
    plot2d(valueVsFluxish, 'Value', 'Fluxish')
    plotHist(bestGuesses[:, 2]) #Pretty sure the target for this is a distribution with a range of ~0.1? Would pretty accuately place it in the spectrum?


