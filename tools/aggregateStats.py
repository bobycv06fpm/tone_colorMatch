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

def plotHist(values, bins=20):
    plt.hist(values, bins)
    plt.show()

def sortOnValue(point):
    [name, hsv, hsv_scaled, bgr, bgr_scaled, fluxish] = point
    #return bestGuesses[1]
    return fluxish
    #return hsv[2]

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


linearFits = []

for faceData in facesData:

    if not faceData['successful']:
        continue

    #for key in faceData['captures']:
        #print('CAPTURES {} -> {}'.format(key, faceData['captures'][key]))

    linearFits.append([faceData['name'], faceData['linearFits'], faceData['bestGuess'], faceData['reflectionArea']])
   # for key in faceData['medianDiffs']:
   #     print('MEDIAN DIFFS {} -> {}'.format(key, faceData['medianDiffs'][key]))


if len(linearFits) == 0:
    print('No Results :(')
else:
    wbBGR = []
    wbHSV = []

    #medianBGRs = []
    linearFitBGRs = []
    #medianHSVs = []
    linearFitHSVs = []
    linearFitBGRScaled = []
    linearFitHSVScaled = []
    #bestGuesses = []
    printPoints = []
    #reflections = []
    fluxishes = []

    for linearFit in linearFits:
        #print('Median Diff :: ' + str(medianDiff))
        name, linearFit, bestGuess, reflectionArea = linearFit
        leftReflection = np.array(linearFit['reflections']['left'])
        rightReflection = np.array(linearFit['reflections']['right'])
        averageReflection = (leftReflection + rightReflection) / 2

        print('----------')
        print('BEST GUESS :: ' + str(bestGuess))
        print('LINEAR FIT :: ' + str(linearFit))
        print('AVERAGE REFLECTION BGR :: ' + str(averageReflection))
        #bestGuessReflection, bestGuessFace = bestGuess
        #bestGuessReflectionHSV = convertRatiosToHueSatValue(bestGuessReflection)
        averageReflectionHSV = convertRatiosToHueSatValue(averageReflection)
        print('AVERAGE REFLECTION HSV :: ' + str(averageReflectionHSV))

        #print('Best Guess vs Average [Hue, Sat] :: {} vs {}'.format(bestGuessReflectionHSV, averageReflectionHSV))

        #print('L :: {} | R :: {} | A :: {}'.format(leftReflection, rightReflection, averageReflection))
        
        leftPoint = np.array(linearFit['regions']['left'])
        rightPoint = np.array(linearFit['regions']['right'])
        chinPoint = np.array(linearFit['regions']['chin'])
        foreheadPoint = np.array(linearFit['regions']['forehead'])

        #leftPointWB = leftPoint
        #rightPointWB = rightPoint
        #chinPointWB = chinPoint
        #foreheadPointWB = foreheadPoint
        #medianBGR = np.median(np.array([leftPoint, rightPoint, chinPoint, foreheadPoint]), axis=0)

        leftPointWB = colorTools.whitebalanceBGRPoints(leftPoint, leftReflection) #Not sure if we should use the average color or the per side... Color diff might be cause by off axis angles?
        rightPointWB = colorTools.whitebalanceBGRPoints(rightPoint, rightReflection)
        chinPointWB = colorTools.whitebalanceBGRPoints(chinPoint, averageReflection)
        foreheadPointWB = colorTools.whitebalanceBGRPoints(foreheadPoint, averageReflection)

        linearFitWB_BGR = np.mean(np.array([leftPointWB, rightPointWB, chinPointWB, foreheadPointWB]), axis=0)
        #medianBGR = chinPointWB

        leftPointWB_HSV = colorTools.bgr_to_hsv(leftPointWB)
        rightPointWB_HSV = colorTools.bgr_to_hsv(rightPointWB)
        chinPointWB_HSV = colorTools.bgr_to_hsv(chinPointWB)
        foreheadPointWB_HSV = colorTools.bgr_to_hsv(foreheadPointWB)
        linearFitWB_HSV = np.mean(np.array([leftPointWB_HSV, rightPointWB_HSV, chinPointWB_HSV, foreheadPointWB_HSV]), axis=0)
        #medianHSV = chinPointHSV

        #bestGuessFaceWB = colorTools.whitebalanceBGRPoints(np.array(bestGuessFace), np.array(bestGuessReflection))
        #bestGuessFaceWB = colorTools.whitebalanceBGRPoints(np.array(bestGuessFace), np.array(bestGuessReflection))
        #bestGuessReflectionWB = colorTools.whitebalanceBGRPoints(np.array(bestGuessReflection), np.array(bestGuessReflection))

        #Just so we can get the brightness value (if we change how we whitebalance down the line [unlikely...])
        linearFitReflectionWB = colorTools.whitebalanceBGRPoints(np.array(averageReflection), np.array(averageReflection))
        linearFitReflectionValue = linearFitReflectionWB[0]
        #print('BEST GUESS WB FROM LINEAR FIT :: {}'.format(bestGuessReflectionWB))


        fluxish = linearFitReflectionValue * reflectionArea
        fluxishes.append(fluxish)

        linearFitWBScaled_BGR = linearFitWB_BGR / fluxish
        linearFitBGRScaled.append(linearFitWBScaled_BGR)
#
#        #bestGuessFaceWB = bestGuessFace
        linearFitWBScaled_HSV = convertRatiosToHueSatValue(linearFitWBScaled_BGR)
        linearFitHSVScaled.append(linearFitWBScaled_HSV)
#        bestGuesses.append([bestGuessHue, bestGuessSat, bestGuessValue])
#
#        wbBGR.append(leftPointWB)
#        wbBGR.append(rightPointWB)
#        wbBGR.append(chinPointWB)
#        wbBGR.append(foreheadPointWB)
        linearFitBGRs.append(linearFitWB_BGR)
#
#        wbHSV.append(leftPointHSV)
#        wbHSV.append(rightPointHSV)
#        wbHSV.append(chinPointHSV)
#        wbHSV.append(foreheadPointHSV)
        linearFitHSVs.append(linearFitWB_HSV)
#
#        #print('{}'.format(name))
#        #print('\tBGR -> Median :: {} || Left :: {} | Right :: {} | Chin :: {} | Forehead :: {}'.format(pts(medianBGR), pts(leftPointWB), pts(rightPointWB), pts(chinPointWB), pts(foreheadPointWB)))
#        #print('\tHSV -> Median :: {} || Left :: {} | Right :: {} | Chin :: {} | Forehead :: {}'.format(pts(medianHSV), pts(leftPointHSV), pts(rightPointHSV), pts(chinPointHSV), pts(foreheadPointHSV)))
        printPoints.append([name, linearFitWB_HSV, linearFitWBScaled_HSV, linearFitWB_BGR, linearFitWBScaled_BGR, fluxish])
        #print('\t{} - HSV -> Median :: {}'.format(name, pts(medianHSV)))

    printPoints.sort(key=sortOnValue)
    #print('\t{} - HSV -> Median :: {}'.format(name, pts(medianHSV)))
    for index, printPoint in enumerate(printPoints):
        print('{}\t{} - (Set Median) (Linear Fits) -> HSV :: {} | HSV Scaled :: {} | BGR :: {} | BGR Scaled :: {} | Fluxish :: {}'.format(index, *printPoint))

    #wbBGR = np.array(wbBGR)
    #wbHSV = np.array(wbHSV)
    #wbHSV[:, 0] = colorTools.rotateHue(wbHSV[:, 0])

    #medianBGRs = np.array(medianBGRs)
    linearFitBGRs = np.array(linearFitBGRs)
    linearFitBGRScaled = np.array(linearFitBGRScaled)
    #medianHSVs = np.array(medianHSVs)
    linearFitHSVs = np.array(linearFitHSVs)
    linearFitHSVScaled = np.array(linearFitHSVScaled)
    #bestGuesses = np.array(bestGuesses)
    linearFitHSVs[:, 0] = colorTools.rotateHue(linearFitHSVs[:, 0])
    linearFitHSVScaled[:, 0] = colorTools.rotateHue(linearFitHSVScaled[:, 0])

    #plot3d(wbBGR, 'Blue', 'Green', 'Red')
    #plot3d(wbHSV, 'Hue', 'Saturation', 'Value')
    #plot3d(medianBGRs, 'Blue', 'Green', 'Red')
    #plot3d(medianHSVs, 'Hue', 'Saturation', 'Value')
    #plotHist(wbHSV[:, 1])
    #valueVsFluxish = np.stack([linearFitHSVs[:, 2], fluxishes], axis=1)
    fluxishVsValue = np.stack([fluxishes, linearFitHSVs[:, 2]], axis=1)
    #print('Values vs Fluxishes :: ' + str(valueVsFluxish))
    print('Fluxishes vs Values :: ' + str(fluxishVsValue))
    plot2d(fluxishVsValue, 'Fluxish', 'Value')
    print('Linear Fit HSV Scaled :: ' + str(linearFitHSVScaled))
    plotHist(linearFitHSVs[:, 2]) #Pretty sure the target for this is a distribution with a range of ~0.1? Would pretty accuately place it in the spectrum?
    plotHist(linearFitHSVScaled[:, 2]) #Pretty sure the target for this is a distribution with a range of ~0.1? Would pretty accuately place it in the spectrum?


