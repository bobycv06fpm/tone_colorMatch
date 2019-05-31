import sys
sys.path.append('../src/')

import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy
import colorTools
import math
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-u", "--user", required=True, default="false", help="The Users user name...")
args = vars(ap.parse_args())
user = args["user"]

#Point to String
def pts(point):
    return '({:.3}, {:.3}, {:.3})'.format(*[float(value) for value in point])

def prettyHist(title, data):
    statsTemplate = '\tMedian\t:: {}\n\tStd\t:: {}'
    print('Histogram - {}'.format(title))
    median = np.median(data)
    SD = np.std(data)
    print(statsTemplate.format(median, SD))
    plotHist(data)

def plot3d(points, xLabel, yLabel, zLabel):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.set_zlabel(zLabel)
    plt.show()

def plot2d(title, points, xLabel, yLabel, rankColumn=False):
    print('Scatter Plot - {}'.format(title))
    colors = np.arange(points.shape[0] * 3).reshape(points.shape[0], 3)
    colors[:] = [0, 0, 1]


    if rankColumn:
        colors[points[:, 2] > 0.2] = [1, 0, 0]

        #ceiling = 0.2

        #colorsRank = points[:, 2]
        #colorsRank[colorsRank > ceiling] = ceiling

        #minColor = min(colorsRank)
        #maxColor = max(colorsRank)

        #colorsRank = (colorsRank - minColor) / (maxColor - minColor)
        #medianRank = np.median(colorsRank)

        #greenChannel= 1 - colorsRank
        #blueChannel = 1 - np.clip(np.abs(colorsRank - medianRank), 0, ceiling) / ceiling
        #redChannel= colorsRank

        #colors = np.stack([redChannel, greenChannel, blueChannel], axis=1)

    plt.scatter(points[:, 0], points[:, 1], 25, colors)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.show()

def plotHist(values, bins=20):
    plt.hist(values, bins)
    plt.show()

def sortOnValue(point):
    [name, bgr_noWB, bgr, bgr_scaled, hsv, hsv_scaled, fluxish, reflectionScore, regionScore, combinedScore, reflectionScores_worst, combinedScore_worst] = point
    #return bestGuesses[1]
    #return fluxish
    return hsv[2]
    #return hsv_scaled[2]
    #return fluxish
    #return hsv_scaled[1]
    #return reflectionScore[0]
    #return regionScore[0]
    #return combinedScore
    #return reflectionScores_worst
    #return combinedScore_worst

#EXPECTS RED TO BE LARGEST VALUE
def convertRatiosToHueSatValue(bgrRatios):
    bgrRatios = np.array(bgrRatios)
    #print('BGR RATIOS :: ' + str(bgrRatios))
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


with open('faceColors-{}.json'.format(user), 'r') as f:
    facesData = f.read()
    facesData = json.loads(facesData)

size = 10


linearFits = []

for faceData in facesData:

    if not faceData['successful']:
        continue

    linearFits.append([faceData['name'], faceData['linearFits'], faceData['bestGuess'], faceData['reflectionArea']])


if len(linearFits) == 0:
    print('No Results :(')
else:
    wbBGR = []
    wbHSV = []

    #medianBGRs = []
    linearFitBGRNoWB = []
    linearFitBGRs = []
    #medianHSVs = []
    linearFitHSVs = []
    linearFitBGRScaled = []
    linearFitHSVScaled = []
    #bestGuesses = []
    printPoints = []
    #reflections = []
    fluxishes = []
    reflectionScores = []
    regionScores = []
    combinedScores = []
    reflectionScores_worst = []
    combinedScores_worst = []

    for linearFit in linearFits:
        name, linearFit, bestGuess, reflectionArea = linearFit
        leftReflection = np.array(linearFit['reflections']['left'])
        rightReflection = np.array(linearFit['reflections']['right'])
        reflectionScore = np.array(linearFit['reflections']['linearityScore'])
        averageReflection = (leftReflection + rightReflection) / 2

        reflectionScores.append(reflectionScore)

        averageReflectionHSV = convertRatiosToHueSatValue(averageReflection)
        
        leftPoint = np.array(linearFit['regions']['left'])
        rightPoint = np.array(linearFit['regions']['right'])
        chinPoint = np.array(linearFit['regions']['chin'])
        foreheadPoint = np.array(linearFit['regions']['forehead'])
        regionScore = np.array(linearFit['regions']['linearityScore'])

        regionScores.append(regionScore)

        combinedScore = np.mean([reflectionScore, regionScore])
        combinedScore_worst = np.max(reflectionScore) + np.max(regionScore)
        reflectionScore_worst = np.max(reflectionScore)

        combinedScores.append(combinedScore)
        reflectionScores_worst.append(reflectionScore_worst)
        combinedScores_worst.append(combinedScore_worst)

        linearFitNoWB_BGR = np.mean(np.array([leftPoint, rightPoint, chinPoint, foreheadPoint]), axis=0)
        linearFitBGRNoWB.append(linearFitNoWB_BGR)

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

        #Just so we can get the brightness value (if we change how we whitebalance down the line [unlikely...])
        linearFitReflectionWB = colorTools.whitebalanceBGRPoints(np.array(averageReflection), np.array(averageReflection))
        linearFitReflectionValue = linearFitReflectionWB[0]

        fluxish = linearFitReflectionValue * reflectionArea
        fluxishes.append(fluxish)

        linearFitWBScaled_BGR = linearFitWB_BGR / fluxish
        linearFitBGRScaled.append(linearFitWBScaled_BGR)

        linearFitWBScaled_HSV = convertRatiosToHueSatValue(linearFitWBScaled_BGR)
        linearFitHSVScaled.append(linearFitWBScaled_HSV)

        linearFitBGRs.append(linearFitWB_BGR)
        linearFitHSVs.append(linearFitWB_HSV)
#
        printPoints.append([name, linearFitNoWB_BGR, linearFitWB_BGR, linearFitWBScaled_BGR, linearFitWB_HSV, linearFitWBScaled_HSV, fluxish, reflectionScore, regionScore, combinedScore, reflectionScore_worst, combinedScore_worst])

    printPoints.sort(key=sortOnValue)
    #print('\t{} - HSV -> Median :: {}'.format(name, pts(medianHSV)))
    for index, printPoint in enumerate(printPoints):
        print('({}) {} - \n\tBGR No WB\t:: {} \n\tBGR\t\t:: {} \n\tBGR Scaled\t:: {} \n\tHSV\t\t:: {} \n\tHSV Scaled\t:: {} \n\tFluxish\t\t:: {}\n\tRef Score\t:: {}\n\tRegion Score\t:: {}\n\tCombined Score\t:: {}\n\tWorst Ref Score\t:: {}\n\tComb Worst Score:: {}'.format(index, *printPoint))


    #medianBGRs = np.array(medianBGRs)
    linearFitBGRs = np.array(linearFitBGRs)
    linearFitBGRScaled = np.array(linearFitBGRScaled)
    #medianHSVs = np.array(medianHSVs)
    linearFitHSVs = np.array(linearFitHSVs)
    linearFitHSVScaled = np.array(linearFitHSVScaled)

    reflectionScores = np.array(reflectionScores)
    meanChannelReflectionScores = np.mean(reflectionScores, axis=1)

    regionScores = np.array(regionScores)
    meanChannelRegionScores = np.mean(regionScores, axis=1)

    combinedScores = np.array(combinedScores)
    reflectionScores_worst = np.array(reflectionScores_worst)

    linearFitHSVs[:, 0] = colorTools.rotateHue(linearFitHSVs[:, 0])
    linearFitHSVScaled[:, 0] = colorTools.rotateHue(linearFitHSVScaled[:, 0])

    #fluxishVsValue = np.stack([fluxishes, linearFitHSVs[:, 2], combinedScores], axis=1)
    #fluxishVsValue = np.stack([fluxishes, linearFitHSVs[:, 2], meanChannelRegionScores], axis=1)
    #fluxishVsValue = np.stack([fluxishes, linearFitHSVs[:, 2], reflectionScores_worst], axis=1)
    fluxishVsValue = np.stack([fluxishes, linearFitHSVs[:, 2], combinedScores_worst], axis=1)

    print('\n---------------\n')
    plot2d('Fluxish vs Value', fluxishVsValue, 'Fluxish', 'Value', True)

    prettyHist('Unscaled Values', linearFitHSVs[:, 2])
    prettyHist('Scaled Values', linearFitHSVScaled[:, 2])
    prettyHist('Fluxish Values', fluxishes)
    prettyHist('Scaled Saturation Values', linearFitHSVScaled[:, 1])
    #prettyHist('Mean Channel REFLECTION Scores', meanChannelReflectionScores)
    #prettyHist('Mean Channel REGION Scores', meanChannelRegionScores)

    #scoresSets = np.stack([meanChannelReflectionScores, meanChannelRegionScores], axis=1)
    #plot2d('REFLECTION vs REGION', scoresSets, 'Reflection', 'Region')

    prettyHist('COMBINED Scores', combinedScores)
    prettyHist('COMBINED WORST REFLECTION Scores', reflectionScores_worst)
    prettyHist('COMBINED WORST Scores', combinedScores_worst)

    #Trying to figure out if the worst offenders in the reflections will stand out in the combined... It seems to.
    #scoresSets = np.stack([combinedScores, reflectionScores_worst], axis=1)
    #plot2d('COMBINED vs REFLECTION WORST CASE', scoresSets, 'Combined', 'Reflection Worst Case')

