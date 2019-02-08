import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import colorsys

bgrLuminanceConsts = np.array([0.0722, 0.7152, 0.2126])
def getRelativeLuminance(bgr):
    return np.sum(bgr * bgrLuminanceConsts)

def sortBy(elem):
    #print(elem)
    #print('elem[1][3] :: ', str(elem[1][3]))
    return elem[0][0]

blacklist = ['doug205', 'doug206', 'doug246', 'doug258', 'doug257', 'doug247', 'doug250', 'doug255', 'doug294', 'doug274', 'doug286', 'doug272', 'doug282', 'doug197', 'doug293', 'doug277', 'doug273', 'doug275', 'doug358']

#whitelist = ['doug196', 'doug198','doug200','doug201','doug210','doug211','doug212','doug213','doug216','doug217','doug219','doug220','doug221','doug223','doug229','doug236','doug237','doug240','doug248','doug251','doug253','doug263']

with open('faceColors.json', 'r') as f:
    faceColors = f.read()
    faceColors = json.loads(faceColors)

size = 10

#lightnessFluxish = []
#correctedLightnessFluxish = []

halfCheekStats = []
halfChinStats = []
halfForeheadStats = []

fullCheekStats = []
fullChinStats = []
fullForeheadStats = []

faceColors = sorted(faceColors, key = sortBy) 

for faceColor in faceColors:
    imageName = faceColor[0]
    noError = faceColor[1]
    if (imageName in blacklist) or not noError:
    #if imageName not in whitelist:
        continue


    [leftCheekHalf, rightCheekHalf, chinHalf, foreheadHalf] = faceColor[2]
    [leftCheekFull, rightCheekFull, chinFull, foreheadFull] = faceColor[3]
    [leftCheekMedianLinearityError, rightCheekMedianLinearityError, chinMedianLinearityError, foreheadMedianLinearityError] = faceColor[4]
    [leftCheekClippingRatio, rightCheekClippingRatio, chinClippingRatio, foreheadClippingRatio] = faceColor[5]
    [leftCheekNoFlashBGR, rightCheekNoFlashBGR, chinNoFlashBGR, foreheadNoFlashBGR] = faceColor[5]

    leftCheekNoFlashLuminance = getRelativeLuminance(leftCheekNoFlashBGR)
    rightCheekNoFlashLuminance = getRelativeLuminance(rightCheekNoFlashBGR)
    chinNoFlashLuminance = getRelativeLuminance(chinNoFlashBGR)
    foreheadNoFlashLuminance = getRelativeLuminance(foreheadNoFlashBGR)

    #HALF 
    [leftFluxishHalf, leftLuminanceHalf, leftHSVHalf, leftBGRHalf, leftLineHalf] = leftCheekHalf
    [rightFluxishHalf, rightLuminanceHalf, rightHSVHalf, rightBGRHalf, rightLineHalf] = rightCheekHalf
    [chinFluxishHalf, chinLuminanceHalf, chinHSVHalf, chinBGRHalf, chinLineHalf] = chinHalf
    [foreheadFluxishHalf, foreheadLuminanceHalf, foreheadHSVHalf, foreheadBGRHalf, foreheadLineHalf] = foreheadHalf

    #FULL
    [leftFluxishFull, leftLuminanceFull, leftHSVFull, leftBGRFull, leftLineFull] = leftCheekFull
    [rightFluxishFull, rightLuminanceFull, rightHSVFull, rightBGRFull, rightLineFull] = rightCheekFull
    [chinFluxishFull, chinLuminanceFull, chinHSVFull, chinBGRFull, chinLineFull] = chinFull
    [foreheadFluxishFull, foreheadLuminanceFull, foreheadHSVFull, foreheadBGRFull, foreheadLineFull] = foreheadFull


    template =  '{} - {} - :: {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}'

    leftHalf = template.format(imageName, 'HALF LEFT', leftFluxishHalf, leftLuminanceHalf, *leftHSVHalf, *leftBGRHalf, *leftLineHalf, leftCheekMedianLinearityError, leftCheekClippingRatio, leftCheekNoFlashLuminance)
    rightHalf = template.format(imageName, 'HALF RIGHT', rightFluxishHalf, rightLuminanceHalf, *rightHSVHalf, *rightBGRHalf, *rightLineHalf, rightCheekMedianLinearityError, rightCheekClippingRatio, rightCheekNoFlashLuminance)
    chinHalf = template.format(imageName, 'HALF CHIN', chinFluxishHalf, chinLuminanceHalf, *chinHSVHalf, *chinBGRHalf, *chinLineHalf, chinMedianLinearityError, chinClippingRatio, chinNoFlashLuminance)
    foreheadHalf = template.format(imageName, 'HALF FOREHEAD', foreheadFluxishHalf, foreheadLuminanceHalf, *foreheadHSVHalf, *foreheadBGRHalf, *foreheadLineHalf, foreheadMedianLinearityError, foreheadClippingRatio, foreheadNoFlashLuminance)

    leftFull = template.format(imageName, 'FULL LEFT', leftFluxishFull, leftLuminanceFull, *leftHSVFull, *leftBGRFull, *leftLineFull, leftCheekMedianLinearityError, leftCheekClippingRatio, leftCheekNoFlashLuminance)
    rightFull = template.format(imageName, 'FULL RIGHT', rightFluxishFull, rightLuminanceFull, *rightHSVFull, *rightBGRFull, *rightLineFull, rightCheekMedianLinearityError, rightCheekClippingRatio, rightCheekNoFlashLuminance)
    chinFull = template.format(imageName, 'FULL CHIN', chinFluxishFull, chinLuminanceFull, *chinHSVFull, *chinBGRFull, *chinLineFull, chinMedianLinearityError, chinClippingRatio, chinNoFlashLuminance)
    foreheadFull = template.format(imageName, 'FULL FOREHEAD', foreheadFluxishFull, foreheadLuminanceFull, *foreheadHSVFull, *foreheadBGRFull, *foreheadLineFull, foreheadMedianLinearityError, foreheadClippingRatio, foreheadNoFlashLuminance)

    print('~~~~~')
    print(leftHalf)
    print(rightHalf)
    print(chinHalf)
    print(foreheadHalf)
    print(leftFull)
    print(rightFull)
    print(chinFull)
    print(foreheadFull)

    halfCheekStats.append(np.array([leftLuminanceHalf, leftFluxishHalf, *leftHSVHalf, *leftBGRHalf, *leftLineHalf, leftCheekMedianLinearityError, leftCheekClippingRatio, leftCheekNoFlashLuminance, (leftLuminanceHalf / leftFluxishHalf)]))
    halfCheekStats.append(np.array([rightLuminanceHalf, rightFluxishHalf, *rightHSVHalf, *rightBGRHalf, *rightLineHalf, rightCheekMedianLinearityError, rightCheekClippingRatio, rightCheekNoFlashLuminance, (rightLuminanceHalf / rightFluxishHalf)]))
    fullCheekStats.append(np.array([leftLuminanceFull, leftFluxishFull, *leftHSVFull, *leftBGRFull, *leftLineFull, leftCheekMedianLinearityError, leftCheekClippingRatio, leftCheekNoFlashLuminance, (leftLuminanceFull / leftFluxishFull)]))
    fullCheekStats.append(np.array([rightLuminanceFull, rightFluxishFull, *rightHSVFull, *rightBGRFull, *rightLineFull, rightCheekMedianLinearityError, rightCheekClippingRatio, rightCheekNoFlashLuminance, (rightLuminanceFull / rightFluxishFull)]))

    halfChinStats.append(np.array([chinLuminanceHalf, chinFluxishHalf, *chinHSVHalf, *chinBGRHalf, *chinLineHalf, chinMedianLinearityError, chinClippingRatio, chinNoFlashLuminance, (chinLuminanceHalf / chinFluxishHalf)]))
    fullChinStats.append(np.array([chinLuminanceFull, chinFluxishFull, *chinHSVFull, *chinBGRFull, *chinLineFull, chinMedianLinearityError, chinClippingRatio, chinNoFlashLuminance, (chinLuminanceFull / chinFluxishFull)]))

    halfForeheadStats.append(np.array([foreheadLuminanceHalf, foreheadFluxishHalf, *foreheadHSVHalf, *foreheadBGRHalf, *foreheadLineHalf, foreheadMedianLinearityError, foreheadClippingRatio, foreheadNoFlashLuminance, (foreheadLuminanceHalf / foreheadFluxishHalf)]))
    fullForeheadStats.append(np.array([foreheadLuminanceFull, foreheadFluxishFull, *foreheadHSVFull, *foreheadBGRFull, *foreheadLineFull, foreheadMedianLinearityError, foreheadClippingRatio, foreheadNoFlashLuminance, (foreheadLuminanceFull/ foreheadFluxishFull)]))

cheekStats = halfCheekStats + fullCheekStats
cheekStats = np.array(cheekStats)
halfCheekStats = np.array(halfCheekStats)
fullCheekStats = np.array(fullCheekStats)

chinStats = halfChinStats + fullChinStats
chinStats = np.array(chinStats)
halfChinStats = np.array(halfChinStats)
fullChinStats = np.array(fullChinStats)

foreheadStats = halfForeheadStats + fullForeheadStats
foreheadStats = np.array(foreheadStats)
halfForeheadStats = np.array(halfForeheadStats)
fullForeheadStats = np.array(fullForeheadStats)

#LUMINANCE VS FLUXISH

fig, axs = plt.subplots(2, 3, sharex=True, tight_layout=True)

compareIndex_y, compareName_y = [13, 'Luminance / Fluxish'] #[5, 'Blue'] [6, 'Green'] [7, 'Red'] [4, 'Value'] [10, 'Linearity Error'], [13, 'Luminance / Fluxish'][0, 'Luminance']
compareIndex_x, compareName_x = [10, 'Linearity Error'] #[5, 'Blue'] [6, 'Green'] [7, 'Red'] [4, 'Value'] [1, 'Fluxish'] [12, 'No Flash Luminance'][11, 'Clean Pixel Ratio']

minFull_x = min(fullCheekStats[:, compareIndex_x])
maxFull_x = max(fullCheekStats[:, compareIndex_x])
full_x_A = np.vstack([fullCheekStats[:, compareIndex_x], np.ones(len(fullCheekStats))]).T
full_slope, full_const = np.linalg.lstsq(full_x_A, fullCheekStats[:, compareIndex_y], rcond=None)[0]

minHalf_x = min(halfCheekStats[:, compareIndex_x])
maxHalf_x = max(halfCheekStats[:, compareIndex_x])
half_x_A = np.vstack([halfCheekStats[:, compareIndex_x], np.ones(len(halfCheekStats))]).T
half_slope, half_const = np.linalg.lstsq(half_x_A, halfCheekStats[:, compareIndex_y], rcond=None)[0]

print('Full Fluxish to Lightness Slope, Constant :: ' + str(full_slope) + ' ' + str(full_const))
print('Half Fluxish to Lightness Slope, Constant :: ' + str(half_slope) + ' ' + str(half_const))

#margin = 10
margin = 0.1
axs[0, 0].plot([minFull_x, maxFull_x], [(full_slope * minFull_x + full_const), (full_slope * maxFull_x + full_const)])
#axs[0].plot([minFull_x, maxFull_x], [(full_slope * minFull_x + full_const - margin), (full_slope * maxFull_x + full_const - margin)])
#axs[0].plot([minFull_x, maxFull_x], [(full_slope * minFull_x + full_const + margin), (full_slope * maxFull_x + full_const + margin)])

axs[0, 0].plot([minHalf_x, maxHalf_x], [(half_slope * minHalf_x + half_const), (half_slope * maxHalf_x + half_const)])
#axs[0].plot([minHalf_x, maxHalf_x], [(half_slope * minHalf_x + half_const - margin), (half_slope * maxHalf_x + half_const - margin)])
#axs[0].plot([minHalf_x, maxHalf_x], [(half_slope * minHalf_x + half_const + margin), (half_slope * maxHalf_x + half_const + margin)])
#axs[0, 0].plot([0, 1], [0, 255])

axs[0, 0].scatter(halfCheekStats[:, compareIndex_x], halfCheekStats[:, compareIndex_y], size, (0, 1, 0))
axs[0, 0].scatter(fullCheekStats[:, compareIndex_x], fullCheekStats[:, compareIndex_y], size, (1, 0, 0))
axs[0, 0].set_title("CHEEK {} vs {}".format(compareName_x, compareName_y))

#axs[0, 1].plot([0, 1], [0, 255])
axs[0, 1].scatter(halfChinStats[:, compareIndex_x], halfChinStats[:, compareIndex_y], size, (0, 1, 0))
axs[0, 1].scatter(fullChinStats[:, compareIndex_x], fullChinStats[:, compareIndex_y], size, (1, 0, 0))
axs[0, 1].set_title("CHIN {} vs {}".format(compareName_x, compareName_y))

#axs[0, 2].plot([0, 1], [0, 255])
axs[0, 2].scatter(halfForeheadStats[:, compareIndex_x], halfForeheadStats[:, compareIndex_y], size, (0, 1, 0))
axs[0, 2].scatter(fullForeheadStats[:, compareIndex_x], fullForeheadStats[:, compareIndex_y], size, (1, 0, 0))
axs[0, 2].set_title("FOREHEAD {} vs {}".format(compareName_x, compareName_y))

axs[0, 0].set_xlabel(compareName_x)
axs[0, 0].set_ylabel(compareName_y)

axs[1, 0].scatter(halfCheekStats[:, compareIndex_x], halfCheekStats[:, 3], size, (0, 1, 0))
axs[1, 0].scatter(fullCheekStats[:, compareIndex_x], fullCheekStats[:, 3], size, (1, 0, 0))
axs[1, 0].set_title("CHEEK {} vs Saturation".format(compareName_x))

axs[1, 1].scatter(halfChinStats[:, compareIndex_x], halfChinStats[:, 3], size, (0, 1, 0))
axs[1, 1].scatter(fullChinStats[:, compareIndex_x], fullChinStats[:, 3], size, (1, 0, 0))
axs[1, 1].set_title("CHIN {} vs Saturation".format(compareName_x))

axs[1, 2].scatter(halfForeheadStats[:, compareIndex_x], halfForeheadStats[:, 3], size, (0, 1, 0))
axs[1, 2].scatter(fullForeheadStats[:, compareIndex_x], fullForeheadStats[:, 3], size, (1, 0, 0))
axs[1, 2].set_title("FOREHEAD {} vs Saturation".format(compareName_x))

axs[1, 0].set_xlabel(compareName_x)
axs[1, 0].set_ylabel('Saturation')
#plt.show()
plt.show()

#RGB
#halfCheekRGB = np.array([colorsys.hsv_to_rgb(*point) for point in halfCheekStats[:, 2:5]])
#fullCheekRGB = np.array([colorsys.hsv_to_rgb(*point) for point in fullCheekStats[:, 2:5]])
#
#halfChinRGB = np.array([colorsys.hsv_to_rgb(*point) for point in halfChinStats[:, 2:5]])
#fullChinRGB = np.array([colorsys.hsv_to_rgb(*point) for point in fullChinStats[:, 2:5]])
#
#halfForeheadRGB = np.array([colorsys.hsv_to_rgb(*point) for point in halfForeheadStats[:, 2:5]])
#fullForeheadRGB = np.array([colorsys.hsv_to_rgb(*point) for point in fullForeheadStats[:, 2:5]])

halfCheekRGB = np.array([np.flip(point, axis=0) for point in halfCheekStats[:, 5:8]])
fullCheekRGB = np.array([np.flip(point, axis=0) for point in fullCheekStats[:, 5:8]])

halfChinRGB = np.array([np.flip(point, axis=0) for point in halfChinStats[:, 5:8]])
fullChinRGB = np.array([np.flip(point, axis=0) for point in fullChinStats[:, 5:8]])

halfForeheadRGB = np.array([np.flip(point, axis=0) for point in halfForeheadStats[:, 5:8]])
fullForeheadRGB = np.array([np.flip(point, axis=0) for point in fullForeheadStats[:, 5:8]])

fig, axs = plt.subplots(3, 3, sharex=True, sharey=True, tight_layout=True)

axs[0, 0].scatter(halfCheekRGB[:, 0], halfCheekRGB[:, 1], size, (0, 1, 0))
axs[0, 0].scatter(fullCheekRGB[:, 0], fullCheekRGB[:, 1], size, (1, 0, 0))
axs[0, 0].set_xlabel('Red')
axs[0, 0].set_ylabel('Green')

axs[0, 1].scatter(halfChinRGB[:, 0], halfChinRGB[:, 1], size, (0, 1, 0))
axs[0, 1].scatter(fullChinRGB[:, 0], fullChinRGB[:, 1], size, (1, 0, 0))
axs[0, 1].set_xlabel('Red')
axs[0, 1].set_ylabel('Green')

axs[0, 2].scatter(halfForeheadRGB[:, 0], halfForeheadRGB[:, 1], size, (0, 1, 0))
axs[0, 2].scatter(fullForeheadRGB[:, 0], fullForeheadRGB[:, 1], size, (1, 0, 0))
axs[0, 2].set_xlabel('Red')
axs[0, 2].set_ylabel('Green')

axs[1, 0].scatter(halfCheekRGB[:, 0], halfCheekRGB[:, 2], size, (0, 1, 0))
axs[1, 0].scatter(fullCheekRGB[:, 0], fullCheekRGB[:, 2], size, (1, 0, 0))
axs[1, 0].set_xlabel('Red')
axs[1, 0].set_ylabel('Blue')

axs[1, 1].scatter(halfChinRGB[:, 0], halfChinRGB[:, 2], size, (0, 1, 0))
axs[1, 1].scatter(fullChinRGB[:, 0], fullChinRGB[:, 2], size, (1, 0, 0))
axs[1, 1].set_xlabel('Red')
axs[1, 1].set_ylabel('Blue')

axs[1, 2].scatter(halfForeheadRGB[:, 0], halfForeheadRGB[:, 2], size, (0, 1, 0))
axs[1, 2].scatter(fullForeheadRGB[:, 0], fullForeheadRGB[:, 2], size, (1, 0, 0))
axs[1, 2].set_xlabel('Red')
axs[1, 2].set_ylabel('Blue')

axs[2, 0].scatter(halfCheekRGB[:, 1], halfCheekRGB[:, 2], size, (0, 1, 0))
axs[2, 0].scatter(fullCheekRGB[:, 1], fullCheekRGB[:, 2], size, (1, 0, 0))
axs[2, 0].set_xlabel('Green')
axs[2, 0].set_ylabel('Blue')

axs[2, 1].scatter(halfChinRGB[:, 1], halfChinRGB[:, 2], size, (0, 1, 0))
axs[2, 1].scatter(fullChinRGB[:, 1], fullChinRGB[:, 2], size, (1, 0, 0))
axs[2, 1].set_xlabel('Green')
axs[2, 1].set_ylabel('Blue')

axs[2, 2].scatter(halfForeheadRGB[:, 1], halfForeheadRGB[:, 2], size, (0, 1, 0))
axs[2, 2].scatter(fullForeheadRGB[:, 1], fullForeheadRGB[:, 2], size, (1, 0, 0))
axs[2, 2].set_xlabel('Green')
axs[2, 2].set_ylabel('Blue')
#axs[0].plot([0, 1], [0, 1])
#axs[0].plot([0, 1], [0, 1])
#axs[0].plot([0, 1], [0, 1])
plt.show()

#VALUE VS FLUXISH

fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)

minFluxish = min(cheekStats[:, 1])
maxFluxish = max(cheekStats[:, 1])

fluxish_A = np.vstack([cheekStats[:, 1], np.ones(len(cheekStats))]).T

FL_m, FL_c = np.linalg.lstsq(fluxish_A, cheekStats[:, 4], rcond=None)[0]
print('Fluxish to Lightness Slope, Constant :: ' + str(FL_m) + ' ' + str(FL_c))
margin = 10
axs[0].plot([minFluxish, maxFluxish], [(FL_m * minFluxish + FL_c), (FL_m * maxFluxish + FL_c)])
#axs[0].plot([minFluxish, maxFluxish], [(FL_m * minFluxish + FL_c - margin), (FL_m * maxFluxish + FL_c - margin)])
#axs[0].plot([minFluxish, maxFluxish], [(FL_m * minFluxish + FL_c + margin), (FL_m * maxFluxish + FL_c + margin)])
axs[0].scatter(cheekStats[:, 1], cheekStats[:, 4], size, (1, 0, 0))
axs[0].set_title("CHEEK Fluxish vs Value")
axs[0].plot([0, 1], [0, 1])

axs[1].scatter(chinStats[:, 1], chinStats[:, 4], size, (1, 0, 0))
axs[1].set_title("CHIN Fluxish vs Value")
axs[1].plot([0, 1], [0, 1])

axs[2].scatter(foreheadStats[:, 1], foreheadStats[:, 4], size, (1, 0, 0))
axs[2].set_title("FOREHEAD Fluxish vs Value")
axs[2].plot([0, 1], [0, 1])

plt.xlabel('Fluxish')
plt.ylabel('Value')
plt.show()

#ColorChannel VS ColorChannel Slopes
fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)

allHalfDataSources = [halfCheekStats, halfChinStats, halfForeheadStats]
allFullDataSources = [fullCheekStats, fullChinStats, fullForeheadStats]

#halfDataSource = halfForeheadStats
#fullDataSource = fullForeheadStats

for i in range(2, 3):
    halfDataSource = allHalfDataSources[i]
    fullDataSource = allFullDataSources[i]

    xChannelIndex, xChannelName = [6, 'Green']
    yChannelIndex, yChannelName = [5, 'Blue']

    for index in range(0, len(halfDataSource)):
        xHalfStat = halfDataSource[index, xChannelIndex]
        yHalfStat = halfDataSource[index, yChannelIndex]

        xFullStat = fullDataSource[index, xChannelIndex]
        yFullStat = fullDataSource[index, yChannelIndex]

        axs[0].plot([xHalfStat, xFullStat], [yHalfStat, yFullStat])
        #axs[0].scatter([xHalfStat, xFullStat], [yHalfStat, yFullStat])

    axs[0].plot([0, 255], [0, 255])
    axs[0].set_xlabel(xChannelName)
    axs[0].set_ylabel(yChannelName)

    #ColorChannel VS ColorChannel Slopes
    xChannelIndex, xChannelName = [7, 'Red']
    yChannelIndex, yChannelName = [5, 'Blue']

    for index in range(0, len(halfDataSource)):
        xHalfStat = halfDataSource[index, xChannelIndex]
        yHalfStat = halfDataSource[index, yChannelIndex]

        xFullStat = fullDataSource[index, xChannelIndex]
        yFullStat = fullDataSource[index, yChannelIndex]

        axs[1].plot([xHalfStat, xFullStat], [yHalfStat, yFullStat])
        #axs[1].scatter([xHalfStat, xFullStat], [yHalfStat, yFullStat])

    axs[1].plot([0, 255], [0, 255])
    axs[1].set_xlabel(xChannelName)
    axs[1].set_ylabel(yChannelName)

    #ColorChannel VS ColorChannel Slopes
    xChannelIndex, xChannelName = [7, 'Red']
    yChannelIndex, yChannelName = [6, 'Green']

    for index in range(0, len(halfDataSource)):
        xHalfStat = halfDataSource[index, xChannelIndex]
        yHalfStat = halfDataSource[index, yChannelIndex]

        xFullStat = fullDataSource[index, xChannelIndex]
        yFullStat = fullDataSource[index, yChannelIndex]

        axs[2].plot([xHalfStat, xFullStat], [yHalfStat, yFullStat])
        #axs[2].scatter([xHalfStat, xFullStat], [yHalfStat, yFullStat])

    axs[2].plot([0, 255], [0, 255])
    axs[2].set_xlabel(xChannelName)
    axs[2].set_ylabel(yChannelName)

plt.show()

#Saturation VS Luminance Slopes

for stat in halfChinStats:
    plt.plot([0, 255], [stat[9], (stat[8] * 255) + stat[9]])
    plt.plot(stat[0], (stat[8] * stat[0] + stat[9]), 'go')

for stat in fullChinStats:
    plt.plot([0, 255], [stat[9], (stat[8] * 255) + stat[9]])
    plt.plot(stat[0], (stat[8] * stat[0] + stat[9]), 'ro')

plt.xlabel('Luminance')
plt.ylabel('Saturation')
plt.show()

#FLUXISH VS HUE

fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)
axs[0].scatter(halfCheekStats[:, 1], halfCheekStats[:, 2], size, (0, 1, 0))
axs[0].scatter(fullCheekStats[:, 1], fullCheekStats[:, 2], size, (1, 0, 0))
axs[0].set_title("CHEEK Fluxish vs Hue")

axs[1].scatter(halfChinStats[:, 1], halfChinStats[:, 2], size, (0, 1, 0))
axs[1].scatter(fullChinStats[:, 1], fullChinStats[:, 2], size, (1, 0, 0))
axs[1].set_title("CHIN Fluxish vs Hue")

axs[2].scatter(halfForeheadStats[:, 1], halfForeheadStats[:, 2], size, (0, 1, 0))
axs[2].scatter(fullForeheadStats[:, 1], fullForeheadStats[:, 2], size, (1, 0, 0))
axs[2].set_title("FOREHEAD Fluxish vs Hue")

plt.xlabel('Fluxish')
plt.ylabel('Hue')
plt.show()

#VALUE VS HUE
#
#fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)
#axs[0].scatter(cheekStats[:, 4], cheekStats[:, 2], size, (1, 0, 0))
#axs[0].set_title("CHEEK Value vs Hue")
#
#axs[1].scatter(chinStats[:, 4], chinStats[:, 2], size, (1, 0, 0))
#axs[1].set_title("CHIN Value vs Hue")
#
#axs[2].scatter(foreheadStats[:, 4], foreheadStats[:, 2], size, (1, 0, 0))
#axs[2].set_title("FOREHEAD Value vs Hue")
#
#plt.xlabel('Value')
#plt.ylabel('Hue')
#plt.show()

#LUMINANCE VS HUE

#fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)
#axs[0].scatter(halfCheekStats[:, 0], halfCheekStats[:, 2], size, (0, 1, 0))
#axs[0].scatter(fullCheekStats[:, 0], fullCheekStats[:, 2], size, (1, 0, 0))
#axs[0].set_title("CHEEK Luminance vs Hue")
#
#axs[1].scatter(halfChinStats[:, 0], halfChinStats[:, 2], size, (0, 1, 0))
#axs[1].scatter(fullChinStats[:, 0], fullChinStats[:, 2], size, (1, 0, 0))
#axs[1].set_title("CHIN Luminance vs Hue")
#
#axs[2].scatter(halfForeheadStats[:, 0], halfForeheadStats[:, 2], size, (0, 1, 0))
#axs[2].scatter(fullForeheadStats[:, 0], fullForeheadStats[:, 2], size, (1, 0, 0))
#axs[2].set_title("FOREHEAD Luminance vs Hue")
#
#plt.xlabel('Luminance')
#plt.ylabel('Hue')
#plt.show()

#FLUXISH VS SAT

fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)
axs[0].scatter(halfCheekStats[:, 1], halfCheekStats[:, 3], size, (0, 1, 0))
axs[0].scatter(fullCheekStats[:, 1], fullCheekStats[:, 3], size, (1, 0, 0))
axs[0].set_title("CHEEK Fluxish vs Saturation")

axs[1].scatter(halfChinStats[:, 1], halfChinStats[:, 3], size, (0, 1, 0))
axs[1].scatter(fullChinStats[:, 1], fullChinStats[:, 3], size, (1, 0, 0))
axs[1].set_title("CHIN Fluxish vs Saturation")

axs[2].scatter(halfForeheadStats[:, 1], halfForeheadStats[:, 3], size, (0, 1, 0))
axs[2].scatter(fullForeheadStats[:, 1], fullForeheadStats[:, 3], size, (1, 0, 0))
axs[2].set_title("FOREHEAD Fluxish vs Saturation")

plt.xlabel('Fluxish')
plt.ylabel('Saturation')
plt.show()

#LUMINANCE VS SAT

fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)
axs[0].scatter(halfCheekStats[:, 3], halfCheekStats[:, 0], size, (0, 1, 0))
axs[0].scatter(fullCheekStats[:, 3], fullCheekStats[:, 0], size, (1, 0, 0))
axs[0].set_title("CHEEK Luminance vs Saturation")

axs[1].scatter(halfChinStats[:, 3], halfChinStats[:, 0], size, (0, 1, 0))
axs[1].scatter(fullChinStats[:, 3], fullChinStats[:, 0], size, (1, 0, 0))
axs[1].set_title("CHIN Luminance vs Saturation")

axs[2].scatter(halfForeheadStats[:, 3], halfForeheadStats[:, 0], size, (0, 1, 0))
axs[2].scatter(fullForeheadStats[:, 3], fullForeheadStats[:, 0], size, (1, 0, 0))
axs[2].set_title("FOREHEAD Luminance vs Saturation")

plt.xlabel('Saturation')
plt.ylabel('Luminance')
plt.show()

#VALUE VS SAT

fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)
axs[0].scatter(halfCheekStats[:, 4], halfCheekStats[:, 3], size, (0, 1, 0))
axs[0].scatter(fullCheekStats[:, 4], fullCheekStats[:, 3], size, (1, 0, 0))
axs[0].set_title("CHEEK Value vs Saturation")

axs[1].scatter(halfChinStats[:, 4], halfChinStats[:, 3], size, (0, 1, 0))
axs[1].scatter(fullChinStats[:, 4], fullChinStats[:, 3], size, (1, 0, 0))
axs[1].set_title("CHIN Value vs Saturation")

axs[2].scatter(halfForeheadStats[:, 4], halfForeheadStats[:, 3], size, (0, 1, 0))
axs[2].scatter(fullForeheadStats[:, 4], fullForeheadStats[:, 3], size, (1, 0, 0))
axs[2].set_title("FOREHEAD Value vs Saturation")

plt.xlabel('Value')
plt.ylabel('Saturation')
plt.show()

#HUE VS SAT

fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)
axs[0].scatter(halfCheekStats[:, 2], halfCheekStats[:, 3], size, (0, 1, 0))
axs[0].scatter(fullCheekStats[:, 2], fullCheekStats[:, 3], size, (1, 0, 0))
axs[0].set_title("CHEEK Hue vs Saturation")

axs[1].scatter(halfChinStats[:, 2], halfChinStats[:, 3], size, (0, 1, 0))
axs[1].scatter(fullChinStats[:, 2], fullChinStats[:, 3], size, (1, 0, 0))
axs[1].set_title("CHIN Hue vs Saturation")

axs[2].scatter(halfForeheadStats[:, 2], halfForeheadStats[:, 3], size, (0, 1, 0))
axs[2].scatter(fullForeheadStats[:, 2], fullForeheadStats[:, 3], size, (1, 0, 0))
axs[2].set_title("FOREHEAD Hue vs Saturation")

plt.xlabel('Hue')
plt.ylabel('Saturation')
plt.show()

#LUMINANCE VS VALUE

fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)
axs[0].scatter(halfCheekStats[:, 0], halfCheekStats[:, 4], size, (0, 1, 0))
axs[0].scatter(fullCheekStats[:, 0], fullCheekStats[:, 4], size, (1, 0, 0))
axs[0].plot([0, 255], [0, 1])
axs[0].set_title("CHEEK Luminance vs Value")

axs[1].scatter(halfChinStats[:, 0], halfChinStats[:, 4], size, (0, 1, 0))
axs[1].scatter(fullChinStats[:, 0], fullChinStats[:, 4], size, (1, 0, 0))
axs[1].set_title("CHIN Luminance vs Value")

axs[2].scatter(halfForeheadStats[:, 0], halfForeheadStats[:, 4], size, (0, 1, 0))
axs[2].scatter(fullForeheadStats[:, 0], fullForeheadStats[:, 4], size, (1, 0, 0))
axs[2].set_title("FOREHEAD Luminance vs Value")

plt.xlabel('Luminance')
plt.ylabel('Value')
plt.show()
