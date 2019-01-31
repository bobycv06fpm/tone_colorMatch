import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

    #HALF 
    [leftFluxishHalf, leftLuminanceHalf, leftHSVHalf, leftLineHalf] = leftCheekHalf
    [rightFluxishHalf, rightLuminanceHalf, rightHSVHalf, rightLineHalf] = rightCheekHalf
    [chinFluxishHalf, chinLuminanceHalf, chinHSVHalf, chinLineHalf] = chinHalf
    [foreheadFluxishHalf, foreheadLuminanceHalf, foreheadHSVHalf, foreheadLineHalf] = foreheadHalf

    #FULL
    [leftFluxishFull, leftLuminanceFull, leftHSVFull, leftLineFull] = leftCheekFull
    [rightFluxishFull, rightLuminanceFull, rightHSVFull, rightLineFull] = rightCheekFull
    [chinFluxishFull, chinLuminanceFull, chinHSVFull, chinLineFull] = chinFull
    [foreheadFluxishFull, foreheadLuminanceFull, foreheadHSVFull, foreheadLineFull] = foreheadFull


    template =  '{} - {} - :: {} | {} | {} | {} | {} | {} | {}'

    leftHalf = template.format(imageName, 'HALF LEFT', leftFluxishHalf, leftLuminanceHalf, *leftHSVHalf, *leftLineHalf)
    rightHalf = template.format(imageName, 'HALF RIGHT', rightFluxishHalf, rightLuminanceHalf, *rightHSVHalf, *rightLineHalf)
    chinHalf = template.format(imageName, 'HALF CHIN', chinFluxishHalf, chinLuminanceHalf, *chinHSVHalf, *chinLineHalf)
    foreheadHalf = template.format(imageName, 'HALF FOREHEAD', foreheadFluxishHalf, foreheadLuminanceHalf, *foreheadHSVHalf, *foreheadLineHalf)

    leftFull = template.format(imageName, 'FULL LEFT', leftFluxishFull, leftLuminanceFull, *leftHSVFull, *leftLineFull)
    rightFull = template.format(imageName, 'FULL RIGHT', rightFluxishFull, rightLuminanceFull, *rightHSVFull, *rightLineFull)
    chinFull = template.format(imageName, 'FULL CHIN', chinFluxishFull, chinLuminanceFull, *chinHSVFull, *chinLineFull)
    foreheadFull = template.format(imageName, 'FULL FOREHEAD', foreheadFluxishFull, foreheadLuminanceFull, *foreheadHSVFull, *foreheadLineFull)

    print('~~~~~')
    print(leftHalf)
    print(rightHalf)
    print(chinHalf)
    print(foreheadHalf)
    print(leftFull)
    print(rightFull)
    print(chinFull)
    print(foreheadFull)

    halfCheekStats.append(np.array([leftLuminanceHalf, leftFluxishHalf, *leftHSVHalf, *leftLineHalf]))
    halfCheekStats.append(np.array([rightLuminanceHalf, rightFluxishHalf, *rightHSVHalf, *rightLineHalf]))
    fullCheekStats.append(np.array([leftLuminanceFull, leftFluxishFull, *leftHSVFull, *leftLineFull]))
    fullCheekStats.append(np.array([rightLuminanceFull, rightFluxishFull, *rightHSVFull, *rightLineFull]))

    halfChinStats.append(np.array([chinLuminanceHalf, chinFluxishHalf, *chinHSVHalf, *chinLineHalf]))
    fullChinStats.append(np.array([chinLuminanceFull, chinFluxishFull, *chinHSVFull, *chinLineFull]))

    halfForeheadStats.append(np.array([foreheadLuminanceHalf, foreheadFluxishHalf, *foreheadHSVHalf, *foreheadLineHalf]))
    fullForeheadStats.append(np.array([foreheadLuminanceFull, foreheadFluxishFull, *foreheadHSVFull, *foreheadLineFull]))

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

fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)

minFluxish = min(cheekStats[:, 1])
maxFluxish = max(cheekStats[:, 1])

fluxish_A = np.vstack([cheekStats[:, 1], np.ones(len(cheekStats))]).T

FL_m, FL_c = np.linalg.lstsq(fluxish_A, cheekStats[:, 0] / 255, rcond=None)[0]
print('Fluxish to Lightness Slope, Constant :: ' + str(FL_m) + ' ' + str(FL_c))
#margin = 10
margin = 0.1
axs[0].plot([minFluxish, maxFluxish], [(FL_m * minFluxish + FL_c), (FL_m * maxFluxish + FL_c)])
axs[0].plot([minFluxish, maxFluxish], [(FL_m * minFluxish + FL_c - margin), (FL_m * maxFluxish + FL_c - margin)])
axs[0].plot([minFluxish, maxFluxish], [(FL_m * minFluxish + FL_c + margin), (FL_m * maxFluxish + FL_c + margin)])

axs[0].scatter(halfCheekStats[:, 1], halfCheekStats[:, 0] / 255, size, (0, 1, 0))
axs[0].scatter(fullCheekStats[:, 1], fullCheekStats[:, 0] / 255, size, (1, 0, 0))
axs[0].set_title("CHEEK Fluxish vs Luminance")

axs[1].scatter(halfChinStats[:, 1], halfChinStats[:, 0] / 255, size, (0, 1, 0))
axs[1].scatter(fullChinStats[:, 1], fullChinStats[:, 0] / 255, size, (1, 0, 0))
axs[1].set_title("CHIN Fluxish vs Luminance")

axs[2].scatter(halfForeheadStats[:, 1], halfForeheadStats[:, 0] / 255, size, (0, 1, 0))
axs[2].scatter(fullForeheadStats[:, 1], fullForeheadStats[:, 0] / 255, size, (1, 0, 0))
axs[2].set_title("FOREHEAD Fluxish vs Luminance")

plt.xlabel('Fluxish')
plt.ylabel('Luminance')
plt.show()

#VALUE VS FLUXISH

#fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)
#
#minFluxish = min(cheekStats[:, 1])
#maxFluxish = max(cheekStats[:, 1])
#
#fluxish_A = np.vstack([cheekStats[:, 1], np.ones(len(cheekStats))]).T
#
#FL_m, FL_c = np.linalg.lstsq(fluxish_A, cheekStats[:, 4], rcond=None)[0]
#print('Fluxish to Lightness Slope, Constant :: ' + str(FL_m) + ' ' + str(FL_c))
#margin = 10
#axs[0].plot([minFluxish, maxFluxish], [(FL_m * minFluxish + FL_c), (FL_m * maxFluxish + FL_c)])
##axs[0].plot([minFluxish, maxFluxish], [(FL_m * minFluxish + FL_c - margin), (FL_m * maxFluxish + FL_c - margin)])
##axs[0].plot([minFluxish, maxFluxish], [(FL_m * minFluxish + FL_c + margin), (FL_m * maxFluxish + FL_c + margin)])
#axs[0].scatter(cheekStats[:, 1], cheekStats[:, 4], size, (1, 0, 0))
#axs[0].set_title("CHEEK Fluxish vs Value")
#
#axs[1].scatter(chinStats[:, 1], chinStats[:, 4], size, (1, 0, 0))
#axs[1].set_title("CHIN Fluxish vs Value")
#
#axs[2].scatter(foreheadStats[:, 1], foreheadStats[:, 4], size, (1, 0, 0))
#axs[2].set_title("FOREHEAD Fluxish vs Value")
#
#plt.xlabel('Fluxish')
#plt.ylabel('Value')
#plt.show()

#Saturation VS Luminance Slopes

for stat in halfChinStats:
    plt.plot([0, 255], [stat[6], (stat[5] * 255) + stat[6]])
    plt.plot(stat[0], (stat[5] * stat[0] + stat[6]), 'go')

for stat in fullChinStats:
    plt.plot([0, 255], [stat[6], (stat[5] * 255) + stat[6]])
    plt.plot(stat[0], (stat[5] * stat[0] + stat[6]), 'ro')

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

fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)
axs[0].scatter(halfCheekStats[:, 0], halfCheekStats[:, 2], size, (0, 1, 0))
axs[0].scatter(fullCheekStats[:, 0], fullCheekStats[:, 2], size, (1, 0, 0))
axs[0].set_title("CHEEK Luminance vs Hue")

axs[1].scatter(halfChinStats[:, 0], halfChinStats[:, 2], size, (0, 1, 0))
axs[1].scatter(fullChinStats[:, 0], fullChinStats[:, 2], size, (1, 0, 0))
axs[1].set_title("CHIN Luminance vs Hue")

axs[2].scatter(halfForeheadStats[:, 0], halfForeheadStats[:, 2], size, (0, 1, 0))
axs[2].scatter(fullForeheadStats[:, 0], fullForeheadStats[:, 2], size, (1, 0, 0))
axs[2].set_title("FOREHEAD Luminance vs Hue")

plt.xlabel('Luminance')
plt.ylabel('Hue')
plt.show()

#LUMINANCE VS SAT

fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)
axs[0].scatter(halfCheekStats[:, 0], halfCheekStats[:, 3], size, (0, 1, 0))
axs[0].scatter(fullCheekStats[:, 0], fullCheekStats[:, 3], size, (1, 0, 0))
axs[0].set_title("CHEEK Luminance vs Saturation")

axs[1].scatter(halfChinStats[:, 0], halfChinStats[:, 3], size, (0, 1, 0))
axs[1].scatter(fullChinStats[:, 0], fullChinStats[:, 3], size, (1, 0, 0))
axs[1].set_title("CHIN Luminance vs Saturation")

axs[2].scatter(halfForeheadStats[:, 0], halfForeheadStats[:, 3], size, (0, 1, 0))
axs[2].scatter(fullForeheadStats[:, 0], fullForeheadStats[:, 3], size, (1, 0, 0))
axs[2].set_title("FOREHEAD Luminance vs Saturation")

plt.xlabel('Luminance')
plt.ylabel('Saturation')
plt.show()

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
