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
cheekStats = []
chinStats = []
foreheadStats = []

faceColors = sorted(faceColors, key = sortBy) 

for faceColor in faceColors:
    imageName = faceColor[0]
    noError = faceColor[1]
    if (imageName in blacklist) or not noError:
    #if imageName not in whitelist:
        continue


    #[fullFlash, halfFlash, corrected, fluxish, leftSide, rightSide] = faceColor
    [leftCheek, rightCheek, chin, forehead] = faceColor[2]

    [leftFluxish, leftLuminance, leftHSV, leftLine] = leftCheek
    [rightFluxish, rightLuminance, rightHSV, rightLine] = rightCheek
    [chinFluxish, chinLuminance, chinHSV, chinLine] = chin
    [foreheadFluxish, foreheadLuminance, foreheadHSV, foreheadLine] = forehead

    #print(imageName + ' :: ' +str(fullFlash) + '\t| ' + str(fluxish))
    #print(imageName + ' :: ' +str(leftSide) + '\t| ' + str(rightSide))
    print('~~~~~')
    print(imageName + ' - LEFT  - :: \t' + str(leftFluxish) + ' | ' + str(leftLuminance) + ' | ' + str(leftHSV[0]) + ' | ' + str(leftHSV[1]) + ' | ' + str(leftHSV[2]) + ' | ' + str(leftLine[0]) + ' | ' + str(leftLine[1]))
    print(imageName + ' - RIGHT - :: \t' + str(rightFluxish) + ' | ' + str(rightLuminance) + ' | ' + str(rightHSV[0]) + ' | ' + str(rightHSV[1])+ ' | ' + str(rightHSV[2]) + ' | ' + str(rightLine[0]) + ' | ' + str(rightLine[1]))
    print(imageName + ' - CHIN  - :: \t' + str(chinFluxish) + ' | ' + str(chinLuminance) + ' | ' + str(chinHSV[0]) + ' | ' + str(chinHSV[1]) + ' | ' + str(chinHSV[2]) + ' | ' + str(chinLine[0]) + ' | ' + str(chinLine[1]))
    print(imageName + ' - FOREHEAD - :: ' + str(foreheadFluxish) + ' | ' + str(foreheadLuminance) + ' | ' + str(foreheadHSV[0]) + ' | ' + str(foreheadHSV[1])+ ' | ' + str(foreheadHSV[2]) + ' | ' + str(foreheadLine[0]) + ' | ' + str(foreheadLine[1]))

    #lightnessFluxish.append(np.array([float(fullFlash[1]), float(fluxish)]))
    #correctedLightnessFluxish.append(np.array([float(corrected[1]), float(fluxish)]))

    cheekStats.append(np.array([float(leftLuminance), float(leftFluxish), float(leftHSV[0]), float(leftHSV[1]), float(leftHSV[2]), float(leftLine[0]), float(leftLine[1])]))
    cheekStats.append(np.array([float(rightLuminance), float(rightFluxish), float(rightHSV[0]), float(rightHSV[1]), float(rightHSV[2]), float(rightLine[0]), float(rightLine[1])]))
    chinStats.append(np.array([float(chinLuminance), float(chinFluxish), float(chinHSV[0]), float(chinHSV[1]), float(chinHSV[2]), float(chinLine[0]), float(chinLine[1])]))
    foreheadStats.append(np.array([float(foreheadLuminance), float(foreheadFluxish), float(foreheadHSV[0]), float(foreheadHSV[1]), float(foreheadHSV[2]), float(foreheadLine[0]), float(foreheadLine[1])]))

#lightnessFluxish = np.array(lightnessFluxish)
#correctedLightnessFluxish = np.array(correctedLightnessFluxish)
cheekStats = np.array(cheekStats)
chinStats = np.array(chinStats)
foreheadStats = np.array(foreheadStats)

#LUMINANCE VS FLUXISH

fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)

minFluxish = min(cheekStats[:, 1])
maxFluxish = max(cheekStats[:, 1])

fluxish_A = np.vstack([cheekStats[:, 1], np.ones(len(cheekStats))]).T

FL_m, FL_c = np.linalg.lstsq(fluxish_A, cheekStats[:, 0], rcond=None)[0]
print('Fluxish to Lightness Slope, Constant :: ' + str(FL_m) + ' ' + str(FL_c))
margin = 10
axs[0].plot([minFluxish, maxFluxish], [(FL_m * minFluxish + FL_c), (FL_m * maxFluxish + FL_c)])
axs[0].plot([minFluxish, maxFluxish], [(FL_m * minFluxish + FL_c - margin), (FL_m * maxFluxish + FL_c - margin)])
axs[0].plot([minFluxish, maxFluxish], [(FL_m * minFluxish + FL_c + margin), (FL_m * maxFluxish + FL_c + margin)])
axs[0].scatter(cheekStats[:, 1], cheekStats[:, 0], size, (1, 0, 0))
axs[0].set_title("CHEEK Fluxish vs Luminance")

axs[1].scatter(chinStats[:, 1], chinStats[:, 0], size, (1, 0, 0))
axs[1].set_title("CHIN Fluxish vs Luminance")

axs[2].scatter(foreheadStats[:, 1], foreheadStats[:, 0], size, (1, 0, 0))
axs[2].set_title("FOREHEAD Fluxish vs Luminance")

plt.xlabel('Fluxish')
plt.ylabel('Luminance')
plt.show()

#FLUXISH VS HUE

fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)
axs[0].scatter(cheekStats[:, 1], cheekStats[:, 2], size, (1, 0, 0))
axs[0].set_title("CHEEK Fluxish vs Hue")

axs[1].scatter(chinStats[:, 1], chinStats[:, 2], size, (1, 0, 0))
axs[1].set_title("CHIN Fluxish vs Hue")

axs[2].scatter(foreheadStats[:, 1], foreheadStats[:, 2], size, (1, 0, 0))
axs[2].set_title("FOREHEAD Fluxish vs Hue")

plt.xlabel('Fluxish')
plt.ylabel('Hue')
plt.show()

#FLUXISH VS SAT

fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)
axs[0].scatter(cheekStats[:, 1], cheekStats[:, 3], size, (1, 0, 0))
axs[0].set_title("CHEEK Fluxish vs Saturation")

axs[1].scatter(chinStats[:, 1], chinStats[:, 3], size, (1, 0, 0))
axs[1].set_title("CHIN Fluxish vs Saturation")

axs[2].scatter(foreheadStats[:, 1], foreheadStats[:, 3], size, (1, 0, 0))
axs[2].set_title("FOREHEAD Fluxish vs Saturation")

plt.xlabel('Fluxish')
plt.ylabel('Saturation')
plt.show()

#HUE VS SAT

fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)
axs[0].scatter(cheekStats[:, 2], cheekStats[:, 3], size, (1, 0, 0))
axs[0].set_title("CHEEK Hue vs Saturation")

axs[1].scatter(chinStats[:, 2], chinStats[:, 3], size, (1, 0, 0))
axs[1].set_title("CHIN Hue vs Saturation")

axs[2].scatter(foreheadStats[:, 2], foreheadStats[:, 3], size, (1, 0, 0))
axs[2].set_title("FOREHEAD Hue vs Saturation")

plt.xlabel('Hue')
plt.ylabel('Saturation')
plt.show()

#LUMINANCE VS VALUE

fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)
axs[0].scatter(cheekStats[:, 0], cheekStats[:, 4], size, (1, 0, 0))
axs[0].set_title("CHEEK Luminance vs Value")

axs[1].scatter(chinStats[:, 0], chinStats[:, 4], size, (1, 0, 0))
axs[1].set_title("CHIN Luminance vs Value")

axs[2].scatter(foreheadStats[:, 0], foreheadStats[:, 4], size, (1, 0, 0))
axs[2].set_title("FOREHEAD Luminance vs Value")

plt.xlabel('Luminance')
plt.ylabel('Value')
plt.show()
