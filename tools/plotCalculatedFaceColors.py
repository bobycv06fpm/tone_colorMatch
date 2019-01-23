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

lightnessFluxish = []
correctedLightnessFluxish = []
perSideLightnessFluxish = []

faceColors = sorted(faceColors, key = sortBy) 

for faceColor in faceColors:
    imageName = faceColor[0]
    noError = faceColor[1]
    if (imageName in blacklist) or not noError:
    #if imageName not in whitelist:
        continue


    #[fullFlash, halfFlash, corrected, fluxish, leftSide, rightSide] = faceColor
    [[leftFluxish, leftLuminance, leftHSV], [rightFluxish, rightLuminance, rightHSV]] = faceColor[2]

    #print(imageName + ' :: ' +str(fullFlash) + '\t| ' + str(fluxish))
    #print(imageName + ' :: ' +str(leftSide) + '\t| ' + str(rightSide))
    print(imageName + ' - LEFT  - :: ' + str(leftFluxish) + ' | ' + str(leftLuminance) + ' | ' + str(leftHSV[0]) + ' | ' + str(leftHSV[1]) + ' | ' + str(leftHSV[2]))
    print(imageName + ' - RIGHT - :: ' + str(rightFluxish) + ' | ' + str(rightLuminance) + ' | ' + str(rightHSV[0]) + ' | ' + str(rightHSV[1])+ ' | ' + str(rightHSV[2]))

    #lightnessFluxish.append(np.array([float(fullFlash[1]), float(fluxish)]))
    #correctedLightnessFluxish.append(np.array([float(corrected[1]), float(fluxish)]))

    perSideLightnessFluxish.append(np.array([float(leftLuminance), float(leftFluxish), float(leftHSV[0]), float(leftHSV[1]), float(leftHSV[2])]))
    perSideLightnessFluxish.append(np.array([float(rightLuminance), float(rightFluxish), float(rightHSV[0]), float(rightHSV[1]), float(rightHSV[2])]))

#lightnessFluxish = np.array(lightnessFluxish)
#correctedLightnessFluxish = np.array(correctedLightnessFluxish)
perSideLightnessFluxish = np.array(perSideLightnessFluxish)

minFluxish = min(perSideLightnessFluxish[:, 1])
maxFluxish = max(perSideLightnessFluxish[:, 1])

fluxish_A = np.vstack([perSideLightnessFluxish[:, 1], np.ones(len(perSideLightnessFluxish))]).T

FL_m, FL_c = np.linalg.lstsq(fluxish_A, perSideLightnessFluxish[:, 0], rcond=None)[0]
print('Fluxish to Lightness Slope, Constant :: ' + str(FL_m) + ' ' + str(FL_c))
margin = 10
plt.plot([minFluxish, maxFluxish], [(FL_m * minFluxish + FL_c), (FL_m * maxFluxish + FL_c)])
plt.plot([minFluxish, maxFluxish], [(FL_m * minFluxish + FL_c - margin), (FL_m * maxFluxish + FL_c - margin)])
plt.plot([minFluxish, maxFluxish], [(FL_m * minFluxish + FL_c + margin), (FL_m * maxFluxish + FL_c + margin)])

#plt.scatter(lightnessFluxish[:, 1], lightnessFluxish[:, 0], size, (1, 0, 0))
plt.scatter(perSideLightnessFluxish[:, 1], perSideLightnessFluxish[:, 0], size, (1, 0, 0))

plt.xlabel('Fluxish')
plt.ylabel('Luminance')
plt.suptitle("Fluxish vs Luminance")
#plt.show()

#plt.scatter(correctedLightnessFluxish[:, 1], correctedLightnessFluxish[:, 0], size, (0, 1, 0))

#plt.xlabel('Fluxish')
#plt.ylabel('Lightness')
#plt.suptitle("Corrected Fluxish vs Lightness")
plt.show()

plt.scatter(perSideLightnessFluxish[:, 1], perSideLightnessFluxish[:, 2], size, (1, 0, 0))

plt.xlabel('Fluxish')
plt.ylabel('Hue')
plt.suptitle("Fluxish vs Hue")
plt.show()


plt.scatter(perSideLightnessFluxish[:, 1], perSideLightnessFluxish[:, 3], size, (1, 0, 0))

plt.xlabel('Fluxish')
plt.ylabel('Saturation')
plt.suptitle("Fluxish vs Saturation")
plt.show()

#plt.scatter(perSideLightnessFluxish[:, 1], perSideLightnessFluxish[:, 3], size, (1, 0, 0))
#
#plt.xlabel('Fluxish')
#plt.ylabel('Lightness')
#plt.suptitle("Fluxish vs Lightness")
#plt.show()

plt.scatter(perSideLightnessFluxish[:, 2], perSideLightnessFluxish[:, 3], size, (1, 0, 0))

plt.xlabel('Hue')
plt.ylabel('Saturation')
plt.suptitle("Hue vs Saturation")
plt.show()

#plt.scatter(perSideLightnessFluxish[:, 4], perSideLightnessFluxish[:, 0], size, (1, 0, 0))
#
#plt.xlabel('Saturation')
#plt.ylabel('Luminance')
#plt.suptitle("Saturation vs Luminance")
#plt.show()
#
#plt.scatter(perSideLightnessFluxish[:, 4], perSideLightnessFluxish[:, 3], size, (1, 0, 0))
#
#plt.xlabel('Saturation')
#plt.ylabel('Lightness')
#plt.suptitle("Saturation vs Lightness")
#plt.show()
#
plt.scatter(perSideLightnessFluxish[:, 0], perSideLightnessFluxish[:, 4], size, (1, 0, 0))

plt.xlabel('Luminance')
plt.ylabel('Value')
plt.suptitle("Luminance vs Value")
plt.show()
