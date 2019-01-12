import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def sortBy(elem):
    print(elem)
    print('elem[1][3] :: ', str(elem[1][3]))
    return elem[1][3]


blacklist = ['doug205', 'doug206', 'doug246', 'doug258', 'doug257', 'doug247', 'doug250', 'doug255', 'doug294', 'doug274', 'doug286', 'doug272', 'doug282', 'doug197', 'doug293', 'doug277', 'doug273', 'doug275']

#whitelist = ['doug196', 'doug198','doug200','doug201','doug210','doug211','doug212','doug213','doug216','doug217','doug219','doug220','doug221','doug223','doug229','doug236','doug237','doug240','doug248','doug251','doug253','doug263']

with open('faceColors.json', 'r') as f:
    faceColors = f.read()
    faceColors = json.loads(faceColors)

size = 10

lightnessFluxish = []
correctedLightnessFluxish = []
perSideLightnessFluxish = []

faceColors = sorted(faceColors, key = sortBy) 

for (imageName, faceColor) in faceColors:
    if imageName in blacklist:
    #if imageName not in whitelist:
        continue

    [fullFlash, halfFlash, corrected, fluxish, leftSide, rightSide] = faceColor

    #print(imageName + ' :: ' +str(fullFlash) + '\t| ' + str(fluxish))
    #print(imageName + ' :: ' +str(leftSide) + '\t| ' + str(rightSide))
    print(imageName + ' :: ' + str(leftSide[1]) + ' | ' + str(leftSide[0]))
    print(imageName + ' :: ' + str(rightSide[1]) + ' | ' + str(rightSide[0]))

    lightnessFluxish.append(np.array([float(fullFlash[1]), float(fluxish)]))
    correctedLightnessFluxish.append(np.array([float(corrected[1]), float(fluxish)]))

    perSideLightnessFluxish.append(np.array([float(leftSide[1]), float(leftSide[0])]))
    perSideLightnessFluxish.append(np.array([float(rightSide[1]), float(rightSide[0])]))

lightnessFluxish = np.array(lightnessFluxish)
correctedLightnessFluxish = np.array(correctedLightnessFluxish)
perSideLightnessFluxish = np.array(perSideLightnessFluxish)

minFluxish = min(lightnessFluxish[:, 1])
maxFluxish = max(lightnessFluxish[:, 1])

fluxish_A = np.vstack([lightnessFluxish[:, 1], np.ones(len(lightnessFluxish))]).T

FL_m, FL_c = np.linalg.lstsq(fluxish_A, lightnessFluxish[:, 0], rcond=None)[0]
print('Fluxish to Lightness Slope, Constant :: ' + str(FL_m) + ' ' + str(FL_c))
plt.plot([minFluxish, maxFluxish], [(FL_m * minFluxish + FL_c), (FL_m * maxFluxish + FL_c)])

#plt.scatter(lightnessFluxish[:, 1], lightnessFluxish[:, 0], size, (1, 0, 0))
plt.scatter(perSideLightnessFluxish[:, 1], perSideLightnessFluxish[:, 0], size, (1, 0, 0))

plt.xlabel('Fluxish')
plt.ylabel('Lightness')
plt.suptitle("Fluxish vs Lightness")
#plt.show()

#plt.scatter(correctedLightnessFluxish[:, 1], correctedLightnessFluxish[:, 0], size, (0, 1, 0))

#plt.xlabel('Fluxish')
#plt.ylabel('Lightness')
#plt.suptitle("Corrected Fluxish vs Lightness")
plt.show()
