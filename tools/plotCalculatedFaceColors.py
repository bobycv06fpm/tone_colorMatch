import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

with open('faceColors.json', 'r') as f:
    faceColors = f.read()
    faceColors = json.loads(faceColors)

size = 250

for (imageName, faceColor) in faceColors:
    [halfFlash, fullFlash, fluxish] = faceColor
    plt.scatter(fluxish, halfFlash[1], size, (0, 255, 0))
    plt.scatter(fluxish, fullFlash[1], size, (0, 0, 255))

plt.xlabel('Fluxish')
plt.ylabel('Lightness')
plt.suptitle("HLS " + names[index])
plt.show()
