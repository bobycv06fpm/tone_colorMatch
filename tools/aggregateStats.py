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

    medianDiffs.append([faceData['name'], faceData['medianDiffs']])
   # for key in faceData['medianDiffs']:
   #     print('MEDIAN DIFFS {} -> {}'.format(key, faceData['medianDiffs'][key]))


if len(medianDiffs) == 0:
    print('No Results :(')
else:
    for medianDiff in medianDiffs:
        #print('Median Diff :: ' + str(medianDiff))
        name, medianDiff = medianDiff
        leftReflection = np.array(medianDiff['reflections']['left'])
        rightReflection = np.array(medianDiff['reflections']['right'])
        averageReflection = (leftReflection + rightReflection) / 2

        #print('L :: {} | R :: {} | A :: {}'.format(leftReflection, rightReflection, averageReflection))
        
        leftPoint = np.array(medianDiff['regions']['left'])
        rightPoint = np.array(medianDiff['regions']['right'])
        chinPoint = np.array(medianDiff['regions']['chin'])
        foreheadPoint = np.array(medianDiff['regions']['forehead'])

        leftPointWB = colorTools.whitebalanceBGRPoints(leftPoint, leftReflection)
        rightPointWB = colorTools.whitebalanceBGRPoints(rightPoint, rightReflection)
        chinPointWB = colorTools.whitebalanceBGRPoints(chinPoint, averageReflection)
        foreheadPointWB = colorTools.whitebalanceBGRPoints(foreheadPoint, averageReflection)
        

        leftPointHSV = colorTools.bgr_to_hsv(leftPointWB)
        rightPointHSV = colorTools.bgr_to_hsv(rightPointWB)
        chinPointHSV = colorTools.bgr_to_hsv(chinPointWB)
        foreheadPointHSV = colorTools.bgr_to_hsv(foreheadPointWB)
        print('{}'.format(name))
        print('\tBGR -> Left :: {} | Right :: {} | Chin :: {} | Forehead :: {}'.format(pts(leftPointWB), pts(rightPointWB), pts(chinPointWB), pts(foreheadPointWB)))
        print('\tHSV -> Left :: {} | Right :: {} | Chin :: {} | Forehead :: {}'.format(pts(leftPointHSV), pts(rightPointHSV), pts(chinPointHSV), pts(foreheadPointHSV)))

