import colorsys
import argparse
import numpy as np
import csv
import os
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

colors = ['g.', 'c.', 'm.', 'y.']

def getLast(arr):
    return arr[-1]

def printStat(values, name):
    print('---')
    print(name)
    median = np.median(values)
    mean = np.mean(values)
    sd = np.std(values)
    smallest = np.min(values)
    largest = np.max(values)

    print('Median %f | Mean %f | SD %f | Min %f | Max %f' % (median, mean, sd, smallest, largest))
    return median

def getMakeupHSVvalues():
    makeupColorPaths = ['../scraped/fenti_colors/fentiColors.csv', '../scraped/bm_colors/bm_colors.csv', '../scraped/makeupForever/makeupForeverColors.csv']
    #makeupColorPaths = ['../scraped/bm_colors/bm_colors.csv']
    
    hsvColors = []

    for makeupColorPath in makeupColorPaths:
        brandColors = []
        with open(makeupColorPath, 'r', newline='') as f:
            colorReader = csv.reader(f, delimiter=' ', quotechar='|')
            for color in colorReader:
                hsvColor = np.array(colorsys.rgb_to_hsv(int(color[0]), int(color[1]), int(color[2])))
                hsvColor = hsvColor / [1, 1, 255]
                brandColors.append(hsvColor)

        hsvColors.append(np.array(brandColors))

    return np.array(hsvColors)

ap = argparse.ArgumentParser()
ap.add_argument("-u", "--username", required=False, default="doug", help="The Users user name...")
ap.add_argument("-s", "--start", required=False, help="Image to start benchmarking run")
ap.add_argument("-e", "--end", required=False, help="Image to end benchmarking run")
ap.add_argument("-z", "--baseline", required=False, default=None, help="Benchmark baseline")

args = vars(ap.parse_args())

username = args["username"]
start = -1
end = -1

if args["start"] is not None:
    start = int(args["start"])

if args["end"] is not None:
    end = int(args["end"])

path = '../images/' + str(username) + '/'

directories = [os.path.join(path, o) for o in os.listdir(path) if os.path.isdir(os.path.join(path, o))]

values = []
reflectionStrength = []
fluxish = []
fluxishRatio = []
names = []
meanReflectionArea = []

findNumer = re.compile(r'(\d+)\/$')

splitValues = []
splitFluxish = []
splitReflections = []
splitSaturation = []
splitHue = []
splitDimensions = []

medianHue = []
medianSaturation = []
medianValue = []

for directory in directories:
    hsvCount_path = directory + '/steps/4.csv'
    imageStat_path = directory + '/reference/imageStats.csv'

    if args["baseline"] is not None:
        hsvCount_path = directory + '/steps/4baseline.csv'

    imageNumber = int(re.compile(r'(\d+)$').search(directory).group(1))

    if (start != -1 and imageNumber < start):
        continue

    if (end != -1 and imageNumber > end):
        continue


    if not os.path.exists(hsvCount_path):
        print('No Results for ' + str(directory))
        continue

    #hsvCounts = []
    with open(hsvCount_path, 'r', newline='') as f:
        hsvCount_Reader = csv.reader(f, delimiter=' ', quotechar='|')
        for row in hsvCount_Reader:
            value = np.array([float(row[0]), float(row[1]), float(row[2])])
            #hsvCounts.append([float(row[0]), float(row[1]), float(row[2])])

    with open(imageStat_path, 'r', newline='') as f:
        imageStat_Reader = csv.reader(f, delimiter=' ', quotechar='|')
        for row in imageStat_Reader:
            #Base
            if row[0] == 'reflectionStrength':
                reflectionStrength.append(float(row[1]))

            elif row[0] == 'testFluxish':
                fluxish.append(float(row[1]))

            elif row[0] == 'fluxishRatio':
                fluxishRatio.append(float(row[1]))

            elif row[0] == 'medianHSV':
                medianHue.append(float(row[1]))
                medianSaturation.append(float(row[2]))
                medianValue.append(float(row[3]))

            elif row[0] == 'meanReflectionDimensions':
                meanReflectionArea.append(float(row[1]) * float(row[2]))

            #Split
            elif row[0] == 'splitReflectionStrength':
                splitReflections.append(float(row[1]))

            elif row[0] == 'splitTestFluxish':
                splitFluxish.append(float(row[1]))

            elif row[0] == 'splitMedianHSV':
                splitHue.append(float(row[1]))
                splitSaturation.append(float(row[2]))
                splitValues.append(float(row[3]))

            elif row[0] == 'splitDimensions':
                splitDimensions.append([float(row[1]), float(row[2])])

    #hsvCounts = np.array(hsvCounts)
    #hsvCounts = np.array(sorted(hsvCounts, key=getLast, reverse=True))
    names.append('doug' + str(imageNumber))
    print('---')
    print(directory)
    print('hsvCounts ::\n' + str(value))

    values.append(value)


print('hue :: ' + str(splitHue))

print('Split Dimensions :: ' + str(splitDimensions))
values = np.array(values)
reflectionStrength = np.array(reflectionStrength)#np.array([[reflection] for reflection in reflectionStrength])
fluxish = np.array(fluxish)#np.array([[flux] for flux in fluxish])
#print(values)

#top2 = values[:, 0:2, :]
#top2hue = values[:, 0:2, :]
#top4sat = values[:, 0:4, :]
#topValue = values[:, 0, 2]
##top1 = values[:, 0, :]
##weightedAverageTop2 = np.array([(((v0[0:3] * v0[3]) + (v1[0:3] * v1[3])) / (v0[3] + v1[3])) for [v0, v1] in top2])
##weightedAverageTop2 = np.array([(((v0[0:3] * v0[3]) + (v1[0:3] * v1[3]) + (v2[0:3] * v2[3])) / (v0[3] + v1[3] + v2[3])) for [v0, v1, v2] in top3])
##weightedAverageTop2 = top1
##print(weightedAverageTop2)
#
#weightedHueAverage = np.array([(((v0[0] * v0[3]) + (v1[0] * v1[3])) / (v0[3] + v1[3])) for [v0, v1] in top2hue])
#weightedSaturationAverage = np.array([(((v0[1] * v0[3]) + (v1[1] * v1[3]) + (v2[1] * v2[3]) + (v3[1] * v3[3])) / (v0[3] + v1[3] + v2[3] + v3[3])) for [v0, v1, v2, v3] in top4sat])
#
#
#values = np.dstack((weightedHueAverage, weightedSaturationAverage, topValue, reflectionStrength, fluxish))[0]
#print('Values :: ' + str(values))
#print('ReflectionStrength :: ' + str(reflectionStrength))
#print('Fluxish :: ' + str(fluxish))
#print('Values shape :: ' + str(values.shape))
#print('ReflectionStrength shape :: ' + str(reflectionStrength.shape))
#print('Fluxish shape :: ' + str(fluxish.shape))
values = np.dstack((values[:, 0], values[:, 1], values[:, 2], reflectionStrength, fluxish, fluxishRatio, (values[:, 2] / reflectionStrength), meanReflectionArea, names))[0]

print('Values :: ' + str(values))

H = 0
S = 1
V = 2
RS = 3
F = 4
VF = 5
VR = 6
A = 7
N = 8

def sortBy(elem):
    return float(elem[S])

values = np.array(sorted(values, key=sortBy))

##print('names :: ' + str([[name] for name in names]))
#print('Weighted Average Top 2 :: ')
valuesMask = values[:, VR].astype(np.float) > 1
#values = values[valuesMask]
#names = np.array(names)
#names = names[valuesMask]


#fluxishValueFit = np.polyfit(values[:, F], values[:, V], 1)

print('Values :: ' + str(values))
print('Name\tHue\t\t\tSaturation\t\tValue\t\t\tValue / RS\tFluxish\t\t\tValue / Fluxish\t\tReflection Area')
for pointsValues in values:
    print(pointsValues[N] + '\t' + pointsValues[H] + '\t' + pointsValues[S] + '\t' + pointsValues[V] + '\t' + pointsValues[VR] + '\t' + pointsValues[F] + '\t' + str(pointsValues[VF]) + '\t' + str(pointsValues[A]))

print('Number of values :: ' + str(len(values)))

#Do this to filter out the obviously terrible values
#if args["baseline"] is not None:
#    values = np.array([value for value in values if value[0] < .1])

values = values[:, H:N].astype(np.float)

allHues = values[:, H]
allSat = values[:, S]
allVal = values[:, V]

medianMedianHue = printStat(allHues, 'Hue')
medianMedianSat = printStat(allSat, 'Saturation')
medianMedianVal = printStat(allVal, 'Value')
meidanMedianVF = printStat(values[:, VF], 'Value / Fluxish')
meidanMedianVR = printStat(values[:, VR], 'Value / ReflectionStrength')

hsvColors = getMakeupHSVvalues()
print('Makeup Colors')
#Test

#Saturation vs Hue
colorCount = 0
for brandColors in hsvColors:
    plt.plot(brandColors[:, H], brandColors[:, S], colors[colorCount])
    colorCount = colorCount + 1

plt.plot(medianHue, medianSaturation, 'b*')
plt.plot([medianMedianHue], [medianMedianSat], 'r*')
plt.xlabel('Hue')
plt.ylabel('Saturation')
plt.show()

#Saturation vs Value
colorCount = 0
for brandColors in hsvColors:
    plt.plot(brandColors[:, V], brandColors[:, S], colors[colorCount])
    colorCount = colorCount + 1

plt.plot(medianValue, medianSaturation, 'b*')
plt.plot([medianMedianVal], [medianMedianSat], 'r*')
plt.xlabel('Value')
plt.ylabel('Saturation')
plt.show()

#plt.plot(values[:, VF], values[:, S], 'b*')
#plt.plot(values[valuesMask][:, VF], values[valuesMask][:, S], 'r*')
##plt.plot([medianMedianVal], [medianMedianSat], 'r*')
#plt.xlabel('Value/Fluxish')
#plt.ylabel('Saturation')
#plt.show()

#plt.plot(values[:, V], values[:, H], 'k*')
#plt.xlabel('Value')
#plt.ylabel('Hue')
#plt.show()

#plt.plot(values[:, 0], values[:, 3], 'k*')
#plt.xlabel('Hue')
#plt.ylabel('Reflection Strength')
#plt.show()

#plt.plot(values[:, V], values[:, S], 'r*')
#plt.xlabel('Value')
#plt.ylabel('Saturation')
#plt.show()

#plt.plot(values[:, 1], values[:, 3], 'r*')
#plt.xlabel('Saturation')
#plt.ylabel('Reflection Strength')
#plt.show()
plt.plot(values[:, F], values[:, H], 'k*')
plt.xlabel('Fluxish')
plt.ylabel('Hue')
plt.show()


plt.plot(values[:, F], values[:, S], 'r*')
plt.xlabel('Fluxish')
plt.ylabel('Saturation')
plt.show()

#plt.plot(values[:, V], values[:, RS], 'r*')
#plt.xlabel('Value')
#plt.ylabel('Reflection Strength')
#plt.show()

plt.plot(values[:, F], values[:, V], 'r*')
plt.xlabel('Fluxish')
plt.ylabel('Value')
plt.show()

#plt.plot(values[:, F] / values[:, V], values[:, H], 'r*')
#plt.xlabel('Fluxish/Value')
#plt.ylabel('Hue')
#plt.show()
#
#plt.plot(values[:, F] / values[:, V], values[:, S], 'r*')
#plt.xlabel('Fluxish/Value')
#plt.ylabel('Saturation')
#plt.show()

#Getting Weird
print('Getting Weird...')
plt.plot(splitFluxish, splitHue, 'b*')
plt.xlabel('Split Fluxish')
plt.ylabel('Split Hue')
plt.show()

plt.plot(splitFluxish, splitSaturation, 'b*')
plt.xlabel('Split Fluxish')
plt.ylabel('Split Saturation')
plt.show()

plt.plot(splitFluxish, splitValues, 'b*')
plt.xlabel('Split Fluxish')
plt.ylabel('Split Value')
plt.show()

#Getting Crazyyy...
#print('Getting Crazzzyyy...')
#plt.plot(values[:, 4], medianHue, 'g*')
#plt.xlabel('Fluxish')
#plt.ylabel('Median Hue')
#plt.show()
#
#plt.plot(values[:, 4], medianSaturation, 'g*')
#plt.xlabel('Fluxish')
#plt.ylabel('Median Saturation')
#plt.show()
#
#plt.plot(values[:, 4], medianValue, 'g*')
#plt.xlabel('Fluxish')
#plt.ylabel('Median Value')
#plt.show()

###3d Plot
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(values[:, 0], values[:, 1], values[:, 2])
#
#for brandColors in hsvColors:
#    ax.scatter(brandColors[:, 0], brandColors[:, 1], brandColors[:, 2])
#    colorCount = colorCount + 1
#
#ax.set_xlabel('Hue')
#ax.set_ylabel('Saturation')
#ax.set_zlabel('Value')
#
#plt.show()

###3d Plot 2
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(values[:, 1], values[:, 2], values[:, 3])
#
#
#ax.set_xlabel('Saturation')
#ax.set_ylabel('Value')
#ax.set_zlabel('Reflection Strength')
#
#plt.show()

