import processPoints
import extractMask
import getPolygons
import colorTools
import numpy as np
import saveStep
import os
import cv2

def benchmarkPoints(username, imageName, points, benchmarkName):
    #FORMAT: cluster = [[hue, saturation, value, # in cluster], ... ]
    linearBGR = colorTools.convert_linearHSV_float_to_linearBGR_float(np.array([points]))
    sBGR = colorTools.convert_linearBGR_float_to_sBGR(linearBGR)
    hsv = np.array(cv2.cvtColor(sBGR.astype('uint8'), cv2.COLOR_BGR2HSV_FULL)) / 255
    hsv = hsv[0]

    #clusters = processPoints.kmeans(username, imageName, points, imageName)
    clusters = processPoints.kmeans(username, imageName, hsv)

    clusterCounts = np.array(clusters)[:, 3]
    clusterCounts = np.sort(clusterCounts)[::-1]

    topThreeClusters = clusters[0:3]
    topThreeClusterCounts = clusterCounts[0:3]

    for index, clusterCount in enumerate(topThreeClusterCounts):
        for cluster in clusters:
            if cluster[3] == clusterCount:
                topThreeClusters[index] = cluster

    saveStep.saveBenchmarkPoints(username, imageName, benchmarkName, topThreeClusters)
    
#    print('BENCHMARK ' + benchmarkName)
#    print('CLUSTER COUNTS :: ', str(clusterCounts))
#    print('1st largest :: ' + str(topThreeClusters[0]))
#    print('2nd largest :: ' + str(topThreeClusters[1]))
#    print('3rd largest :: ' + str(topThreeClusters[2]))

def benchmarkValue(username, imageName, value, benchmarkName):
    saveStep.saveBenchmarkValue(username, imageName, benchmarkName, str(value))

def benchmarkPoint(username, imageName, point, benchmarkName):
    saveStep.saveBenchmarkPoints(username, imageName, benchmarkName, [point])

def benchmarkImage(username, imageName, image, shape, benchmarkName, clippedMask=None):
    image = colorTools.convert_linearBGR_float_to_linearHSV_float(image)
    [polygons, error] = getPolygons.getPolygons(image, shape)
    [(points, averageValue), error] = extractMask.extractMask(username, image, polygons, clippedMask)
    benchmarkPoints(points, imageName, benchmarkName)

def clearBenchmarks(username, imageName):
    path = saveStep.benchmarkPathBuilder(username, imageName)
    if os.path.exists(path):
        for file in os.listdir(path):
            filePath = path + file
            if os.path.isfile(filePath):
                os.remove(path + file)
            elif os.path.isdir(filePath):
                os.rmdir(filePath)

def logError(username, error, imageName):
    saveStep.touchBenchmark(username, imageName)
    path = saveStep.benchmarkPathBuilder(username, imageName)
    path = path + 'error.txt'
    with open(path, 'w') as f:
        f.write(error)

