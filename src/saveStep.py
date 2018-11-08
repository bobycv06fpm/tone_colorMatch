import cv2
import csv
import os
import numpy as np
import colorTools
import json

root = '../../'

def referencePathBuilder(username, fileName, file='', extension=''):
    return root + 'images/' + username + '/' + fileName + '/reference/' + file + extension

def benchmarkPathBuilder(username, fileName, benchmarkName=''):
    extension = '.csv' if benchmarkName != '' else ''
    return root + 'images/' + username + '/' + fileName + '/benchmark/' + benchmarkName + extension

def stepPathBuilder(username, fileName, step='', ext='.csv', meta=''):
    extension = ext if step != '' else ''
    return root + 'images/' + username + '/' + fileName + '/steps/'+ str(step) + meta + extension

#def calibrationPathBuilder(username):
#    return '../calibrations/' + username + '/cameraResponseFunction.csv'

def touchReference(username, fileName):
    path = referencePathBuilder(username, fileName)
    if not os.path.exists(path):
        os.makedirs(path)
        os.chmod(path, 0o777)

def touchBenchmark(username, fileName):
    path = benchmarkPathBuilder(username, fileName)
    if not os.path.exists(path):
        os.makedirs(path)
        os.chmod(path, 0o777)

def touchSteps(username, fileName):
    path = stepPathBuilder(username, fileName)
    if not os.path.exists(path):
        os.makedirs(path)
        os.chmod(path, 0o777)

def saveImageStep(username, fileName, image_float, step, meta=''):
    touchSteps(username, fileName)
    path = stepPathBuilder(username, fileName, step, '.PNG', meta)
    image = np.clip(image_float * 255, 0, 255).astype('uint8')
    cv2.imwrite(path, image)
    os.chmod(path, 0o777)

def loadImageStep(username, fileName, step, meta=''):
    path = stepPathBuilder(username, fileName, step, '.PNG', meta)
    image = cv2.imread(path)
    return image

def saveShapeStep(username, fileName, shape, step):
    touchSteps(username, fileName)
    path = stepPathBuilder(username, fileName, step)
    with open(path, 'w', newline='') as f:
        shapeWriter = csv.writer(f, delimiter=' ', quotechar='|')
        shapeWriter.writerows(shape)

    os.chmod(path, 0o777)

def loadShapeStep(username, fileName, step):
    path = stepPathBuilder(username, fileName, step)
    shape = []
    with open(path, 'r', newline='') as f:
        shapeReader = csv.reader(f, delimiter=' ', quotechar='|')
        for row in shapeReader:
            shape.append([int(i) for i in row])

    return np.array(shape)

def savePointsStep(username, fileName, points, step, meta=''):
    touchSteps(username, fileName)
    path = stepPathBuilder(username, fileName, step, '.csv', meta)
    with open(path, 'w', newline='') as f:
        pointWriter = csv.writer(f, delimiter=' ', quotechar='|')
        pointWriter.writerows(points)

    os.chmod(path, 0o777)

def loadPointsStep(username, fileName, step, meta=''):
    path = stepPathBuilder(username, fileName, step, '.csv', meta)
    points = []
    with open(path, 'r', newline='') as f:
        pointReader = csv.reader(f, delimiter=' ', quotechar='|')
        for row in pointReader:
            points.append([float(i) for i in row])

    return np.array(points)


def saveAverageFlashContribution(username, averageFlashContribution, fileName, step):
    touchSteps(username, fileName)
    path = stepPathBuilder(username, fileName, step, '.txt', '_averageFlashContribution')
    with open(path, 'w') as f:
        f.write(str(averageFlashContribution))

    os.chmod(path, 0o777)

def loadAverageFlashContribution(username, fileName, step):
    path = stepPathBuilder(username, fileName, step, '.txt', '_averageFlashContribution')
    with open(path, 'r') as f:
        averageFlashContribution = f.read()

    return float(averageFlashContribution)

def saveColor(username, fileName, color, step):
    touchSteps(username, fileName)
    path = stepPathBuilder(username, fileName, step, '.txt', '_averageColor')
    with open(path, 'w') as f:
        f.write(str(color[0]) + ',' + str(color[1]) + ',' + str(color[2]))

    os.chmod(path, 0o777)

def loadColor(username, fileName, step):
    path = stepPathBuilder(username, fileName, step, '.txt', '_averageColor')
    with open(path, 'r') as f:
        point = [float(i) for i in f.readline().split(',')]

    return point

#def drawPointsAndSaveBGR(username, bgr_float, shape, fileName, reference):
#    touchReference(username, fileName)
#    [image, error] = colorTools.convert_linearBGR_float_to_sBGR(bgr_float)
#    for (i, j) in shape:
#       cv2.circle(image, (i, j), 3, (0, 0, 255), -1)
#    image = bgr_float * 255
#    path = referencePathBuilder(username, fileName, reference, '.PNG')
#    cv2.imwrite(path, image)
#    os.chmod(path, 0o777)


def saveReferenceImageSBGR(username, fileName, image, reference):
    touchReference(username, fileName)
    #image = bgr_float * 255
    #image = bgr_float
    path = referencePathBuilder(username, fileName, reference, '.PNG')
    cv2.imwrite(path, image)
    os.chmod(path, 0o777)

def saveReferenceImageLinearBGR(username, fileName, bgr_float, reference):
    touchReference(username, fileName)
    image = (bgr_float * 255).astype('uint8')
    #[image, error] = colorTools.convert_linearBGR_float_to_sBGR(bgr_float)
    #image = bgr_float * 255
    #image = bgr_float
    path = referencePathBuilder(username, fileName, reference, '.PNG')
    cv2.imwrite(path, image)
    os.chmod(path, 0o777)

def saveReferenceImageBGR(username, fileName, bgr_float, reference):
    touchReference(username, fileName)
    bgr_float_clipped = np.clip(bgr_float, 0.0, 1.0)
    image = colorTools.convert_linearBGR_float_to_sBGR(bgr_float_clipped)
    #image = bgr_float * 255
    #image = bgr_float
    path = referencePathBuilder(username, fileName, reference, '.PNG')
    cv2.imwrite(path, image)
    os.chmod(path, 0o777)

def saveReferenceImageHSV(username, fileName, hsv_float, reference):
    bgr_float  = colorTools.convert_linearHSV_float_to_linearBGR_float(hsv_float)
    saveReferenceImageBGR(username, fileName, bgr_float, reference)

def logMeasurement(username, fileName, measurementName, measurementValue):
    path = referencePathBuilder(username, fileName, 'measurementLog', '.txt')
    with open(path, 'a+') as f:
        f.write(measurementName + ", " + measurementValue + "\n")

    os.chmod(path, 0o777)

def resetLogFile(username, fileName):
    touchReference(username, fileName)
    path = referencePathBuilder(username, fileName, 'measurementLog', '.txt')
    open(path, 'w').close()
    os.chmod(path, 0o777)

def logTimeMeasurement(username, fileName, measurementName, programStartTime, stepStartTime, endTime):
    stepElapsedTime = endTime - stepStartTime
    totalElapsedTime = endTime - programStartTime
    logMeasurement(fileName, 'Step Elapsed Time ' + measurementName, str(stepElapsedTime))
    logMeasurement(fileName, 'Total Elapsed Time ' + measurementName, str(totalElapsedTime))

def saveBenchmarkPoints(username, fileName, benchmarkName, points):
    touchBenchmark(fileName)
    path = benchmarkPathBuilder(username, fileName, benchmarkName)
    with open(path, 'w', newline='') as f:
        benchWriter = csv.writer(f, delimiter=' ', quotechar='|')
        benchWriter.writerows(points)

    os.chmod(path, 0o777)

def saveBenchmarkValue(username, fileName, benchmarkName, value):
    touchBenchmark(fileName)
    path = benchmarkPathBuilder(username, fileName, benchmarkName)
    with open(path, 'w', newline='') as f:
        f.write(value)

    os.chmod(path, 0o777)
    
#def saveCameraResponseFunction(CRF, userName):
#    path = calibrationPathBuilder(userName)
#    with open(path, 'w') as f:
#        pointWriter = csv.writer(f)
#        pointWriter.writerows(CRF)
#
#    os.chmod(path, 0o777)
#
#def getCameraResponseFunction(userName):
#    path = calibrationPathBuilder(userName)
#    CRF = []
#    with open(path, 'r') as f:
#        crfReader = csv.reader(f)
#        for row in crfReader:
#            row = row[0][1:-1].split(" ")
#            row = [value for value in row if len(value) > 0]
#            print("Row :: " + str(row))
#            CRF.append([float(i) for i in row])
#
#    os.chmod(path, 0o777)
#
#    return np.array(CRF)

#def getAsShotWhiteBalance(username, fileName):
#    path = root + 'images/' + username + '/' + fileName + '/' + fileName + '-whiteBalance.txt'
#
#    parsedValues = []
#    with open(path, 'r', newline='') as f:
#        #Yea Yea I know this is nasty. Change so app just sends json or csv 
#        wbValuesString = f.readline()
#        roughParseValues = wbValuesString.split('], [')
#        parsedStringValues = [value.lstrip('[]\n').rstrip('[]\n').split(', ') for value in roughParseValues]
#        parsedValues = [[float(coordValue) for coordValue in value] for value in parsedStringValues]
#
#    for value in parsedValues:
#        if value != parsedValues[0]:
#            raise NameError('Not All White Balance Values Match!')
#
#    return parsedValues[0]

def getAsShotWhiteBalance(username, fileName):
    path = root + 'images/' + username + '/' + fileName + '/' + fileName + '-metadata.txt'

    with open(path) as f:
        data = json.load(f)

    return [data[0]['whiteBalance']['x'], data[0]['whiteBalance']['y']]

def savePlot(username, fileName, name, plot):
    path = referencePathBuilder(username, fileName, name, '.png')
    plot.savefig(path, dpi=200)

def saveImageStat(username, filename, statName, statValue):
    path = referencePathBuilder(username, filename, 'imageStats', '.csv')

    with open(path, 'a', newline='') as f:
        benchWriter = csv.writer(f, delimiter=' ', quotechar='|')
        benchWriter.writerows([[statName, *statValue]])

    os.chmod(path, 0o777)


