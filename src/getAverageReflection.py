import utils
import saveStep
import cv2
import numpy as np
import colorTools
import alignImages

#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

def interiorBoundingRect(tl, tr, bl, br):
    topLeftX = tl[0] if tl[0] > bl[0] else bl[0]
    topLeftY = tl[1] if tl[1] > tr[1] else tr[1]

    targetX = tr[0] if tr[0] < br[0] else br[0]
    targetY = bl[1] if bl[1] < br[1] else br[1]

    width = targetX - topLeftX
    height = targetY - topLeftY

    return (topLeftX, topLeftY, width, height)

def getLeftEye(username, imageName, image, fullFlash_sBGR, shape):
    (x, y, w, h) = interiorBoundingRect(shape[43], shape[44], shape[47], shape[46])
    leftEye = image[y:y+h, x:x+w]
    left_sBGR = fullFlash_sBGR[y:y+h, x:x+w]
    saveStep.saveReferenceImageBGR(username, imageName, leftEye, 'LeftEye')
    return leftEye, left_sBGR

def getRightEye(username, imageName, image, fullFlash_sBGR, shape):
    (x, y, w, h) = interiorBoundingRect(shape[37], shape[38], shape[41], shape[40])
    rightEye = image[y:y+h, x:x+w]
    right_sBGR = fullFlash_sBGR[y:y+h, x:x+w]
    saveStep.saveReferenceImageBGR(username, imageName, rightEye, 'RightEye')
    return rightEye, right_sBGR

def getEyeWidth(username, imageName, image, shape):
    RightEdge = shape[36] - [50, 0]
    LeftEdge = shape[45] + [50, 0]

    RightEye = list(shape[36:42])
    LeftEye = list(shape[42:48])
    EyePoints = list([RightEdge, LeftEdge])
    EyePoints += LeftEye
    EyePoints += RightEye
    eyesBB = cv2.boundingRect(np.array(EyePoints))
    eyesImg = np.ascontiguousarray(image[eyesBB[1]:(eyesBB[1] + eyesBB[3]), eyesBB[0]:(eyesBB[0] + eyesBB[2])])

    rightPoint = shape[36]
    leftPoint = shape[39]

    corner = np.array([eyesBB[0], eyesBB[1]])
    rightCroppedPoint = tuple(rightPoint - corner)
    leftCroppedPoint = tuple(leftPoint - corner)
    cv2.line(eyesImg, rightCroppedPoint, leftCroppedPoint, (0, 255, 0), 1)

    rightEyeWidth = np.sum((rightPoint - leftPoint) ** 2) ** .5

    rightPoint = shape[42]
    leftPoint = shape[45]

    rightCroppedPoint = tuple(rightPoint - corner)
    leftCroppedPoint = tuple(leftPoint - corner)
    cv2.line(eyesImg, rightCroppedPoint, leftCroppedPoint, (0, 255, 0), 1)
    leftEyeWidth = np.sum((rightPoint - leftPoint) ** 2) ** .5

    saveStep.saveReferenceImageBGR(username, imageName, eyesImg, 'EyesWidth')

    return (rightEyeWidth + leftEyeWidth) / 2

def getDistance(shape, pointA, pointB):
    delta = (shape[pointB] - shape[pointA]) ** 2
    return (delta[0] + delta[1]) ** .5

def createNewGroup(pixelCoord, pixelValue, isClipped):
    [x, y] = pixelCoord
    if not isClipped:
        newGroup = {
            "numPixels": 1,
            "numClippedPixels":0,
            "totalPixels": 1,
            "sumBGR": np.array(pixelValue),
            "lowestX": x,
            "lowestY": y,
            "highestX": x,
            "highestY": y
        }
    else:
        newGroup = {
            "numPixels": 0,
            "numClippedPixels":1,
            "totalPixels": 1,
            "sumBGR": np.array([0.0, 0.0, 0.0]),
            "lowestX": x,
            "lowestY": y,
            "highestX": x,
            "highestY": y
        }

    return newGroup

def combineGroups(group1Id, group2Id, pixelToGroup, groups):
    group1 = groups[group1Id]
    group2 = groups[group2Id]

    group1["numPixels"] = group1["numPixels"] + group2["numPixels"]
    group1["numClippedPixels"] = group1["numClippedPixels"] + group2["numClippedPixels"]
    group1["totalPixels"] = group1["totalPixels"] + group2["totalPixels"]

    group1["sumBGR"] += group2["sumBGR"]

    group1["lowestX"] = group1["lowestX"] if group1["lowestX"] < group2["lowestX"] else group2["lowestX"]
    group1["lowestY"] = group1["lowestY"] if group1["lowestY"] < group2["lowestY"] else group2["lowestY"]
    group1["highestX"] = group1["highestX"] if group1["highestX"] > group2["highestX"] else group2["highestX"]
    group1["highestY"] = group1["highestY"] if group1["highestY"] > group2["highestY"] else group2["highestY"]

    del groups[group2Id]

    for (pixel, group) in pixelToGroup.items():
        if group == group2Id:
            pixelToGroup[pixel] = group1Id

    return [pixelToGroup, groups, group1Id]

def addPixelToGroup(group, pixelCoord, pixelValue, isClipped):
    [x, y] = pixelCoord

    group["totalPixels"] = group["totalPixels"] + 1
    if not isClipped:
        group["numPixels"] = group["numPixels"] + 1
        group["sumBGR"] += pixelValue
    else:
        group["numClippedPixels"] = group["numClippedPixels"] + 1

    group["lowestX"] = group["lowestX"] if group["lowestX"] < x else x
    group["lowestY"] = group["lowestY"] if group["lowestY"] < y else y
    group["highestX"] = group["highestX"] if group["highestX"] > x else x
    group["highestY"] = group["highestY"] if group["highestY"] > y else y

    return group

def getScreenReflection(username, imageName, eyeBGR, eye_sBGR, eyeName, cameraWB_CIE_xy_coords):
    groupCounter = 0
    pixelToGroup = {}
    groups = {}
    largestGroupId = 0

    eyeBGR_d65 = colorTools.whitebalance_from_asShot_to_d65(eyeBGR, cameraWB_CIE_xy_coords)
#    cv2.imshow('d65', eyeBGR_d65)
#    cv2.waitKey(0)
    eyeHSV = colorTools.convert_linearBGR_float_to_linearHSV_float(eyeBGR_d65)

#    plt.clf()
#    plt.plot(eyeHSV[:, :, 1], eyeHSV[:, :, 2], 'k.')

    minimumBrightnessValue = getMinimumBrightnessReflectionThreshold(eyeHSV)
    eyeHSVBrightnessMask = eyeHSV[:, :, 2] > minimumBrightnessValue
    eyeHSVMasked = eyeHSV[eyeHSVBrightnessMask]
    maximumSaturationValue = np.median(eyeHSVMasked[:, 1])

#    potentialValues = eyeHSVMasked[eyeHSVMasked[:, 1] < maximumSaturationValue]
#    print("Potential Values :: " + str(potentialValues))
#    plt.plot(potentialValues[:, 1], potentialValues[:, 2], 'r.')
#    plt.show()


    #Used to check if value is clipped
    #eyeHSV_NoWB = colorTools.convert_linearBGR_float_to_linearHSV_float(eyeBGR)
    #cv2.imshow('Linear', eyeBGR)
    #cv2.imshow('sBGR', eye_sBGR)

    largestValueMask = np.amax(eye_sBGR, axis=2)
    saveStep.saveReferenceImageSBGR(username, imageName, largestValueMask, eyeName + '_sBGR_eye')
    #eyeHSV_NoWB = colorTools.convert_linearBGR_float_to_linearHSV_float(eye_sBGR)

    maxPixelValue = 255

    for x in range(1, len(eyeBGR)):
        for y in range(1, len(eyeBGR[x])):
            neighborLeftGroup = pixelToGroup[(x - 1, y)] if (x - 1, y) in pixelToGroup else False
            neighborTopGroup = pixelToGroup[(x, y - 1)] if (x, y - 1) in pixelToGroup else False

            [b, g, r] = eyeBGR[x, y] #CHECK THAT ARRAY SUPPORTS VALUES LARGE ENOUGH?? IS PRECISION LOST? (probably not...)
            [h, s, v] = eyeHSV[x, y]

            isClipped = False

            #if s < .58 and v > 180:
            #if s < .45 and v > 100:
            #if v > minimumBrightnessValue and s < maximumSaturationValue:
            #    print('V, S :: ' + str(v) + ', ' + str(s))

            if (v > minimumBrightnessValue) and (s < maximumSaturationValue): #Scale proportionally to brightness in rest of eye image
                #print('Reflection...largest value ' + str(largestValueMask[x, y]))
                if largestValueMask[x, y] >= maxPixelValue:#1:
                    #print('Clipped Reflection...hsv ' + str([h, s, v]) + '| rgb ' + str([r, g, b]))
                    isClipped = True

                if neighborLeftGroup:
                    groups[neighborLeftGroup] = addPixelToGroup(groups[neighborLeftGroup], [x, y], [b, g, r], isClipped)
                    pixelToGroup[(x, y)] = neighborLeftGroup
                    largestGroupId = neighborLeftGroup if groups[neighborLeftGroup]["totalPixels"] > groups[largestGroupId]["totalPixels"] else largestGroupId
                elif neighborTopGroup:
                    groups[neighborTopGroup] = addPixelToGroup(groups[neighborTopGroup], [x, y], [b, g, r], isClipped)
                    pixelToGroup[(x, y)] = neighborTopGroup
                    largestGroupId = neighborTopGroup if groups[neighborTopGroup]["totalPixels"] > groups[largestGroupId]["totalPixels"] else largestGroupId
                else:
                    pixelToGroup[(x, y)] = groupCounter
                    groups[groupCounter] = createNewGroup([x, y], [b, g, r], isClipped)
                    if largestGroupId in groups:
                        largestGroupId = groupCounter if groups[groupCounter]["totalPixels"] > groups[largestGroupId]["totalPixels"] else largestGroupId
                    else:
                        largestGroupId = groupCounter
                    groupCounter = groupCounter + 1


                if (neighborLeftGroup and neighborTopGroup) and (neighborLeftGroup != neighborTopGroup):
                    [pixelToGroup, groups, combinedGroupId] = combineGroups(neighborLeftGroup, neighborTopGroup, pixelToGroup, groups)
                    if (largestGroupId == neighborTopGroup) or (largestGroupId == neighborLeftGroup):
                        largestGroupId = combinedGroupId
                    else:
                        largestGroupId = combinedGroupId if groups[combinedGroupId]["totalPixels"] > groups[largestGroupId]["totalPixels"] else largestGroupId
                    

    if len(groups) == 0:
        raise NameError('Could not find reflection in ' + eyeName)

    averageBGR = groups[largestGroupId]["sumBGR"][0] / groups[largestGroupId]["numPixels"]

    (x, y, w, h) = cv2.boundingRect(np.array([(groups[largestGroupId]["lowestX"], groups[largestGroupId]["lowestY"]), (groups[largestGroupId]["highestX"], groups[largestGroupId]["highestY"])]))
    reflection = eyeBGR[x:x+w, y:y+h]

    #Find the values that are not reflections
    reflectionHSV = eyeHSV[x:x+w, y:y+h]
    med = np.median(reflectionHSV[:, :, 2])
    sd = np.std(reflection[:, :, 2])
    lowerBound = med - sd
    lowerBoundMask = reflectionHSV[:, :, 2] < lowerBound

    #Build Non Clipped Mask
    reflection_sBGR = largestValueMask[x:x+w, y:y+h]
    reflectionMask = reflection_sBGR >= maxPixelValue
    reflection[reflectionMask] = [0, 0, 0]


    #Count Valid Pixels
    countNonClippedOnes = np.ones(reflection.shape)
    countNonClippedOnes[reflectionMask] = 0
    countNonClippedOnes[lowerBoundMask] = 0
    countNonClipped = np.sum(countNonClippedOnes)


    #Count All 'Reflection' Pixels
    #totalPixelsOnes = reflection.shape[0] * reflection.shape[1]
    totalPixelsOnes = np.ones(reflection.shape)
    totalPixelsOnes[lowerBoundMask] = 0
    totalPixels = np.sum(totalPixelsOnes)

    clippedPixelsRatio = (totalPixels - countNonClipped) / totalPixels
    print('Clipping Ratio :: ' + str(clippedPixelsRatio))


    reflection[lowerBoundMask] = [0, 0, 0]

    validPixelsMask = np.logical_not(np.logical_or(lowerBoundMask, reflectionMask))
    reflectionPoints = reflection[validPixelsMask]
    sumBGR = np.sum(reflectionPoints, axis=0)
    numPixels = reflectionPoints.shape[0]

    #cv2.imshow('reflection masked', eye_sBGR[x:x+w, y:y+h])
    #cv2.imshow('reflection', reflection)
    #cv2.waitKey(0)
    
    saveStep.saveReferenceImageBGR(username, imageName, reflection, eyeName + '_reflection')
    saveStep.logMeasurement(username, imageName, eyeName + ' OK Pixels In Reflection', str(groups[largestGroupId]["numPixels"]))
    saveStep.logMeasurement(username, imageName, eyeName + ' Clipped Pixels In Reflection', str(groups[largestGroupId]["numClippedPixels"]))
    saveStep.logMeasurement(username, imageName, eyeName + ' Average BGR', str(averageBGR))

    #print('Total Pixels :: ' + str(groups[largestGroupId]["totalPixels"]))
    #print('Clipped Pixels :: ' + str(groups[largestGroupId]["numClippedPixels"]))

    #clippedPixelsRatio = groups[largestGroupId]["numClippedPixels"] / groups[largestGroupId]["totalPixels"]
    clippingThreshold = .5
    #if clippedPixelsRatio > clippingThreshold:
        #return [None, "Clipped Pixel ratio in " + eyeName + " excedded clipping threshold ratio of " + str(clippingThreshold) + ". Ratio :: " + str(clippedPixelsRatio)]
    print('Clipping Ratio :: ' + eyeName + ' ' + str(clippedPixelsRatio))
    if clippedPixelsRatio > clippingThreshold:
        raise NameError("Clipped Pixel ratio in " + eyeName + " excedded clipping threshold ratio of " + str(clippingThreshold) + ". Ratio :: " + str(clippedPixelsRatio))

    #return [[groups[largestGroupId]["sumBGR"], groups[largestGroupId]["numPixels"]], None]
    return [sumBGR, numPixels, [w, h]]

def getMinimumBrightnessReflectionThreshold(eyeHSV):
    values = eyeHSV[:, :, 2]
    values = values.reshape(values.shape[0], values.shape[1])
    SD = np.std(values)
    mean = np.mean(values)
    return mean + (2 * SD)

def getEyeCrops(capture):
    (x, y, w, h) = capture.landmarks.getLeftEyeBB()
    leftEye = capture.image[y:y+h, x:x+w]

    (x, y, w, h) = capture.landmarks.getRightEyeBB()
    rightEye = capture.image[y:y+h, x:x+w]

    return [leftEye, rightEye]

def getAverageScreenReflectionColor(noFlashCapture, halfFlashCapture, fullFlashCapture, cameraWB_CIE_xy_coords):
    [noFlashLeftEyeCrop, noFlashRightEyeCrop] = getEyeCrops(noFlashCapture)
    [halfFlashLeftEyeCrop, halfFlashRightEyeCrop] = getEyeCrops(halfFlashCapture)
    [fullFlashLeftEyeCrop, fullFlashRightEyeCrop] = getEyeCrops(fullFlashCapture)

    [noFlashLeftEyeCrop, halfFlashLeftEyeCrop, fullFlashLeftEyeCrop] = alignImages.cropAndAlignEyes(noFlashLeftEyeCrop, halfFlashLeftEyeCrop, fullFlashLeftEyeCrop)
    [noFlashRightEyeCrop, halfFlashRightEyeCrop, fullFlashRightEyeCrop] = alignImages.cropAndAlignEyes(noFlashRightEyeCrop, halfFlashRightEyeCrop, fullFlashRightEyeCrop)

    cv2.imshow('no flash left eye', noFlashLeftEyeCrop)
    cv2.imshow('no flash right eye', noFlashRightEyeCrop)
    cv2.imshow('half flash left eye', halfFlashLeftEyeCrop)
    cv2.imshow('half flash right eye', halfFlashRightEyeCrop)
    cv2.imshow('full flash left eye', fullFlashLeftEyeCrop)
    cv2.imshow('full flash right eye', fullFlashRightEyeCrop)
    cv2.waitKey(0)








#def getAverageScreenReflectionColor(username, imageName, image, fullFlash_sBGR, imageShape, cameraWB_CIE_xy_coords):
#    leftEye, left_sBGR = getLeftEye(username, imageName, image, fullFlash_sBGR, imageShape)
#    rightEye, right_sBGR = getRightEye(username, imageName, image, fullFlash_sBGR, imageShape)
#
#    #if leftEyeError is not None:
#    eyeWidth = getEyeWidth(username, imageName, image, imageShape)
#    print('Eye Width :: ' + str(eyeWidth))
#
#    leftWidth = 0
#    leftHeight = 0
#    rightWidth = 0
#    rightHeight = 0
#
#    try:
#        leftEyeReflection = getScreenReflection(username, imageName, leftEye, left_sBGR, 'leftEye', cameraWB_CIE_xy_coords)
#    except:
#        print('Setting Left to None!')
#        leftAverageBGR = None
#        leftFluxish = None
#    else:
#        [leftSumBGR, leftNumPixels, [leftWidth, leftHeight]] = leftEyeReflection
#        leftAverageBGR = leftSumBGR / leftNumPixels
#
#        leftWidth = leftWidth / eyeWidth
#        leftHeight = leftHeight / eyeWidth
#
#        leftFluxish = leftWidth * leftHeight * max(leftAverageBGR)
#        print('Left Num Pixels :: ' + str(leftNumPixels))
#        print('Left reflection Width, Height, Area, Fluxish :: ' + str(leftWidth) + ' ' + str(leftHeight) + ' ' + str(leftWidth * leftHeight) + ' ' + str(leftWidth * leftHeight * max(leftAverageBGR)))
#
#    try:
#        rightEyeReflection = getScreenReflection(username, imageName, rightEye, right_sBGR, 'rightEye', cameraWB_CIE_xy_coords)
#    except:
#        print('Setting Right to None!')
#        rightAverageBGR = None
#        rightFluxish = None
#    else:
#        [rightSumBGR, rightNumPixels, [rightWidth, rightHeight]] = rightEyeReflection
#        rightAverageBGR = rightSumBGR / rightNumPixels
#
#        rightWidth = rightWidth / eyeWidth
#        rightHeight = rightHeight / eyeWidth
#
#        rightFluxish = rightWidth * rightHeight * max(rightAverageBGR)
#        print('Right Num Pixels :: ' + str(rightNumPixels))
#        print('Right reflection Width, Height, Area, Fluxish :: ' + str(rightWidth) + ' ' + str(rightHeight) + ' ' + str(rightWidth * rightHeight) + ' ' + str(rightWidth * rightHeight * max(rightAverageBGR)))
#
#
#
#    return [[leftAverageBGR, leftFluxish, [leftWidth, leftHeight]], [rightAverageBGR, rightFluxish, [rightWidth, rightHeight]]]
