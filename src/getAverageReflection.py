import utils
import saveStep
import cv2
import numpy as np
import colorTools
import alignImages
import cropTools
import colorsys

#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

#def interiorBoundingRect(tl, tr, bl, br):
#    topLeftX = tl[0] if tl[0] > bl[0] else bl[0]
#    topLeftY = tl[1] if tl[1] > tr[1] else tr[1]
#
#    targetX = tr[0] if tr[0] < br[0] else br[0]
#    targetY = bl[1] if bl[1] < br[1] else br[1]
#
#    width = targetX - topLeftX
#    height = targetY - topLeftY
#
#    return (topLeftX, topLeftY, width, height)
#
#def getLeftEye(username, imageName, image, fullFlash_sBGR, shape):
#    (x, y, w, h) = interiorBoundingRect(shape[43], shape[44], shape[47], shape[46])
#    leftEye = image[y:y+h, x:x+w]
#    left_sBGR = fullFlash_sBGR[y:y+h, x:x+w]
#    saveStep.saveReferenceImageBGR(username, imageName, leftEye, 'LeftEye')
#    return leftEye, left_sBGR
#
#def getRightEye(username, imageName, image, fullFlash_sBGR, shape):
#    (x, y, w, h) = interiorBoundingRect(shape[37], shape[38], shape[41], shape[40])
#    rightEye = image[y:y+h, x:x+w]
#    right_sBGR = fullFlash_sBGR[y:y+h, x:x+w]
#    saveStep.saveReferenceImageBGR(username, imageName, rightEye, 'RightEye')
#    return rightEye, right_sBGR
#
#def getEyeWidth(username, imageName, image, shape):
#    RightEdge = shape[36] - [50, 0]
#    LeftEdge = shape[45] + [50, 0]
#
#    RightEye = list(shape[36:42])
#    LeftEye = list(shape[42:48])
#    EyePoints = list([RightEdge, LeftEdge])
#    EyePoints += LeftEye
#    EyePoints += RightEye
#    eyesBB = cv2.boundingRect(np.array(EyePoints))
#    eyesImg = np.ascontiguousarray(image[eyesBB[1]:(eyesBB[1] + eyesBB[3]), eyesBB[0]:(eyesBB[0] + eyesBB[2])])
#
#    rightPoint = shape[36]
#    leftPoint = shape[39]
#
#    corner = np.array([eyesBB[0], eyesBB[1]])
#    rightCroppedPoint = tuple(rightPoint - corner)
#    leftCroppedPoint = tuple(leftPoint - corner)
#    cv2.line(eyesImg, rightCroppedPoint, leftCroppedPoint, (0, 255, 0), 1)
#
#    rightEyeWidth = np.sum((rightPoint - leftPoint) ** 2) ** .5
#
#    rightPoint = shape[42]
#    leftPoint = shape[45]
#
#    rightCroppedPoint = tuple(rightPoint - corner)
#    leftCroppedPoint = tuple(leftPoint - corner)
#    cv2.line(eyesImg, rightCroppedPoint, leftCroppedPoint, (0, 255, 0), 1)
#    leftEyeWidth = np.sum((rightPoint - leftPoint) ** 2) ** .5
#
#    saveStep.saveReferenceImageBGR(username, imageName, eyesImg, 'EyesWidth')
#
#    return (rightEyeWidth + leftEyeWidth) / 2
#
#def getDistance(shape, pointA, pointB):
#    delta = (shape[pointB] - shape[pointA]) ** 2
#    return (delta[0] + delta[1]) ** .5
#
#def createNewGroup(pixelCoord, pixelValue, isClipped):
#    [x, y] = pixelCoord
#    if not isClipped:
#        newGroup = {
#            "numPixels": 1,
#            "numClippedPixels":0,
#            "totalPixels": 1,
#            "sumBGR": np.array(pixelValue),
#            "lowestX": x,
#            "lowestY": y,
#            "highestX": x,
#            "highestY": y
#        }
#    else:
#        newGroup = {
#            "numPixels": 0,
#            "numClippedPixels":1,
#            "totalPixels": 1,
#            "sumBGR": np.array([0.0, 0.0, 0.0]),
#            "lowestX": x,
#            "lowestY": y,
#            "highestX": x,
#            "highestY": y
#        }
#
#    return newGroup
#
#def combineGroups(group1Id, group2Id, pixelToGroup, groups):
#    group1 = groups[group1Id]
#    group2 = groups[group2Id]
#
#    group1["numPixels"] = group1["numPixels"] + group2["numPixels"]
#    group1["numClippedPixels"] = group1["numClippedPixels"] + group2["numClippedPixels"]
#    group1["totalPixels"] = group1["totalPixels"] + group2["totalPixels"]
#
#    group1["sumBGR"] += group2["sumBGR"]
#
#    group1["lowestX"] = group1["lowestX"] if group1["lowestX"] < group2["lowestX"] else group2["lowestX"]
#    group1["lowestY"] = group1["lowestY"] if group1["lowestY"] < group2["lowestY"] else group2["lowestY"]
#    group1["highestX"] = group1["highestX"] if group1["highestX"] > group2["highestX"] else group2["highestX"]
#    group1["highestY"] = group1["highestY"] if group1["highestY"] > group2["highestY"] else group2["highestY"]
#
#    del groups[group2Id]
#
#    for (pixel, group) in pixelToGroup.items():
#        if group == group2Id:
#            pixelToGroup[pixel] = group1Id
#
#    return [pixelToGroup, groups, group1Id]
#
#def addPixelToGroup(group, pixelCoord, pixelValue, isClipped):
#    [x, y] = pixelCoord
#
#    group["totalPixels"] = group["totalPixels"] + 1
#    if not isClipped:
#        group["numPixels"] = group["numPixels"] + 1
#        group["sumBGR"] += pixelValue
#    else:
#        group["numClippedPixels"] = group["numClippedPixels"] + 1
#
#    group["lowestX"] = group["lowestX"] if group["lowestX"] < x else x
#    group["lowestY"] = group["lowestY"] if group["lowestY"] < y else y
#    group["highestX"] = group["highestX"] if group["highestX"] > x else x
#    group["highestY"] = group["highestY"] if group["highestY"] > y else y
#
#    return group
#
#def getScreenReflection(username, imageName, eyeBGR, eye_sBGR, eyeName, cameraWB_CIE_xy_coords):
#    groupCounter = 0
#    pixelToGroup = {}
#    groups = {}
#    largestGroupId = 0
#
#    eyeBGR_d65 = colorTools.whitebalance_from_asShot_to_d65(eyeBGR, cameraWB_CIE_xy_coords)
##    cv2.imshow('d65', eyeBGR_d65)
##    cv2.waitKey(0)
#    eyeHSV = colorTools.convert_linearBGR_float_to_linearHSV_float(eyeBGR_d65)
#
##    plt.clf()
##    plt.plot(eyeHSV[:, :, 1], eyeHSV[:, :, 2], 'k.')
#
#    minimumBrightnessValue = getMinimumBrightnessReflectionThreshold(eyeHSV)
#    eyeHSVBrightnessMask = eyeHSV[:, :, 2] > minimumBrightnessValue
#    eyeHSVMasked = eyeHSV[eyeHSVBrightnessMask]
#    maximumSaturationValue = np.median(eyeHSVMasked[:, 1])
#
##    potentialValues = eyeHSVMasked[eyeHSVMasked[:, 1] < maximumSaturationValue]
##    print("Potential Values :: " + str(potentialValues))
##    plt.plot(potentialValues[:, 1], potentialValues[:, 2], 'r.')
##    plt.show()
#
#
#    #Used to check if value is clipped
#    #eyeHSV_NoWB = colorTools.convert_linearBGR_float_to_linearHSV_float(eyeBGR)
#    #cv2.imshow('Linear', eyeBGR)
#    #cv2.imshow('sBGR', eye_sBGR)
#
#    largestValueMask = np.amax(eye_sBGR, axis=2)
#    saveStep.saveReferenceImageSBGR(username, imageName, largestValueMask, eyeName + '_sBGR_eye')
#    #eyeHSV_NoWB = colorTools.convert_linearBGR_float_to_linearHSV_float(eye_sBGR)
#
#    maxPixelValue = 255
#
#    for x in range(1, len(eyeBGR)):
#        for y in range(1, len(eyeBGR[x])):
#            neighborLeftGroup = pixelToGroup[(x - 1, y)] if (x - 1, y) in pixelToGroup else False
#            neighborTopGroup = pixelToGroup[(x, y - 1)] if (x, y - 1) in pixelToGroup else False
#
#            [b, g, r] = eyeBGR[x, y] #CHECK THAT ARRAY SUPPORTS VALUES LARGE ENOUGH?? IS PRECISION LOST? (probably not...)
#            [h, s, v] = eyeHSV[x, y]
#
#            isClipped = False
#
#            #if s < .58 and v > 180:
#            #if s < .45 and v > 100:
#            #if v > minimumBrightnessValue and s < maximumSaturationValue:
#            #    print('V, S :: ' + str(v) + ', ' + str(s))
#
#            if (v > minimumBrightnessValue) and (s < maximumSaturationValue): #Scale proportionally to brightness in rest of eye image
#                #print('Reflection...largest value ' + str(largestValueMask[x, y]))
#                if largestValueMask[x, y] >= maxPixelValue:#1:
#                    #print('Clipped Reflection...hsv ' + str([h, s, v]) + '| rgb ' + str([r, g, b]))
#                    isClipped = True
#
#                if neighborLeftGroup:
#                    groups[neighborLeftGroup] = addPixelToGroup(groups[neighborLeftGroup], [x, y], [b, g, r], isClipped)
#                    pixelToGroup[(x, y)] = neighborLeftGroup
#                    largestGroupId = neighborLeftGroup if groups[neighborLeftGroup]["totalPixels"] > groups[largestGroupId]["totalPixels"] else largestGroupId
#                elif neighborTopGroup:
#                    groups[neighborTopGroup] = addPixelToGroup(groups[neighborTopGroup], [x, y], [b, g, r], isClipped)
#                    pixelToGroup[(x, y)] = neighborTopGroup
#                    largestGroupId = neighborTopGroup if groups[neighborTopGroup]["totalPixels"] > groups[largestGroupId]["totalPixels"] else largestGroupId
#                else:
#                    pixelToGroup[(x, y)] = groupCounter
#                    groups[groupCounter] = createNewGroup([x, y], [b, g, r], isClipped)
#                    if largestGroupId in groups:
#                        largestGroupId = groupCounter if groups[groupCounter]["totalPixels"] > groups[largestGroupId]["totalPixels"] else largestGroupId
#                    else:
#                        largestGroupId = groupCounter
#                    groupCounter = groupCounter + 1
#
#
#                if (neighborLeftGroup and neighborTopGroup) and (neighborLeftGroup != neighborTopGroup):
#                    [pixelToGroup, groups, combinedGroupId] = combineGroups(neighborLeftGroup, neighborTopGroup, pixelToGroup, groups)
#                    if (largestGroupId == neighborTopGroup) or (largestGroupId == neighborLeftGroup):
#                        largestGroupId = combinedGroupId
#                    else:
#                        largestGroupId = combinedGroupId if groups[combinedGroupId]["totalPixels"] > groups[largestGroupId]["totalPixels"] else largestGroupId
#                    
#
#    if len(groups) == 0:
#        raise NameError('Could not find reflection in ' + eyeName)
#
#    averageBGR = groups[largestGroupId]["sumBGR"][0] / groups[largestGroupId]["numPixels"]
#
#    (x, y, w, h) = cv2.boundingRect(np.array([(groups[largestGroupId]["lowestX"], groups[largestGroupId]["lowestY"]), (groups[largestGroupId]["highestX"], groups[largestGroupId]["highestY"])]))
#    reflection = eyeBGR[x:x+w, y:y+h]
#
#    #Find the values that are not reflections
#    reflectionHSV = eyeHSV[x:x+w, y:y+h]
#    med = np.median(reflectionHSV[:, :, 2])
#    sd = np.std(reflection[:, :, 2])
#    lowerBound = med - sd
#    lowerBoundMask = reflectionHSV[:, :, 2] < lowerBound
#
#    #Build Non Clipped Mask
#    reflection_sBGR = largestValueMask[x:x+w, y:y+h]
#    reflectionMask = reflection_sBGR >= maxPixelValue
#    reflection[reflectionMask] = [0, 0, 0]
#
#
#    #Count Valid Pixels
#    countNonClippedOnes = np.ones(reflection.shape)
#    countNonClippedOnes[reflectionMask] = 0
#    countNonClippedOnes[lowerBoundMask] = 0
#    countNonClipped = np.sum(countNonClippedOnes)
#
#
#    #Count All 'Reflection' Pixels
#    #totalPixelsOnes = reflection.shape[0] * reflection.shape[1]
#    totalPixelsOnes = np.ones(reflection.shape)
#    totalPixelsOnes[lowerBoundMask] = 0
#    totalPixels = np.sum(totalPixelsOnes)
#
#    clippedPixelsRatio = (totalPixels - countNonClipped) / totalPixels
#    print('Clipping Ratio :: ' + str(clippedPixelsRatio))
#
#
#    reflection[lowerBoundMask] = [0, 0, 0]
#
#    validPixelsMask = np.logical_not(np.logical_or(lowerBoundMask, reflectionMask))
#    reflectionPoints = reflection[validPixelsMask]
#    sumBGR = np.sum(reflectionPoints, axis=0)
#    numPixels = reflectionPoints.shape[0]
#
#    #cv2.imshow('reflection masked', eye_sBGR[x:x+w, y:y+h])
#    #cv2.imshow('reflection', reflection)
#    #cv2.waitKey(0)
#    
#    saveStep.saveReferenceImageBGR(username, imageName, reflection, eyeName + '_reflection')
#    saveStep.logMeasurement(username, imageName, eyeName + ' OK Pixels In Reflection', str(groups[largestGroupId]["numPixels"]))
#    saveStep.logMeasurement(username, imageName, eyeName + ' Clipped Pixels In Reflection', str(groups[largestGroupId]["numClippedPixels"]))
#    saveStep.logMeasurement(username, imageName, eyeName + ' Average BGR', str(averageBGR))
#
#    #print('Total Pixels :: ' + str(groups[largestGroupId]["totalPixels"]))
#    #print('Clipped Pixels :: ' + str(groups[largestGroupId]["numClippedPixels"]))
#
#    #clippedPixelsRatio = groups[largestGroupId]["numClippedPixels"] / groups[largestGroupId]["totalPixels"]
#    clippingThreshold = .5
#    #if clippedPixelsRatio > clippingThreshold:
#        #return [None, "Clipped Pixel ratio in " + eyeName + " excedded clipping threshold ratio of " + str(clippingThreshold) + ". Ratio :: " + str(clippedPixelsRatio)]
#    print('Clipping Ratio :: ' + eyeName + ' ' + str(clippedPixelsRatio))
#    if clippedPixelsRatio > clippingThreshold:
#        raise NameError("Clipped Pixel ratio in " + eyeName + " excedded clipping threshold ratio of " + str(clippingThreshold) + ". Ratio :: " + str(clippedPixelsRatio))
#
#    #return [[groups[largestGroupId]["sumBGR"], groups[largestGroupId]["numPixels"]], None]
#    return [sumBGR, numPixels, [w, h]]
#
#def getMinimumBrightnessReflectionThreshold(eyeHSV):
#    values = eyeHSV[:, :, 2]
#    values = values.reshape(values.shape[0], values.shape[1])
#    SD = np.std(values)
#    mean = np.mean(values)
#    return mean + (2 * SD)
#
def getEyeCrops(capture):
    (lx, ly, w, h) = capture.landmarks.getLeftEyeBB()
    leftEye = capture.image[ly:ly+h, lx:lx+w]

    (rx, ry, w, h) = capture.landmarks.getRightEyeBB()
    rightEye = capture.image[ry:ry+h, rx:rx+w]

    return [[leftEye, [lx, ly]], [rightEye, [rx, ry]]]

def getEyeRegionCrops(capture):
    (x, y, w, h) = capture.landmarks.getLeftEyeRegionBB()
    leftEye = capture.image[y:y+h, x:x+w]

    (x, y, w, h) = capture.landmarks.getRightEyeRegionBB()
    rightEye = capture.image[y:y+h, x:x+w]

    return [leftEye, rightEye]

def blur(img):
    #return cv2.GaussianBlur(img, (15, 15), 0)
    #return cv2.bilateralFilter(img,15,75,75)
    return cv2.medianBlur(img, 9)


def getReflectionBB(mask):
    img = mask.astype('uint8') * 255
    im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    contour = contours[max_index]

    return cv2.boundingRect(contour)

def maskReflection(noFlash, halfFlash, fullFlash):
    noFlashGrey = np.sum(blur(noFlash), axis=2)
    halfFlashGrey = np.sum(blur(halfFlash), axis=2)
    fullFlashGrey = np.sum(blur(fullFlash), axis=2)

    halfFlashGrey = np.clip(halfFlashGrey.astype('int32') - noFlashGrey, 0, (256 * 3))
    fullFlashGrey = np.clip(fullFlashGrey.astype('int32') - noFlashGrey, 0, (256 * 3))

    noFlashMin = np.min(noFlashGrey)
    noFlashMask = noFlashGrey > (noFlashMin + 10)

    halfFlashMedian = np.median(halfFlashGrey)
    halfFlashStd = np.std(halfFlashGrey)
    halfFlashUpper = halfFlashMedian + (3 * halfFlashStd)
    halfFlashMask = halfFlashGrey > halfFlashUpper

    fullFlashMedian = np.median(fullFlashGrey)
    fullFlashStd = np.std(fullFlashGrey)
    fullFlashUpper = fullFlashMedian + (2 * fullFlashStd)
    fullFlashMask = fullFlashGrey > fullFlashUpper

    flashMask = np.logical_and(halfFlashMask, fullFlashMask)
    flashMask = np.logical_and(flashMask, np.logical_not(noFlashMask))
    return flashMask


def stretchBW(image):
    #median = np.median(image)
    #sd = np.std(image)
    #lower = median - (3 * sd)
    #lower = lower if lower > 0 else 0
    #upper = median + (3 * sd)
    #upper = upper if upper < 256 else 255

    lower = np.min(image)
    upper = np.max(image)

    #print('MEDIAN :: ' + str(median))
    #print('SD :: ' + str(sd))
    #print('LOWER :: ' + str(lower))
    #print('UPPER :: ' + str(upper))

    #bounds = np.copy(gray)
    #bounds[bounds < lower] = lower
    #bounds[bounds > upper] = upper

    numerator = (image - lower).astype('int32')
    denominator = (upper - lower).astype('int32')
    #stretched = (numerator.astype('int32') / denominator.astype('int32'))
    stretched = (numerator / denominator)
    #stretched = np.clip(stretched * 255, 0, 255).astype('uint8')
    stretched = np.clip(stretched * 255, 0, 255).astype('uint8')
    return stretched

def getAverageScreenReflectionColor(noFlashCapture, halfFlashCapture, fullFlashCapture, saveStep):
    [[noFlashLeftEyeCrop, noFlashLeftEyeCoord], [noFlashRightEyeCrop, noFlashRightEyeCoord]] = getEyeCrops(noFlashCapture)
    [[halfFlashLeftEyeCrop, halfFlashLeftEyeCoord], [halfFlashRightEyeCrop, halfFlashRightEyeCoord]] = getEyeCrops(halfFlashCapture)
    [[fullFlashLeftEyeCrop, fullFlashLeftEyeCoord], [fullFlashRightEyeCrop, fullFlashRightEyeCoord]] = getEyeCrops(fullFlashCapture)

    [noFlashLeftEyeRegionCrop, noFlashRightEyeRegionCrop] = getEyeRegionCrops(noFlashCapture)
    [halfFlashLeftEyeRegionCrop, halfFlashRightEyeRegionCrop] = getEyeRegionCrops(halfFlashCapture)
    [fullFlashLeftEyeRegionCrop, fullFlashRightEyeRegionCrop] = getEyeRegionCrops(fullFlashCapture)

    [leftEyeOffsets, [noFlashLeftEyeCrop, halfFlashLeftEyeCrop, fullFlashLeftEyeCrop]] = alignImages.cropAndAlignEyes(noFlashLeftEyeCrop, halfFlashLeftEyeCrop, fullFlashLeftEyeCrop)
    [rightEyeOffsets, [noFlashRightEyeCrop, halfFlashRightEyeCrop, fullFlashRightEyeCrop]] = alignImages.cropAndAlignEyes(noFlashRightEyeCrop, halfFlashRightEyeCrop, fullFlashRightEyeCrop)

    [noFlashLeftEyeRegionCrop, halfFlashLeftEyeRegionCrop, fullFlashLeftEyeRegionCrop] = cropTools.cropImagesToOffsets([noFlashLeftEyeRegionCrop, halfFlashLeftEyeRegionCrop, fullFlashLeftEyeRegionCrop], np.array(leftEyeOffsets))
    #[noFlashLeftEyeRegionCrop, halfFlashLeftEyeRegionCrop, fullFlashLeftEyeRegionCrop] = [noFlashLeftEyeCrop, halfFlashLeftEyeCrop, fullFlashLeftEyeCrop]
    [noFlashRightEyeRegionCrop, halfFlashRightEyeRegionCrop, fullFlashRightEyeRegionCrop] = cropTools.cropImagesToOffsets([noFlashRightEyeRegionCrop, halfFlashRightEyeRegionCrop, fullFlashRightEyeRegionCrop], np.array(rightEyeOffsets))
    #[noFlashRightEyeRegionCrop, halfFlashRightEyeRegionCrop, fullFlashRightEyeRegionCrop] = [noFlashRightEyeCrop, halfFlashRightEyeCrop, fullFlashRightEyeCrop]
    


    leftRemainder = np.clip(np.abs((2 * halfFlashLeftEyeRegionCrop.astype('int32')) - (fullFlashLeftEyeRegionCrop.astype('int32') + noFlashLeftEyeRegionCrop.astype('int32'))), 0, 255).astype('uint8')
    rightRemainder = np.clip(np.abs((2 * halfFlashRightEyeRegionCrop.astype('int32')) - (fullFlashRightEyeRegionCrop.astype('int32') + noFlashRightEyeRegionCrop.astype('int32'))), 0, 255).astype('uint8')

    leftRemainderMask = np.max(leftRemainder, axis=2) > 6
    leftRemainderMask = leftRemainderMask.astype('uint8') * 255
    #leftRemainderMask = np.stack((leftRemainderMask, leftRemainderMask, leftRemainderMask), axis=-1)
    rightRemainderMask = np.max(rightRemainder, axis=2) > 6
    rightRemainderMask = rightRemainderMask.astype('uint8') * 255
    #rightRemainderMask = np.stack((rightRemainderMask, rightRemainderMask, rightRemainderMask), axis=-1)

    #cv2.imshow('left linearity', np.hstack((leftRemainder, leftRemainderMask)))
    #cv2.imshow('right linearity', np.hstack((rightRemainder, rightRemainderMask)))
    #cv2.waitKey(0)
    saveStep.saveReferenceImageBGR(leftRemainderMask, 'Left Remainder Eye Mask')
    saveStep.saveReferenceImageBGR(rightRemainderMask, 'Right Remainder Eye Mask')

    leftEyeGreyReflectionMask = maskReflection(noFlashLeftEyeCrop, halfFlashLeftEyeCrop, fullFlashLeftEyeCrop)
    rightEyeGreyReflectionMask = maskReflection(noFlashRightEyeCrop, halfFlashRightEyeCrop, fullFlashRightEyeCrop)

    fullFlashEyeStripCoords = np.array(fullFlashCapture.landmarks.getEyeStripBB())

    eyeStripCoordDiff_left = np.array(fullFlashLeftEyeCoord) - fullFlashEyeStripCoords[0:2]
    eyeStripCoordDiff_right = np.array(fullFlashRightEyeCoord) - fullFlashEyeStripCoords[0:2]

    (x, y, w, h) = fullFlashEyeStripCoords
    fullFlashEyeStripXStart = x
    fullFlashEyeStripXEnd = x + w
    fullFlashEyeStrip = fullFlashCapture.image[y:y+h, x:x+w]
    #fullFlashEyeStrip = halfFlashCapture.image[y:y+h, x:x+w]

    eyeStripCoordDiff_left += leftEyeOffsets[2]
    eyeStripCoordDiff_right += rightEyeOffsets[2]


    #leftEyeReflectionMask = np.stack((leftEyeGreyReflectionMask, leftEyeGreyReflectionMask, leftEyeGreyReflectionMask), axis=-1)
    #rightEyeReflectionMask = np.stack((rightEyeGreyReflectionMask, rightEyeGreyReflectionMask, rightEyeGreyReflectionMask), axis=-1)

    #leftEye = halfFlashLeftEyeCrop * leftEyeReflectionMask
    #rightEye = halfFlashRightEyeCrop * rightEyeReflectionMask

    #reflections = []

    [x, y, w, h] = getReflectionBB(leftEyeGreyReflectionMask)
    #[bb_x, bb_y] = np.array([x, y]) + eyeStripCoordDiff_left
    #cv2.rectangle(fullFlashEyeStrip, (bb_x, bb_y), (bb_x + w, bb_y + h), (0, 0, 255), 1)

    eyeSlitTop = y + eyeStripCoordDiff_left[1]
    eyeSlitBottom = y + h + eyeStripCoordDiff_right[1]


    leftEyeCenter = np.array([x + int(w / 2), y + int(h / 2)])
    print('Left Eye Center! :: ' + str(leftEyeCenter))
    eyeStripCoordDiff_left += leftEyeCenter

    leftEyeX = eyeStripCoordDiff_left[0]

    #cv2.circle(fullFlashEyeStrip, (eyeStripCoordDiff_left[0], eyeStripCoordDiff_left[1]), 5, (0, 255, 0), -1)

    leftEyeReflection = halfFlashLeftEyeCrop[y:y+h, x:x+w]

    leftHighMask = np.max(leftEyeReflection, axis=2) < 253
    leftLowMask = np.min(leftEyeReflection, axis=2) >= 2

    leftEyeMask = np.logical_and(leftHighMask, leftLowMask)
    leftEyePoints = leftEyeReflection[leftEyeMask]
    leftClipRatio = leftEyePoints.shape[0] / (leftEyeMask.shape[0] * leftEyeMask.shape[1])
    print('LEFT CLIP RATIO :: ' + str(leftClipRatio))
    if leftClipRatio < .9:
        print("TOO MUCH CLIPPING!")
        raise NameError('Not enough clean non-clipped pixels in left eye reflections')
    #else:
    #    leftReflectionMedian = np.median(leftEyePoints, axis=0) * 2 #Multiply by 2 because we got the value from the half flash
    #    leftReflectionArea = w * h
    #    #leftReflectionValue = np.max(leftReflectionMedian)
    #    reflections.append([leftReflectionMedian, leftReflectionArea])
    #print("LEFT EYE POINTS :: " + str(leftEyePoints))
    leftReflectionMedian = np.median(leftEyePoints, axis=0) * 2 #Multiply by 2 because we got the value from the half flash
    leftReflectionWidth = w 
    leftReflectionHeight = h
    leftReflectionValue = np.max(leftReflectionMedian)


    [x, y, w, h] = getReflectionBB(rightEyeGreyReflectionMask)

    eyeSlitTop = y + eyeStripCoordDiff_left[1] if y + eyeStripCoordDiff_left[1] < eyeSlitTop else eyeSlitTop
    eyeSlitBottom = y + h + eyeStripCoordDiff_right[1] if y + h + eyeStripCoordDiff_right[1] > eyeSlitBottom else eyeSlitBottom


    #eyeSlitHSL = colorsys.rgb_to_hsl(leftReflectionMedian[2], leftReflectionMedian[1], leftReflectionMedian[0])

    #[bb_x, bb_y] = np.array([x, y]) + eyeStripCoordDiff_right
    #cv2.rectangle(fullFlashEyeStrip, (bb_x, bb_y), (bb_x + w, bb_y + h), (0, 0, 255), 1)

    rightEyeCenter = [x + int(w / 2), y + int(h / 2)]
    print('Right Eye Center! :: ' + str(rightEyeCenter))
    eyeStripCoordDiff_right += rightEyeCenter
    #rightEyeX = eyeStripCoordDiff_right[0] + x

    #cv2.circle(fullFlashEyeStrip, (eyeStripCoordDiff_right[0], eyeStripCoordDiff_right[1]), 5, (0, 0, 255), -1)
    #cv2.imshow('full flash eye strip', fullFlashEyeStrip)
    #cv2.waitKey(0)
    #cv2.waitKey(0)

    rightEyeReflection = halfFlashRightEyeCrop[y:y+h, x:x+w]

    rightHighMask = np.max(rightEyeReflection, axis=2) < 253
    rightLowMask = np.min(rightEyeReflection, axis=2) >= 2

    rightEyeMask = np.logical_and(rightHighMask, rightLowMask)
    rightEyePoints = rightEyeReflection[rightEyeMask]
    rightClipRatio = rightEyePoints.shape[0] / (rightEyeMask.shape[0] * rightEyeMask.shape[1])
    print('RIGHT CLIP RATIO :: ' + str(rightClipRatio))
    if rightClipRatio < .9:
        print("TOO MUCH CLIPPING!")
        raise NameError('Not enough clean non-clipped pixels in right eye reflections')
    #else:
    #    rightReflectionMedian = np.median(rightEyePoints, axis=0) * 2 #Multiply by 2 because we got the value from the half flash
    #    rightReflectionArea = w * h
    #    #rightReflectionValue = np.max(rightReflectionMedian)
    #    reflections.append([rightReflectionMedian, rightReflectionArea])
    #print("RIGHT EYE POINTS :: " + str(rightEyePoints))
    rightReflectionMedian = np.median(rightEyePoints, axis=0) * 2 #Multiply by 2 because we got the value from the half flash
    rightReflectionWidth = w
    rightReflectionHeight = h
    rightReflectionValue = np.max(rightReflectionMedian)
    
    
    rightEyeX = eyeStripCoordDiff_right[0]
    eyeDistance = int((leftEyeX - rightEyeX) / 3)

    print('Eyes left, right :: ' + str(leftEyeX) + ' ' + str(rightEyeX))
    leftEyeLeftEdge = (leftEyeX + eyeDistance) if (leftEyeX + eyeDistance) < fullFlashEyeStrip.shape[1] else fullFlashEyeStrip.shape[1]
    leftEyeRightEdge = (leftEyeX - eyeDistance) if (leftEyeX - eyeDistance) > 0 else 0

    rightEyeLeftEdge = (rightEyeX + eyeDistance) if (rightEyeX + eyeDistance) < fullFlashEyeStrip.shape[1] else fullFlashEyeStrip.shape[1]
    rightEyeRightEdge = (rightEyeX - eyeDistance) if (rightEyeX - eyeDistance) > 0 else 0
    #print('Edges left, right :: ' + str(leftEdge) + ' ' + str(rightEdge))

    #eyeSlitCrop = fullFlashEyeStrip[eyeSlitTop - 50:eyeSlitBottom + 50, rightEdge:leftEdge]

    margin = 50
    leftEyeSlit = fullFlashEyeStrip[eyeSlitTop - margin:eyeSlitBottom + margin, leftEyeRightEdge:leftEyeLeftEdge]
    leftEyeSlitMiddle = int(leftEyeSlit.shape[1]/2)
    leftEyeSlit[:, (leftEyeSlitMiddle - int(leftEyeSlitMiddle / 3)):(leftEyeSlitMiddle + int(leftEyeSlitMiddle / 3))] = 0

    rightEyeSlit = fullFlashEyeStrip[eyeSlitTop - margin:eyeSlitBottom + margin, rightEyeRightEdge:rightEyeLeftEdge]
    rightEyeSlitMiddle = int(rightEyeSlit.shape[1]/2)
    rightEyeSlit[:, (rightEyeSlitMiddle - int(rightEyeSlitMiddle / 3)):(rightEyeSlitMiddle + int(rightEyeSlitMiddle / 3))] = 0

    #cv2.imshow('left', leftEyeSlit)
    #cv2.imshow('right', rightEyeSlit)
    #cv2.waitKey(0)


    #eyeSlitCrop = fullFlashEyeStrip[eyeSlitTop - 50:eyeSlitBottom + 50, rightEdge:leftEdge]
    #eyeSlitCrop = fullFlashEyeStrip[eyeSlitTop - 50:eyeSlitBottom + 50, rightEyeX:leftEyeX]


    #eyeSlitCrop = cv2.GaussianBlur(eyeSlitCrop, (51, 51), 0)
    leftEyeSlitHLS = cv2.cvtColor(leftEyeSlit, cv2.COLOR_BGR2HLS_FULL)
    rightEyeSlitHLS = cv2.cvtColor(rightEyeSlit, cv2.COLOR_BGR2HLS_FULL)
    #print('Eye Slit HLS :: ' + str(eyeSlitHLS))
    #eyeSlitH = eyeSlitHLS[:, :, 0]
    #eyeSlitH = cv2.Sobel(eyeSlitH, cv2.CV_64F, 1, 1, ksize=5)
    #eyeSlitH = (eyeSlitH < 26).astype('int32') * 255
    #eyeSlitHOriginal = eyeSlitHLS[:, :, 0]
    #eyeSlitH = np.copy(eyeSlitHOriginal).astype('int32')
    #eyeSlitHigh = np.clip(eyeSlitH - 85, 0, 255) #Shift Hue by 1/3 of range to move red/yellow away from Hue boundary
    #eyeSlitLow = np.clip(eyeSlitH + 170, 0, 255)

    #eyeSlitH[eyeSlitHOriginal >= 85] = eyeSlitHigh[eyeSlitHOriginal >= 85]
    #eyeSlitH[eyeSlitHOriginal < 85] = eyeSlitLow[eyeSlitHOriginal < 85]

    #hueBound = 230
    leftEyeSlitH = (leftEyeSlitHLS[:, :, 0]).astype('int32')
    #leftEyeSlitH[leftEyeSlitH > hueBound] = 0
    #leftEyeSlitH = stretchBW(leftEyeSlitH)
    rightEyeSlitH = (rightEyeSlitHLS[:, :, 0]).astype('int32')
    #rightEyeSlitH[rightEyeSlitH > hueBound] = 0
    #rightEyeSlitH = stretchBW(rightEyeSlitH)



    leftEyeSlitL = (leftEyeSlitHLS[:, :, 1]).astype('int32')
    rightEyeSlitL = (rightEyeSlitHLS[:, :, 1]).astype('int32')
    leftEyeSlitS = (leftEyeSlitHLS[:, :, 2]).astype('int32')
    rightEyeSlitS = (rightEyeSlitHLS[:, :, 2]).astype('int32')

    leftEyeSlitDiff = np.clip(leftEyeSlitS - leftEyeSlitL, 0, 255).astype('uint8')
    rightEyeSlitDiff = np.clip(rightEyeSlitS - rightEyeSlitL, 0, 255).astype('uint8')
    #eyeSlitTest = cv2.GaussianBlur(eyeSlitTest, (51, 51), 0)

    #med = np.median(eyeSlitTest)
    #sd = np.std(eyeSlitTest)

    #lowerbound = med #+ (2 * sd)

    #eyeSlitTest4 = (eyeSlitTest > lowerbound).astype('uint8') * 255


    #eyeSlitTest2 = cv2.Sobel(eyeSlitTest, cv2.CV_16U, 1, 1, ksize=5)
    #eyeSlitTest3 = cv2.GaussianBlur(eyeSlitTest2, (5, 5), 0)

    #cv2.imshow('Eye Slit Crop H', eyeSlitH.astype('uint8'))
    #cv2.imshow('Eye Slit Crop L', eyeSlitL.astype('uint8'))
    #cv2.imshow('Eye Slit Crop S', eyeSlitS.astype('uint8'))
    #cv2.imshow('Eye Slit Crop Test', eyeSlitTest.astype('uint8'))
   # cv2.imshow('Eye Slit Crop Test 2', eyeSlitTest2.astype('uint8'))
   # cv2.imshow('Eye Slit Crop Test 3', eyeSlitTest3.astype('uint8'))
    #cv2.imshow('Eye Slit Crop Test 4', eyeSlitTest4.astype('uint8'))


    #cv2.imshow('left no stretch', leftEyeSlitDiff)
    #cv2.imshow('right no stretch', rightEyeSlitDiff)

    leftEyeSlitDiff1 = stretchBW(leftEyeSlitDiff)
    rightEyeSlitDiff1 = stretchBW(rightEyeSlitDiff)
    
    #margin = 20
    #leftCenter = int(leftEyeSlitDiff.shape[0] / 2)
    #cv2.line(leftEyeSlitDiff, (0, leftCenter), (leftEyeSlitDiff.shape[1], leftCenter), (255, 255, 255))
    #cv2.line(leftEyeSlitDiff, (0, leftCenter + margin), (leftEyeSlitDiff.shape[1], leftCenter + margin), (255, 255, 255))
    #cv2.line(leftEyeSlitDiff, (0, leftCenter - margin), (leftEyeSlitDiff.shape[1], leftCenter - margin), (255, 255, 255))

    #rightCenter = int(rightEyeSlitDiff.shape[0] / 2)
    #cv2.line(rightEyeSlitDiff, (0, rightCenter), (rightEyeSlitDiff.shape[1], rightCenter), (255, 255, 255))
    #cv2.line(rightEyeSlitDiff, (0, rightCenter + margin), (rightEyeSlitDiff.shape[1], rightCenter + margin), (255, 255, 255))
    #cv2.line(rightEyeSlitDiff, (0, rightCenter - margin), (rightEyeSlitDiff.shape[1], rightCenter - margin), (255, 255, 255))

    #cv2.imshow('Left Slit Crop', leftEyeSlitDiff1)
    #cv2.imshow('Right Slit Crop', rightEyeSlitDiff1)

    threshold = 20
    leftEyeSlitDiff2 = (leftEyeSlitDiff1 > threshold).astype('uint8') * 255
    rightEyeSlitDiff2 = (rightEyeSlitDiff1 > threshold).astype('uint8') * 255

    #cv2.imshow('Left Slit Crop mask', leftEyeSlitDiff)
    #cv2.imshow('Right Slit Crop mask', rightEyeSlitDiff)

    kernel = np.ones((7, 5), np.uint8)
    #kernel = np.ones((7, 3), np.uint8)
    #algo = cv2.MORPH_CLOSE
    algo = cv2.MORPH_CROSS
    print('ALGO :: ' + str(algo))

    leftEyeSlitDiff3 = cv2.morphologyEx(leftEyeSlitDiff2, algo, kernel)
    rightEyeSlitDiff3 = cv2.morphologyEx(rightEyeSlitDiff2, algo, kernel)

    leftEyeSlitDiff3 = cv2.morphologyEx(leftEyeSlitDiff3, cv2.MORPH_OPEN, kernel)
    rightEyeSlitDiff3 = cv2.morphologyEx(rightEyeSlitDiff3, cv2.MORPH_OPEN, kernel)

    margin = 40
    leftCenter = int(leftEyeSlitDiff.shape[0] / 2)
#    cv2.line(leftEyeSlitDiff3, (0, leftCenter), (leftEyeSlitDiff3.shape[1], leftCenter), (255, 255, 255))
#    cv2.line(leftEyeSlitDiff3, (0, leftCenter + margin), (leftEyeSlitDiff3.shape[1], leftCenter + margin), (255, 255, 255))
#    cv2.line(leftEyeSlitDiff3, (0, leftCenter - margin), (leftEyeSlitDiff3.shape[1], leftCenter - margin), (255, 255, 255))
#
    rightCenter = int(rightEyeSlitDiff.shape[0] / 2)
#    cv2.line(rightEyeSlitDiff3, (0, rightCenter), (rightEyeSlitDiff3.shape[1], rightCenter), (255, 255, 255))
#    cv2.line(rightEyeSlitDiff3, (0, rightCenter + margin), (rightEyeSlitDiff3.shape[1], rightCenter + margin), (255, 255, 255))
#    cv2.line(rightEyeSlitDiff3, (0, rightCenter - margin), (rightEyeSlitDiff3.shape[1], rightCenter - margin), (255, 255, 255))

    leftEyeRows = leftEyeSlitDiff3[leftCenter - margin:leftCenter + margin]
    leftEyeRows = leftEyeRows == 0
    rightEyeRows = rightEyeSlitDiff3[rightCenter - margin:rightCenter + margin]
    rightEyeRows = rightEyeRows == 0

    #cv2.imshow('left eye rows', leftEyeRows.astype('uint8')*255)
    #cv2.imshow('right eye rows', rightEyeRows.astype('uint8')*255)


    leftEyeBB = getReflectionBB(leftEyeRows)
    cv2.line(leftEyeSlitDiff1, (leftEyeBB[0], leftEyeBB[1]), (leftEyeBB[0], leftEyeBB[1] + leftEyeBB[3]), (255, 255, 255))
    cv2.line(leftEyeSlitDiff1, (leftEyeBB[0] + leftEyeBB[2], leftEyeBB[1]), (leftEyeBB[0] + leftEyeBB[2], leftEyeBB[1] + leftEyeBB[3]), (255, 255, 255))
    print('left eye bb :: ' + str(leftEyeBB))
    leftEyeWidth = leftEyeBB[2]

    rightEyeBB = getReflectionBB(rightEyeRows)
    cv2.line(rightEyeSlitDiff1, (rightEyeBB[0], rightEyeBB[1]), (rightEyeBB[0], rightEyeBB[1] + rightEyeBB[3]), (255, 255, 255))
    cv2.line(rightEyeSlitDiff1, (rightEyeBB[0] + rightEyeBB[2], rightEyeBB[1]), (rightEyeBB[0] + rightEyeBB[2], rightEyeBB[1] + rightEyeBB[3]), (255, 255, 255))
    print('right eye bb :: ' + str(rightEyeBB))
    rightEyeWidth = rightEyeBB[2]

    #NOTE: USING MAX MIGHT BE MORE ACCURATE....
    averageEyeWidth = int(round((rightEyeWidth + leftEyeWidth) / 2))

    print('RIGHT EYE WIDTH :: ' + str(rightEyeWidth))
    print('LEFT EYE WIDTH :: ' + str(leftEyeWidth))
    print('AVERAGE EYE WIDTH :: ' + str(averageEyeWidth))

    #blur = 5
    #leftEyeSlitDiff = cv2.GaussianBlur(leftEyeSlitDiff, (blur, blur), 0)
    #rightEyeSlitDiff = cv2.GaussianBlur(rightEyeSlitDiff, (blur, blur), 0)

    #threshold = 64
    #leftEyeSlitDiff = (leftEyeSlitDiff > threshold).astype('uint8') * 255
    #rightEyeSlitDiff = (rightEyeSlitDiff > threshold).astype('uint8') * 255

    leftEyeSlitStack = np.vstack((leftEyeSlitH.astype('uint8'), leftEyeSlitS.astype('uint8'), leftEyeSlitDiff1, leftEyeSlitDiff2, leftEyeSlitDiff3))
    rightEyeSlitStack = np.vstack((rightEyeSlitH.astype('uint8'), rightEyeSlitS.astype('uint8'), rightEyeSlitDiff1, rightEyeSlitDiff2, rightEyeSlitDiff3))

    #cv2.imshow('Eye Mask Comparison', np.hstack((rightEyeSlitStack, leftEyeSlitStack)))
    #cv2.waitKey(0)

    valuesDiff = np.abs((rightReflectionMedian - leftReflectionMedian))
    leftReflectionHSV = colorsys.rgb_to_hsv(leftReflectionMedian[2], leftReflectionMedian[1], leftReflectionMedian[0])
    rightReflectionHSV = colorsys.rgb_to_hsv(rightReflectionMedian[2], rightReflectionMedian[1], rightReflectionMedian[0])

    print('rightReflectionMedian :: ' + str(rightReflectionMedian))
    print('right HSV :: ' + str(rightReflectionHSV))
    print('leftReflectionMedian :: ' + str(leftReflectionMedian))
    print('left HSV :: ' + str(leftReflectionHSV))

    hueDiff = np.abs(leftReflectionHSV[0] - rightReflectionHSV[0])
    satDiff = np.abs(leftReflectionHSV[1] - rightReflectionHSV[1])

    print('HUE and SAT diff :: ' + str(hueDiff) + ' | ' + str(satDiff)) 

    print('Values Diff :: ' + str(valuesDiff))


    averageMedian = (leftReflectionMedian + rightReflectionMedian) / 2
    #averageMedian = rightReflectionMedian
    averageValue = (leftReflectionValue + rightReflectionValue) / 2
    #averageValue = rightReflectionValue

    leftReflectionArea = (leftReflectionWidth / averageEyeWidth) * (leftReflectionHeight / averageEyeWidth)
    rightReflectionArea = (rightReflectionWidth / averageEyeWidth) * (rightReflectionHeight / averageEyeWidth)
    averageArea = (leftReflectionArea + rightReflectionArea) / 2
    #averageArea = rightReflectionArea

    fluxish = averageArea * averageValue

    return [averageMedian, fluxish]


#def getAverageScreenReflectionColor(username, imageName, image, fullFlash_sBGR, imageShape, cameraWB_CIE_xy_coords):
#    leftEye, left_sBGR = getLeftEye(username, imageName, image, fullFlash_sBGR, imageShape)
#    rightEye, right_sBGR = getRightEye(username, imageName, image, fullFlash_sBGR, imageShape)
#
#    #if leftEyeError is not None:
#    eyeWidth = getEyeWidth(username, imageName, image, imageShape)
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
