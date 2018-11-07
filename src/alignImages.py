import utils
import numpy as np
import cv2
import imutils
import colorTools
import math

BLUR_SIZE = 11

def updateShape(shape, xyDiff):
    newShape = np.copy(shape)
    (dX, dY) = xyDiff
    for index, value in enumerate(shape):
        (vX, vY) = value
        newShape[index] = (vX - dX, vY - dY)

    return newShape

X = 0
Y = 1
W = 2
H = 3

bbMargin = .01 #make the bounding boxes % larger in each Coord (2x% total in each direction)

def getPreparedAlt(image, shape):
    gray = cv2.cvtColor(np.clip(image * 255, 0, 255).astype('uint8'), cv2.COLOR_BGR2GRAY)

    shapeCulled = shape[17:]

    grayHull = cv2.convexHull(shapeCulled)
    
    mask = gray.copy()
    mask.fill(0)
    mask = cv2.fillConvexPoly(mask, grayHull, 1).astype('bool')

    gray = cv2.GaussianBlur(gray, (11, 11), 0)
    prepped = cv2.Laplacian(gray, cv2.CV_64F)
    #prepped = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=5)

    preppedMasked = prepped * mask
    return preppedMasked

#def getPreparedBaseImage(baseImage, baseShape):
#    grayBase = cv2.cvtColor(np.clip(baseImage * 255, 0, 255).astype('uint8'), cv2.COLOR_BGR2GRAY)
#
#    #CREATE A MASK FOR THE INNER FACE REGION
#    baseShapeCroppedCulled = baseShape[17:]
#
#    grayHullBase = cv2.convexHull(baseShapeCroppedCulled)
#    
#    baseMask = grayBase.copy()
#    baseMask.fill(0)
#    baseMask = cv2.fillConvexPoly(baseMask, grayHullBase, 1).astype('bool')
#
#    #GET POINTS IN MASK
#    baseFacePoints = grayBase[baseMask] # Only take the median of points in the face
#
#    #gamma = 1.0
#    #grayBase = ((grayBase/255 ** (1/gamma)) * 255).astype('uint8')
#    grayBase = cv2.GaussianBlur(grayBase, (BLUR_SIZE, BLUR_SIZE), 0)
#
#    #MODIFIED AUTOCANNY
#    #sigma = .33
#    sigma = .25
#
#    maskMedian = np.median(baseFacePoints)
#    baseLower = int(max(0, (1.0 - sigma) * maskMedian))
#    baseUpper = int(min(255, (1.0 + sigma) * maskMedian))
#    edgeBase = cv2.Canny(grayBase, baseLower, baseUpper)
#    edgeBase = edgeBase * baseMask
#    #Need to convert to a float64 for phaseCorrelation np.float64(edgeBase)
#    return edgeBase
#
#
#
#def getPreparedFlashImage(flashImage, flashShape, gamma):
#    grayFlash = cv2.cvtColor(np.clip(flashImage * 255, 0, 255).astype('uint8'), cv2.COLOR_BGR2GRAY)
#
#    #CREATE A MASK FOR THE INNER FACE REGION
#    flashShapeCroppedCulled = flashShape[17:]
#
#    grayHullFlash = cv2.convexHull(flashShapeCroppedCulled)
#
#    flashMask = grayFlash.copy()
#    flashMask.fill(0)
#    flashMask = cv2.fillConvexPoly(flashMask, grayHullFlash, 1).astype('bool')
#
#    #GET POINTS IN MASK
#    #gamma = 1.3
#    #grayFlash = ((grayFlash/255 ** (1/gamma)) * 255).astype('uint8')
#    flashFacePoints = grayFlash[flashMask] # Only take the median of points in the face
#    grayFlash = cv2.GaussianBlur(grayFlash, (15, 15), 0)
#
#    #MODIFIED AUTOCANNY
#    #sigma = .33
#    sigma = .25
#
#    maskMedian = np.median(flashFacePoints)
#
#    flashLower = int(max(0, (1.0 - sigma) * maskMedian))
#    flashUpper = int(min(255, (1.0 + sigma) * maskMedian))
#    edgeFlash = cv2.Canny(grayFlash, flashLower, flashUpper)
#
#    edgeFlash = edgeFlash * flashMask
#    return edgeFlash

def calculateOffset(preparedBaseImage, preparedFlashImage):
    (offset, response) = cv2.phaseCorrelate(np.float64(preparedBaseImage), np.float64(preparedFlashImage))
    offset = list(offset)
    offset = [round(value) for value in offset]
    print("Offset :: " + str(offset))
    return offset

X=0
Y=1

def getXOffset(elem):
    return elem[0]

def getYOffset(elem):
    return elem[1]

def getOrder(elem):
    return elem[3]

def cropToOffset(imagesSortedByOffset, axis):
    cropped = []

    targetOffset = imagesSortedByOffset[0][axis] #SmallestOffsetValue
    largestOffset = imagesSortedByOffset[-1][axis] #LargestOffsetValue
    axisSizeReduction = abs(targetOffset - largestOffset) #Used to figure out how much to reduce the length of each axis

    for capture in imagesSortedByOffset:
        imageCapture = capture[2]
        imageOffset = capture[axis]

        coordDelta = abs(targetOffset - imageOffset)
        axisSizeDelta = abs(axisSizeReduction - coordDelta)

        shapeMask = [0, 0]
        shapeMask[axis] = coordDelta

        (image, shape, mask) = imageCapture
        if axis == Y:
            image = image[coordDelta:, :]
            mask = mask[coordDelta:, :]

            if axisSizeDelta != 0:
                image = image[0:-axisSizeDelta, :]
                mask = mask[0:-axisSizeDelta, :]
                capture.append([0, coordDelta, -axisSizeDelta])
            else:
                capture.append([0, coordDelta, None])

        else:
            image = image[:, coordDelta:]
            mask = mask[:, coordDelta:]

            if axisSizeDelta != 0:
                image = image[:, 0:-axisSizeDelta]
                mask = mask[:, 0:-axisSizeDelta]
                capture.append([1, coordDelta, -axisSizeDelta])
            else:
                capture.append([1, coordDelta, None])

        shape = shape - shapeMask

        capture[2] = (image, shape, mask)
        capture[axis] = 0

        cropped.append(capture)

    return cropped


def cropAndAlign(base, fullFlash, topFlash, bottomFlash, imageName):
    (baseImage, baseShape, baseMask) = base
    (fullFlashImage, fullFlashShape, fullFlashMask) = fullFlash
    #(fullFlashImage_sBGR, fullFlashShape_sBGR, fullFlashMask_sBGR) = fullFlash_sBGR
    (topFlashImage, topFlashShape, topFlashMask) = topFlash
    (bottomFlashImage, bottomFlashShape, bottomFlashMask) = bottomFlash


    #preparedBaseImage = getPreparedBaseImage(baseImage, baseShape)
    #preparedFullFlashImage = getPreparedFlashImage(fullFlashImage, fullFlashShape, 1.1)
    #preparedTopFlashImage = getPreparedFlashImage(topFlashImage, topFlashShape, 1.1)
    #preparedBottomFlashImage = getPreparedFlashImage(bottomFlashImage, bottomFlashShape, 1.1)

    preparedBaseImage = getPreparedAlt(baseImage, baseShape)
    preparedFullFlashImage = getPreparedAlt(fullFlashImage, fullFlashShape)
    preparedTopFlashImage = getPreparedAlt(topFlashImage, topFlashShape)
    preparedBottomFlashImage = getPreparedAlt(bottomFlashImage, bottomFlashShape)

    #cv2.imshow("Prepared", np.hstack([cv2.resize(preparedBaseImage, (0, 0), fx=.5, fy=.5), cv2.resize(preparedTopFlashImage, (0, 0), fx=.5, fy=.5), cv2.resize(preparedBottomFlashImage, (0, 0), fx=.5, fy=.5), cv2.resize(preparedFullFlashImage, (0, 0), fx=.5, fy=.5)]))
    #cv2.waitKey(0)

    baseOffset = [0, 0] #All offsets are relative to baseImage
    fullFlashOffset = calculateOffset(preparedBaseImage, preparedFullFlashImage)
    #fullFlashOffset_sBGR = fullFlashOffset.copy()
    topFlashOffset = calculateOffset(preparedBaseImage, preparedTopFlashImage)
    bottomFlashOffset = calculateOffset(preparedBaseImage, preparedBottomFlashImage)

    justOffsets = [baseOffset, fullFlashOffset, topFlashOffset, bottomFlashOffset]#, fullFlashOffset_sBGR]
    justOffsets.sort(key=getXOffset, reverse=True)
    largestX = justOffsets[0][X]

    justOffsets.sort(key=getYOffset, reverse=True)
    largestY = justOffsets[0][Y]

    baseOffset.append(base)
    baseOffset.append(0) #Want to keep track of ordering...

    fullFlashOffset.append(fullFlash)
    fullFlashOffset.append(1)

    #fullFlashOffset_sBGR.append(fullFlash_sBGR)
    #fullFlashOffset_sBGR.append(2)

    topFlashOffset.append(topFlash)
    topFlashOffset.append(2)

    bottomFlashOffset.append(bottomFlash)
    bottomFlashOffset.append(3)

    offsets = [baseOffset, fullFlashOffset, topFlashOffset, bottomFlashOffset]

    offsets.sort(key=getXOffset)
    offsets = cropToOffset(offsets, X)

    offsets.sort(key=getYOffset)
    offsets = cropToOffset(offsets, Y)

    offsets.sort(key=getOrder)

    #print('Offsets :: ' + str(offsets))

    [baseCropped, fullFlashCropped, topFlashCropped, bottomFlashCropped] = [offset[2] for offset in offsets]

    #cv2.imshow("Cropped", np.hstack([cv2.resize(baseCropped[0], (0, 0), fx=.5, fy=.5), cv2.resize(topFlashCropped[0], (0, 0), fx=.5, fy=.5), cv2.resize(bottomFlashCropped[0], (0, 0), fx=.5, fy=.5), cv2.resize(fullFlashCropped[0], (0, 0), fx=.5, fy=.5)]))
    #cv2.waitKey(0)


    offsetDistance = math.sqrt(math.pow(largestX, 2) + math.pow(largestY, 2))
    #print("Offset Distance! :: " + str(offsetDistance))
    #if offsetDistance > 100:
        #return [None, 'Possible Alignment Error. Total offset distance :: ' + str(offsetDistance) + '. Returning Early']
    croppedImages = [baseCropped, fullFlashCropped, topFlashCropped, bottomFlashCropped]
    crop = [[offset[-2], offset[-1]] for offset in offsets]

    return [croppedImages, crop]
