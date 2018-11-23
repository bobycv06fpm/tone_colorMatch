import utils
import numpy as np
import cv2
import colorTools
import math
import cropTools

BLUR_SIZE = 11

def updateShape(shape, xyDiff):
    newShape = np.copy(shape)
    (dX, dY) = xyDiff
    for index, value in enumerate(shape):
        (vX, vY) = value
        newShape[index] = (vX - dX, vY - dY)

    return newShape

#X = 0
#Y = 1
#W = 2
#H = 3
#
#bbMargin = .01 #make the bounding boxes % larger in each Coord (2x% total in each direction)

def getPreparedAlt(capture):
    gray = cv2.cvtColor(np.clip(capture.image * 255, 0, 255).astype('uint8'), cv2.COLOR_BGR2GRAY)

    shapeCulled = capture.landmarks.getInteriorPoints()

    grayHull = cv2.convexHull(shapeCulled)
    
    mask = gray.copy()
    mask.fill(0)
    mask = cv2.fillConvexPoly(mask, grayHull, 1).astype('bool')

    gray = cv2.GaussianBlur(gray, (11, 11), 0)
    prepped = cv2.Laplacian(gray, cv2.CV_16S)
    #prepped = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=5)

    preppedMasked = prepped * mask
    return np.float32(preppedMasked)

#def getPreparedNoFlashImage(noFlashImage, noFlashShape):
#    grayNoFlash = cv2.cvtColor(np.clip(noFlashImage * 255, 0, 255).astype('uint8'), cv2.COLOR_BGR2GRAY)
#
#    #CREATE A MASK FOR THE INNER FACE REGION
#    noFlashShapeCroppedCulled = noFlashShape[17:]
#
#    grayHullNoFlash = cv2.convexHull(noFlashShapeCroppedCulled)
#    
#    noFlashMask = grayNoFlash.copy()
#    noFlashMask.fill(0)
#    noFlashMask = cv2.fillConvexPoly(noFlashMask, grayHullNoFlash, 1).astype('bool')
#
#    #GET POINTS IN MASK
#    noFlashFacePoints = grayNoFlash[noFlashMask] # Only take the median of points in the face
#
#    #gamma = 1.0
#    #grayNoFlash = ((grayNoFlash/255 ** (1/gamma)) * 255).astype('uint8')
#    grayNoFlash = cv2.GaussianBlur(grayNoFlash, (BLUR_SIZE, BLUR_SIZE), 0)
#
#    #MODIFIED AUTOCANNY
#    #sigma = .33
#    sigma = .25
#
#    maskMedian = np.median(noFlashFacePoints)
#    noFlashLower = int(max(0, (1.0 - sigma) * maskMedian))
#    noFlashUpper = int(min(255, (1.0 + sigma) * maskMedian))
#    edgeNoFlash = cv2.Canny(grayNoFlash, noFlashLower, noFlashUpper)
#    edgeNoFlash = edgeNoFlash * noFlashMask
#    #Need to convert to a float64 for phaseCorrelation np.float64(edgeNoFlash)
#    return edgeNoFlash
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

def calculateOffset(preparedNoFlashImage, preparedFlashImage):
    (offset, response) = cv2.phaseCorrelate(preparedNoFlashImage, preparedFlashImage)
    offset = list(offset)
    offset = [round(value) for value in offset]
    print("Offset :: " + str(offset))
    return offset

#X=0
#Y=1
#
#def getXOffset(elem):
#    return elem[0]
#
#def getYOffset(elem):
#    return elem[1]
#
#def getOrder(elem):
#    return elem[3]

#def cropToOffset(imagesSortedByOffset, axis):
#    cropped = []
#
#    targetOffset = imagesSortedByOffset[0][axis] #SmallestOffsetValue
#    largestOffset = imagesSortedByOffset[-1][axis] #LargestOffsetValue
#    axisSizeReduction = abs(targetOffset - largestOffset) #Used to figure out how much to reduce the length of each axis
#
#    for capture in imagesSortedByOffset:
#        imageCapture = capture[2]
#        imageOffset = capture[axis]
#
#        coordDelta = abs(targetOffset - imageOffset)
#        axisSizeDelta = abs(axisSizeReduction - coordDelta)
#
#        shapeMask = [0, 0]
#        shapeMask[axis] = coordDelta
#
#        (image, shape, mask) = imageCapture
#        if axis == Y:
#            image = image[coordDelta:, :]
#            mask = mask[coordDelta:, :]
#
#            if axisSizeDelta != 0:
#                image = image[0:-axisSizeDelta, :]
#                mask = mask[0:-axisSizeDelta, :]
#                capture.append([0, coordDelta, -axisSizeDelta])
#            else:
#                capture.append([0, coordDelta, None])
#
#        else:
#            image = image[:, coordDelta:]
#            mask = mask[:, coordDelta:]
#
#            if axisSizeDelta != 0:
#                image = image[:, 0:-axisSizeDelta]
#                mask = mask[:, 0:-axisSizeDelta]
#                capture.append([1, coordDelta, -axisSizeDelta])
#            else:
#                capture.append([1, coordDelta, None])
#
#        shape = shape - shapeMask
#
#        capture[2] = (image, shape, mask)
#        capture[axis] = 0
#
#        cropped.append(capture)
#
#    return cropped


def cropAndAlign(noFlashCapture, halfFlashCapture, fullFlashCapture):
    #(noFlashImage, noFlashShape, noFlashMask) = noFlash
    #(halfFlashImage, halfFlashShape, halfFlashMask) = halfFlash
    #(halfFlashImage_sBGR, halfFlashShape_sBGR, halfFlashMask_sBGR) = halfFlash_sBGR
    #(fullFlashImage, fullFlashShape, fullFlashMask) = fullFlash


    #preparedNoFlashImage = getPreparedNoFlashImage(noFlashImage, noFlashShape)
    #preparedHalfFlashImage = getPreparedFlashImage(halfFlashImage, halfFlashShape, 1.1)
    #preparedFullFlashImage = getPreparedFlashImage(fullFlashImage, fullFlashShape, 1.1)

    print("Preparing Images")
    preparedNoFlashImage = getPreparedAlt(noFlashCapture)
    preparedHalfFlashImage = getPreparedAlt(halfFlashCapture)
    preparedFullFlashImage = getPreparedAlt(fullFlashCapture)
    print("Done Preparing Images")

    #cv2.imshow("Prepared", np.hstack([cv2.resize(preparedNoFlashImage, (0, 0), fx=.5, fy=.5), cv2.resize(preparedFullFlashImage, (0, 0), fx=.5, fy=.5), cv2.resize(preparedBottomFlashImage, (0, 0), fx=.5, fy=.5), cv2.resize(preparedHalfFlashImage, (0, 0), fx=.5, fy=.5)]))
    #cv2.waitKey(0)

    print("Calculating Offset")
    noFlashOffset = [0, 0] #All offsets are relative to noFlashImage
    halfFlashOffset = calculateOffset(preparedNoFlashImage, preparedHalfFlashImage)
    fullFlashOffset = calculateOffset(preparedNoFlashImage, preparedFullFlashImage)
    print("Done Calculating Offset")

    print('Cropping to offsets!')
    cropTools.cropToOffsets([noFlashCapture, halfFlashCapture, fullFlashCapture], np.array([noFlashOffset, halfFlashOffset, fullFlashOffset]))
    print('Done Cropping to offsets!')
    #halfFlashOffset_sBGR = halfFlashOffset.copy()

    #justOffsets = [noFlashOffset, halfFlashOffset, fullFlashOffset]#, halfFlashOffset_sBGR]
    #justOffsets.sort(key=getXOffset, reverse=True)
    #largestX = justOffsets[0][X]

    #justOffsets.sort(key=getYOffset, reverse=True)
    #largestY = justOffsets[0][Y]

    #noFlashOffset.append(noFlash)
    #noFlashOffset.append(0) #Want to keep track of ordering...

    #halfFlashOffset.append(halfFlash)
    #halfFlashOffset.append(1)

    ##halfFlashOffset_sBGR.append(halfFlash_sBGR)
    ##halfFlashOffset_sBGR.append(2)

    #fullFlashOffset.append(fullFlash)
    #fullFlashOffset.append(2)

    #offsets = [noFlashOffset, halfFlashOffset, fullFlashOffset]

    #offsets.sort(key=getXOffset)
    #offsets = cropToOffset(offsets, X)

    #offsets.sort(key=getYOffset)
    #offsets = cropToOffset(offsets, Y)

    #offsets.sort(key=getOrder)

    ##print('Offsets :: ' + str(offsets))

    #[noFlashCropped, halfFlashCropped, fullFlashCropped] = [offset[2] for offset in offsets]

    ##offsetDistance = math.sqrt(math.pow(largestX, 2) + math.pow(largestY, 2))
    ##print("Offset Distance! :: " + str(offsetDistance))
    ##if offsetDistance > 100:
    #    #return [None, 'Possible Alignment Error. Total offset distance :: ' + str(offsetDistance) + '. Returning Early']
    #croppedImages = [noFlashCropped, halfFlashCropped, fullFlashCropped]
    #crop = [[offset[-2], offset[-1]] for offset in offsets]

    #return croppedImages
