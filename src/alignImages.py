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

#def rankAlignment(image,

def getMask(img, points):
    hull = cv2.convexHull(points)
    mask = img.copy()
    mask.fill(0)
    mask = cv2.fillConvexPoly(mask, hull, 1).astype('bool')
    return mask

def stretchHistogram(gray, mask=None):
    #TEST

    if mask is not None:
        clippedHigh = gray != 255
        clippedLow = gray != 0

        mask = np.logical_and(mask, clippedHigh)
        mask = np.logical_and(mask, clippedLow)

        grayPoints = gray[mask]
    else:
        grayPoints = gray.flatten()

    median = np.median(grayPoints)
    sd = np.std(grayPoints)
    lower = median - (3 * sd)
    lower = lower if lower > 0 else 0
    upper = median + (3 * sd)
    upper = upper if upper < 256 else 255

    #print('MEDIAN :: ' + str(median))
    #print('SD :: ' + str(sd))
    #print('LOWER :: ' + str(lower))
    #print('UPPER :: ' + str(upper))

    bounds = np.copy(gray)
    bounds[bounds < lower] = lower
    bounds[bounds > upper] = upper

    numerator = bounds - lower
    denominator = upper - lower
    #stretched = (numerator.astype('int32') / denominator.astype('int32'))
    stretched = (numerator / denominator)
    #stretched = np.clip(stretched * 255, 0, 255).astype('uint8')
    stretched = np.clip(stretched * 255, 0, 255)

#    cv2.imshow('stretched', cv2.resize(stretched, (0, 0), fx=1/2, fy=1/2))
#    cv2.imshow('original', cv2.resize(gray, (0, 0), fx=1/2, fy=1/2))
    #END TEST
    return stretched


#def getPreparedFlash(gray, points):
#    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#    #shapeCulled = capture.landmarks.getInteriorPoints()
#    
#
#    grayHull = cv2.convexHull(points)
#
#    mask = gray.copy()
#    mask.fill(0)
#    mask = cv2.fillConvexPoly(mask, grayHull, 1).astype('bool')
#
#
#    #TEST
#    grayPoints = gray[mask]
#    median = np.median(grayPoints)
#    sd = np.std(grayPoints)
#    lower = median - (2 * sd)
#    upper = median + (2 * sd)
#    test = np.copy(gray)
#    test[test < lower] = lower
#    test[test > upper] = upper
#
#    numerator = test - lower
#    denominator = upper - lower
#    stretched = (numerator.astype('int32') / denominator.astype('int32'))
#    stretched = np.clip(stretched * 255, 0, 255).astype('uint8')
#
#    #cv2.imshow('stretched', cv2.resize(stretched, (0, 0), fx=1/2, fy=1/2))
#    #cv2.imshow('original', cv2.resize(gray, (0, 0), fx=1/2, fy=1/2))
#    #END TEST
#    gray = stretched
#
#    #gray = cv2.GaussianBlur(gray, (11, 11), 0)
#    #blur = 31 #Works will with eyes.. Ksize5
#    blur = 41
#    gray = cv2.GaussianBlur(gray, (blur, blur), 0)
#    #gray = cv2.GaussianBlur(gray, (45, 45), 0)
#    #prepped = cv2.Laplacian(gray, cv2.CV_16S)
#    #prepped = cv2.Laplacian(gray, cv2.CV_32F)
#
#    prepped = cv2.Sobel(gray, cv2.CV_16S, 1, 1, ksize=5)
#
##    sigma = .5
##    med = np.median(gray)
##    lower = int(max(0, (1.0 - sigma) * med))
##    upper = int(min(0, (1.0 + sigma) * med))
##    prepped = cv2.Canny(gray, lower, upper)
#
#    #prepped = cv2.GaussianBlur(prepped, (11, 11), 0)
#
#    #cv2.imshow('prepped', prepped)
#
#    preppedMasked = prepped * mask
#    
#    #cv2.imshow('prepped masked', preppedMasked.astype('uint8'))
#    #cv2.imshow('prepped masked',cv2.resize(preppedMasked.astype('uint8'), (0, 0), fx=1/2, fy=1/2))
#    #cv2.waitKey(0)
#    return np.float32(preppedMasked)

def getPrepared(gray, mask):
    #gray = cv2.GaussianBlur(gray, (11, 11), 0)
    #blur = 41 #Works will with eyes.. Ksize5
    #blur = 101
    #gray = cv2.GaussianBlur(gray, (blur, blur), 0)
    #gray = cv2.bilateralFilter(gray,30,300,300)
    #cv2.imshow('before gray',cv2.resize((gray * mask).astype('uint8'), (0, 0), fx=1/2, fy=1/2))

    #blur = 1#21#41
    #gray = gray * mask
    #cv2.imshow('gray', gray)
    #gray = cv2.bilateralFilter(gray,30,300,300)
    #gray = cv2.GaussianBlur(gray, (blur, blur), 0)

    #gray = cv2.GaussianBlur(gray, (blur, blur), 0)
    #cv2.imshow('after gray',cv2.resize((gray * mask).astype('uint8'), (0, 0), fx=1/2, fy=1/2))
    #gray = cv2.blur(gray,(30,30))
    #gray = cv2.GaussianBlur(gray, (45, 45), 0)
    #prepped = cv2.Laplacian(gray, cv2.CV_16S)
    #prepped = cv2.Laplacian(gray, cv2.CV_16S)
    prepped = cv2.Sobel(gray, cv2.CV_16S, 1, 1, ksize=5)
    #cv2.imshow('prepped', prepped.astype('uint8'))
    #cv2.waitKey(0)
    #prepped = cv2.GaussianBlur(prepped, (1, 3), 0)
    #sigma = .8
    #med = np.median(gray)
    #lower = int(max(0, (1.0 - sigma) * med))
    #upper = int(min(0, (1.0 + sigma) * med))
    #prepped = cv2.Canny(gray, lower, upper)

    #cv2.imshow('prepped', prepped)

    preppedMasked = prepped * mask
    #cv2.imshow('prepped',cv2.resize(preppedMasked.astype('uint8'), (0, 0), fx=1/2, fy=1/2))
    
    #cv2.imshow('prepped masked', preppedMasked.astype('uint8'))
    #cv2.imshow('prepped masked',cv2.resize(preppedMasked.astype('uint8'), (0, 0), fx=1/2, fy=1/2))
    #cv2.waitKey(0)
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

#def calculateOffset(preparedNoFlashImage, preparedFlashImage):
def calculateOffset(offsetImage, targetImage):
    #(offset, response) = cv2.phaseCorrelate(offsetImage, targetImage)
    (offset, response) = cv2.phaseCorrelate(targetImage, offsetImage)
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


def getPoints(bb):
    (x, y, w, h) = bb
    return np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])

def scaleBB(bb, scale, dimensions):
    (x, y, w, h) = bb
    print('dimensions :: ' + str(dimensions))
    (maxHeight, maxWidth, depth) = dimensions

    addWidth = w * scale
    addHeight = h * scale

    x -= math.floor(addWidth/2)
    x = x if x >= 0 else 0

    w += math.ceil(addWidth)
    w = w if (w + x) <= maxWidth else maxWidth

    y -= math.floor(addHeight/2)
    y = y if y >= 0 else 0

    h += math.ceil(addHeight)
    h = h if (h + y) <= maxHeight else maxHeight

    return (x, y, w, h)

def getLuminosity(image):#, image, blur=101, dimensions=3):
    #values = np.max(image, axis=2)
    #if dimensions == 3:
    #    values = np.floor(np.sum(image.astype('int32'), axis=2) / 3).astype('uint8')
    #else:
    #    values = image

    blur = 101
    #blur = 301
    luminosity = cv2.GaussianBlur(image.astype('uint8'), (blur, blur), 0)
    luminosity[luminosity == 0] = 1
    #cv2.imshow(capture.name + ' luminosity!', luminosity)
    return luminosity

#def getLuminosity(bb, image):
#    (x, y, w, h) = bb
#    crop = image[y:y+h, x:x+w]
#    values = np.max(crop, axis=2)
#    luminosity = cv2.GaussianBlur(values, (101, 101), 0)
#    return luminosity
    #minvalues = np.min(crop, axis=2)
    #cv2.imshow('crop max values', bluredValues)
    #cv2.imshow('crop min values', minvalues)
    #cv2.waitKey(0)

#def prepEye(capture, eyeBB):
#    scaledBB = scaleBB(eyeBB, .5, capture.image.shape)
#    #luminosityField = getLuminosity(scaledBB, capture.image)
#    luminosityField = getLuminosity(capture.image)
#    scaledPoints = getPoints(scaledBB)
#
#    return getPreparedAlt(capture.image, scaledPoints)
#
#def alignEyes(noFlashCapture, halfFlashCapture, fullFlashCapture):
#    print("Preparing Images")
#
#    #scaledBB = scaleBB(noFlashCapture.landmarks.getLeftEyeBB(), .5, capture.image.shape)
#
#    prepared_leftEye_noFlash = prepEye(noFlashCapture, noFlashCapture.landmarks.getLeftEyeBB())
#    prepared_leftEye_halfFlash = prepEye(halfFlashCapture, halfFlashCapture.landmarks.getLeftEyeBB())
#    prepared_leftEye_fullFlash = prepEye(fullFlashCapture, fullFlashCapture.landmarks.getLeftEyeBB())
#    cv2.imshow('prepped left eye no flash', prepared_leftEye_noFlash)
#    cv2.imshow('prepped left eye half flash', prepared_leftEye_halfFlash)
#    cv2.imshow('prepped left eye full flash', prepared_leftEye_fullFlash)
#    cv2.waitKey(0)
#
#    prepared_rightEye_noFlash = prepEye(noFlashCapture, noFlashCapture.landmarks.getRightEyeBB())
#    prepared_rightEye_halfFlash = prepEye(halfFlashCapture, halfFlashCapture.landmarks.getRightEyeBB())
#    prepared_rightEye_fullFlash = prepEye(fullFlashCapture, fullFlashCapture.landmarks.getRightEyeBB())
#    cv2.imshow('prepped right eye no flash', prepared_rightEye_noFlash)
#    cv2.imshow('prepped right eye half flash', prepared_rightEye_halfFlash)
#    cv2.imshow('prepped right eye full flash', prepared_rightEye_fullFlash)
#    cv2.waitKey(0)
#
#    print("Done Preparing Images")
def getPreparedEye(gray):
    gray = cv2.bilateralFilter(gray,30,300,300)
    prepped = cv2.Sobel(gray, cv2.CV_16S, 1, 1, ksize=5)
    return np.float32(prepped)

def cropAndAlignEyes(noFlashEye, halfFlashEye, fullFlashEye):

    noFlashGrey = np.sum(noFlashEye, axis=2) / 3
    halfFlashGrey = np.sum(halfFlashEye, axis=2) / 3
    fullFlashGrey = np.sum(fullFlashEye, axis=2) / 3

    noFlashGreyStretched = stretchHistogram(noFlashGrey).astype('uint8')
    halfFlashGreyStretched = stretchHistogram(halfFlashGrey).astype('uint8')
    fullFlashGreyStretched = stretchHistogram(fullFlashGrey).astype('uint8')

    preparedNoFlashImage = getPreparedEye(noFlashGreyStretched)
    preparedHalfFlashImage = getPreparedEye(halfFlashGreyStretched)
    preparedFullFlashImage = getPreparedEye(fullFlashGreyStretched)

    noFlashOffset = [0, 0]
    halfFlashOffset = [0, 0]
    fullFlashOffset = calculateOffset(preparedFullFlashImage, preparedHalfFlashImage)

    print('no flash offset :: ' + str(noFlashOffset))
    print('half flash offset :: ' + str(halfFlashOffset))
    print('full flash offset :: ' + str(fullFlashOffset))

    [noFlashEyeCropped, halfFlashEyeCropped, fullFlashEyeCropped] = cropTools.cropImagesToOffsets([noFlashEye, halfFlashEye, fullFlashEye], np.array([noFlashOffset, halfFlashOffset, fullFlashOffset]))

    return [noFlashEyeCropped, halfFlashEyeCropped, fullFlashEyeCropped]

def cropAndAlign(noFlashCapture, halfFlashCapture, fullFlashCapture):
    #(noFlashImage, noFlashShape, noFlashMask) = noFlash
    #(halfFlashImage, halfFlashShape, halfFlashMask) = halfFlash
    #(halfFlashImage_sBGR, halfFlashShape_sBGR, halfFlashMask_sBGR) = halfFlash_sBGR
    #(fullFlashImage, fullFlashShape, fullFlashMask) = fullFlash


    #preparedNoFlashImage = getPreparedNoFlashImage(noFlashImage, noFlashShape)
    #preparedHalfFlashImage = getPreparedFlashImage(halfFlashImage, halfFlashShape, 1.1)
    #preparedFullFlashImage = getPreparedFlashImage(fullFlashImage, fullFlashShape, 1.1)

    #luminosityNoFlash = getLuminosity(noFlashCapture)
    #luminosityHalfFlash = getLuminosity(halfFlashCapture.image)
    #luminosityFullFlash = getLuminosity(fullFlashCapture.image)
    #cv2.waitKey(0)

    #noFlashMultiplier = luminosityHalfFlash / luminosityNoFlash
    #fullFlashMultiplier = luminosityHalfFlash / luminosityFullFlash


    #noFlashGrey = np.max(noFlashCapture.image, axis=2)
    #halfFlashGrey = np.max(halfFlashCapture.image, axis=2)
    #fullFlashGrey = np.max(fullFlashCapture.image, axis=2)

    print('one')
    noFlashGrey = np.sum(noFlashCapture.image, axis=2) / 3#.astype('uint8')
    halfFlashGrey = np.sum(halfFlashCapture.image, axis=2) / 3#.astype('uint8')
    fullFlashGrey = np.sum(fullFlashCapture.image, axis=2) / 3#.astype('uint8')

    #cv2.imshow('fullFlashGrey', fullFlashGrey.astype('uint8'))
    #cv2.waitKey(0)


    print('two')
    noFlashMask = getMask(noFlashGrey, noFlashCapture.landmarks.getInteriorPoints())
    halfFlashMask = getMask(halfFlashGrey, halfFlashCapture.landmarks.getInteriorPoints())
    fullFlashMask = getMask(fullFlashGrey, fullFlashCapture.landmarks.getInteriorPoints())


    print('three')
    noFlashGreyStretched = stretchHistogram(noFlashGrey, noFlashMask)
    halfFlashGreyStretched = stretchHistogram(halfFlashGrey, halfFlashMask)
    fullFlashGreyStretched = stretchHistogram(fullFlashGrey, fullFlashMask)

    print('four')
    #REMOVE ONCE PARTIAL ILLUMINATION ON NO FLASH IS ADDED?
    luminosityNoFlash = getLuminosity(noFlashGreyStretched)
    luminosityHalfFlash = getLuminosity(halfFlashGreyStretched)
    luminosityFullFlash = getLuminosity(fullFlashGreyStretched)

    print('five')
    noFlashMultiplier = luminosityHalfFlash / luminosityNoFlash
    fullFlashMultiplier = luminosityHalfFlash / luminosityFullFlash

    print('six')
    noFlashGreyScaled = np.clip(np.floor(noFlashGreyStretched * noFlashMultiplier), 0, 255)#.astype('uint8')
    fullFlashGreyScaled = np.clip(np.floor(fullFlashGreyStretched * fullFlashMultiplier), 0, 255)#.astype('uint8')
    #noFlashGreyScaled = noFlashGreyStretched
    #fullFlashGreyScaled = fullFlashGreyStretched
    print('seven')


    #noFlashResized = cv2.resize(noFlashGrey, (0, 0), fx=1/2, fy=1/2)
    #noFlashScaledResized = cv2.resize(noFlashGreyScaled, (0, 0), fx=1/2, fy=1/2)
    #noFlashStacked = np.dstack(([noFlashResized], [noFlashScaledResized]))[0]
    #print('no flash stacked shape' + str(noFlashStacked.shape))
    #print('no flash stacked' + str(noFlashStacked))

    #halfFlashResized = cv2.resize(halfFlashGrey, (0, 0), fx=1/2, fy=1/2)
    #halfFlashGreyStretchedResized = cv2.resize(halfFlashGreyStretched, (0, 0), fx=1/2, fy=1/2)
    #halfFlashStacked = np.dstack(([halfFlashResized], [halfFlashGreyStretchedResized]))[0]

    #fullFlashResized = cv2.resize(fullFlashGrey, (0, 0), fx=1/2, fy=1/2)
    #fullFlashScaledResized = cv2.resize(fullFlashGreyScaled, (0, 0), fx=1/2, fy=1/2)
    #fullFlashStacked = np.dstack(([fullFlashResized], [fullFlashScaledResized]))[0]

    #cv2.imshow('No Flash', noFlashStacked)
    #cv2.imshow('Half Flash', halfFlashStacked)
    #cv2.imshow('Full Flash', fullFlashStacked)
    #cv2.waitKey(0)

    #print('No Flash Scaled :: ' + str(noFlashGreyScaled))

#    (x, y, w, h) = noFlashCapture.landmarks.getRightEyeInnerBB()
#    noFlashGreyScaled[y:y+h, x:x+w] = 0
#    (x, y, w, h) = noFlashCapture.landmarks.getLeftEyeInnerBB()
#    noFlashGreyScaled[y:y+h, x:x+w] = 0
#
#    (x, y, w, h) = halfFlashCapture.landmarks.getRightEyeInnerBB()
#    halfFlashGrey[y:y+h, x:x+w] = 0
#    (x, y, w, h) = halfFlashCapture.landmarks.getLeftEyeInnerBB()
#    halfFlashGrey[y:y+h, x:x+w] = 0
#
#    (x, y, w, h) = fullFlashCapture.landmarks.getRightEyeInnerBB()
#    fullFlashGreyScaled[y:y+h, x:x+w] = 0
#    (x, y, w, h) = fullFlashCapture.landmarks.getLeftEyeInnerBB()
#    fullFlashGreyScaled[y:y+h, x:x+w] = 0

    #cv2.imshow('no flash scaled :: ', cv2.resize(noFlashGreyScaled.astype('uint8'), (0, 0), fx=1/2, fy=1/2))
    #cv2.imshow('half flash scaled :: ', cv2.resize(halfFlashGrey.astype('uint8'), (0, 0), fx=1/2, fy=1/2))
    #cv2.imshow('full flash scaled :: ', cv2.resize(fullFlashGreyScaled.astype('uint8'), (0, 0), fx=1/2, fy=1/2))
    #cv2.waitKey(0)

    print("Preparing Images")
    preparedNoFlashImage = getPrepared(noFlashGreyScaled, noFlashMask)#noFlashCapture.landmarks.getInteriorPoints())
    preparedHalfFlashImage = getPrepared(halfFlashGreyStretched, halfFlashMask)#halfFlashCapture.landmarks.getInteriorPoints())
    preparedFullFlashImage = getPrepared(fullFlashGreyScaled, fullFlashMask)#fullFlashCapture.landmarks.getInteriorPoints())
    print("Done Preparing Images")

    #cv2.imshow("Prepared", np.hstack([cv2.resize(preparedNoFlashImage, (0, 0), fx=.5, fy=.5), cv2.resize(preparedFullFlashImage, (0, 0), fx=.5, fy=.5), cv2.resize(preparedBottomFlashImage, (0, 0), fx=.5, fy=.5), cv2.resize(preparedHalfFlashImage, (0, 0), fx=.5, fy=.5)]))
    #cv2.waitKey(0)

    print("Calculating Offset")
    noFlashOffset = calculateOffset(preparedNoFlashImage, preparedHalfFlashImage)#[0, 0] #All offsets are relative to noFlashImage
    #noFlashOffset = calculateOffset(preparedNoFlashImage, preparedHalfFlashImage)#[0, 0] #All offsets are relative to noFlashImage
    #halfFlashOffset = calculateOffset(preparedFullFlashImage, preparedHalfFlashImage)
    halfFlashOffset = [0, 0]#calculateOffset(preparedHalfFlashImage, preparedFullFlashImage)
    fullFlashOffset = calculateOffset(preparedFullFlashImage, preparedHalfFlashImage)
    print("Done Calculating Offset")

    print('Cropping to offsets!')
    cropTools.cropToOffsets([noFlashCapture, halfFlashCapture, fullFlashCapture], np.array([noFlashOffset, halfFlashOffset, fullFlashOffset]))
    print('Done Cropping to offsets!')

    #blur = 31
    #halfFlashImageBlur = cv2.GaussianBlur(halfFlashCapture.image, (blur, blur), 0)
    #fullFlashImageBlur = cv2.GaussianBlur(fullFlashCapture.image, (blur, blur), 0)
    #synNoFlash = np.clip((2 * halfFlashImageBlur.astype('int32')) - (fullFlashImageBlur.astype('int32')), 0, 255).astype('uint8')
    #synNoFlash = halfFlashCapture.image#np.clip((2 * halfFlashCapture.image.astype('int32')) - (fullFlashCapture.image.astype('int32')), 0, 255).astype('uint8')

    #cv2.imshow('checker', cv2.resize(synNoFlash, (0, 0), fx=1/2, fy=1/2))
    #cv2.imshow('actual', cv2.resize(noFlashCapture.image, (0, 0), fx=1/2, fy=1/2))
    #cv2.waitKey(0)

    #synNoFlashLuminosity = getLuminosity(synNoFlash)
    #noFlashLuminosity = getLuminosity(noFlashCapture.image)

    #synNoFlashGrey = np.floor(np.sum(synNoFlash.astype('int32'), axis=2) / 3).astype('uint8')
    #synNoFlashGreyScaled = synNoFlashGrey
    #noFlashGrey = np.floor(np.sum(noFlashCapture.image.astype('int32'), axis=2) / 3).astype('uint8')

    #cv2.imshow('before scaling', cv2.resize(synNoFlashGrey, (0, 0), fx=1/2, fy=1/2))
    #synNoFlashMultiplier = noFlashLuminosity / synNoFlashLuminosity
    #synNoFlashGreyScaled = np.clip(np.floor(synNoFlashGrey * synNoFlashMultiplier), 0, 255).astype('uint8')
    #cv2.imshow('after scaling', cv2.resize(synNoFlashGreyScaled, (0, 0), fx=1/2, fy=1/2))
    #cv2.waitKey(0)

#    cv2.imshow('synthetic no flash scaled :: ', cv2.resize(synNoFlashGreyScaled, (0, 0), fx=1/2, fy=1/2))
#    cv2.imshow('no flash :: ', cv2.resize(noFlashGrey, (0, 0), fx=1/2, fy=1/2))
#    cv2.waitKey(0)

    #synNoFlashGreyMask = getMask(synNoFlashGreyScaled, halfFlashCapture.landmarks.getInteriorPoints())
    #cv2.imshow('before stretch', cv2.resize(synNoFlashGreyScaled, (0, 0), fx=1/2, fy=1/2))
    #synNoFlashGreyScaled = stretchHistogram(synNoFlashGreyScaled, synNoFlashGreyMask)
    #cv2.imshow('after stretch', cv2.resize(synNoFlashGreyScaled, (0, 0), fx=1/2, fy=1/2))
    #cv2.waitKey(0)

    #noFlashGreyMask = getMask(noFlashGrey, noFlashCapture.landmarks.getInteriorPoints())
    #cv2.imshow('before stretch', cv2.resize(noFlashGrey, (0, 0), fx=1/2, fy=1/2))
    #noFlashGrey = stretchHistogram(noFlashGrey, noFlashGreyMask)
    #cv2.imshow('after stretch', cv2.resize(noFlashGrey, (0, 0), fx=1/2, fy=1/2))
    #cv2.waitKey(0)


    #synNoFlashLuminosity = getLuminosity(synNoFlashGreyScaled, 201, 2)
    #noFlashLuminosity = getLuminosity(noFlashGrey, 201, 2)

    #synNoFlashMultiplier = noFlashLuminosity / synNoFlashLuminosity
    #cv2.imshow('before scale', cv2.resize(synNoFlashGreyScaled, (0, 0), fx=1/2, fy=1/2))
    #synNoFlashGreyScaled = np.clip(np.floor(synNoFlashGreyScaled * synNoFlashMultiplier), 0, 255).astype('uint8')
    #synNoFlashGreyScaled = cv2.GaussianBlur(synNoFlashGreyScaled, (51, 51), 0)
    #cv2.imshow('syn scale', cv2.resize(synNoFlashGreyScaled, (0, 0), fx=1/2, fy=1/2))
    #cv2.imshow('actual scale', cv2.resize(noFlashGrey, (0, 0), fx=1/2, fy=1/2))

    #blur = 11
    #subtractFromSyn = np.clip(synNoFlashGreyScaled.astype('int32') - noFlashGrey.astype('int32'), 0, 255).astype('uint8')
    #subtractLowPass = cv2.GaussianBlur(subtractFromSyn, (blur, blur), 0)
    #subtractLowPass = np.clip(subtractFromSyn.astype('int32') - subtractLowPass.astype('int32'), 0, 255).astype('uint8')

    #addToSyn =  np.clip(noFlashGrey.astype('int32') - synNoFlashGreyScaled.astype('int32'), 0, 255).astype('uint8')
    #addLowPass = cv2.GaussianBlur(addToSyn, (blur, blur), 0)
    #addLowPass = np.clip(addToSyn.astype('int32') - addLowPass.astype('int32'), 0, 255).astype('uint8')

    ##cv2.imshow('SubDiff', np.clip(subtractLowPass, 0, 255).astype('uint8'))
    ##cv2.imshow('AddDiff', np.clip(addLowPass, 0, 255).astype('uint8'))
    ##cv2.waitKey(0)

    #cv2.imshow('before add/sub', cv2.resize(synNoFlashGreyScaled, (0, 0), fx=1/2, fy=1/2))
    #synNoFlashGreyScaled += addLowPass
    #synNoFlashGreyScaled -= subtractLowPass
    #cv2.imshow('after add/sub', cv2.resize(synNoFlashGreyScaled, (0, 0), fx=1/2, fy=1/2))
    #cv2.waitKey(0)

    #preparedSynNoFlashImage = getPreparedNoFlash(synNoFlashGreyScaled, synNoFlashGreyMask)
    #preparedNoFlashImage = getPreparedNoFlash(noFlashGrey, noFlashGreyMask)

    #noFlashOffset = calculateOffset(preparedNoFlashImage, preparedSynNoFlashImage)
    #halfFlashOffset = [0, 0]#calculateOffset(preparedHalfFlashImage, preparedFullFlashImage)
    #fullFlashOffset = [0, 0]

    #cropTools.cropToOffsets([noFlashCapture, halfFlashCapture, fullFlashCapture], np.array([noFlashOffset, halfFlashOffset, fullFlashOffset]))


    #noFlashGrey = np.max(noFlashCapture.image, axis=2)
    #halfFlashGrey = np.max(halfFlashCapture.image, axis=2)
    #fullFlashGrey = np.max(fullFlashCapture.image, axis=2)

    #syntheticNoFlash = np.clip(2 * halfFlashGrey.astype('int32') - fullFlashGrey.astype('int32'), 0, 255).astype('uint8')
    #cv2.imshow('synthetic no flash', syntheticNoFlash)
    #cv2.imshow('no flash', noFlashGrey)
    #cv2.waitKey(0)

    #preparedNoFlash = getPreparedNoFlash(noFlashGrey, noFlashCapture.landmarks.getInteriorPoints())
    #preparedSyntheticNoFlash = getPreparedNoFlash(syntheticNoFlash, halfFlashCapture.landmarks.getInteriorPoints())

    #noFlashOffset = calculateOffset(preparedNoFlash, preparedSyntheticNoFlash)
    #halfFlashOffset = [0, 0]
    #fullFlashOffset = [0, 0]
    #cropTools.cropToOffsets([noFlashCapture, halfFlashCapture, fullFlashCapture], np.array([noFlashOffset, halfFlashOffset, fullFlashOffset]))

    #noFlashGrey = np.max(noFlashCapture.image, axis=2)
    #halfFlashGrey = np.max(halfFlashCapture.image, axis=2)
    #fullFlashGrey = np.max(fullFlashCapture.image, axis=2)
    #syntheticNoFlash = np.clip(2 * halfFlashGrey.astype('int32') - fullFlashGrey.astype('int32'), 0, 255).astype('uint8')

    #justReflection = np.clip(syntheticNoFlash.astype('int32') - noFlashGrey.astype('int32'), 0, 255).astype('uint8')
    #cv2.imshow('Just Reflection?', cv2.resize(justReflection, (0, 0), fx=1/2, fy=1/2))
    #cv2.waitKey(0)




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
