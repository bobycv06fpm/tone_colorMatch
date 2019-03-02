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

def getMask(img, points):
    hull = cv2.convexHull(points)
    mask = img.copy()
    mask.fill(0)
    mask = cv2.fillConvexPoly(mask, hull, 1).astype('bool')
    return mask

def stretchHistogram(gray, mask=None):
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

    bounds = np.copy(gray)
    bounds[bounds < lower] = lower
    bounds[bounds > upper] = upper

    numerator = bounds - lower
    denominator = upper - lower
    stretched = (numerator / denominator)
    stretched = np.clip(stretched * 255, 0, 255)
    return stretched

def getPrepared(gray, mask):
    prepped = cv2.Sobel(gray, cv2.CV_16S, 1, 1, ksize=9)
    preppedMasked = prepped * mask
    return np.float32(preppedMasked)

#def calculateOffset(preparedNoFlashImage, preparedFlashImage):
def calculateOffset(offsetImage, targetImage):
    #(offset, response) = cv2.phaseCorrelate(offsetImage, targetImage)
    (offset, response) = cv2.phaseCorrelate(targetImage, offsetImage)
    offset = list(offset)
    offset = [round(value) for value in offset]
    print("Offset :: " + str(offset))
    return np.array(offset)

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
    blur = 101
    luminosity = cv2.GaussianBlur(image.astype('uint8'), (blur, blur), 0)
    luminosity[luminosity == 0] = 1
    return luminosity

def getPreparedEye(gray):
    gray = cv2.bilateralFilter(np.clip(gray, 0, 255).astype('uint8'),30,150,150)
    prepped = cv2.Sobel(gray, cv2.CV_16S, 1, 1, ksize=5)
    return np.float32(prepped)

def cropAndAlignEyes(eyes):
    greyEyes = [np.min(eye, axis=2) for eye in eyes] #Sort of counter intuitive, but using min we basically isolate white values/reflections
    stretchedEyes = [stretchHistogram(greyEye) for greyEye in greyEyes]
    preparedEyes = [getPreparedEye(stretchedEye) for stretchedEye in stretchedEyes]

    #stretchedShow = np.hstack(stretchedEyes)
    #preparedShow = np.hstack(preparedEyes)
    #cv2.imshow('prepared', np.vstack((stretchedShow.astype('uint8'), preparedShow)).astype('uint8'))
    #cv2.waitKey(0)

    #middleEyeIndex = math.floor(len(eyes) / 2)
    #eyeOffsets = [calculateOffset(preparedEye, preparedEyes[middleEyeIndex]) for preparedEye in preparedEyes]
    relativeEyeOffsets = [calculateOffset(preparedEye, preparedEyes[index - 1 if index > 0 else 0]) for index, preparedEye in enumerate(preparedEyes)]

    eyeOffsets = [relativeEyeOffsets[0]]
    for relativeEyeOffset in relativeEyeOffsets[1:]:
        eyeOffsets.append(eyeOffsets[-1] + relativeEyeOffset)

    for index, eyeOffset in enumerate(eyeOffsets):
        print('Eye Offset {} :: {}'.format(index, eyeOffset))

    return cropTools.cropImagesToOffsets(eyes, eyeOffsets)

def getOffsetMagnitude(offsets, imageShape):
    offsets = np.array(offsets)
    XOffsetMagnitude = (max(offsets[:, 0]) - min(offsets[:, 0])) / imageShape[1]
    YOffsetMagnitude = (max(offsets[:, 1]) - min(offsets[:, 1])) / imageShape[0]

    return (XOffsetMagnitude**2 + YOffsetMagnitude**2) ** 0.5

def getLandmarkOffsetMagnitude(captures, landmarkIndex):
    offsetsFromZero = np.array([capture.landmarks.landmarkPoints[landmarkIndex] for capture in captures])
    offsets = offsetsFromZero - offsetsFromZero[0]#[minXOffset, minYOffset]
    return getOffsetMagnitude(offsets, captures[0].image.shape)

def cropAndAlignCaptures(captures):
    landmarkOffsetMagnitude = getLandmarkOffsetMagnitude(captures, 25)#Users right Eye Outside Point

    print('one')
    greyImages = [np.mean(capture.image, axis=2) for capture in captures]

    print('two')
    interiorPoints = [capture.landmarks.getInteriorPoints() for capture in captures]
    masks = [getMask(greyImage, interiorPoints) for greyImage, interiorPoints in zip(greyImages, interiorPoints)]

    print('three')
    stretchedImages = [stretchHistogram(image, mask) for image, mask in zip(greyImages, masks)]
    #stretchedImages = greyImages
    
    #stretchedShow = np.hstack(stretchedImages)
    #stretchedShow = cv2.resize(stretchedShow, (0, 0), fx=1/3, fy=1/3)
    #cv2.imshow('stretched', stretchedShow.astype('uint8'))
    #cv2.waitKey(0)

    print('four')
    preparedImages = [getPrepared(stretchedImage, mask) for stretchedImage, mask in zip(stretchedImages, masks)]
    print("Done Preparing Images")

    #preparedShow = np.hstack(preparedImages)
    #preparedShow = cv2.resize(preparedShow, (0, 0), fx=1/3, fy=1/3)
    #cv2.imshow('prepared', preparedShow)
    #cv2.waitKey(0)

    print("Calculating Offset")
    #middleImageIndex = math.floor(len(captures) / 2)
    relativeImageOffsets = [calculateOffset(preparedImage, preparedImages[index - 1 if index > 0 else 0]) for index, preparedImage in enumerate(preparedImages)]

    imageOffsets = [relativeImageOffsets[0]]
    for relativeImageOffset in relativeImageOffsets[1:]:
        imageOffsets.append(imageOffsets[-1] + relativeImageOffset)

    print('Image Offsets :: ' + str(imageOffsets))
    print("Done Calculating Offset")

    alignedOffsetMagnitude = getOffsetMagnitude(imageOffsets, captures[0].image.shape)

    print('Landmark, Aligned values :: ' + str(landmarkOffsetMagnitude) + ' | ' + str(alignedOffsetMagnitude))

    if  (alignedOffsetMagnitude > 0.05) or (landmarkOffsetMagnitude > 0.05):
        raise NameError('Probable Error Stacking. Alignment of Landmark Offset Mag is too large. Landmark Mag :: ' + str(landmarkOffsetMagnitude) + ', Alignment Mag :: ' + str(alignedOffsetMagnitude))

    print('Cropping to offsets!')
    cropTools.cropToOffsets(captures, np.array(imageOffsets))
    print('Done Cropping to offsets!')
