import numpy as np
import cv2

X = 0
Y = 1
SHAPE_X = 1
SHAPE_Y = 0

def getSecond(arr):
    return arr[1]

def getThird(arr):
    return arr[2]

def cropToAxis(captures, offsets, axis):
    print('Cropping to Axis')
    #imageSets = np.dstack((images, offset, np.arange(len(offset))))
    OFFSET = 1

    #captureSets = zip(captures, offsets)
    captureSets = []
    for index, capture in enumerate(captures):
        captureSets.append([capture, offsets[index]])

    captureSets = np.array(sorted(captureSets, key=getSecond))

    if captureSets[0, OFFSET] < 0:
        captureSets[:, OFFSET] += abs(captureSets[0, OFFSET])

    #print('Capture Sets :: ' + str(captureSets))

    maxOffset = captureSets[-1, OFFSET]

    #cropped = []
    for captureSet in captureSets:
        [capture, offset] = captureSet
        start = offset

        if axis == Y:
            end = capture.image.shape[SHAPE_Y] - (maxOffset - offset)
            capture.image = capture.image[start:end, :]
            capture.mask = capture.mask[start:end, :]
            capture.landmarks.landmarkPoints[:, Y] -= start
            #capture.landmarks.sourceLandmarkPoints[:, 1] -= start

        else:
            end = capture.image.shape[SHAPE_X] - (maxOffset - offset)
            capture.image = capture.image[:, start:end]
            capture.mask = capture.mask[:, start:end]
            capture.landmarks.landmarkPoints[:, X] -= start
            #capture.landmarks.sourceLandmarkPoints[:, 0] -= start

def cropToOffsets(captures, offsets):
    print('Offsets :: ' + str(offsets))
    cropToAxis(captures, offsets[:, X], X)
    cropToAxis(captures, offsets[:, Y], Y)

def cropImagesToAxis(images, offsets, axis):
    print('Cropping to Axis')
    OFFSET = 1

    imageSets = []
    for index, capture in enumerate(images):
        imageSets.append([capture, offsets[index], index])

    imageSets = np.array(sorted(imageSets, key=getSecond))

    if imageSets[0, OFFSET] < 0:
        imageSets[:, OFFSET] += abs(imageSets[0, OFFSET])

    maxOffset = imageSets[-1, OFFSET]

    cropped = []
    for imageSet in imageSets:
        [image, offset, order] = imageSet
        start = offset

        if axis == Y:
            end = image.shape[SHAPE_Y] - (maxOffset - offset)
            image = image[start:end, :]
        else:
            end = image.shape[SHAPE_X] - (maxOffset - offset)
            image = image[:, start:end]

        cropped.append([image, offset, order])

    croppedImages = np.array(sorted(cropped, key=getThird))
    return croppedImages[:, 0], croppedImages[:, 1]

def cropImagesToOffsets(images, offsets):
    images = np.array(images)
    offsets = np.array(offsets)
    updatedOffsets = np.copy(offsets)
    images, xOffsets = cropImagesToAxis(images, offsets[:, X], X)
    images, yOffsets = cropImagesToAxis(images, offsets[:, Y], Y)

    updatedOffsets[:, X] = xOffsets
    updatedOffsets[:, Y] = yOffsets

    return images, updatedOffsets

#TODO: ARE LANDMARKS LABLED BACKWARKS?!?! YEESH
def cropCapturesToLandmark(captures, landmarkIndex):
    offsetsFromZero = np.array([capture.landmarks.landmarkPoints[landmarkIndex] for capture in captures])
    offsets = offsetsFromZero - offsetsFromZero[0]#[minXOffset, minYOffset]
    cropToOffsets(captures, offsets)

