import numpy as np

def getSecond(arr):
    return arr[1]

def cropToAxis(captures, offsets, axis):
    print('Cropping to Axis')
    #imageSets = np.dstack((images, offset, np.arange(len(offset))))
    OFFSET = 1

    #captureSets = zip(captures, offsets)
    captureSets = []
    for index, capture in enumerate(captures):
        captureSets.append([capture, offsets[index]])

    captureSets = np.array(sorted(captureSets, key=getSecond))

    if captureSets[0, 1] < 0:
        captureSets[:, 1] += abs(captureSets[0, OFFSET])

    print('Capture Sets :: ' + str(captureSets))

    maxOffset = captureSets[-1, OFFSET]

    #cropped = []
    for captureSet in captureSets:
        [capture, offset] = captureSet
        start = offset#maxCrop - offset
        end = capture.image.shape[axis] - (maxOffset - offset)

        if axis == 0:
            capture.image = capture.image[start:end, :]
            capture.mask = capture.mask[start:end, :]
            capture.landmarks.landmarkPoints[:, 1] -= start
            #capture.landmarks.sourceLandmarkPoints[:, 1] -= start

        else:
            capture.image = capture.image[:, start:end]
            capture.mask = capture.mask[:, start:end]
            capture.landmarks.landmarkPoints[:, 0] -= start
            #capture.landmarks.sourceLandmarkPoints[:, 0] -= start

def cropToOffsets(captures, offsets):
    print('Offsets :: ' + str(offsets))
    cropToAxis(captures, offsets[:, 0], 0)
    cropToAxis(captures, offsets[:, 1], 1)

def cropImagesToAxis(images, offsets, axis):
    print('Cropping to Axis')
    OFFSET = 1

    imageSets = []
    for index, capture in enumerate(images):
        imageSets.append([capture, offsets[index], index])

    imageSets = np.array(sorted(imageSets, key=getSecond))

    if imageSets[0, 1] < 0:
        imageSets[:, 1] += abs(imageSets[0, OFFSET])

    #print('Capture Sets :: ' + str(imageSets))

    maxOffset = imageSets[-1, OFFSET]

    cropped = []
    for imageSet in imageSets:
        [image, offset, order] = imageSet
        start = offset#maxCrop - offset
        end = image.shape[axis] - (maxOffset - offset)

        if axis == 0:
            image = image[start:end, :]
        else:
            image = image[:, start:end]

        cropped.append([image, order])

    croppedImages = np.array(sorted(cropped, key=getSecond))
    return croppedImages[:, 0]

def cropImagesToOffsets(images, offsets):
    print('Offsets :: ' + str(offsets))
    images = cropImagesToAxis(images, offsets[:, 0], 0)
    images = cropImagesToAxis(images, offsets[:, 1], 1)
    return images
