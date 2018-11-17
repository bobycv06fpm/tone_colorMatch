import numpy as np

def getSecond(arr):
    return arr[1]

def cropToAxis(captures, offsets, axis):
    print('Cropping to Axis')
    #imageSets = np.dstack((images, offset, np.arange(len(offset))))
    CAPTURE = 0
    OFFSET = 1

    #captureSets = zip(captures, offsets)
    captureSets = []
    for index, capture in enumerate(captures):
        captureSets.append([capture, offsets[index]])

    captureSets = np.array(sorted(captureSets, key=getSecond))

    if captureSets[0, 1] < 0:
        captureSets[:, 1] += abs(captureSets[0, OFFSET])

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
            capture.landmarks.sourceLandmarkPoints[:, 1] -= start

        else:
            capture.image = capture.image[:, start:end]
            capture.mask = capture.mask[:, start:end]
            capture.landmarks.landmarkPoints[:, 0] -= start
            capture.landmarks.sourceLandmarkPoints[:, 0] -= start

def cropToOffsets(captures, offsets):
    print('Offsets :: ' + str(offsets))
    cropToAxis(captures, offsets[:, 0], 0)
    cropToAxis(captures, offsets[:, 1], 1)
