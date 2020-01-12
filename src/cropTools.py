"""Set of functions to help with image and capture cropping"""
import numpy as np

X = 0
Y = 1
SHAPE_X = 1
SHAPE_Y = 0

def __getSecond(arr):
    """Lambda helper function"""
    return arr[1]

def __getThird(arr):
    """Lambda helper function"""
    return arr[2]

def __cropImagesToAxis(images, offsets, axis):
    """Crop images by the offset amount in the axis provided, the resulting images are all equal in size"""
    OFFSET = 1

    imageSets = []
    for index, capture in enumerate(images):
        imageSets.append([capture, offsets[index], index])

    imageSets = np.array(sorted(imageSets, key=__getSecond))

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

    croppedImages = np.array(sorted(cropped, key=__getThird))
    return croppedImages[:, 0], croppedImages[:, 1]

def cropImagesToOffsets(images, offsets):
    """Crops the images to the offsets provided"""

    imageDimensions = np.array([image.shape for image in images])
    minHeight = np.min(imageDimensions[:, 0])
    minWidth = np.min(imageDimensions[:, 1])
    #print('Min Width :: {}, Min Height :: {}'.format(minHeight, minWidth))

    #Occasionally there is data that is 1 px smaller. Probably a rounding issue on the cropping app side. On todo list...
    images = np.array([image[0:minHeight, 0:minWidth] for image in images])

    #images = np.array(images)
    offsets = np.array(offsets)
    updatedOffsets = np.copy(offsets)
    images, xOffsets = __cropImagesToAxis(images, offsets[:, X], X)
    images, yOffsets = __cropImagesToAxis(images, offsets[:, Y], Y)

    updatedOffsets[:, X] = xOffsets
    updatedOffsets[:, Y] = yOffsets

    return images, updatedOffsets

def cropCapturesToFaceOffsets(captures, offsets):
    """Crops face image in captures to the offsets provided. Updates landmarks, masks, BBs as well"""
    images = [capture.faceImage for capture in captures]
    croppedImages, updatedOffsets = cropImagesToOffsets(images, offsets)
    for capture, croppedImage, updatedOffset in zip(captures, croppedImages, updatedOffsets):
        capture.faceImage = croppedImage
        capture.landmarks.cropLandmarkPoints(updatedOffset)
        capture.faceMask = capture.faceMask[updatedOffset[1]:updatedOffset[1] + capture.faceImage.shape[0], updatedOffset[0]:updatedOffset[0] + capture.faceImage.shape[1]]
        capture.leftEyeBB -= updatedOffset
        capture.rightEyeBB -= updatedOffset

def getEyeImagesCroppedToOffsets(captures, offsets, side):
    """Returns specified eye image cropped to offsets"""

    if side == 'left':
        imageSets = np.array([capture.leftEyeImage for capture in captures])
    elif side == 'right':
        imageSets = np.array([capture.rightEyeImage for capture in captures])
    elif side is None:
        raise ValueError("Eye not specified with side argument")
    else:
        raise ValueError("Invalid side argument")

    croppedImages, _ = cropImagesToOffsets(imageSets, offsets)

    return croppedImages

def cropImagesToParentBB(images, BBs):
    """Take a set of images and a set of bounding boxes and find the BB that encompases all of the provided BBs. Crop all images to the parent BB"""
    BBPoints = np.array([[BB[0], BB[1], BB[0] + BB[2], BB[1] + BB[3]] for BB in BBs])
    parentBBPoints = [np.min(BBPoints[:, 0]), np.min(BBPoints[:, 1]), np.max(BBPoints[:, 2]), np.max(BBPoints[:, 3])]
    return [image[parentBBPoints[1]:parentBBPoints[3], parentBBPoints[0]:parentBBPoints[2]] for image in images]
