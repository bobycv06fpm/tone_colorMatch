import cv2
import numpy as np

def loadImage(path):
    image = cv2.imread(path)
    if image is None:
        raise NameError('Failed to load image :: ' + path)

    if image.shape[0] < image.shape[1]:
        print('Rotating Image')
        image = np.rot90(image, 3)

    return image

def loadImages(username, fileName, extension='PNG'): 
    pathRoot = "../images/" + username + "/" + fileName + "/" + fileName

    baseImagePath = pathRoot + '-base.' + extension
    baseImage = loadImage(baseImagePath)

    fullFlashImagePath = pathRoot + '-fullFlash.' + extension
    fullFlashImage = loadImage(fullFlashImagePath)

    topFlashImagePath = pathRoot + '-topFlash.' + extension
    topFlashImage = loadImage(topFlashImagePath)

    bottomFlashImagePath = pathRoot + '-bottomFlash.' + extension
    bottomFlashImage = loadImage(bottomFlashImagePath)

    return [baseImage, fullFlashImage, topFlashImage, bottomFlashImage]
