import cv2
import numpy as np
import os

root = os.path.expanduser('~/Projects/tone/')

def loadImage(path):
    image = cv2.imread(path)
    if image is None:
        raise NameError('Failed to load image :: ' + path)

    if image.shape[0] < image.shape[1]:
        print('Rotating Image')
        image = np.rot90(image, 3)

    return image

def loadImages(username, fileName, extension='PNG'): 
    pathRoot = os.path.join(root, "images/", username, fileName, fileName)

    noFlashPath = pathRoot + '-3.' + extension
    noFlash = loadImage(noFlashPath)

    halfFlashPath = pathRoot + '-2.' + extension
    halfFlash = loadImage(halfFlashPath)

    fullFlashPath = pathRoot + '-1.' + extension
    fullFlash = loadImage(fullFlashPath)

    return [noFlash, halfFlash, fullFlash]
