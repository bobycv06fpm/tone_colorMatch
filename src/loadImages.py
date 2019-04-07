import cv2
import numpy as np
import os

#root = os.path.expanduser('~/Projects/tone/')
root = os.path.expanduser('/home/dmacewen/Projects/tone/')

def isEye(fileName, side=''):
    extension = '{}Eye.PNG'.format(side)
    return (fileName[-1 * len(extension):] == extension)

def isImage(fileName):
    extension = 'PNG'
    return (fileName[-1 * len(extension):] == extension)

def loadImage(path):
    image = cv2.imread(path)
    if image is None:
        raise NameError('Failed to load image :: ' + path)

    #Not Really needed anymore...
    #if image.shape[0] < image.shape[1]:
    #    print('Rotating Image')
    #    image = np.rot90(image, 3)

    return image

def loadImages(username, fileName, extension='PNG'): 
    leftEyes = loadEyes(username, fileName, 'left', extension)
    rightEyes = loadEyes(username, fileName, 'right', extension)
    faces = loadFaces(username, fileName)

    imageSets = []
    for index, face in enumerate(faces):
        imageSet = []
        imageSet.append(face)
        if (index < len(leftEyes)) and (index < len(rightEyes)):
            imageSet.append(leftEyes[index])
            imageSet.append(rightEyes[index])
        else:
            imageSet.append([])
            imageSet.append([])

        imageSets.append(imageSet)



    #allLeftEyes = np.vstack(leftEyes)
    #allRightEyes = np.vstack(rightEyes)
    #allFaces = np.hstack(faces)
    #cv2.imshow('Left', allLeftEyes)
    #cv2.imshow('Right', allRightEyes)
    #cv2.imshow('Faces', allFaces)
    #cv2.waitKey(0)
    #print('Faces :: {} | Left Eyes :: {} | Right Eyes :: {}'.format(faces, leftEyes, rightEyes))
    return np.array(imageSets)
    #pathRoot = os.path.join(root, "images/", username, fileName)

    #files = os.listdir(pathRoot)
    #imagePaths = sorted([os.path.join(pathRoot, imageFile) for imageFile in files if (imageFile[-1 * len(extension):] == extension)])

    #images = [loadImage(imagePath) for imagePath in imagePaths]
    #return images

def loadEyes(username, fileName, side, extension='Eye.PNG'): 
    pathRoot = os.path.join(root, "images/", username, fileName)

    files = os.listdir(pathRoot)
    imagePaths = sorted([os.path.join(pathRoot, imageFile) for imageFile in files if isEye(imageFile, side)])

    images = [loadImage(imagePath) for imagePath in imagePaths]
    return images

def loadFaces(username, fileName, extension='PNG'): 
    pathRoot = os.path.join(root, "images/", username, fileName)

    files = os.listdir(pathRoot)
    imagePaths = sorted([os.path.join(pathRoot, imageFile) for imageFile in files if (isImage(imageFile) and not isEye(imageFile))])

    images = [loadImage(imagePath) for imagePath in imagePaths]
    return images
