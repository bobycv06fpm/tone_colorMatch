import json
import numpy as np
import os
import cv2

bgrLuminanceConsts = np.array([0.0722, 0.7152, 0.2126])
def getLuminance(bgr):
    return np.sum(bgr * bgrLuminanceConsts)

def getKeyValue(imageStats):
    #return getLuminance(imageStats['fullFlashValues']['left'])
    return np.mean(imageStats['reflectionValues']['left'][0])

def getSecondaryStats(imageStats):
    region = 'right'
    fullFlashLuminance = getLuminance(imageStats['fullFlashValues'][region])
    linearity = imageStats['linearity'][region]
    cleanRatio = imageStats['cleanRatio'][region]
    noFlashLuminance = getLuminance(imageStats['noFlashValues'][region])
    return [fullFlashLuminance, linearity, cleanRatio, noFlashLuminance]

save = True

root = '../../'
path = root + 'images/'
user = 'doug'
userPath = os.path.join(path, user)

with open('faceColors.json', 'r') as f:
    faceColors = f.read()
    faceColors = json.loads(faceColors)


faceColors = [faceColor for faceColor in faceColors if faceColor['successful']]
faceColors = sorted(faceColors, key = getKeyValue) 

sampleSize = 7
sampleSize -= 1

mod = int(len(faceColors) / sampleSize)

imageComparison = None

for index, image in enumerate(faceColors):
    if index % mod == 0:
        imagePathRoot = os.path.join(userPath, image['name'])
        imagePaths = [os.path.join(imagePathRoot, '{}-{}.PNG'.format(image['name'], str(imageId))) for imageId in [1, 2, 3]]
        referenceImagePaths = [os.path.join(imagePathRoot, 'reference', imageId + '.PNG') for imageId in ['half_WhitebalancedImage', 'Diff_masked']]
        imagePaths += referenceImagePaths

        print('{} :: {}\t| {}'.format(image['name'], getKeyValue(image), getSecondaryStats(image)))

        imageSet = None
        blankImage = None

        for path in imagePaths:
            image = cv2.imread(path)

            if image.shape[0] < image.shape[1]:
                image = np.rot90(image, 3)

            ratio = 2#8
            smallImage = cv2.resize(image, (0, 0), fx=1/ratio, fy=1/ratio)

            if imageSet is None:
                imageSet = smallImage
                blankImage = np.zeros(smallImage.shape, dtype='uint8')
            else:
                if (smallImage.shape[0] < blankImage.shape[0]) or (smallImage.shape[1] < blankImage.shape[1]):
                    newImage = blankImage.copy()
                    newImage[0:smallImage.shape[0], 0:smallImage.shape[1]] = smallImage
                    smallImage = newImage

                imageSet = np.hstack([imageSet, smallImage])


        if imageComparison is None:
            imageComparison = imageSet
        else:
            imageComparison = np.vstack([imageComparison, imageSet])

if save:
    cv2.imwrite('./compare.PNG', imageComparison)
else:
    cv2.imshow('Side By Side', imageComparison)
    cv2.waitKey(0)

