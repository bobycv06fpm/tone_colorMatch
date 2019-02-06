import json
import numpy as np
import os
import cv2

# 0  ImageName
# 1  noError
# 2  Half Flash
# - 0  Half Left Cheek
# - - 0  Half Left Fluxish
# - - 1  Half Left Luminance
# - - 2  Half Left HSV
# - - - 0  Hue
# - - - 1  Saturation
# - - - 2  Value
# - - 3  Half Left BGR
# - - - 0  Blue
# - - - 1  Green
# - - - 2  Red
# - - 4  Half Left Line
# - - - 0  Slope
# - - - 1  Intercept
# - 1  Half Right Cheek
# - - 0  Half Right Fluxish
# - - 1  Half Right Luminance
# - - 2  Half Right HSV
# - - - 0  Hue
# - - - 1  Saturation
# - - - 2  Value
# - - 3  Half Right BGR
# - - - 0  Blue
# - - - 1  Green
# - - - 2  Red
# - - 4  Half Right Line
# - - - 0  Slope
# - - - 1  Intercept
# - 2  Half Chin 
# - - 0  Half Chin Fluxish
# - - 1  Half Chin Luminance
# - - 2  Half Chin HSV
# - - - 0  Hue
# - - - 1  Saturation
# - - - 2  Value
# - - 3  Half Chin BGR
# - - - 0  Blue
# - - - 1  Green
# - - - 2  Red
# - - 4  Half Chin Line
# - - - 0  Slope
# - - - 1  Intercept
# - 3  Half Forehead 
# - - 0  Half Forehead Fluxish
# - - 1  Half Forehead Luminance
# - - 2  Half Forehead HSV
# - - - 0  Hue
# - - - 1  Saturation
# - - - 2  Value
# - - 3  Half Forehead BGR
# - - - 0  Blue
# - - - 1  Green
# - - - 2  Red
# - - 4  Half Forehead Line
# - - - 0  Slope
# - - - 1  Intercept
# 3  Full Flash
# - 0  Full Left Cheek
# - - 0  Full Left Fluxish
# - - 1  Full Left Luminance
# - - 2  Full Left HSV
# - - - 0  Hue
# - - - 1  Saturation
# - - - 2  Value
# - - 3  Full Left BGR
# - - - 0  Blue
# - - - 1  Green
# - - - 2  Red
# - - 4  Full Left Line
# - - - 0  Slope
# - - - 1  Intercept
# - 1  Full Right Cheek
# - - 0  Full Right Fluxish
# - - 1  Full Right Luminance
# - - 2  Full Right HSV
# - - - 0  Hue
# - - - 1  Saturation
# - - - 2  Value
# - - 3  Full Right BGR
# - - - 0  Blue
# - - - 1  Green
# - - - 2  Red
# - - 4  Full Right Line
# - - - 0  Slope
# - - - 1  Intercept
# - 2  Full Chin 
# - - 0  Full Chin Fluxish
# - - 1  Full Chin Luminance
# - - 2  Full Chin HSV
# - - - 0  Hue
# - - - 1  Saturation
# - - - 2  Value
# - - 3  Full Chin BGR
# - - - 0  Blue
# - - - 1  Green
# - - - 2  Red
# - - 4  Full Chin Line
# - - - 0  Slope
# - - - 1  Intercept
# - 3  Full Forehead 
# - - 0  Full Forehead Fluxish
# - - 1  Full Forehead Luminance
# - - 2  Full Forehead HSV
# - - - 0  Hue
# - - - 1  Saturation
# - - - 2  Value
# - - 3  Full Forehead BGR
# - - - 0  Blue
# - - - 1  Green
# - - - 2  Red
# - - 4  Full Forehead Line
# - - - 0  Slope
# - - - 1  Intercept

def getKeyValue(imageStats):
    #return imageStats[3][0][1] #Full Flash Left Luminance
    return imageStats[3][0][0] #Full Flash Left Fluxish

root = '../../'
path = root + 'images/'
user = 'doug'
userPath = os.path.join(path, user)

with open('faceColors.json', 'r') as f:
    faceColors = f.read()
    faceColors = json.loads(faceColors)


faceColors = [faceColor for faceColor in faceColors if faceColor[1]]
faceColors = sorted(faceColors, key = getKeyValue) 

sampleSize = 5
sampleSize -= 1

mod = int(len(faceColors) / sampleSize)

imageComparison = None

for index, image in enumerate(faceColors):
    if index % mod == 0:
        imageName = image[0]
        imagePathRoot = os.path.join(userPath, imageName)
        imagePaths = [os.path.join(imagePathRoot, '{}-{}.PNG'.format(imageName, str(imageId))) for imageId in [1, 2, 3]]
        referenceImagePaths = [os.path.join(imagePathRoot, 'reference', imageId + '.PNG') for imageId in ['half_WhitebalancedImage', 'Diff_masked']]
        imagePaths += referenceImagePaths

        print('{} :: {}'.format(imageName, getKeyValue(image)))
        imageSet = None
        blankImage = None

        for path in imagePaths:
            image = cv2.imread(path)

            ratio = 8
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

cv2.imshow('Side By Side', imageComparison)
cv2.waitKey(0)

