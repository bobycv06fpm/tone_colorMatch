from imutils import face_utils
import imutils
import numpy as np
import cv2
import utils
import saveStep

def drawSpot(image, location):
    (y, x) = location
    cv2.circle(image, (x, y), 3, (0, 0, 255), -1)

#def drawPolygons(username, img, polygons, imageName):
#    overlay = img.copy()
#    output = img.copy()
#
#    colors = np.array([(19, 199, 109), (79, 76, 240), (230, 159, 23),
#	      (168, 100, 168), (158, 163, 32), (163, 38, 32), (180, 42, 220)])
#
#    colors = colors / 255
#
#    for (index, polygon) in enumerate(polygons):
#        print(polygon)
#
#        hull = cv2.convexHull(np.asarray(polygon))
#        cv2.drawContours(overlay, [hull], -1, colors[index], -1)
#
#    cv2.addWeighted(overlay, .75, output, .25, 0, output)
#    saveStep.saveReferenceImageBGR(username, imageName, output, 'polygons')

#Coordinates same as align step
# Image is an array of rows
# Rows are arrays of pixels
# Directions are based on person in image (inherited from dlib)

# +---X--->
# | <- Right
# Y
# | Left ->
# v

#Points defined here: https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/ (NOTE: Points are in 0 indexed array while site starts at 1)

X = 0
Y = 1

def getEyePoints(imageShape):
    (start, end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (x, y, w, h) = cv2.boundingRect(np.array([imageShape[start:end]]))

    #Bottom two points for left eye bounding box
    leftEyeRight = [x, y + h]
    leftEyeLeft = [x + w, y + h]

    (start, end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (x, y, w, h) = cv2.boundingRect(np.array([imageShape[start:end]]))

    rightEyeRight = [x, y + h]
    rightEyeLeft = [x + w, y + h]

    bottomY = leftEyeRight[Y] if leftEyeRight[Y] > rightEyeRight[Y] else rightEyeRight[Y]
    leftEyeLeft[Y] = bottomY
    leftEyeRight[Y] = bottomY
    rightEyeLeft[Y] = bottomY
    rightEyeRight[Y] = bottomY

    return [leftEyeLeft, leftEyeRight, rightEyeLeft, rightEyeRight]

def getMouthPoints(imageShape):
    (start, end) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
    (x, y, w, h) = cv2.boundingRect(np.array([imageShape[start:end]]))

    mouthTopRight = [x, y]
    mouthTopLeft = [x + w, y]
    mouthBottomRight = [x, y + h]
    mouthBottomLeft = [x + w, y + h]

    return [mouthTopLeft, mouthTopRight, mouthBottomLeft, mouthBottomRight]

def getJawShape(imageShape):
    jawTopRight = imageShape[6]
    jawBottomRight = imageShape[7]
    jawBottomLeft = imageShape[9]
    jawTopLeft = imageShape[10]

    topY = jawTopRight[Y] if jawTopRight[Y] < jawTopLeft[Y] else jawTopLeft[Y]
    jawTopRight[Y] = topY
    jawTopLeft[Y] = topY

    bottomY = jawBottomRight[Y] if jawBottomRight[Y] > jawBottomLeft[Y] else jawBottomLeft[Y]
    jawBottomRight[Y] = bottomY
    jawBottomLeft[Y] = bottomY
    
    return [jawTopLeft, jawTopRight, jawBottomLeft, jawBottomRight]


def getPolygons(imageShape):
    print('starting get polygons')
    #points = []
    polygons = []

    [leftEyeLeft, leftEyeRight, rightEyeLeft, rightEyeRight] = getEyePoints(imageShape)
    [mouthTopLeft, mouthTopRight, mouthBottomLeft, mouthBottomRight] = getMouthPoints(imageShape)
    [jawTopLeft, jawTopRight, jawBottomLeft, jawBottomRight] = getJawShape(imageShape)

    #Ordering defines boundary (traced point to point)
    #points.append(leftEyeLeft)
    #points.append(leftEyeRight)

    #points.append(mouthTopLeft)
    #points.append(mouthBottomLeft)
    #points.append(mouthBottomRight)
    #points.append(mouthTopRight)

    #points.append(rightEyeLeft)
    #points.append(rightEyeRight)

    #points.append(jawTopRight)
    #points.append(jawBottomRight)
    #points.append(jawBottomLeft)
    #points.append(jawTopLeft)

    #RIGHT TOP CHEEK
    (rightCheekSlope, rightCheekIntercept) = utils.getLineConsts(rightEyeRight, jawTopRight)
    (mouthX, mouthY) = mouthTopRight
    topRightCheekPointY = mouthY
    topRightCheekPointX = utils.getXValue(topRightCheekPointY, rightCheekSlope, rightCheekIntercept)
    cheekRightTop = [rightEyeRight, rightEyeLeft, mouthTopRight, [topRightCheekPointX, topRightCheekPointY]]
    polygons.append(cheekRightTop)

    #RIGHT BOTTOM CHEEK
    (mouthX, mouthY) = mouthBottomRight
    bottomRightCheekPointY = mouthY
    bottomRightCheekPointX = utils.getXValue(bottomRightCheekPointY, rightCheekSlope, rightCheekIntercept)
    cheekRightBottom = [[topRightCheekPointX, topRightCheekPointY], mouthTopRight, mouthBottomRight, [bottomRightCheekPointX, bottomRightCheekPointY]]
    polygons.append(cheekRightBottom)

    #LEFT TOP CHEEK
    (leftCheekSlope, leftCheekIntercept) = utils.getLineConsts(leftEyeLeft, jawTopLeft)
    (mouthX, mouthY) = mouthTopLeft
    topLeftCheekPointY = mouthY
    topLeftCheekPointX = utils.getXValue(topLeftCheekPointY, leftCheekSlope, leftCheekIntercept)
    cheekLeftTop = [leftEyeRight, leftEyeLeft, [topLeftCheekPointX, topLeftCheekPointY], mouthTopLeft]
    polygons.append(cheekLeftTop)

    #LEFT BOTTOM CHEEK
    (mouthX, mouthY) = mouthBottomLeft
    bottomLeftCheekPointY = mouthY
    bottomLeftCheekPointX = utils.getXValue(bottomLeftCheekPointY, leftCheekSlope, leftCheekIntercept)
    cheekLeftBottom = [mouthTopLeft, [topLeftCheekPointX, topLeftCheekPointY], [bottomLeftCheekPointX, bottomLeftCheekPointY], mouthBottomLeft]
    polygons.append(cheekLeftBottom)

    #CHIN
    chinTopRight = (bottomRightCheekPointX, bottomRightCheekPointY)
    chinTopLeft = (bottomLeftCheekPointX, bottomLeftCheekPointY)
    chinTop = [chinTopRight, chinTopLeft, jawTopLeft, jawTopRight]
    polygons.append(chinTop)

    chinBottom = [jawTopRight, jawTopLeft, jawBottomLeft, jawBottomRight]
    polygons.append(chinBottom)

    #polygonImage = image.copy()
    #for (index, point) in enumerate(points):
    #    startPoint = tuple(point)
    #    endPoint = tuple(points[index - 1]) #will access last point as end for first list line
    #    RED = (0, 0, 255)
    #    cv2.line(polygonImage, startPoint, endPoint, RED, 2)
    #    #drawSpot(img, point)

    #drawPolygons(username, polygonImage, polygons, imageName)
    print('Done Getting Polygons!')
    return np.array(polygons)

#Return One polygon of all points
def getFullFacePolygon(imageShape):
    [polygons, error] = getPolygons(imageShape)
    if error is not None:
        return error

    allPoints = []
    for polygon in polygons:
        allPoints = allPoints + list(polygon)

    return [[np.array(allPoints)], None]



