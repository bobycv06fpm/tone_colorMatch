def getWidth(image):
    return image["image"].shape[1]

def getHeight(image):
    return image["image"].shape[0]

def getImageWidth(image):
    return image.shape[1]

def getImageHeight(image):
    return image.shape[0]

def getLineConsts(pointA, pointB):
    (Ax, Ay) = pointA
    (Bx, By) = pointB

    if(Bx == Ax):
        return ('vertical', Ax)

    slope = (By - Ay) / (Bx - Ax)
    intercept = Ay - (slope * Ax)

    return (slope, intercept)

def getXValue(yValue, slope, intercept):
    if slope == 'vertical':
        return int(intercept)

    return int(round((yValue / slope) - (intercept / slope)))

def getLowestYValue(A, B):
    (Ax, Ay) = A
    (Bx, By) = B

    if Ay < By:
        return By
    return Ay

def setToLowestYValue(newY, points):
    updatedPoints = []
    for (y, x) in points:
        updatedPoints.append((newY, x))
    
    return updatedPoints
