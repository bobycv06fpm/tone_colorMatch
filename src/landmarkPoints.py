import numpy as np
import cv2

#Dlib face landmarkPoints url (indexed from 1, not 0...) https://www.pyimagesearch.com/wp-content/uploads/2017/04/facial_landmarkPoints_68markup-768x619.jpg
#Apple face landmarkPoints url https://i.stack.imgur.com/2p1la.png

#Algo Specific Features (For generating bounding boxes... might as well use all the data)
#Format: [start, end)

APPLE_JAW = (40, 51)
APPLE_RIGHT_EYEBROW = (0, 4)
APPLE_LEFT_EYEBROW = (4, 8) 

APPLE_RIGHT_EYE = (8, 16) 
APPLE_LEFT_EYE = (16, 24) 

APPLE_NOSE = (51, 63) 
APPLE_MOUTH = (24, 40) 

DLIB_JAW = (0, 17)
DLIB_RIGHT_EYEBROW = (17, 22)
DLIB_LEFT_EYEBROW = (22, 27) 

DLIB_RIGHT_EYE = (36, 42) 
DLIB_LEFT_EYE = (42, 48) 

DLIB_NOSE = (27, 36) 
DLIB_MOUTH = (48, 68) 

class Landmarks:

    #General Rule Of thumb... Try and match points between Apple and Dlib with the source that has fewer points

    #All Points are defined (left and right) from the perspective of the face

    # [0, 10]  Jaw Outline
    # [11, 14] Right Eyebrow
    # [15, 18] Left Eyebrow
    # [19, 24] Right Eye
    # [25, 30] Left Eye
    # [31, 38] Nose
    #   [31, 33] Nose Bridge
    #   [34, 38] Nostrils
    # [39, 48] Lips
    #   [39, 45] Top Lip
    #   [46, 48] Bottom Lip


    source = ''
    #sourceLandmarkPoints = []
    landmarkPoints = []
    
    def __init__(self, source, landmarkPoints, imageSize):
        #self.sourceLandmarkPoints = np.array(landmarkPoints)
        self.source = source
        if source == 'apple':
            self.landmarkPoints = self.convertAppleLandmarks(landmarkPoints, imageSize)
            #self.sourceLandmarkPoints = imageSize[0] - self.sourceLandmarkPoints[:, 1]
        elif source == 'dlib':
            self.landmarkPoints = self.convertDLibLandmarks(landmarkPoints)

    def convertAppleLandmarks(self, sourceLandmarkPoints, imageSize):
        landmarkPoints = []

        #Jaw
        landmarkPoints = landmarkPoints + sourceLandmarkPoints[40:51]

        #Right Eyebrow
        landmarkPoints = landmarkPoints + sourceLandmarkPoints[0:4]

        #Left Eyebrow
        landmarkPoints = landmarkPoints + sourceLandmarkPoints[4:8]

        #Right Eye
        rightEye = [
                sourceLandmarkPoints[8],
                sourceLandmarkPoints[9], 
                sourceLandmarkPoints[11], 
                sourceLandmarkPoints[12], 
                sourceLandmarkPoints[13], 
                sourceLandmarkPoints[15]]

        landmarkPoints = landmarkPoints + rightEye

        #Left Eye
        leftEye = [
                sourceLandmarkPoints[16],
                sourceLandmarkPoints[17], 
                sourceLandmarkPoints[19], 
                sourceLandmarkPoints[20], 
                sourceLandmarkPoints[21], 
                sourceLandmarkPoints[23]]

        landmarkPoints = landmarkPoints + leftEye

        #Nose
        landmarkPoints = landmarkPoints + sourceLandmarkPoints[60:63]
        landmarkPoints = landmarkPoints + sourceLandmarkPoints[53:58]

        #Lips
        landmarkPoints = landmarkPoints + [sourceLandmarkPoints[33]]
        landmarkPoints = landmarkPoints + sourceLandmarkPoints[24:33]

        #Flip Y coordinates... Indexed off of Bottom, not top..
        landmarkPoints = np.array(landmarkPoints)
        landmarkPoints[:, 1] = imageSize[0] - landmarkPoints[:, 1]
        return landmarkPoints

    def convertDLibLandmarks(self, landmarkPoints):
        landmarkPoints = []

        #Jaw
        jaw = [ sourceLandmarkPoints[0], #1
                sourceLandmarkPoints[1], #2
                sourceLandmarkPoints[3], #4
                sourceLandmarkPoints[5], #6
                sourceLandmarkPoints[6], #7
                sourceLandmarkPoints[8], #9
                sourceLandmarkPoints[10],#11
                sourceLandmarkPoints[11],#12
                sourceLandmarkPoints[13],#14
                sourceLandmarkPoints[15],#16
                sourceLandmarkPoints[16]]#17

        landmarkPoints = landmarkPoints + jaw

        #Right Eyebrow
        rightEyebrow = [
                sourceLandmarkPoints[17],#18
                sourceLandmarkPoints[18],#19
                sourceLandmarkPoints[20],#21
                sourceLandmarkPoints[21]]#22

        landmarkPoints = landmarkPoints + rightEyebrow

        #Left Eyebrow
        leftEyebrow = [
                sourceLandmarkPoints[22],#23
                sourceLandmarkPoints[23],#24
                sourceLandmarkPoints[25],#26
                sourceLandmarkPoints[26]]#27

        landmarkPoints = landmarkPoints + leftEyebrow

        #Right Eye
        landmarkPoints = landmarkPoints + sourceLandmarkPoints[36:42]

        #Left Eye
        landmarkPoints = landmarkPoints + sourceLandmarkPoints[42:48]

        #Nose
        noseBridge = [
                sourceLandmarkPoints[27], #28
                sourceLandmarkPoints[28], #29
                sourceLandmarkPoints[30]] #31

        noseNostril = sourceLandmarkPoints[31:36]

        landmarkPoints = landmarkPoints + noseBridge + noseNostril

        #Lips
        topLip = sourceLandmarkPoints[48:55]
        bottomLip = sourceLandmarkPoints[57:60]

        landmarkPoints = landmarkPoints + topLip + bottomLip

        return np.array(landmarkPoints)

    def getRightEye(self):
        return np.array(self.landmarkPoints[19:25])

    def getRightEyeRegion(self):
        return np.array(list(self.landmarkPoints[19:25]) + list(self.landmarkPoints[11:15]))

    def getRightEyeBB(self):
        return cv2.boundingRect(self.getRightEye())

    def getRightEyeRegionBB(self):
        return cv2.boundingRect(self.getRightEyeRegion())

    def getRightEyeInnerBB(self):
        innerEyePoints = np.array(list(self.landmarkPoints[26:28]) + list(self.landmarkPoints[29:31]))
        return cv2.boundingRect(innerEyePoints)

    def getLeftEye(self):
        return np.array(self.landmarkPoints[25:31])

    def getLeftEyeRegion(self):
        return np.array(list(self.landmarkPoints[25:31]) + list(self.landmarkPoints[15:19]))

    def getLeftEyeBB(self):
        return cv2.boundingRect(self.getLeftEye())

    def getLeftEyeRegionBB(self):
        return cv2.boundingRect(self.getLeftEyeRegion())

    def getLeftEyeInnerBB(self):
        innerEyePoints = np.array(list(self.landmarkPoints[20:22]) + list(self.landmarkPoints[23:25]))
        return cv2.boundingRect(innerEyePoints)

    def getMouth(self):
        return np.array(self.landmarkPoints[39:49])
        return cv2.boundingRect(self.getMouth())

    def getMouthBB(self):
        mouthPoints = self.landmarkPoints[39:49]
        return cv2.boundingRect(np.array(mouthPoints))

    def getFaceBB(self):
        return cv2.boundingRect(np.array(sourceLandmarkPoints))

    def getInteriorPoints(self):
        return np.array(list(self.landmarkPoints[11:]) + list(self.landmarkPoints[4:7]))

    def getEyesPoints(self):
        return self.landmarkPoints[11:34]
        #return cv2.boundingRect(np.array(eyes))


    #def getInteriorPoints(self):
    #    return self.landmarkPoints[19:31]

