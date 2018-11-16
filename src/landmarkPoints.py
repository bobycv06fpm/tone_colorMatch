from imutils import face_utils
import imutils
import numpy as np

#Dlib face landmarks url (indexed from 1, not 0...) https://www.pyimagesearch.com/wp-content/uploads/2017/04/facial_landmarks_68markup-768x619.jpg
#Apple face landmarks url https://i.stack.imgur.com/2p1la.png

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

class landmarks:

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
    sourceLandmarks = []
    landmarks = []
    
    def __init__(self, source, landmarks):
        self.sourceLandmarks = landmarks
        self.source = source
        if source == 'apple':
            self.landmarks = self.convertAppleLandmarks(landmarks)
        else if source == 'dlib':
            self.landmarks = self.convertDLibLandmarks(landmarks)

    def convertAppleLandmarks(sourceLandmarks):
        landmarks = []

        #Jaw
        landmarks = landmarks + sourceLandmarks[40:51]

        #Right Eyebrow
        landmarks = landmarks + sourceLandmarks[0:4]

        #Left Eyebrow
        landmarks = landmarks + sourceLandmarks[4:8]

        #Right Eye
        rightEye = [
                sourceLandmarks[8],
                sourceLandmarks[9], 
                sourceLandmarks[11], 
                sourceLandmarks[12], 
                sourceLandmarks[13], 
                sourceLandmarks[15]]

        landmarks = landmarks + rightEye

        #Left Eye
        leftEye = [
                sourceLandmarks[16],
                sourceLandmarks[17], 
                sourceLandmarks[19], 
                sourceLandmarks[20], 
                sourceLandmarks[21], 
                sourceLandmarks[23]]

        landmarks = landmarks + leftEye

        #Nose
        landmarks = landmarks + sourceLandmarks[60:63]
        landmarks = landmarks + sourceLandmarks[53:58]

        #Lips
        landmarks = landmarks + [sourceLandmarks[33]]
        landmarks = landmarks + sourceLandmarks[24:33]

        return landmarks

    def convertDLibLandmarks(landmarks):
        landmarks = []

        #Jaw
        jaw = [ sourceLandmarks[0], #1
                sourceLandmarks[1], #2
                sourceLandmarks[3], #4
                sourceLandmarks[5], #6
                sourceLandmarks[6], #7
                sourceLandmarks[8], #9
                sourceLandmarks[10],#11
                sourceLandmarks[11],#12
                sourceLandmarks[13],#14
                sourceLandmarks[15],#16
                sourceLandmarks[16]]#17

        landmarks = landmarks + jaw

        #Right Eyebrow
        rightEyebrow = [
                sourceLandmarks[17],#18
                sourceLandmarks[18],#19
                sourceLandmarks[20],#21
                sourceLandmarks[21]]#22

        landmarks = landmarks + rightEyebrow

        #Left Eyebrow
        leftEyebrow = [
                sourceLandmarks[22],#23
                sourceLandmarks[23],#24
                sourceLandmarks[25],#26
                sourceLandmarks[26]]#27

        landmarks = landmarks + leftEyebrow

        #Right Eye
        landmarks = landmarks + sourceLandmarks[36:42]

        #Left Eye
        landmarks = landmarks + sourceLandmarks[42:48]

        #Nose
        noseBridge = [
                sourceLandmarks[27], #28
                sourceLandmarks[28], #29
                sourceLandmarks[30]] #31

        noseNostril = sourceLandmarks[31:36]

        landmarks = landmarks + noseBridge + noseNostril

        #Lips
        topLip = sourceLandmarks[48:55]
        bottomLip = sourceLandmarks[57:60]

        landmarks = landmarks + topLip + bottomLip

        return landmarks

    def getRightEyeBB():
        if source == 'apple':
            (start, end) = APPLE_RIGHT_EYE
            rightEyePoints = sourceLandmarks[start:end]
        else:
            (start, end) = DLIB_RIGHT_EYE
            rightEyePoints = sourceLandmarks[start:end]

        return cv2.boundingRect(np.array(rightEyePoints))

    def getLeftEyeBB():
        if source == 'apple':
            (start, end) = APPLE_LEFT_EYE
            leftEyePoints = sourceLandmarks[start:end]
        else:
            (start, end) = DLIB_LEFT_EYE
            leftEyePoints = sourceLandmarks[start:end]

        return cv2.boundingRect(np.array(leftEyePoints))

    def getMouthBB():
        if source == 'apple':
            (start, end) = APPLE_MOUTH
            mouthPoints = sourceLandmarks[start:end]
        else:
            (start, end) = DLIB_MOUTH
            mouthPoints = sourceLandmarks[start:end]

        return cv2.boundingRect(np.array(mouthPoints))

    def

