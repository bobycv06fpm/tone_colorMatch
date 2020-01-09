import numpy as np
import dlib
import utils 
import cv2
import colorsys

def detectFace(image, predictor, detector):

    try:
        #ratio = image.shape[0] / 500
        ratio = 5

        smallImage = cv2.resize(image, (0, 0), fx=1/ratio, fy=1/ratio)
        gray = cv2.cvtColor(smallImage, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)
        shape = predictor(gray, rects[0])
        shape = face_utils.shape_to_np(shape)
        shape = shape * ratio
        return [image, shape]
    except:
        raise ValueError("Could Not Detect Face")

