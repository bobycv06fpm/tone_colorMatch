import sys
sys.path.append('../src/')

import argparse
import runSteps
import saveStep
from loadImages import loadImages
import cv2
import numpy as np

def run(username, imageName):
    images = loadImages(username, imageName)
    combined = np.hstack(images)
    cv2.imwrite('./combinedCompressed.png', combined, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    images = np.array([image.astype('int32') for image in images])

    diffed = images[:-1] - images[1:]
    negativeMasks = [diff < 0 for diff in diffed]
    diffed = [np.abs(diff).astype('uint8') for diff in diffed]

    #for index, diff in enumerate(diffed):
    #    cv2.imwrite('./diff{}.png'.format(index), diff)

    #for index, negativeMask in enumerate(negativeMasks):
    #    cv2.imwrite('./negativeMask{}.png'.format(index), diff)

    diffedCombined = np.hstack(diffed)
    negativeMaskCombined = np.hstack(negativeMasks).astype('uint8')

    cv2.imwrite('./diffedCombined.png', diffedCombined, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    cv2.imwrite('./negativeMaskCombined.png', negativeMaskCombined, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    #cv2.imshow('combined', combined)
    #cv2.waitKey(0)


def strToBool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentError('Boolean value expected. i.e. true, false, yes, no')

ap = argparse.ArgumentParser()
#ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
ap.add_argument("-u", "--username", required=True, default="false", help="The Users user name...")
ap.add_argument("-n", "--name", required=True, help="path and root name of image, i.e. images/doug")
args = vars(ap.parse_args())

imageName = args["name"]
username = args["username"]

run(username, imageName)

