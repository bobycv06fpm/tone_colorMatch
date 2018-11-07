import argparse
import numpy as np
import os
import re
import runSteps
import dlib
import multiprocessing as mp

root = '../../'
path = root + 'images/'
hsvCount_path = 'steps/4.csv'
imageStat_path = 'reference/imageStats.csv'

userDirectories = [(os.path.join(path, o), o) for o in os.listdir(path) if os.path.isdir(os.path.join(path, o))]

allImageArgs = []
deletePaths = []

for (userDirectoryPath, username) in userDirectories:
    allImageArgs += [(username, imageName, False, True) for imageName in os.listdir(userDirectoryPath) if os.path.isdir(os.path.join(userDirectoryPath, imageName))]
    deletePaths += [os.path.join(userDirectoryPath, imageName) for imageName in os.listdir(userDirectoryPath) if os.path.isdir(os.path.join(userDirectoryPath, imageName))]

print('deletePaths :: ' + str(deletePaths))
for deletePath in deletePaths:
    hsvCountPath = os.path.join(deletePath, hsvCount_path)
    if os.path.exists(hsvCountPath):
        os.remove(hsvCountPath)

    imageStatPath = os.path.join(deletePath, imageStat_path)
    if os.path.exists(imageStatPath):
        os.remove(imageStatPath)

errors = []
with mp.Pool() as pool:
    faceColors = pool.starmap_async(runSteps.run, allImageArgs)
    try:
        errors = faceColors.get()
    except NameError as err:
        print('Uncaught Error :: {}'.format(err))
    
    print('Errors ::\n {}'.format(errors))

