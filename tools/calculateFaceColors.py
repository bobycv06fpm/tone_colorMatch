import sys
sys.path.insert(0, '../src')

import runSteps
import argparse
import numpy as np
import os
import re
import multiprocessing as mp
import json

root = '../../'
path = root + 'images/'
user = 'doug'
userPath = path + user
hsvCount_path = 'steps/4.csv'
imageStat_path = 'reference/imageStats.csv'

minDirectory = 196
#minDirectory = 240

#userDirectories = [(os.path.join(userPath, o), o) for o in os.listdir(userPath) if os.path.isdir(os.path.join(userPath, o))]
userDirectories = [o for o in os.listdir(userPath) if os.path.isdir(os.path.join(userPath, o))]

filteredUserDirectories = [o for o in userDirectories if int(re.search( r'[0-9]+$', o, re.M|re.I).group()) >= minDirectory]

print('filted user directories :: ' + str(filteredUserDirectories))

#faceColors = []
#
#for imageName in filteredUserDirectories:
#    print('Image Name :: ' + str(imageName))
#    try:
#        result = runSteps.run(user, imageName, fast=False, saveStats=False)
#    except Exception as e:
#        print('Error Processing ' + str(imageName) + ' | ' + str(e))
#    else:
#        print('result :: ' + str(result))
#        faceColors.append([imageName, result])

allImageArgs = [(user, imageName) for imageName in filteredUserDirectories]

errors = []
with mp.Pool() as pool:
    faceColors = pool.starmap_async(runSteps.run, allImageArgs)
    print('\t\tDONE!\t\t')
    try:
        results = faceColors.get()
    except Exception as err:
        print('Uncaught Error :: {}'.format(err))
    else:
        faceColorsJson = json.dumps(results)
        
        with open('faceColors.json', 'w') as f:
            f.write(faceColorsJson)
        
        os.chmod('faceColors.json', 0o777)
