import cv2
import csv
import os
import numpy as np

pathToHex = '../../scraped/makeupForever/makeupForeverColors'
pathToCSV = '../../scraped/makeupForever/makeupForeverColors.csv'

colors = []

with open(pathToHex, 'r', newline='') as f:
    for line in f.readlines():
        hexRGB = line.lstrip(' #').rstrip()
        hexR = hexRGB[0:2]
        hexG = hexRGB[2:4]
        hexB = hexRGB[4:6]
        colors.append([int(hexR, 16), int(hexG, 16), int(hexB, 16)])

with open(pathToCSV, 'w', newline='') as f:
    colorWriter = csv.writer(f, delimiter=' ', quotechar='|')
    colorWriter.writerows(colors)

os.chmod(pathToCSV, 0o777)
print(colors)

