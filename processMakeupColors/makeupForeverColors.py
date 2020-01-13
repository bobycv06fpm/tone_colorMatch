"""Process scraped Makeup Forever colors into csv file"""
import csv
import os

root = '../../'
pathToHex = root + 'scraped/makeupForever/makeupForeverColors'
pathToCSV = root + 'scraped/makeupForever/makeupForeverColors.csv'

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
