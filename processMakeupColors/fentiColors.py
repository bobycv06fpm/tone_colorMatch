"""Process scraped fenti colors into csv file"""
import csv
import os
import cv2
import numpy as np

root = '../../'

pathToImageDir = root + 'scraped/fenti_colors/'
pathToCSV = root + 'scraped/fenti_colors/fentiColors.csv'
pathsToImages = [os.path.join(pathToImageDir, img) for img in os.listdir(pathToImageDir) if img[-3:] == 'jpg']

colors = []

for pathToImage in pathsToImages:
    colorSample = cv2.imread(pathToImage)
    colorSampleMean = np.round(np.mean(colorSample, axis=(0, 1))).astype('int32')
    colorSampleMean = np.flip(colorSampleMean, axis=0)
    colors.append(colorSampleMean)


with open(pathToCSV, 'w', newline='') as f:
    colorWriter = csv.writer(f, delimiter=' ', quotechar='|')
    colorWriter.writerows(colors)

os.chmod(pathToCSV, 0o777)
print(colors)
