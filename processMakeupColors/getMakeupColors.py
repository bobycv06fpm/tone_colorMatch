"""Process raw scraped content into usable (approximate) colors for comparison"""
from os import listdir
from os.path import join
import math
import colorsys
import cv2

ROOT = '../../'

def getImageWidth(image):
    return image.shape[1]

def getImageHeight(image):
    return image.shape[0]

def getFentiColors(trueColor):
    fentiColorFiles = listdir(ROOT + 'scraped/fenti_colors')
    fentiColors = []
    #print(fentiColors)

    for file in fentiColorFiles:
        image = cv2.imread(join(ROOT + 'scraped/fenti_colors', file))
        fentiColors.append(image)

    width = getImageWidth(fentiColors[0])
    height = getImageHeight(fentiColors[0])
    numPixels = width * height
    fenti_averages = []
    fenti_scaled = []

    for color in fentiColors:
        average = [0, 0, 0]
        for w in range(0, width):
            for h in range(0, height):
                average[0] = average[0] + color[h, w, 0]
                average[1] = average[1] + color[h, w, 1]
                average[2] = average[2] + color[h, w, 2]

        average[0] = math.floor(average[0] / numPixels)
        average[1] = math.floor(average[1] / numPixels)
        average[2] = math.floor(average[2] / numPixels)
        fenti_averages.append(average)

        if(trueColor):
            fenti_scaled.append([average[2]/255, average[1]/255, average[0]/255])
        else:
            fenti_scaled.append([0, 0, 1])

    fenti_r = []
    fenti_g = []
    fenti_b = []

    for color in fenti_averages:
        fenti_b.append(color[0])
        fenti_g.append(color[1])
        fenti_r.append(color[2])


    return ((fenti_r, fenti_g, fenti_b), fenti_scaled)

def getMakeupForeverColors(trueColor):
    makeupForeverHex = open(ROOT + "scraped/makeupForever/makeupForeverColors", "r")

    mf_r = []
    mf_g = []
    mf_b = []

    mf_scaled = []

    for line in makeupForeverHex.readlines():
        h = line.lstrip(' #')
        (r, g, b) = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
        mf_r.append(r)
        mf_g.append(g)
        mf_b.append(b)

        if trueColor:
            mf_scaled.append([r/255, g/255, b/255])
        else:
            mf_scaled.append([1, 0, 0])


    return ((mf_r, mf_g, mf_b), mf_scaled)


def getBareMineralsColors(trueColor):
    bareMineralsRGB = open(ROOT + "scraped/bm_colors/bm_colors", "r")

    bm_r = []
    bm_g = []
    bm_b = []
    bm_scaled = []

    bareMineralsArr = bareMineralsRGB.readlines()

    for line in bareMineralsArr:
        [r, g, b] = line.rstrip("\n").split(" ")
        bm_r.append(int(r))
        bm_g.append(int(g))
        bm_b.append(int(b))

        if trueColor:
            bm_scaled.append([int(r)/255, int(g)/255, int(b)/255])
        else:
            bm_scaled.append([0, 1, 0])

    return ((bm_r, bm_g, bm_b), bm_scaled)

def convertRGBToHSV(r_values, g_values, b_values):
    h_values = []
    s_values = []
    v_values = []
    for (i, r) in enumerate(r_values):
        g = g_values[i]
        b = b_values[i]

        (h, s, v) = colorsys.rgb_to_hsv(r, g, b)
        h_values.append(h)
        s_values.append(s)
        v_values.append(v)

    return (h_values, s_values, v_values)



#trueColor = True
#plotRGB(trueColor, plt)
#plotHSV(trueColor, plt)

#plt.show()
