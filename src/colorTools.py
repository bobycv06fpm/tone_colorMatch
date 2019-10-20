import colorsys
import numpy as np
import cv2

TARGET_FLASH_CONTRIBUTION = 150

def scaleToMaxHSVValue(points, white):
    print('scaling to max hsv Value')
    (wb_b, wb_g, wb_r) = white
    (wb_h, wb_s, wb_v) = colorsys.rgb_to_hsv(wb_r, wb_g, wb_b)

    scale = 1 / wb_v

    scaledBGR = []
    for point in points:
        (b, g, r) = point
        (h, s, v) = colorsys.rgb_to_hsv(r, g, b)
        v = v * scale 
        v = v if v < 1 else 1
        (r, g, b) = colorsys.hsv_to_rgb(h, s, v)
        scaledBGR.append((b, g, r))

    return scaledBGR

def bgr_to_hsv(pixel):
    [b, g, r] = pixel
    return colorsys.rgb_to_hsv(r, g, b)

def bgr_to_hsv_and_scale(pixel, multiplier):
    [b, g, r] = pixel
    [h, s, v] = colorsys.rgb_to_hsv(r, g, b)
    scaled_v = multiplier * v
    if scaled_v > 255:
        scaled_v = 0

    return [h, s, scaled_v]

def rgb_to_hsv(pixel):
    return colorsys.rgb_to_hsv(pixel[0], pixel[1], pixel[2])

def hsv_to_rgb(pixel):
    return colorsys.hsv_to_rgb(pixel[0], pixel[1], pixel[2])

def hsv_to_bgr(pixel):
    [r, g, b] = colorsys.hsv_to_rgb(pixel[0], pixel[1], pixel[2])
    return [b, g, r]

def rgb_to_plot_rgb(pixel):
    [r, g, b] = pixel
    return [r/255, g/255, b/255]

def convertRGBToHSV(points):
    return np.apply_along_axis(rgb_to_hsv, 1, points)

def convertBGRToHSV(points):
    return np.apply_along_axis(bgr_to_hsv, 1, points)

def convertHSVToRGB(points):
    return np.apply_along_axis(hsv_to_rgb, 1, points)

def convertRGBToPlotRGB(points):
    return np.apply_along_axis(rgb_to_plot_rgb, 1, points)

def convertBGRToHSVAndScale(points, averageFlashContribution):
    #multiplier = TARGET_FLASH_CONTRIBUTION / averageFlashContribution
    multiplier = 1
    print('Multiplier :: ' + str(multiplier))
    #converted = cv2.cvtColor(np.array([points.astype('uint8')]), cv2.COLOR_BGR2HSV)[0]
    converted = np.apply_along_axis(bgr_to_hsv, 1, points)
    return converted

# https://www.rapidtables.com/convert/color/hsv-to-rgb.html Modified to Scaled Value. Check notes October 3, 2019
def hueSatToProportionalBGR(hue, sat): 
    #Prime just means V is divided out
    c_prime = sat
    x_prime = sat * (1 - abs((((hue * 360) / 60) % 2) - 1))

    if hue < (60 / 360):
        (r_prime, g_prime, b_prime) = (c_prime, x_prime, 0)
    elif hue < (120 / 360):
        (r_prime, g_prime, b_prime) = (x_prime, c_prime, 0)
    elif hue < (180 / 360):
        (r_prime, g_prime, b_prime) = (0, c_prime, x_prime)
    elif hue < (240 / 360):
        (r_prime, g_prime, b_prime) = (0, x_prime, c_prime)
    elif hue < (300 / 360):
        (r_prime, g_prime, b_prime) = (x_prime, 0, c_prime)
    elif hue < (360 / 360):
        (r_prime, g_prime, b_prime) = (c_prime, 0, x_prime)
    else:
        print('ERROR EXTRACTING HUE SAT')

    r = 1
    g = (g_prime + 1 - sat) / (r_prime + 1 - sat)
    b = (b_prime + 1 - sat) / (r_prime + 1 - sat)

    return [b, g, r]

H = B = 0
S = G = 1
V = R = 2
#Algo from http://coecsl.ece.illinois.edu/ge423/spring05/group8/finalproject/hsv_writeup.pdf
#Also https://stackoverflow.com/questions/3018313/algorithm-to-convert-rgb-to-hsv-and-hsv-to-rgb-in-range-0-255-for-both
#Vectorized by me
def naiveBGRtoHSV(bgrImage, isFloat=True):
    if isFloat:
        bgrImage *= 255
    else:
        bgrImage = bgrImage.astype(np.float)

    hsvImage = np.copy(bgrImage)

    minValues = np.min(bgrImage, axis=2)
    maxValues = np.max(bgrImage, axis=2)

    delta = maxValues - minValues

    mask_deltaAlmostZero = delta < .00001
    mask_zeroMax = maxValues == 0

    maxValues[mask_zeroMax] = 1 #Setting to one to avoid divide by zero. Fix at end
    delta[mask_deltaAlmostZero] = 1 #Setting to one to avoid divide by zero. Fix at end

    #Set The Value
    hsvImage[:, :, V] = maxValues / 255

    #Set Saturation
    hsvImage[:, :, S] = (delta / maxValues)

    #Set Hue
    hsvImage[:, :, H] = 1.0
    mask_maxIsBlue = bgrImage[:, :, B] == maxValues
    mask_maxIsGreen = bgrImage[:, :, G] == maxValues
    mask_maxIsRed = bgrImage[:, :, R] == maxValues

    hsvImage[mask_maxIsBlue, H] = ((bgrImage[:, :, R] - bgrImage[:, :, G]) / delta)[mask_maxIsBlue]
    hsvImage[mask_maxIsGreen, H] = ((bgrImage[:, :, B] - bgrImage[:, :, R]) / delta)[mask_maxIsGreen]
    hsvImage[mask_maxIsRed, H] = ((bgrImage[:, :, G] - bgrImage[:, :, B]) / delta)[mask_maxIsRed]

    hsvImage[:, :, H] *= 60
    hsvImage[mask_maxIsBlue, H] += 240
    hsvImage[mask_maxIsGreen, H] += 120
    mask_negativeHue = hsvImage[:, :, H] < 0
    hsvImage[mask_negativeHue, H] += 360
    hsvImage[:, :, H] = hsvImage[:, :, H] / 360
    
    #Return Not So Early Statement from Saturation
    hsvImage[mask_deltaAlmostZero, H] = 0
    hsvImage[mask_deltaAlmostZero, S] = 0

    hsvImage[mask_zeroMax, H] = 0
    hsvImage[mask_zeroMax, S] = 0#float('nan')
    hsvImage[mask_zeroMax, V] = 0

    return hsvImage

#def naiveHSVtoBGR(hsvImage):
#    bgrImage = np.copy(hsvImage)
#
#    mask_zeroSaturation = hsvImage[:, :, S] == 0
#
#    bgrImage[mask_zeroSaturation] = np.ones(3) * hsvImage[mask_zeroSaturation][:, V]
#    return bgrImage


#def hsvToRGB(points):
#    rgb_points = []
#    for (h, s, v) in points:
#        h = h if h < 1 else 1
#        s = s if s < 1 else 1
#        v = v if v < 255 else 255
#        rgb_points.append(colorsys.hsv_to_rgb(h, s, v))
#
#    return rgb_points

def filterHSV(hsvPoints):
    filteredHSV = []

    for (i, hsv) in enumerate(hsvPoints):
        (h, s, v) = hsv
        if(h <= .1 and v > 0):
            filteredHSV.append(hsv)

    return filteredHSV

def rgbArraysToHSVArrays(rgb_arrays):
    (r_values, g_values, b_values) = rgb_arrays

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

#########################################

#sRGB -> linearRGB and back taken from sRGB wikipedia https://en.wikipedia.org/wiki/SRGB#Specification_of_the_transformation
GAMMA = 2.4
#GAMMA = 2.6
ALPHA = 0.055
Small_Values_Const = 12.92

One_Over_Gamma = 1 / GAMMA
One_Plus_Alpha = 1 + ALPHA

#NOTE: We are actually passing this BGR but it doesnt much matter because the subpixels are caluclated independently
def sBGR_to_linearBGR_image_largeValues(image):
    return ((image + ALPHA) / One_Plus_Alpha) ** GAMMA

def sBGR_to_linearBGR_image_smallValues(image):
    return image / Small_Values_Const

def linearBGR_to_sBGR_image_largeValues(image):
    return (One_Plus_Alpha * (image ** One_Over_Gamma)) - ALPHA

def linearBGR_to_sBGR_image_smallValues(image):
    return image * Small_Values_Const

#def convert_linearHSV_float_point_to_correctedHSV_float_point(linearHSV_float):
#    linearRGB_float = colorsys.hsv_to_rgb(linearHSV_float[0], linearHSV_float[1], linearHSV_float[2])
#    [sRGB, error] = convert_linearBGR_float_to_sBGR(np.array(linearRGB_float))
#    sRGB = sRGB / 255
#    correctedHSV = colorsys.hsv_to_rgb(sRGB[0], sRGB[1], sRGB[2])
#    return np.array(correctedHSV)
#
def convert_sBGR_to_linearBGR_float(image, isFloat=False):
    if isFloat:
        image_float = image
    else:
        image_float = image / 255

    smallValuesMask = (image_float <= .04045)

    #convertedSmallValues = sBGR_to_linearBGR_image_smallValues(image_float)
    convertedLargeValues = sBGR_to_linearBGR_image_largeValues(image_float)
    image_float /= Small_Values_Const

    #convertedSmallValues *= smallValuesMask
    convertedLargeValues *= (~smallValuesMask)
    image_float *= smallValuesMask

    #print('(almost) Done :: sBGR -> linearBGR float')
    return image_float + convertedLargeValues

def convert_sBGR_to_linearBGR_float_fast(image):
    image_float = image / 255

    largeValuesMask = (image_float > .04045)
    largeValues = image_float[largeValuesMask]

    largeValues += ALPHA
    largeValues /= One_Plus_Alpha
    largeValues **= GAMMA

    image_float[largeValuesMask] = largeValues
    image_float[np.logical_not(largeValuesMask)] /= Small_Values_Const

    return image_float

def convert_linearBGR_to_sBGR_float_fast(image):
    smallValuesMask = (image <= .0031308)

    convertedLargeValues = linearBGR_to_sBGR_image_largeValues(image)
    image *= Small_Values_Const

    convertedLargeValues *= (~smallValuesMask)
    image *= smallValuesMask

    return image + convertedLargeValues

def convert_linearBGR_float_to_sBGR(image):
    #print('Starting :: linearBGR float -> sBGR')
    smallValuesMask = (image <= .0031308)

    #convertedSmallValues = linearBGR_to_sBGR_image_smallValues(image)
    convertedLargeValues = linearBGR_to_sBGR_image_largeValues(image)
    image *= Small_Values_Const

    #convertedSmallValues *= smallValuesMask
    convertedLargeValues *= (~smallValuesMask)
    image *= smallValuesMask

    #print('(almost) Done :: linearBGR float -> sBGR')
    return (image + convertedLargeValues) * 255

def convert_linearBGR_float_to_linearHSV_float(image_float):
    return naiveBGRtoHSV(image_float)

def convert_linearHSV_float_to_linearBGR_float(image_float):
    #image = image_float * 255
    #scaledImage = np.clip(image_float * 255, 0, 255).astype('uint8')
    scaledImage = (image_float * 255).astype('uint16')
    #Any way to avoid converting it to a uint?? Losing precision :(
    #Any way to avoid converting it to a uint?? Losing precision :(
    hsv = cv2.cvtColor(scaledImage, cv2.COLOR_HSV2BGR_FULL)
    return hsv / 255

def convertSingle_sValue_to_linearValue(value):
    if value <= 0.04045:
        return sBGR_to_linearBGR_image_smallValues(value)
    else:
        return sBGR_to_linearBGR_image_largeValues(value)

def convertSingle_linearValue_to_sValue(point):
    point = point / 255
    print('Point :: ' + str(point))
    sBGR_point = []
    for value in point:
        if value <= .0031308:
            sBGR_point.append(linearBGR_to_sBGR_image_smallValues(value))
        else:
            sBGR_point.append(linearBGR_to_sBGR_image_largeValues(value))

    sBGR_point = np.array(sBGR_point)
    sBGR_point *= 255
    return np.clip(sBGR_point, 0, 255).astype('uint8')

#def whitebalanceBGR_float(image, wb):
#    #wbMultiplier = [1, 1, 1]
#    #sortedWB = sorted(wb)
#    #targetValue = sortedWB[1]
#    targetValue = max(wb)
#    #targetValue = min(wb)
#    wbMultiplier = [targetValue, targetValue, targetValue] / wb
#    return image * wbMultiplier

def whitebalanceBGR(capture, wb):
    if not np.all(wb.astype('bool')):
        print('Trying to WB to a 0 value!')
        raise NameError('Trying to WB to a 0 value!')

    targetValue = max(wb)
    wbMultiplier = [targetValue, targetValue, targetValue] / wb
    capture.image = (capture.image * wbMultiplier)#.astype('uint16')

def whitebalanceBGRImage(image, wb):
    if not np.all(wb.astype('bool')):
        print('Trying to WB to a 0 value!')
        raise NameError('Trying to WB to a 0 value!')

    targetValue = max(wb)
    wbMultiplier = [targetValue, targetValue, targetValue] / wb
    return (image * wbMultiplier).astype('uint16')

def whitebalanceBGRPoints(points, wb):
    if not np.all(wb.astype('bool')):
        print('Trying to WB to a 0 value!')
        raise NameError('Trying to WB to a 0 value!')

    targetValue = max(wb)
    wbMultiplier = [targetValue, targetValue, targetValue] / wb
    return points * wbMultiplier

def getRelativeLuminance(points):
    points = np.array(points)
    bgrLuminanceConsts = np.array([0.0722, 0.7152, 0.2126])
    return np.sum(points * bgrLuminanceConsts, axis=1)

def getRelativeLuminanceImage(image):
    #points = np.array(points)
    bgrLuminanceConsts = np.array([0.0722, 0.7152, 0.2126])
    return np.sum(image * bgrLuminanceConsts, axis=2)


def get_sRGB_and_sHSV(points_float):
    print('Points Float :: ' + str(points_float))
    #points = np.array([(points_float * 255).astype('uint8')])
    points = np.clip(points_float * 255, 0, 255).astype('uint8')
    print('Points :: ' + str(points))
    rgb = cv2.cvtColor(points, cv2.COLOR_HSV2RGB_FULL)
    print('RGB :: ' + str(rgb))
    sRGB = convert_linearBGR_float_to_sBGR(rgb/255)
    print('sRGB :: ' + str(sRGB))
    hsv = cv2.cvtColor(sRGB.astype('uint8'), cv2.COLOR_RGB2HSV_FULL)
    print('HSV :: ' + str(hsv))
    plot_hsv = hsv[0] / [255, 255, 1]
    plot_rgb = sRGB[0]

    plot_hsv_filtered = []
    plot_rgb_filtered = []
    for (index, hsv_point) in enumerate(plot_hsv):
        if (hsv_point[0] < .1):
            plot_hsv_filtered.append(hsv_point)
            plot_rgb_filtered.append(plot_rgb[index])

    return (plot_rgb_filtered, plot_hsv_filtered)

def convert_CIE_xy_to_unscaledBGR(x, y):
    z = 1 - x - y

    #print('x y z :: ' + str(x) + ' ' + str(y) + ' ' + str(z))
    #This converts the xyz value to linear RGB
    #From http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html (sRGB, d65)
    RGBConversionMatrix = np.array([[3.2404542, -1.5371385, -0.4985314], [-0.9692660, 1.8760108, 0.0415560], [0.0556434, -0.2040259, 1.0572252]])

    rgb = np.dot(RGBConversionMatrix, np.array([x, y, z]))
    bgr = np.flipud(rgb)

    return bgr

def whitebalance_from_asShot_to_d65(image, x, y):
    asShotBGR = convert_CIE_xy_to_unscaledBGR(x, y)
    #print('As Shot BGR :: ' + str(asShotBGR))

    targetBGR = convert_CIE_xy_to_unscaledBGR(0.31271, 0.32902) #Illuminant D65
    #print('Target BGR :: ' + str(targetBGR))

    bgrMultiplier = asShotBGR / targetBGR
    #print('BGR Multiplier :: ' + str(bgrMultiplier))

    return image * bgrMultiplier

def rotateHue(hue):
    hue = hue.copy()
    shiftMask = hue <= 2/3
    hue[shiftMask] += 1/3
    hue[np.logical_not(shiftMask)] -= 2/3
    return hue

def unRotateHue(hue):
    hue = hue.copy()
    shiftMask = hue >= 1/3
    hue[shiftMask] -= 1/3
    hue[np.logical_not(shiftMask)] += 2/3
    return hue

