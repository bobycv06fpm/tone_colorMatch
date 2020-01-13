"""Set of tools for working with color"""
import colorsys
import numpy as np

def bgr_to_hsv(pixel):
    """Convert a single pixel from BGR to HSV"""
    [b, g, r] = pixel
    return colorsys.rgb_to_hsv(r, g, b)

# https://www.rapidtables.com/convert/color/hsv-to-rgb.html Modified to Scaled Value. Check notes October 3, 2019
def hueSatToBGRRatio(hue, sat):
    """Returns the ratio of Green and Blue to Red from Hue and Sat. Use result for white balancing"""
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
#+Vectorized
def naiveBGRtoHSV(bgrImage, isFloat=True):
    """Convert full image from BGR to HSV without having to convert to integer"""
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
    hsvImage[mask_zeroMax, S] = 0
    hsvImage[mask_zeroMax, V] = 0

    return hsvImage

#sRGB -> linearRGB and back taken from sRGB wikipedia https://en.wikipedia.org/wiki/SRGB#Specification_of_the_transformation
GAMMA = 2.4
ALPHA = 0.055
Small_Values_Const = 12.92

One_Over_Gamma = 1 / GAMMA
One_Plus_Alpha = 1 + ALPHA

def __linearBGR_to_sBGR_image_largeValues(image):
    return (One_Plus_Alpha * (image ** One_Over_Gamma)) - ALPHA

def convert_sBGR_to_linearBGR_float(image):
    """Convert sBGR to linearBGR. Takes image as [0, 255]. Returns image as [0, 1] float"""
    image_float = image / 255

    largeValuesMask = (image_float > .04045)
    largeValues = image_float[largeValuesMask]

    largeValues += ALPHA
    largeValues /= One_Plus_Alpha
    largeValues **= GAMMA

    image_float[largeValuesMask] = largeValues
    image_float[np.logical_not(largeValuesMask)] /= Small_Values_Const

    return image_float

def convert_linearBGR_float_to_sBGR(image):
    """Convert linearBGR to sBGR . Takes image as [0, 1] float. Returns image as [0, 255] float"""
    smallValuesMask = (image <= .0031308)

    convertedLargeValues = __linearBGR_to_sBGR_image_largeValues(image)
    image *= Small_Values_Const

    convertedLargeValues *= (~smallValuesMask)
    image *= smallValuesMask

    return (image + convertedLargeValues) * 255

#https://en.wikipedia.org/wiki/Luma_(video)
def getRelativeLuminance(points):
    """Returns the relative luminance for the BGR points"""
    points = np.array(points)
    bgrLuminanceConsts = np.array([0.0722, 0.7152, 0.2126])
    return np.sum(points * bgrLuminanceConsts, axis=1)

def __convert_CIE_xy_to_unscaledBGR(x, y):
    z = 1 - x - y

    #This converts the xyz value to linear RGB
    #From http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html (sRGB, d65)
    RGBConversionMatrix = np.array([[3.2404542, -1.5371385, -0.4985314], [-0.9692660, 1.8760108, 0.0415560], [0.0556434, -0.2040259, 1.0572252]])

    rgb = np.dot(RGBConversionMatrix, np.array([x, y, z]))
    bgr = np.flipud(rgb)

    return bgr

def whitebalance_from_asShot_to_d65(image, x, y):
    """Whitebalances image from as shot WB to d65 white point"""
    asShotBGR = __convert_CIE_xy_to_unscaledBGR(x, y)
    targetBGR = __convert_CIE_xy_to_unscaledBGR(0.31271, 0.32902) #Illuminant D65
    bgrMultiplier = asShotBGR / targetBGR
    return image * bgrMultiplier

def rotateHue(hue):
    """Linear hue shift. Helps when working with hues around 0 like skin tone often is"""
    hue = hue.copy()
    shiftMask = hue <= 2/3
    hue[shiftMask] += 1/3
    hue[np.logical_not(shiftMask)] -= 2/3
    return hue
