"""Mask out Clipped Values"""
import numpy as np

def getClippedMask(img, shadowPixels=1):
    """Returns a mask covering pixels that are blown out or too small"""
    highlightPixels = np.iinfo(img.dtype).max - 1 #Blown Out Highlights

    isSmallSubPixelMask = img < shadowPixels
    isLargeSubPixelMask = img > highlightPixels

    isClippedSubPixelMask = np.logical_or(isSmallSubPixelMask, isLargeSubPixelMask)
    isClippedPixelMask = np.any(isClippedSubPixelMask, axis=2)

    return isClippedPixelMask
