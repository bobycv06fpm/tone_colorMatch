import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plotPoints(pixels, markers=[[]]):
    step = 100
    print("Plotting RGB Points")
    print("Printing " + str(len(pixels)) + " pixels in steps of " + str(step))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')

    scaled = []
    unscaled_x = []
    unscaled_y = []
    unscaled_z = []

    #for pixel in pixels:
    for index in range(0, len(pixels), step):
        pixel = pixels[index]
        scaled.append((pixel[2]/255, pixel[1]/255, pixel[0]/255))
        unscaled_x.append(pixel[2])
        unscaled_y.append(pixel[1])
        unscaled_z.append(pixel[0])

    for index, marker in enumerate(markers):
        for point in marker:
            if index != 0:
                scaled.append((0, 0, 1))
            else:
                scaled.append((0, 1, 0))
            unscaled_x.append(point[2])
            unscaled_y.append(point[1])
            unscaled_z.append(point[0])


    ax.scatter(unscaled_x, unscaled_y, unscaled_z, 'z', 20, scaled, True)
    plt.show()

def plotHSV(hsvValues, rgb):
    hue = []
    sat = []
    val = []
    scaled = []
    step = 100

    print("Plotting Points")
    print("Printing " + str(len(hsvValues)) + " pixels in steps of " + str(step))

    for i in range(0, len(hsvValues), step):
        hsv = hsvValues[i]
        (h, s, v) = hsv
        hue.append(h)
        sat.append(s)
        val.append(v)

        scaled.append([rgb[i][0], rgb[i][1], rgb[i][2]])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Hue')
    ax.set_ylabel('Saturation')
    ax.set_zlabel('Value')

    base_size = 20

    ax.scatter(hue, sat, val, 'z', base_size, scaled, False)
    plt.show()

def hsvMultiplot(sets):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Hue')
    ax.set_ylabel('Saturation')
    ax.set_zlabel('Value')

    for series in sets:
        ((h, s, v), scaled, size) = series
        ax.scatter(h, s, v, 'z', size, scaled, False)

    plt.show()

def scaleRGBValues(rgb_values):
    rgb_scaled = []
    for rgb in rgb_values:
        (r, g, b) = rgb
        r = r / 255
        g = g / 255
        b = b / 255

        rgb_scaled.append((r, g, b))

    return rgb_scaled

