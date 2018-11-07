from sklearn.cluster import KMeans
import saveStep
import numpy as np

#def scaleHSV(points):
#    max_value = 0
#    for point in points:
#        (h, s, v) = point
#        if v > max_value:
#            max_value = v
#
#    scale = 1 / max_value
#    scaled = []
#    for point in points:
#        (h, s, v) = point
#        v = v * scale
#        scaled.append((h, s, v))
#
#    return scaled

NUMBER_OF_CLUSTERS=4
#NUMBER_OF_CLUSTERS=5
#NUMBER_OF_CLUSTERS=10

def kmeans(username, imageName, points):
    print('Starting k means')
    kmeans = KMeans(n_clusters=NUMBER_OF_CLUSTERS, n_jobs=-1).fit(points[:, 0:2])
    medianValue = np.median(points[:, 2])
    print('Median Value :: ' + str(medianValue))
    if imageName is not None:
        saveStep.logMeasurement(username, imageName, 'pixels in mask', str(len(kmeans.labels_)))

    clusterCount = []
    for _ in range(NUMBER_OF_CLUSTERS):
        clusterCount.append(0)

    for label in kmeans.labels_:
        clusterCount[label] = clusterCount[label] + 1

    if imageName is not None:
        for (i, cluster) in enumerate(kmeans.cluster_centers_):
            saveStep.logMeasurement(username, imageName, 'cluster ' + str(i), str(cluster) + str(clusterCount[i]))

    clustersWithCount = []
    for (index, clusterCenter) in enumerate(kmeans.cluster_centers_):
        clusterCenterList = clusterCenter.tolist()
        clusterCenterList.append(medianValue)
        clusterCenterList.append(clusterCount[index])
        clustersWithCount.append(clusterCenterList)

    return clustersWithCount
