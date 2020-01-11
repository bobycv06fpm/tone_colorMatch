"""Set of tools to simplify formatting pipeline data for output or transfer"""
def __getReflectionMap(leftReflection, rightReflection):
    """Builds a dictionary containting left and right reflection"""
    value = {}
    value['left'] = [float(value) for value in leftReflection]
    value['right'] = [float(value) for value in rightReflection]

    return value

def __getResponse(imageName, successful, captureSets=None, linearFits=None, bestGuess=None, averageReflectionArea=None):
    """Builds a dictionary containing all of the output values helpful for analyzing how well the image pipeline is performing"""
    response = {}
    response['name'] = imageName
    response['successful'] = successful
    response['captures'] = {}
    response['linearFits'] = linearFits
    response['bestGuess'] = bestGuess
    response['reflectionArea'] = averageReflectionArea

    if not successful:
        return response

    for captureSet in captureSets:
        faceRegions, leftEyeReflection, rightEyeReflection = captureSet
        key = faceRegions.capture.name
        response['captures'][key] = {}
        response['captures'][key]['regions'] = faceRegions.getRegionMapValue()
        response['captures'][key]['reflections'] = __getReflectionMap(leftEyeReflection, rightEyeReflection)

    return response

def getSuccessfulResponse(imageName, captureSets, linearFits, bestGuess, averageReflectionArea):
    """Returns the response for image pipeline success"""
    return __getResponse(imageName, True, captureSets, linearFits, bestGuess, averageReflectionArea)

def getFailureResponse(imageName):
    """Returns the response for image pipeline failure"""
    return __getResponse(imageName, False)
