import cv2
import csv
import os
import numpy as np
import colorTools
import json
import psycopg2
import getVersion

IMAGES_DIR = '/home/dmacewen/Projects/tone/images/'

#Turn into an abstraction around state? Encapsulate DB and File System?
# Load Images
# Load DB info
# Load and Save reference data
class State:

    def __init__(self, user_id, capture_id=None):
        self.version = getVersion.getVersion()
        print("SERVER VERSION :: " + str(self.version))

        self.user_id = user_id
        self.capture_id = capture_id
        self.capture_directory = None 
        self.capture_metadata = None

        try:
            #Do not love storing password in plain text in code....
            self.conn = psycopg2.connect(dbname="tone",
                                    user="postgres",
                                    port="5434",
                                    password="dirty vent unroof")

        except (Exception, psycopg2.Error) as error:
            print("Error while fetch data from Postrgesql", error)
            raise NameError("Error while fetch data from Postrgesql", error)

        if self.capture_id is not None:
            captureQuery = 'SELECT capture_id, session_id, capture_metadata FROM captures WHERE (user_id=%s AND capture_id=%s)'
            data = (self.user_id, self.capture_id)
        else:
            captureQuery = 'SELECT capture_id, session_id, capture_metadata FROM captures WHERE (user_id=%s AND capture_date=(SELECT MAX(capture_date) FROM captures WHERE user_id=%s))'
            data = (self.user_id, self.user_id)


        with self.conn.cursor() as cursor:
            cursor.execute(captureQuery, data)
            capture = cursor.fetchone()

        self.capture_id = capture[0]
        self.session_id = capture[1]
        self.capture_metadata = capture[2]
        self.capture_directory = os.path.join(IMAGES_DIR, str(user_id), str(self.session_id), str(self.capture_id))

        if not os.path.isdir(self.capture_directory):
            raise NameError("Capture Directory Does Not Exist :: {}".format(self.capture_directory))

        #self.images = self.loadImages()

    def saveCaptureResults(self, calibrated_skin_color, matched_skin_color_id):
        upsertCaptureResult = 'INSERT INTO capture_results (capture_id, user_id, backend_version, calibrated_skin_color, matched_skin_color_id) VALUES (%s, %s, %s, %s, %s) ON CONFLICT (capture_id) DO UPDATE SET (processed_date, backend_version, calibrated_skin_color, matched_skin_color_id)=ROW(NOW()::TIMESTAMP, EXCLUDED.backend_version, EXCLUDED.calibrated_skin_color, EXCLUDED.matched_skin_color_id)'
        data = (self.capture_id, int(self.user_id), self.version, calibrated_skin_color, matched_skin_color_id)
        print('Capture Results Data :: {}'.format(data))

        with self.conn.cursor() as cursor:
            cursor.execute(upsertCaptureResult, data)
            self.conn.commit()

    def referencePathBuilder(self, file='', extension=''):
        return os.path.join(self.capture_directory, 'reference', file + extension)

    def stepPathBuilder(self, step='', ext='.csv', meta=''):
        extension = ext if step != '' else ''
        return os.path.join(self.capture_directory, 'steps', str(step) + meta + extension)

    def touchReference(self):
        path = self.referencePathBuilder()
        if not os.path.exists(path):
            os.makedirs(path)
            os.chmod(path, 0o777)

    def deleteReference(self):
        path = self.referencePathBuilder()

        if os.path.isdir(path):
            for fileName in os.listdir(path):
                fileToRemove = os.path.join(path, fileName)
                print('Removing :: ' + str(fileToRemove))
                os.remove(fileToRemove)

    def touchSteps(self):
        path = self.stepPathBuilder()
        if not os.path.exists(path):
            os.makedirs(path)
            os.chmod(path, 0o777)

    def imageName(self):
        return '{}-{}-{}'.format(self.user_id, self.session_id, self.capture_id)

    def loadImages(self): 
        leftEyeFileTemplate = "{}_leftEye.PNG"
        rightEyeFileTemplate = "{}_rightEye.PNG"
        faceFileTemplate = "{}.PNG"

        imageSets = []
        for capture_number in range(1, 9):
            faceFile = faceFileTemplate.format(capture_number)
            leftEyeFile = leftEyeFileTemplate.format(capture_number)
            rightEyeFile = rightEyeFileTemplate.format(capture_number)

            faceFilePath = os.path.join(self.capture_directory, faceFile)
            leftEyeFilePath = os.path.join(self.capture_directory, leftEyeFile)
            rightEyeFilePath = os.path.join(self.capture_directory, rightEyeFile)

            isFacePathValid = os.path.isfile(faceFilePath)
            isLeftEyePathValid = os.path.isfile(leftEyeFilePath)
            isRightEyePathValid = os.path.isfile(rightEyeFilePath)

            if not (isFacePathValid and isLeftEyePathValid and isRightEyePathValid):
                raise NameError('Face, Left Eye, or Right Eye Path is not valid :: {}'.format(faceFilePath))

            face = cv2.imread(faceFilePath)
            leftEye = cv2.imread(leftEyeFilePath)
            rightEye = cv2.imread(rightEyeFilePath)

            if (face is None) or (leftEye is None) or (rightEye is None):
                raise NameError('Face, Left Eye or Right Eye image could not be read :: {}'.format(faceFilePath))

            imageSets.append([face, leftEye, rightEye])

        return imageSets

    def saveImageStep(self, image, step, meta=''):
        self.touchSteps()
        path = self.stepPathBuilder(step, '.PNG', meta)
        #image = np.clip(image_float * 255, 0, 255).astype('uint8')
        cv2.imwrite(path, image.astype('uint8'))
        os.chmod(path, 0o777)

    def saveMaskStep(self, mask, step, meta=''):
        self.touchSteps()
        path = self.stepPathBuilder(step, '.PNG', meta)
        image = np.clip(mask * 255, 0, 255).astype('uint8')
        cv2.imwrite(path, image.astype('uint8'))
        os.chmod(path, 0o777)

    def saveShapeStep(self, shape, step):
        self.touchSteps()
        path = self.stepPathBuilder(step)
        with open(path, 'w', newline='') as f:
            shapeWriter = csv.writer(f, delimiter=' ', quotechar='|')
            shapeWriter.writerows(shape)

        os.chmod(path, 0o777)

    def savePointsStep(self, points, step, meta=''):
        self.touchSteps()
        path = self.stepPathBuilder(step, '.csv', meta)
        with open(path, 'w', newline='') as f:
            pointWriter = csv.writer(f, delimiter=' ', quotechar='|')
            pointWriter.writerows(points)

        os.chmod(path, 0o777)

    def saveReferenceImageSBGR(self, image, reference):
        self.touchReference()
        #image = bgr_float * 255
        #image = bgr_float
        path = self.referencePathBuilder(reference, '.PNG')
        cv2.imwrite(path, image)
        os.chmod(path, 0o777)

    def saveReferenceImageLinearBGR(self, bgr, reference):
        self.touchReference()
        image = bgr
        #[image, error] = colorTools.convert_linearBGR_float_to_sBGR(bgr_float)
        #image = bgr_float * 255
        #image = bgr_float
        path = self.referencePathBuilder(reference, '.PNG')
        cv2.imwrite(path, image)
        os.chmod(path, 0o777)

    def saveReferenceImageBGR(self, bgr, reference):
        self.touchReference()
        #bgr_float_clipped = np.clip(bgr_float, 0.0, 1.0)
        #image = colorTools.convert_linearBGR_float_to_sBGR(bgr_float_clipped)
        #image = bgr * 255
        image = bgr
        path = self.referencePathBuilder(reference, '.PNG')
        cv2.imwrite(path, image)
        os.chmod(path, 0o777)

    def logMeasurement(self, measurementName, measurementValue):
        path = self.referencePathBuilder('measurementLog', '.txt')
        with open(path, 'a+') as f:
            f.write(measurementName + ", " + measurementValue + "\n")

        os.chmod(path, 0o777)

    def resetLogFile(self):
        self.touchReference()
        path = self.referencePathBuilder('measurementLog', '.txt')
        open(path, 'w').close()
        os.chmod(path, 0o777)

    def getMetadata(self):
        return self.capture_metadata

    def getAsShotWhiteBalance(self):
        whiteBalanceDict = self.capture_metadata[0]['whiteBalance']
        return [whiteBalanceDict['x'], whiteBalanceDict['y']]

    #def getMetadata(self):
    #    path = os.path.join(self.root, 'images/', self.username, self.fileName, self.fileName + '-metadata.txt')

    #    with open(path) as f:
    #        data = json.load(f)

    #    defaultImageTransforms = {}
    #    defaultImageTransforms["isGammaSBGR"] = True
    #    defaultImageTransforms["isRotated"] = True

    #    for capture in data:
    #        if not "imageTransforms" in capture:
    #            capture["imageTransforms"] = defaultImageTransforms

    #    return data

    def savePlot(self, name, plot):
        path = self.referencePathBuilder(name, '.jpg')
        plot.savefig(path, dpi=500)
        plot.close()

    def saveImageStat(self, statName, statValue):
        path = self.referencePathBuilder('imageStats', '.csv')

        with open(path, 'a', newline='') as f:
            benchWriter = csv.writer(f, delimiter=' ', quotechar='|')
            benchWriter.writerows([[statName, *statValue]])

        os.chmod(path, 0o777)

