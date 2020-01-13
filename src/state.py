"""
Handle state management, including DB access and saving/reading files
   Abstract away saving state locally vs saving state in AWS
"""
import os
import io
import sys
import cv2
import numpy as np
import psycopg2
import boto3
from logger import getLogger

boto3_mock_module_path = '../../boto_local_cache/'
if os.path.isdir(boto3_mock_module_path): #Directory should only exist in Dev
    sys.path.append(os.path.abspath(boto3_mock_module_path))
    import boto_local_cache as boto3

LOGGER = getLogger(__name__, 'app')
TONE_USER_CAPTURES_BUCKET = 'tone-user-captures'

class State:
    """State provides convenience methods for accessing DB, File System, and stores parsed information that maybe be used in processing. Should abstract away running in production vs Dev"""

    def __init__(self, user_id, capture_id=None, isProduction=False):
        self.s3 = boto3.client('s3')
        self.version = '0.0.1' #GET EB VERSION SOMEHOW?
        LOGGER.info("NOTE: THIS IS STATIC - SERVER VERSION :: %s", self.version)

        self.user_id = user_id
        self.capture_id = capture_id
        self.capture_key_root = None
        self.capture_metadata = None

        try:

            LOGGER.info("Opening connection to DB")

            self.conn = psycopg2.connect(dbname="ebdb",
                                         host="",
                                         user="toneDatabase",
                                         port="5432",
                                         password="")

            LOGGER.info("Opened connection to DB")

        except Exception as error:
            LOGGER.error("Error establishing connection to Postrgesql\n%s", error)
            raise

        if self.capture_id is not None:
            captureQuery = ('SELECT capture_id, session_id, capture_metadata '
                            'FROM captures '
                            'WHERE (user_id=%s AND capture_id=%s)')

            data = (self.user_id, self.capture_id)
        else:
            captureQuery = ('SELECT capture_id, session_id, capture_metadata '
                            'FROM captures '
                            'WHERE (user_id=%s AND capture_date=(SELECT MAX(capture_date) FROM captures WHERE user_id=%s))')

            data = (self.user_id, self.user_id)


        with self.conn.cursor() as cursor:
            cursor.execute(captureQuery, data)
            capture = cursor.fetchone()

        self.capture_id = capture[0]
        self.session_id = capture[1]
        self.capture_metadata = capture[2]
        self.capture_key_root = '{}/{}/{}'.format(user_id, self.session_id, self.capture_id)

    def __referencePathBuilder(self, file='', extension=''):
        """Returns path to reference file for capture set"""
        return os.path.join(self.capture_key_root, 'reference', file + extension)

    def __fetchImage(self, key):
        """Returns image specified by key. Fetches from S3 or S3 local cache and parses into numpy array"""
        try:
            LOGGER.info('FETCHING :: %s - %s', TONE_USER_CAPTURES_BUCKET, key)
            resp = self.s3.get_object(Bucket=TONE_USER_CAPTURES_BUCKET, Key=key)
            raw = resp['Body'].read()
            raw_buffer = io.BytesIO(raw).getbuffer()
            np_array = np.asarray(raw_buffer)

            #OPEN CV HAS CAUSED ISSUES IN THE PAST. https://stackoverflow.com/questions/51124056/opencvimread-operation-stuck-in-elastic-beanstalk
            decoded = cv2.imdecode(np_array, 1)
        except Exception as e:
            LOGGER.error('Error in Fetch! :: %s', e)
            raise Exception

        return decoded


    def saveCaptureResults(self, calibrated_skin_color, matched_skin_color_id):
        """Save capture result object to the DB"""

        upsertCaptureResult = ('INSERT INTO capture_results (capture_id, user_id, backend_version, calibrated_skin_color, matched_skin_color_id, successful) '
                               'VALUES (%s, %s, %s, %s, %s, %s) '
                               'ON CONFLICT (capture_id) '
                               'DO UPDATE SET (processed_date, backend_version, calibrated_skin_color, matched_skin_color_id, successful)= '
                               'ROW(NOW()::TIMESTAMP, '
                               'EXCLUDED.backend_version, '
                               'EXCLUDED.calibrated_skin_color, '
                               'EXCLUDED.matched_skin_color_id, '
                               'EXCLUDED.successful)')

        data = (self.capture_id, int(self.user_id), self.version, calibrated_skin_color, matched_skin_color_id, True)
        LOGGER.info('Capture Results Data :: %s', data)

        with self.conn.cursor() as cursor:
            cursor.execute(upsertCaptureResult, data)
            self.conn.commit()

    def errorProccessing(self):
        """Call after catching an exception. Logs failure to database"""

        upsertCaptureResult = ('INSERT INTO capture_results (capture_id, user_id, backend_version, calibrated_skin_color, matched_skin_color_id, successful) '
                               'VALUES (%s, %s, %s, %s, %s, %s) '
                               'ON CONFLICT (capture_id) '
                               'DO UPDATE SET (processed_date, backend_version, calibrated_skin_color, matched_skin_color_id, successful)= '
                               'ROW(NOW()::TIMESTAMP, '
                               'EXCLUDED.backend_version, '
                               'EXCLUDED.calibrated_skin_color, '
                               'EXCLUDED.matched_skin_color_id, '
                               'EXCLUDED.successful)')

        data = (self.capture_id, int(self.user_id), self.version, None, None, False)

        with self.conn.cursor() as cursor:
            cursor.execute(upsertCaptureResult, data)
            self.conn.commit()

    def imageName(self):
        """Returns image name"""
        return '{}-{}-{}'.format(self.user_id, self.session_id, self.capture_id)

    def saveExposurePointImage(self, key, images):
        """Save a copy of the brightest capture with the point the camera exposured to drawn on the face"""
        exposurePoints = [meta['exposurePoint'] for meta in self.capture_metadata]

        filtered = [np.copy(img[0]) for img in images]

        drawn = []
        for img, point in zip(filtered, exposurePoints):
            width, height, p = img.shape
            print('{} ,{}, {}'.format(width, height, p))

            exposureX = int(width * point[1])
            exposureY = int(height * point[0])

            print('{}, {}'.format(exposureX, exposureY))


            cv2.circle(img, (exposureY, exposureX), 5, (255, 0, 0))
            drawn.append(img)

        stacked = np.hstack(drawn)
        key = self.__referencePathBuilder('exposurePoints', '.png')
        self.storeImage(key, stacked)
        #cv2.imshow('stacked', stacked)
        #cv2.waitKey(0)

    def storeImage(self, key, img, extension='.png'):
        """Saves image in S3 or local S3 cache at provided key"""
        img_encoded = io.BytesIO(cv2.imencode(extension, img)[1]).getvalue()
        self.s3.put_object(Bucket=TONE_USER_CAPTURES_BUCKET, Key=key, Body=img_encoded)

    def loadImages(self):
        """Returns Array of arrays containing the [face, leftEye, rightEye] images"""
        leftEyeFileTemplate = "{}_leftEye.png"
        rightEyeFileTemplate = "{}_rightEye.png"
        faceFileTemplate = "{}.png"

        imageSets = []
        for capture_number in range(1, 9):
        #for capture_number in range(1, 16):
            faceFile = faceFileTemplate.format(capture_number)
            leftEyeFile = leftEyeFileTemplate.format(capture_number)
            rightEyeFile = rightEyeFileTemplate.format(capture_number)

            faceFileKey = os.path.join(self.capture_key_root, faceFile)
            leftEyeFileKey = os.path.join(self.capture_key_root, leftEyeFile)
            rightEyeFileKey = os.path.join(self.capture_key_root, rightEyeFile)

            face = self.__fetchImage(faceFileKey)
            leftEye = self.__fetchImage(leftEyeFileKey)
            rightEye = self.__fetchImage(rightEyeFileKey)

            if (face is None) or (leftEye is None) or (rightEye is None):
                raise ValueError('Face, Left Eye or Right Eye image could not be read :: {}'.format(faceFileKey))

            imageSets.append([face, leftEye, rightEye])

        return imageSets

    def saveReferenceImageBGR(self, bgr, reference):
        """Save image reference folder"""
        key = self.__referencePathBuilder(reference, '.png')
        self.storeImage(key, bgr)

    def getValidatedMetadata(self):
        """Returns metadata if it is valid (exposure variables match between captures), raises error if not"""

        expectedISO = self.capture_metadata[0]["iso"]
        expectedExposure = self.capture_metadata[0]["exposureTime"]
        expectedWB = self.capture_metadata[0]["whiteBalance"]

        if not 'faceImageTransforms' in self.capture_metadata[0]:
            raise ValueError('No Face Image Transforms in Metadata')

        for captureMetadata in self.capture_metadata:
            iso = captureMetadata["iso"]
            exposure = captureMetadata["exposureTime"]
            wb = captureMetadata["whiteBalance"]

            if (iso != expectedISO) or (exposure != expectedExposure) or (wb['x'] != expectedWB['x']) or (wb['y'] != expectedWB['y']):
                raise ValueError('Metadata Does Not Match in all Captures')

        return self.capture_metadata

    def savePlot(self, name, plot):
        """Save plot to reference folder"""
        extension = '.jpg'
        key = self.__referencePathBuilder(name, extension) #new matplotlib requires png
        path = '/tmp/{}{}'.format(self.imageName(), extension)
        plot.savefig(path, optimize=True)
        plot.close()

        LOGGER.info('Saved chart to %s', path)

        plotImg = cv2.imread(path)
        self.storeImage(key, plotImg, extension)
        os.remove(path)
