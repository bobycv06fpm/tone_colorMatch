import cv2
import csv
import os
import io
import numpy as np
import colorTools
import json
import psycopg2
import getVersion
import boto3
from logger import getLogger

#COMMENT OUT FOR PROD
import sys
sys.path.append(os.path.abspath('../../boto_local_cache/'))
import boto_local_cache as boto3 

logger = getLogger(__name__)
TONE_USER_CAPTURES_BUCKET = 'tone-user-captures'

class State:

    def __init__(self, user_id, capture_id=None, isProduction=False):
        #NOTE: THIS IS STATIC
        self.version = '0.0.1' #GET EB VERSION SOMEHOW?? #getVersion.getVersion(COLOR_MATCH_DIR)
        logger.info("NOTE: THIS IS STATIC - SERVER VERSION :: " + str(self.version))

        self.user_id = user_id
        self.capture_id = capture_id
        self.capture_key_root = None 
        self.capture_metadata = None

        try:

            logger.info("Opening connection to DB")

            #Do not love storing password in plain text in code....
            self.conn = psycopg2.connect(dbname="ebdb",
                                    host="aa7a9qu9bzxsgc.cz5sm4eeyiaf.us-west-2.rds.amazonaws.com",
                                    user="toneDatabase",
                                    port="5432",
                                    password="mr9pkatYVlX5pD9HjGRDJEzJ0NFpoC")
            #self.conn = psycopg2.connect(dbname="tone",
            #                        user="postgres",
            #                        port="5434",
            #                        password="dirty vent unroof")

            logger.info("Opened connection to DB")

        except Exception as error:
            logger.error("Error while fetch data from Postrgesql\n{}".format(error))
            raise 

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
        self.capture_key_root = '{}/{}/{}'.format(user_id, self.session_id, self.capture_id)
        self.s3 = boto3.client('s3')

    def saveCaptureResults(self, calibrated_skin_color, matched_skin_color_id):
        upsertCaptureResult = 'INSERT INTO capture_results (capture_id, user_id, backend_version, calibrated_skin_color, matched_skin_color_id, successful) VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (capture_id) DO UPDATE SET (processed_date, backend_version, calibrated_skin_color, matched_skin_color_id, successful)=ROW(NOW()::TIMESTAMP, EXCLUDED.backend_version, EXCLUDED.calibrated_skin_color, EXCLUDED.matched_skin_color_id, EXCLUDED.successful)'
        data = (self.capture_id, int(self.user_id), self.version, calibrated_skin_color, matched_skin_color_id, True)
        logger.info('Capture Results Data :: {}'.format(data))

        with self.conn.cursor() as cursor:
            cursor.execute(upsertCaptureResult, data)
            self.conn.commit()

    def errorProccessing(self):
        upsertCaptureResult = 'INSERT INTO capture_results (capture_id, user_id, backend_version, calibrated_skin_color, matched_skin_color_id, successful) VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (capture_id) DO UPDATE SET (processed_date, backend_version, calibrated_skin_color, matched_skin_color_id, successful)=ROW(NOW()::TIMESTAMP, EXCLUDED.backend_version, EXCLUDED.calibrated_skin_color, EXCLUDED.matched_skin_color_id, EXCLUDED.successful)'
        data = (self.capture_id, int(self.user_id), self.version, None, None, False)

        with self.conn.cursor() as cursor:
            cursor.execute(upsertCaptureResult, data)
            self.conn.commit()

    def referencePathBuilder(self, file='', extension=''):
        return os.path.join(self.capture_key_root, 'reference', file + extension)

    def imageName(self):
        return '{}-{}-{}'.format(self.user_id, self.session_id, self.capture_id)

    def fetchImage(self, key):
        try:
            logger.info('FETCHING :: {} - {}'.format(TONE_USER_CAPTURES_BUCKET, key))
            resp = self.s3.get_object(Bucket=TONE_USER_CAPTURES_BUCKET, Key=key)
            raw = resp['Body'].read()
            raw_buffer = io.BytesIO(raw).getbuffer()
            np_array =  np.asarray(raw_buffer)

            #NOTE: OPEN CV HAS CAUSED ISSUES IN THE PAST. https://stackoverflow.com/questions/51124056/opencvimread-operation-stuck-in-elastic-beanstalk
            decoded = cv2.imdecode(np_array, 1) 
        except Exception as e:
            logger.error('Error in Fetch! :: {}'.format(e))
            raise Exception

        return decoded

    def saveExposurePointImage(self, key, images):
        exposurePoints = [meta['exposurePoint'] for meta in self.capture_metadata]
        print('EXPOSURE POINTS :: {}'.format(exposurePoints))

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
        key = self.referencePathBuilder('exposurePoints', '.png')
        self.storeImage(key, stacked)
        #cv2.imshow('stacked', stacked)
        #cv2.waitKey(0)

    def storeImage(self, key, img, extension='.png'):
        img_encoded = io.BytesIO(cv2.imencode(extension, img)[1]).getvalue()
        self.s3.put_object(Bucket=TONE_USER_CAPTURES_BUCKET, Key=key, Body=img_encoded)

    def loadImages(self): 
        leftEyeFileTemplate = "{}_leftEye.png"
        rightEyeFileTemplate = "{}_rightEye.png"
        faceFileTemplate = "{}.png"

        imageSets = []
        for capture_number in range(1, 9):
        #for capture_number in range(1, 9, 2):
        #for capture_number in range(1, 16):
            faceFile = faceFileTemplate.format(capture_number)
            leftEyeFile = leftEyeFileTemplate.format(capture_number)
            rightEyeFile = rightEyeFileTemplate.format(capture_number)

            faceFileKey = os.path.join(self.capture_key_root, faceFile)
            leftEyeFileKey = os.path.join(self.capture_key_root, leftEyeFile)
            rightEyeFileKey = os.path.join(self.capture_key_root, rightEyeFile)

            face = self.fetchImage(faceFileKey)
            leftEye = self.fetchImage(leftEyeFileKey)
            rightEye = self.fetchImage(rightEyeFileKey)

            if (face is None) or (leftEye is None) or (rightEye is None):
                raise ValueError('Face, Left Eye or Right Eye image could not be read :: {}'.format(faceFileKey))

            imageSets.append([face, leftEye, rightEye])

        return imageSets

    def saveReferenceImageSBGR(self, image, reference):
        key = self.referencePathBuilder(reference, '.png')
        self.storeImage(key, image)

    def saveReferenceImageLinearBGR(self, bgr, reference):
        key = self.referencePathBuilder(reference, '.png')
        self.storeImage(key, bgr)

    def saveReferenceImageBGR(self, bgr, reference):
        key = self.referencePathBuilder(reference, '.png')
        self.storeImage(key, bgr)

    def getMetadata(self):
        return self.capture_metadata

    def getAsShotWhiteBalance(self):
        whiteBalanceDict = self.capture_metadata[0]['whiteBalance']
        return [whiteBalanceDict['x'], whiteBalanceDict['y']]

    def savePlot(self, name, plot):
        extension='.jpg'
        key = self.referencePathBuilder(name, extension) #new matplotlib requires png
        path = '/tmp/{}{}'.format(self.imageName(), extension)
        plot.savefig(path, optimize=True)
        plot.close()

        logger.info('Saved chart to {}'.format(path))

        plotImg = cv2.imread(path)
        self.storeImage(key, plotImg, extension)
        os.remove(path)

