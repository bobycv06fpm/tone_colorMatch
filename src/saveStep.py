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

logger = getLogger(__name__)

#IMAGES_DIR = '/home/dmacewen/Projects/tone/images/'
#COLOR_MATCH_DIR = '/home/dmacewen/Projects/tone/tone_colorMatch/'
TONE_USER_CAPTURES_BUCKET = 'tone-user-captures'

#Turn into an abstraction around state? Encapsulate DB and File System?
# Load Images
# Load DB info
# Load and Save reference data
class State:

    def __init__(self, user_id, capture_id=None, isProduction=False):
        # Create logger()
        #self.version = getVersion.getVersion(COLOR_MATCH_DIR)
        self.version = '0.0.1' #GET EB VERSION SOMEHOW?? #getVersion.getVersion(COLOR_MATCH_DIR)
        logger.info("NOTE: THIS IS STATIC - SERVER VERSION :: " + str(self.version))

        self.user_id = user_id
        self.capture_id = capture_id
        self.capture_key_root = None 
        self.capture_metadata = None

        try:
            #NOTE: THIS IS STATIC - priniTEMP
            #Do not love storing password in plain text in code....

            logger.info("Opening connection to DB")

            if isProduction:
                self.conn = psycopg2.connect(dbname="ebdb",
                                        host="aa7a9qu9bzxsgc.cz5sm4eeyiaf.us-west-2.rds.amazonaws.com",
                                        user="toneDatabase",
                                        port="5432",
                                        password="mr9pkatYVlX5pD9HjGRDJEzJ0NFpoC")
            else:
                self.conn = psycopg2.connect(dbname="tone",
                                        user="postgres",
                                        port="5434",
                                        password="dirty vent unroof")

            logger.info("Opened connection to DB")

#            if 'RDS_HOSTNAME' in os.environ:
#                self.conn = psycopg2.connect(dbname=os.environ['RDS_DB_NAME'],
#                                        user=os.environ['RDS_USERNAME'],
#                                        password=os.environ['RDS_PASSWORD'],
#                                        host=os.environ['RDS_HOSTNAME'],
#                                        port=os.environ['RDS_PORT'])
#            else:
#                #print('CANNOT CONNECT TO DB!')
#                logger.warning('Cannot Connect to DB - RDS_HOSTNAME not present')


        except (Exception, psycopg2.Error) as error:
            logger.warning("Error while fetch data from Postrgesql :: {}".format(error))
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
        #self.capture_key_root = os.path.join(IMAGES_DIR, str(user_id), str(self.session_id), str(self.capture_id))
        self.capture_key_root = '{}/{}/{}'.format(user_id, self.session_id, self.capture_id)

        self.s3 = boto3.client('s3')

        #if not os.path.isdir(self.capture_key_root):
        #    raise NameError("Capture Directory Does Not Exist :: {}".format(self.capture_key_root))

        #self.images = self.loadImages()


    def saveCaptureResults(self, calibrated_skin_color, matched_skin_color_id):
        upsertCaptureResult = 'INSERT INTO capture_results (capture_id, user_id, backend_version, calibrated_skin_color, matched_skin_color_id) VALUES (%s, %s, %s, %s, %s) ON CONFLICT (capture_id) DO UPDATE SET (processed_date, backend_version, calibrated_skin_color, matched_skin_color_id)=ROW(NOW()::TIMESTAMP, EXCLUDED.backend_version, EXCLUDED.calibrated_skin_color, EXCLUDED.matched_skin_color_id)'
        data = (self.capture_id, int(self.user_id), self.version, calibrated_skin_color, matched_skin_color_id)
        #print('Capture Results Data :: {}'.format(data))
        logger.info('Capture Results Data :: {}'.format(data))

        with self.conn.cursor() as cursor:
            cursor.execute(upsertCaptureResult, data)
            self.conn.commit()

    def referencePathBuilder(self, file='', extension=''):
        #s3_client.get_object(Bucket=TONE_USER_CAPTURES_BUCKET, Key=metadataPath))
        #'{}/{}/{}/reference'.format(self.user_id, self.session_id, self.capture_id)
        return os.path.join(self.capture_key_root, 'reference', file + extension)

    #CAN THIS BE REMOVED? ALL "STEP" THINGS?
   # def stepPathBuilder(self, step='', ext='.csv', meta=''):
   #     #'{}/{}/{}/step'.format(self.user_id, self.session_id, self.capture_id)
   #     extension = ext if step != '' else ''
   #     return os.path.join(self.capture_key_root, 'steps', str(step) + meta + extension)

    #def touchReference(self):
    #    path = self.referencePathBuilder()
    #    if not os.path.exists(path):
    #        os.makedirs(path)
    #        os.chmod(path, 0o777)

    #def deleteReference(self):
    #    path = self.referencePathBuilder()

    #    if os.path.isdir(path):
    #        for fileName in os.listdir(path):
    #            fileToRemove = os.path.join(path, fileName)
    #            print('Removing :: ' + str(fileToRemove))
    #            os.remove(fileToRemove)

   # def touchSteps(self):
   #     path = self.stepPathBuilder()
   #     if not os.path.exists(path):
   #         os.makedirs(path)
   #         os.chmod(path, 0o777)

    def imageName(self):
        return '{}-{}-{}'.format(self.user_id, self.session_id, self.capture_id)

    def fetchImage(self, key):
        #print('FETCHING :: {} - {}'.format(TONE_USER_CAPTURES_BUCKET, key))
        try:
            print("FETCHING...DOUBLE RUNNING?? :: {}".format(key))
            logger.info('FETCHING :: {} - {}'.format(TONE_USER_CAPTURES_BUCKET, key))
            resp = self.s3.get_object(Bucket=TONE_USER_CAPTURES_BUCKET, Key=key)
            logger.info('RESPONSE :: {}'.format(resp))
            raw = resp['Body'].read()
            logger.info('Read Response Body | length :: {}'.format(len(raw)))
            raw_buffer = io.BytesIO(raw).getbuffer()
            logger.info('Got Raw Buffer')
            np_array =  np.asarray(raw_buffer)
            logger.info('Got NP Array')
            decoded = cv2.imdecode(np_array, 1)
            logger.info('Decoded Raw Buffer to Image')
        except Exception as e:
            print('ERROR IN FETCH!')
            logger.warn('Error in Fetch! :: {}'.format(e))

        return decoded

    def storeImage(self, key, img, extension='.png'):
        img_encoded = io.BytesIO(cv2.imencode(extension, img)[1]).getvalue()
        self.s3.put_object(Bucket=TONE_USER_CAPTURES_BUCKET, Key=key, Body=img_encoded)

    def loadImages(self): 
        leftEyeFileTemplate = "{}_leftEye.png"
        rightEyeFileTemplate = "{}_rightEye.png"
        faceFileTemplate = "{}.png"

        imageSets = []
        for capture_number in range(1, 9):
            faceFile = faceFileTemplate.format(capture_number)
            leftEyeFile = leftEyeFileTemplate.format(capture_number)
            rightEyeFile = rightEyeFileTemplate.format(capture_number)

            faceFileKey = os.path.join(self.capture_key_root, faceFile)
            leftEyeFileKey = os.path.join(self.capture_key_root, leftEyeFile)
            rightEyeFileKey = os.path.join(self.capture_key_root, rightEyeFile)

            #isFacePathValid = os.path.isfile(faceFilePath)
            #isLeftEyePathValid = os.path.isfile(leftEyeFilePath)
            #isRightEyePathValid = os.path.isfile(rightEyeFilePath)

            #if not (isFacePathValid and isLeftEyePathValid and isRightEyePathValid):
            #    raise NameError('Face, Left Eye, or Right Eye Path is not valid :: {}'.format(faceFilePath))

            #face = cv2.imread(faceFilePath)
            face = self.fetchImage(faceFileKey)

            #leftEye = cv2.imread(leftEyeFilePath)
            leftEye = self.fetchImage(leftEyeFileKey)

            #rightEye = cv2.imread(rightEyeFilePath)
            rightEye = self.fetchImage(rightEyeFileKey)

            if (face is None) or (leftEye is None) or (rightEye is None):
                raise NameError('Face, Left Eye or Right Eye image could not be read :: {}'.format(faceFileKey))

            imageSets.append([face, leftEye, rightEye])

        return imageSets

    #def saveImageStep(self, image, step, meta=''):
    #    self.touchSteps()
    #    path = self.stepPathBuilder(step, '.png', meta)
    #    #image = np.clip(image_float * 255, 0, 255).astype('uint8')
    #    cv2.imwrite(path, image.astype('uint8'))
    #    os.chmod(path, 0o777)

    #def saveMaskStep(self, mask, step, meta=''):
    #    self.touchSteps()
    #    path = self.stepPathBuilder(step, '.png', meta)
    #    image = np.clip(mask * 255, 0, 255).astype('uint8')
    #    cv2.imwrite(path, image.astype('uint8'))
    #    os.chmod(path, 0o777)

    #def saveShapeStep(self, shape, step):
    #    self.touchSteps()
    #    path = self.stepPathBuilder(step)
    #    with open(path, 'w', newline='') as f:
    #        shapeWriter = csv.writer(f, delimiter=' ', quotechar='|')
    #        shapeWriter.writerows(shape)

    #    os.chmod(path, 0o777)

    #def savePointsStep(self, points, step, meta=''):
    #    self.touchSteps()
    #    path = self.stepPathBuilder(step, '.csv', meta)
    #    with open(path, 'w', newline='') as f:
    #        pointWriter = csv.writer(f, delimiter=' ', quotechar='|')
    #        pointWriter.writerows(points)

    #    os.chmod(path, 0o777)

    def saveReferenceImageSBGR(self, image, reference):
        #self.touchReference()
        #image = bgr_float * 255
        #image = bgr_float
        key = self.referencePathBuilder(reference, '.png')
        self.storeImage(key, image)
        #cv2.imwrite(path, image)
        #os.chmod(path, 0o777)

    def saveReferenceImageLinearBGR(self, bgr, reference):
        #self.touchReference()
        #image = bgr
        #[image, error] = colorTools.convert_linearBGR_float_to_sBGR(bgr_float)
        #image = bgr_float * 255
        #image = bgr_float
        key = self.referencePathBuilder(reference, '.png')
        self.storeImage(key, bgr)
        #cv2.imwrite(path, image)
        #os.chmod(path, 0o777)

    def saveReferenceImageBGR(self, bgr, reference):
        #self.touchReference()
        #bgr_float_clipped = np.clip(bgr_float, 0.0, 1.0)
        #image = colorTools.convert_linearBGR_float_to_sBGR(bgr_float_clipped)
        #image = bgr * 255
        #image = bgr
        key = self.referencePathBuilder(reference, '.png')
        self.storeImage(key, bgr)
        #cv2.imwrite(path, image)
        #os.chmod(path, 0o777)

    #def logMeasurement(self, measurementName, measurementValue):
    #    path = self.referencePathBuilder('measurementLog', '.txt')
    #    with open(path, 'a+') as f:
    #        f.write(measurementName + ", " + measurementValue + "\n")

    #    os.chmod(path, 0o777)

    #def resetLogFile(self):
    #    self.touchReference()
    #    path = self.referencePathBuilder('measurementLog', '.txt')
    #    open(path, 'w').close()
    #    os.chmod(path, 0o777)

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
        #print('NOT IMPLEMENTED')
        #return
        #path = self.referencePathBuilder(name, '.jpg')
        extension='.jpg'
        key = self.referencePathBuilder(name, extension) #new matplotlib requires png
        path = '/tmp/{}{}'.format(self.imageName(), extension)
        plot.savefig(path, optimize=True)
        plot.close()

        logger.info('Saved chart to {}'.format(path))

        plotImg = cv2.imread(path)
        self.storeImage(key, plotImg, extension)
        os.remove(path)

    #def saveImageStat(self, statName, statValue):
    #    path = self.referencePathBuilder('imageStats', '.csv')

    #    with open(path, 'a', newline='') as f:
    #        benchWriter = csv.writer(f, delimiter=' ', quotechar='|')
    #        benchWriter.writerows([[statName, *statValue]])

    #    os.chmod(path, 0o777)

