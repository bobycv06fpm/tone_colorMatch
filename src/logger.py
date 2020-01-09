import logging
import logging.handlers
import os

def getLogger(name, location):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    LOG_FILE = '/opt/python/log/tone-colormatch-{}.log'.format(location) #For Prod
    LOG_DIR = '/opt/python/log/'
    isProd = os.path.isdir(LOG_DIR) #Directory only exists in prodution

    if not isProd:
        LOG_FILE = 'tone-colormatch-{}.log'.format(location)

    handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=1048576, backupCount=5)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
