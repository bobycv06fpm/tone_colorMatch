#from flask import Flask, escape, request
import logging
import logging.handlers
from runSteps import run2
import json

from wsgiref.simple_server import make_server


# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

isProduction = False
# Handler 
if __name__ == '__main__':
    LOG_FILE = 'tone-colormatch-server.log'
else:
    LOG_FILE = '/opt/python/log/tone-colormatch-server.log'
    isProduction = True

handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=1048576, backupCount=5)
handler.setLevel(logging.INFO)

# Formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add Formatter to Handler
handler.setFormatter(formatter)

# add Handler to Logger
logger.addHandler(handler)

welcome = 'Probably not what you are looking for...'

def application(environ, start_response):
    path    = environ['PATH_INFO']
    method  = environ['REQUEST_METHOD']
    if method == 'POST':
        try:
            if path == '/':
                request_body_size = int(environ['CONTENT_LENGTH'])
                request_body = environ['wsgi.input'].read(request_body_size).decode()
                logger.info("Received message: %s" % request_body)
                requestForm = json.loads(request_body)

                user_id_key = 'user_id'
                session_id_key = 'session_id'
                capture_id_key = 'capture_id'

                if user_id_key not in requestForm:
                    logger.warning('No user_id')
                user_id = requestForm[user_id_key]

                if session_id_key not in requestForm:
                    logger.warning('No session_id')
                session_id = requestForm[session_id_key]

                if capture_id_key not in requestForm:
                    logger.warning('No capture_id')
                capture_id = requestForm[capture_id_key]
                
                logger.info('PROCESSING :: user_id :: {}, session_id :: {}, capture_id :: {}'.format(user_id, session_id, capture_id))
                isSuccessful = run2(user_id, capture_id, isProduction)['successful']
                if isSuccessful:
                    logger.info('IS SUCCESSFUL :: user_id :: {}, session_id :: {}, capture_id :: {}'.format(user_id, session_id, capture_id))
                else:
                    logger.warning('NOT SUCCESSFUL :: user_id :: {}, session_id :: {}, capture_id :: {}'.format(user_id, session_id, capture_id))

                response = str(isSuccessful)

            elif path == '/scheduled':
                logger.info("Received task %s scheduled at %s", environ['HTTP_X_AWS_SQSD_TASKNAME'], environ['HTTP_X_AWS_SQSD_SCHEDULED_AT'])
                response = str(True)
        except (TypeError, ValueError) as e:
            logger.warning('Error retrieving request body for async work. {}'.format(e))
            response = str(False)
    else:
        response = welcome

    status = '200 OK'
    headers = [('Content-type', 'text/html')]

    start_response(status, headers)
    return [response.encode('utf-8')]


if __name__ == '__main__':
    httpd = make_server('', 8000, application)
    print("Serving on port 8000...")
    httpd.serve_forever()

