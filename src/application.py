"""
Lightweight server that runs in the Elastic Beanstalk worker
    The worker environment (where this is running) recieves an SQS message and makes a POST request with the message contents
    This server calls the image processing pipeline with the arguments
"""
from wsgiref.simple_server import make_server
import json
from runSteps import run

from logger import getLogger
LOGGER = getLogger(__name__, 'server')

isProduction = __name__ != '__main__'

def application(environ, start_response):
    """Worker environment looks for this function (named application) to set up the server"""
    path = environ['PATH_INFO']
    method = environ['REQUEST_METHOD']
    if method == 'POST':
        try:
            if path == '/':
                request_body_size = int(environ['CONTENT_LENGTH'])
                request_body = environ['wsgi.input'].read(request_body_size).decode()
                requestForm = json.loads(request_body)

                user_id_key = 'user_id'
                session_id_key = 'session_id'
                capture_id_key = 'capture_id'

                if user_id_key not in requestForm:
                    LOGGER.warning('No user_id')
                user_id = requestForm[user_id_key]

                if session_id_key not in requestForm:
                    LOGGER.warning('No session_id')
                session_id = requestForm[session_id_key]

                if capture_id_key not in requestForm:
                    LOGGER.warning('No capture_id')
                capture_id = requestForm[capture_id_key]

                LOGGER.info('PROCESSING :: user_id :: %s, session_id :: %s, capture_id :: %s', user_id, session_id, capture_id)
                isSuccessful = run(user_id, capture_id, isProduction)['successful']

                if isSuccessful:
                    LOGGER.info('IS SUCCESSFUL :: user_id :: %s, session_id :: %s, capture_id :: %s', user_id, session_id, capture_id)
                else:
                    LOGGER.warning('NOT SUCCESSFUL :: user_id :: %s, session_id :: %s, capture_id :: %s', user_id, session_id, capture_id)

                response = str(isSuccessful)

            elif path == '/scheduled':
                LOGGER.info("Received task %s scheduled at %s", environ['HTTP_X_AWS_SQSD_TASKNAME'], environ['HTTP_X_AWS_SQSD_SCHEDULED_AT'])
                response = str(True)
        except Exception as e:
            LOGGER.warning('Error retrieving request body for async work. %s', e)
            response = str(False)
    else:
        response = 'Probably not what you are looking for...'

    status = '200 OK'
    headers = [('Content-type', 'text/html')]

    start_response(status, headers)
    return [response.encode('utf-8')]

if not isProduction:
    httpd = make_server('', 8000, application)
    print("Serving on port 8000...")
    httpd.serve_forever()
