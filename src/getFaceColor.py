import json
import argparse
import psycopg2
import time
import runSteps

import logger
logger = logger.getLogger(__name__, 'app')

#Do not love storing password in plain text in code....
conn = psycopg2.connect(dbname="ebdb",
                        host="aa7a9qu9bzxsgc.cz5sm4eeyiaf.us-west-2.rds.amazonaws.com",
                        user="toneDatabase",
                        port="5432",
                        password="mr9pkatYVlX5pD9HjGRDJEzJ0NFpoC")

def getMessageForCaptureId(capture_id, session_id, user_id):
    colorMatchMessage = {}
    colorMatchMessage['user_id'] = user_id
    colorMatchMessage['session_id'] = session_id
    colorMatchMessage['capture_id'] = capture_id
    return colorMatchMessage

def getMessagesForSessionId(session_id, user_id):
    allCapturesForSessionQuery = 'SELECT capture_id, session_id, user_id FROM captures WHERE (user_id=%s AND session_id=%s)'
    data = (user_id, session_id)

    with conn.cursor() as cursor:
            cursor.execute(allCapturesForSessionQuery, data)
            captures = capture = cursor.fetchall()

    return [getMessageForCaptureId(*capture) for capture in captures]

def getMessagesForUserId(user_id):
    print("Getting Messages for user_id {}".format(user_id))
    allCapturesForSessionQuery = 'SELECT capture_id, session_id, user_id FROM captures WHERE (user_id=%s)'
    data = (user_id)

    with conn.cursor() as cursor:
            cursor.execute(allCapturesForSessionQuery, data)
            captures = capture = cursor.fetchall()

    #print("Raw Captures :: {}".format(captures))
    return [getMessageForCaptureId(*capture) for capture in captures]

ap = argparse.ArgumentParser()

ap.add_argument("-u", "--user_id", required=True, help="The user id you want to process")
ap.add_argument("-s", "--session_id", required=False, help="The session id you want to process")
ap.add_argument("-c", "--capture_id", required=False, help="The capture id you want to process")

# Capture # -> Process this capture
# Session # -> Process all captures for this session
# User # -> Process all captures for this user

args = vars(ap.parse_args())

messages = []

if (args["capture_id"] is not None) and (args["session_id"] is not None) and (args["user_id"] is not None):
    user_id = args["user_id"]
    session_id = args["session_id"]
    capture_id = args["capture_id"]
    message = getMessageForCaptureId(capture_id, session_id, user_id)
    messages.append(message)
elif (args["session_id"] is not None) and (args["user_id"] is not None):
    user_id = args["user_id"]
    session_id = args["session_id"]
    messages += getMessagesForSessionId(session_id, user_id)
elif (args["user_id"] is not None):
    user_id = args["user_id"]
    messages += getMessagesForUserId(user_id)
else:
    print("Incorrect Argumets")

print('Messages :: {}'.format(messages))

for message in messages:
    #addMessageToSQS(message)

    try:
        response = runSteps.run(message['user_id'], message['capture_id'])
        print('Response :: {}'.format(response))
    except Exception as err:
        logger.error(err)
        pass

    #time.sleep(10)

print("Done")
