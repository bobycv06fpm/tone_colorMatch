#from wsgiref.simple_server import make_server
from flask import Flask, escape, request
from runSteps import run2

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
#@app.route('/')
def colormatchRequest():
    if request.method == 'POST':
        user_id_key = 'user_id'
        session_id_key = 'session_id'
        capture_id_key = 'capture_id'

        requestForm = request.form

        if user_id_key not in request.form:
            return 'No user_id'
        user_id = requestForm[user_id_key]

        if session_id_key not in request.form:
            return 'No session_id'
        session_id = requestForm[session_id_key]

        if capture_id_key not in request.form:
            return 'No capture_id'
        capture_id = requestForm[capture_id_key]
        
        print('PROCESSING :: user_id :: {}, session_id :: {}, capture_id :: {}'.format(user_id, session_id, capture_id))
        isSuccessful = run2(user_id, capture_id)['successful']
        return str(isSuccessful)
        #return 'user_id :: {}, session_id :: {}, capture_id :: {}'.format(user_id, session_id, capture_id)
    else:
        return 'Probably Not What You Are Looking For....'


if __name__=="__main__":
    app.run()
