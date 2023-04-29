from flask import Flask, request, session
# from flask_restful import Resource, Api
from flask_cors import CORS, cross_origin
from Model import Model


# create a Flask instance
app = Flask(__name__)
app.secret_key = 'super secret key'
app.config['CORS_HEADERS'] = 'Content-Type'

# Cross Origin Resource Sharing
# CORS(app)
# create an API object
# api = Api(app)

# load the speech recognition model
# MODEL_DIR = "../models/asr-lch-optim:adamw-scheduler:oncecycle-data:full-epochs:30"
MODEL_DIR = "../models/kaggle-epochs_14-rnn_type_LSTM-rnn_dim_512-n_rnn_layers_5-n_cnn_layers_3"
model = Model(MODEL_DIR)

cors = CORS(app, resources={r"/speech-recognition": {"origins": "http://localhost:5000"}})

@app.route('/speech-recognition/', methods=['POST'])
@cross_origin(origin='localhost', headers=['Content-Type'])
def post():
    session.permanent = True
    
    audio_file = request.files['audio_file']
    print(audio_file)
    recognition = model.recognize(audio=audio_file)
    return recognition


# api.add_resource(SpeechRecognition, '/speech-recognition/<string:audio_path>')


if __name__ == '__main__':
    # app.config['SESSION_TYPE'] = 'filesystem'

    session.init_app(app)

    app.run(debug=True)
