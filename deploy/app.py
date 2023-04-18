from flask import Flask, request
from flask_restful import Resource, Api
from flask_cors import CORS
from Model import Model


# create a Flask instance
app = Flask(__name__)

# Cross Origin Resource Sharing
# CORS(app)

# create an API object
# api = Api(app)

# load the speech recognition model
MODEL_DIR = "../models/asr-lch-optim:adamw-scheduler:oncecycle-data:full-epochs:30"
model = Model(MODEL_DIR)


# class SpeechRecognition(Resource):
    # TODO: audio_path vs. audio_url
@app.route('/speech-recognition/<path:audio_path>')
def get(audio_path):
    # audio_path = request.args.get('audio_path')
    print(audio_path)
    recognition = model.recognize('/' + audio_path)
    return recognition


# api.add_resource(SpeechRecognition, '/speech-recognition/<string:audio_path>')    # это url?


if __name__ == '__main__':
    app.run(debug=True)