from flask import Flask, request, session, jsonify, make_response
from flask_cors import CORS, cross_origin
from Model import Model


# load the speech recognition model
MODEL_DIR = "../models/kaggle-epochs_14-rnn_type_LSTM-rnn_dim_512-n_rnn_layers_5-n_cnn_layers_3"
model = Model(MODEL_DIR)

# create a Flask instance
app = Flask(__name__)
app.secret_key = 'super secret key'

# Cross Origin Resource Sharing
CORS(app)


# method for CORS preflight request
def _build_cors_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add('Access-Control-Allow-Headers', "*")
    response.headers.add('Access-Control-Allow-Methods', "*")
    return response


# method for CORSification of response with a recognition
def _corsify_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


# @cross_origin(origin='localhost', headers=['Content-Type'])
@app.route('/speech-recognition/', methods=['POST', 'OPTION'])
def post():
    session.permanent = True

    if request.method == "OPTIONS":
        # CORS preflight
        return _build_cors_preflight_response()

    elif request.method == "POST":
        # the actual request following the preflight
        audio_file = request.files['audio_file']
        print(audio_file)
        recognition = model.recognize(audio=audio_file)
        # return recognition
        return _corsify_actual_response(jsonify(recognition))


if __name__ == '__main__':
    session.init_app(app)

    app.run(debug=True)