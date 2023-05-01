import requests
import pickle

from pydub import AudioSegment


def load_audio_file(audio_path: str):
    _audio_path_noformat, format = audio_path.rsplit('.', maxsplit=1)
    audio_file = AudioSegment.from_file(audio_path, format)
    return audio_file


audio_path = "/home/pigeon_gcc/Desktop/speech-recognition/data/Iskander/flac/10000-9999999-0001.flac"
audio_file = load_audio_file(audio_path)
audio_file = pickle.dumps(audio_file)

# response = requests.post('http://127.0.0.1:5000/speech-recognition/', files={'audio_file': audio_file})
response = requests.post('http://185.188.183.103:5000/speech-recognition/', files={'audio_file': audio_file})

print(response.text)