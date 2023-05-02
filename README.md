# Speech Recognition Model

## Repository structure

- `Report.pdf` - detailed technical report on the model.
- `model.ipynb` - notebook with all the code required to train and evaluate the model
- `deploy` and `run.sh` - folder and bash script with all the code required to run the model on a server.
- `requirements.txt` - file for reproduction of the Python virtual environment used in the code.

## Try to Recognize now!

The model is deployed. To use it, send your speech audio file using the following bash command:

```
curl \
-F "audio_file=@path_to_audio_file" \
http://185.188.183.103:5000/speech-recognition/
```

There are no limitations on audio format.
