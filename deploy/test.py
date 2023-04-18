from Model import Model

model_dir = "models/asr-lch-optim:adamw-scheduler:oncecycle-data:full-epochs:30"

model = Model(model_dir)

audio_path = "data/Iskander/flac/10000-9999999-0002.flac"

recognition = model.recognize(audio_path)

print(recognition)