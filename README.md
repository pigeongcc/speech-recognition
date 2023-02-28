# speech-recognition

## Pipeline

1. WAV to torch.Tensor (+ pics from https://pytorch.org/tutorials/intermediate/speech_command_classification_with_torchaudio_tutorial.html#formatting-the-data)

2. torch.Tensor to Mel-Spectrogram

3. Augmentations

4. RNN/CNN

5. Language Model ?