import torchaudio
from pydub import AudioSegment
import torch
from torch import nn
from utils import get_device


class AudioTransform:

    def __init__(self, text_transform) -> None:
        self.text_transform = text_transform

        self.train_audio_transforms = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
            torchaudio.transforms.TimeMasking(time_mask_param=35)
        )
        self.valid_audio_transforms = torchaudio.transforms.MelSpectrogram()
    
    def _load_audio(self, audio_path: str):
        print(audio_path)
        _audio_path, format = audio_path.rsplit('.', maxsplit=1)
        flac_audio_path = _audio_path + ".flac"

        import os
        print()
        files = [f for f in os.listdir('.') if os.path.isfile(f)]
        for f in files:
            print(f)
        print()

        audio = AudioSegment.from_file(audio_path, format)
        audio = audio.set_frame_rate(16000)
        audio = audio.set_channels(1)
        audio.export(flac_audio_path, format="flac")

        waveform, _ = torchaudio.load(flac_audio_path)
        sample = (waveform, None, "", None, None, None)
        
        return sample
    
    # def _process_audio(self, sample: tuple):
    #     device, _ = get_device()

    #     spectrogram, label, input_length, label_length = \
    #         self._data_processing((sample,))
    #     spectrogram, label = spectrogram.to(device), label.to(device)
        
    #     return spectrogram, label, input_length, label_length
    
    def _data_processing(self, data, data_type="train"):
        spectrograms = []
        labels = []
        input_lengths = []
        label_lengths = []
        for (waveform, _, utterance, _, _, _) in data:
            if data_type == 'train':
                spec = self.train_audio_transforms(waveform).squeeze(0).transpose(0, 1)
            else:
                spec = self.valid_audio_transforms(waveform).squeeze(0).transpose(0, 1)
            spectrograms.append(spec)
            # labels are lists of integer character ids
            label = torch.Tensor(self.text_transform.text_to_int(utterance.lower()))
            labels.append(label)
            # input_lengths, label_lengths are used in loss function
            input_lengths.append(spec.shape[0]//2)
            label_lengths.append(len(label))

        spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
        labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

        return spectrograms, labels, input_lengths, label_lengths
    
    def transform(self, audio_path: str):
        device, _ = get_device()

        audio_sample = self._load_audio(audio_path)
        spectrogram, label, input_length, label_length = self._data_processing((audio_sample,))
        spectrogram, label = spectrogram.to(device), label.to(device)

        return spectrogram, label, input_length, label_length

