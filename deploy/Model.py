import json
import torch
import pickle
from pprint import pprint
# from ctcdecode import CTCBeamDecoder

from SpeechRecognitionModel.SpeechRecognitionModel import SpeechRecognitionModel
from GreedyDecoder import GreedyDecoder
from AudioTransform import AudioTransform
from TextTransform import TextTransform
from utils import get_device


class Model():

    def __init__(self,
                 model_dir: str) -> None:

        self.text_transform = TextTransform()

        self.audio_transform = AudioTransform(self.text_transform)
        self.greedy_decoder = GreedyDecoder(self.text_transform)

        self.model, self.hparams = self._load_model(model_dir)
        self.model.eval()

    def _load_model(self, model_dir: str):

        def _get_model_paths(model_dir):
            experiment_name = model_dir.rsplit('/', maxsplit=1)[-1]
            model_path = f'{model_dir}/{experiment_name}.pth'
            hparams_path = f'{model_dir}/{experiment_name}.json'
            return experiment_name, model_path, hparams_path

        _, model_path, hparams_path = _get_model_paths(model_dir)
        
        device, _ = get_device()

        with open(hparams_path, 'r') as f:
            hparams = json.load(f)

        if 'rnn_type' not in hparams:
            hparams['rnn_type'] = "GRU"

        model = SpeechRecognitionModel(
            hparams['rnn_type'], hparams['n_cnn_layers'], hparams['n_rnn_layers'], hparams['rnn_dim'],
            hparams['n_class'], hparams['n_feats'], hparams['stride'], hparams['dropout']
            ).to(device)

        model.load_state_dict(torch.load(model_path, map_location=device))

        return model, hparams

    def recognize(self,
                  audio = None,
                  collapse_repeated: bool = True,):
        
        #audio = pickle.load(audio)
        print(type(audio))
        audio.save(audio.filename)

        audio_path = audio.filename

        spectrogram, label, input_length, label_length = self.audio_transform.transform(audio_path)

        output = self.model(spectrogram)

        greedy_pred, label = self.greedy_decoder.decode(output, label, label_length, collapse_repeated=collapse_repeated)
        # beams, beam_preds = BeamSearchDecoder(output)
        
        print(f"\nNegative log likelihood matrix shape: {output.shape}")
        print("\nGREEDY DECODING")
        print(f"Decoded indices:\n{torch.argmax(output, dim=2)}\n")
        print(f"Prediction (len {len(greedy_pred[0])}): {greedy_pred}\n")

        # print("\nBEAM SEARCH DECODING")
        # print(f"Top beam:\n{beams}")
        # print()
        # print(f"Target (len {len(label[0])}): {label}")
        # print(f"Prediction (len {len(beam_preds[0])}): {beam_preds}")

        return greedy_pred[0]