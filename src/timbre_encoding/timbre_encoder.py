# timbre_encoding/timbre_encoder.py

import os
import sys
import torch
import torch.nn as nn
from .models.lstm import LSTMSpeakerEncoder
from utils.config import SpeakerEncoderConfig
from utils.audio import AudioProcessor
from utils.read_json import read_json


class SpkEncoderHelper(nn.Module):
    def __init__(self, root_path=None, use_cuda=False):
        super(SpkEncoderHelper, self).__init__()
        # 모델 및 설정 파일 경로 설정
        self.config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pretrained_timbre_enc/config.json')
        if root_path:
            self.config_path = os.path.join(root_path, self.config_path)
        # 설정 로드
        self.config_dict = read_json(self.config_path)
        self.config = SpeakerEncoderConfig(self.config_dict)
        self.config.from_dict(self.config_dict)
        # 모델 초기화
        self.speaker_encoder = LSTMSpeakerEncoder(
            self.config.model_params["input_dim"],
            self.config.model_params["proj_dim"],
            self.config.model_params["lstm_dim"],
            self.config.model_params["num_lstm_layers"],
        )
        self.use_cuda = use_cuda

        # 오디오 프로세서 초기화
        self.speaker_encoder_ap = AudioProcessor(**self.config.audio)
        self.speaker_encoder_ap.do_sound_norm = True
        self.speaker_encoder_ap.do_trim_silence = True

    def forward(self, wav_files, infer=False):
        embeds = torch.zeros(len(wav_files), self.speaker_encoder.proj_dim)
        device = torch.device('cuda' if self.use_cuda else 'cpu')
        self.speaker_encoder.to(device)
        for i, wav_file in enumerate(wav_files):
            waveform = self.speaker_encoder_ap.load_wav(wav_file, sr=self.speaker_encoder_ap.sample_rate)
            spec = self.speaker_encoder_ap.melspectrogram(waveform)
            spec = torch.from_numpy(spec.T).to(device).unsqueeze(0)
            embed = self.speaker_encoder.compute_embedding(spec, infer=infer)
            embeds[i] = embed
        return embeds

def timbre_enc(wav_path):
    load_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pretrained_timbre_enc/best_model.pth.tar')
    loaded_state_dict = torch.load(load_path, map_location='cpu', weights_only=True)
    use_cuda = torch.cuda.is_available()

    # SpkEncoderHelper 초기화
    spk_encoder_helper = SpkEncoderHelper(root_path='.', use_cuda=use_cuda)
    spk_encoder_helper.load_state_dict(loaded_state_dict)
    spk_encoder_helper.to(torch.device('cuda' if use_cuda else 'cpu'))
    embeds = spk_encoder_helper.forward([wav_path], infer=True)
    embedding = embeds[0].detach().cpu().numpy()
    return embedding