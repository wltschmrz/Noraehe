# <src/pitch_detecting/audio_processor.py>

import os
import shlex
import librosa
import crepe
import numpy as np
import logging
import shutil
from yt_dlp import YoutubeDL
import demucs.separate
import torch
import tensorflow as tf
from src.pitch_detecting.vocal_range import VocalRange
from src.pitch_detecting.embedding_utils import add_or_search_embedding

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self, youtube_url, save_dir, original_wav_path=None):
        
        self.yt_url = youtube_url                   # 'Youtube 다운받을 링크
        self.data_file_path = save_dir              # ./data
        self.origin_wav_path = original_wav_path    # 분리 전 원곡 .wav 파일 경로
        self.vocal_wav_path = None                  # 분리된 vocal 음원 .wav 파일 경로

        self.yt_title = None                        # Youtube 제목
        self.yt_id = None                           # Youtube 링크 ID

        self.audio = None                           # Audio 값
        self.sr = None                              # Sampling rate
        self.time = None                            # time array
        self.frequency = None                       # 시간 당 frequency 값

        self.frame_length = 2048                    # hyperparam1 for pitch detecting
        self.hop_length = 512                       # hyperparam2 for pitch detecting
        self.rms_threshold = 0.04                   # RMS 에너지 임계값 (이 값보다 에너지가 낮으면 무음으로 간주)
        self.trim_duration = 20                     # 오디오 전후 cliping할 sec
        self.to_deletes = []                        # 지울 dir 목록

    def download_audio(self):
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(self.data_file_path, '%(id)s.%(ext)s'),
            'quiet': True,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }],
        }
        with YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(self.yt_url, download=True)
            self.yt_title = info_dict.get('title', None)
            self.yt_id = info_dict.get('id', None)
            self.origin_wav_path = os.path.join(self.data_file_path, f"{self.yt_id}.wav")
            logger.info(f"Downloaded youtube to .wav: {self.yt_title}, {self.yt_url}")           

    def separate_vocals(self, mode='yt'):
        """
        origin_wav_path와 Demucs를 이용해, Vocal 분리 및 vocal_wav_path 경로 값 채움.
        input:
            mode (str): 'yt'라면 origin은 yt 인것. 'user'라면 user 목소리인것.
        """
        assert mode in ['yt', 'user'], "Incorrect mode. while .separate_vocal()"
        super_file_name = 'yt' if mode == 'yt' else 'user'
        output_dir = os.path.join(self.data_file_path, super_file_name)
        command = f'--two-stems vocals -n htdemucs -o {output_dir} "{self.origin_wav_path}"'
        demucs.separate.main(shlex.split(command))
        logger.info(f"Separated vocals from {self.origin_wav_path}")
        # 보컬 wav 경로 설정
        sub_file_name = self.yt_id if mode == 'yt' else os.path.splitext(os.path.basename(self.origin_wav_path))[0]
        self.vocal_wav_path = os.path.join(self.data_file_path, f"{super_file_name}/htdemucs/{sub_file_name}/vocals.wav")

        self.to_deletes.append(output_dir)

    def detect_pitch_range(self):
        self.audio, self.sr = librosa.load(self.vocal_wav_path, sr=self.sr)
        self.time, self.frequency, confidence, activation = crepe.predict(self.audio, self.sr, viterbi=True)
        # Calculate RMS energy
        rms = librosa.feature.rms(y=self.audio.astype(float), frame_length=self.frame_length, hop_length=self.hop_length)[0]
        rms_time = librosa.frames_to_time(np.arange(len(rms)), sr=self.sr, hop_length=self.hop_length)

        # Interpolate RMS values at pitch times
        rms_values_at_pitch_times = np.interp(self.time, rms_time, rms)

        # Filter out frequencies where RMS is below threshold
        frequency_array = np.array(self.frequency)
        frequency_array[rms_values_at_pitch_times < self.rms_threshold] = np.nan

        # Convert time to NumPy array for efficient indexing
        time_array = np.array(self.time)

        # Trim the first and last `trim_duration` seconds
        valid_indices = (time_array >= self.trim_duration) & (time_array <= (time_array[-1] - self.trim_duration))
        frequency_array = frequency_array[valid_indices]

        # Remove NaN values for percentile calculations
        valid_frequencies = frequency_array[~np.isnan(frequency_array)]
        if len(valid_frequencies) == 0:
            raise ValueError("No valid frequency data after applying RMS threshold and trimming.")

        # Calculate the 20th and 80th percentiles (Q1 and Q3)
        Q1, Q3 = np.percentile(valid_frequencies, 20), np.percentile(valid_frequencies, 80)
        IQR = Q3 - Q1

        # Calculate bounds for outlier removal
        lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

        # Remove outliers
        frequency_array = np.where((frequency_array >= lower_bound) & (frequency_array <= upper_bound), frequency_array, np.nan)

        # 주파수를 MIDI 음계로 변환
        midi_values = 69 + 12 * np.log2(frequency_array / 440.0)
        midi_values = midi_values[~np.isnan(midi_values)]  # Remove NaN values

        if len(midi_values) == 0:
            raise ValueError("No valid MIDI values after outlier removal.")

        # Calculate minimum and maximum MIDI values
        min_midi, max_midi = np.min(midi_values), np.max(midi_values)
        note_range = (VocalRange.midi_to_note_name(min_midi), VocalRange.midi_to_note_name(max_midi))

        # Convert MIDI values to note names
        return note_range

    def cleanup_files(self, remove_org_wav=True):
        if remove_org_wav and os.path.exists(self.origin_wav_path):
            os.remove(self.origin_wav_path)
            logger.info(f"Deleted: {self.origin_wav_path}")
        for dir_path in self.to_deletes:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
                logger.info(f"Deleted: {dir_path}")

    def process(self, faiss, sbert):
        try:
            self.download_audio()
            self.separate_vocals()
            range = self.detect_pitch_range()
            add_or_search_embedding(faiss, self.vocal_wav_path, sbert, self.yt_id, self.yt_title, range)
            return range
        
        except Exception as e:
           logging.error(f"Error occurred while processing in AudioProcessor.process(): {e}")
        
        finally:
           faiss.save()
           self.cleanup_files()

    def vocal_base_searching(self, faiss):
        try:
            self.separate_vocals(mode='user')
            range = self.detect_pitch_range()
            similar_sets = add_or_search_embedding(faiss, self.vocal_wav_path)
            return similar_sets, range
        
        except Exception as e:
            logging.error(f"Error occurred while processing in AudioProcessor.vocal_base_searching(): {e}")

        finally:
            self.cleanup_files(remove_org_wav=False)
