import glob
import time
from pathlib import Path
from typing import List, NamedTuple

import librosa
import numpy as np
import pyopenjtalk
import resampy
import sklearn.neighbors._partition_nodes
import sklearn.utils._typedefs
import torch
import yaml
from espnet2.text.token_id_converter import TokenIDConverter
from espnet2.bin.tts_inference import Text2Speech
from sklearn.preprocessing import StandardScaler

from ..core.coeiroink import get_metas_dict
from ...model import AccentPhrase, AudioQuery
from ...synthesis_engine import SynthesisEngineBase


class EspnetSettings(NamedTuple):
    acoustic_model_config_path: str
    acoustic_model_path: str


class EspnetModel:
    def __init__(self, settings: EspnetSettings, use_gpu=False, speed_scale=1.0):
        self.device = 'cuda' if use_gpu else 'cpu'
        self.tts_model = Text2Speech(
            settings.acoustic_model_config_path,
            settings.acoustic_model_path,
            device=self.device,
            seed=0,
            # Only for FastSpeech & FastSpeech2 & VITS
            speed_control_alpha=speed_scale,
            # Only for VITS
            noise_scale=0.333,
            noise_scale_dur=0.333,
        )

        with open(settings.acoustic_model_config_path) as f:
            config = yaml.safe_load(f)
        self.token_id_converter = TokenIDConverter(
            token_list=config["token_list"],
            unk_symbol="<unk>",
        )

    def make_voice(self, tokens, seed=0):
        np.random.seed(seed)
        torch.manual_seed(seed)
        text_ints = np.array(self.token_id_converter.tokens2ids(tokens), dtype=np.int64)
        with torch.no_grad():
            wave = self.tts_model(text_ints)["wav"]
        wave = wave.view(-1).cpu().numpy()
        return wave

    @classmethod
    def get_espnet_model(cls, acoustic_model_path, acoustic_model_config_path, use_gpu, speed_scale=1.0):
        settings = EspnetSettings(
            acoustic_model_config_path=acoustic_model_config_path,
            acoustic_model_path=acoustic_model_path,
        )
        return cls(settings, use_gpu=use_gpu, speed_scale=speed_scale)

    @classmethod
    def get_character_model(cls, use_gpu, speaker_id, speed_scale=1.0):
        uuid = None
        metas = get_metas_dict()
        for meta in metas:
            for style in meta['styles']:
                if speaker_id == style['id']:
                    uuid = meta['speaker_uuid']
        if uuid is None:
            raise Exception("Not Found Speaker Directory")

        acoustic_model_folder_path = f"./speaker_info/{uuid}/model/{speaker_id}"
        model_files = sorted(glob.glob(acoustic_model_folder_path + '/*.pth'))

        return cls.get_espnet_model(
            acoustic_model_path=model_files[0],
            acoustic_model_config_path=f"{acoustic_model_folder_path}/config.yaml",
            use_gpu=use_gpu,
            speed_scale=speed_scale
        )


class MockSynthesisEngine(SynthesisEngineBase):
    def __init__(self, **kwargs):
        self.speakers = kwargs["speakers"]

        self.default_sampling_rate = 44100
        self.use_gpu = False

        metas = get_metas_dict()
        self.previous_speaker_id = metas[0]['styles'][0]['id']
        self.previous_speed_scale = 1.0

        self.current_speaker_models: EspnetModel = \
            EspnetModel.get_character_model(use_gpu=self.use_gpu,
                                            speaker_id=self.previous_speaker_id,
                                            speed_scale=self.previous_speed_scale)

    @staticmethod
    def replace_phoneme_length(accent_phrases: List[AccentPhrase], speaker_id: int) -> List[AccentPhrase]:
        return accent_phrases

    @staticmethod
    def replace_mora_pitch(accent_phrases: List[AccentPhrase], speaker_id: int) -> List[AccentPhrase]:
        return accent_phrases

    def _synthesis_impl(self, query: AudioQuery, speaker_id: int, text: str = '') -> np.ndarray:
        start_time = time.time()
        tokens = self.query2tokens_prosody(query, text)

        if self.previous_speaker_id != speaker_id or self.previous_speed_scale != query.speedScale:
            self.current_speaker_models = None
            self.current_speaker_models = EspnetModel.get_character_model(use_gpu=self.use_gpu,
                                                                          speaker_id=speaker_id,
                                                                          speed_scale=1/query.speedScale)
            self.previous_speaker_id = speaker_id
            self.previous_speed_scale = query.speedScale
        wave = self.current_speaker_models.make_voice(tokens)

        # trim
        wave, _ = librosa.effects.trim(wave)

        # volume
        if query.volumeScale != 1:
            wave *= query.volumeScale

        # add sil
        if query.prePhonemeLength != 0 or query.postPhonemeLength != 0:
            pre_pause = np.zeros(int(self.default_sampling_rate * query.prePhonemeLength))
            post_pause = np.zeros(int(self.default_sampling_rate * query.postPhonemeLength))
            wave = np.concatenate([pre_pause, wave, post_pause], 0)

        # resampling
        if query.outputSamplingRate != self.default_sampling_rate:
            wave = resampy.resample(
                wave,
                self.default_sampling_rate,
                query.outputSamplingRate,
                filter="kaiser_fast",
            )

        rtf = (time.time() - start_time)
        print(f"Synthesis Time: {rtf}")
        return wave

    @staticmethod
    def query2tokens_prosody(query: AudioQuery, text=''):
        question_flag = False
        if query.kana != '':
            if query.kana[-1] in ['?', '？']:
                question_flag = True
        if text != '':
            if text[-1] in ['?', '？']:
                question_flag = True
        tokens = ['^']
        for i, accent_phrase in enumerate(query.accent_phrases):
            up_token_flag = False
            for j, mora in enumerate(accent_phrase.moras):
                if mora.consonant:
                    tokens.append(mora.consonant.lower())
                if mora.vowel == 'N':
                    tokens.append(mora.vowel)
                else:
                    tokens.append(mora.vowel.lower())
                if accent_phrase.accent == j+1 and j+1 != len(accent_phrase.moras):
                    tokens.append(']')
                if accent_phrase.accent-1 >= j+1 and up_token_flag is False:
                    tokens.append('[')
                    up_token_flag = True
            if i+1 != len(query.accent_phrases):
                if accent_phrase.pause_mora:
                    tokens.append('_')
                else:
                    tokens.append('#')
        if question_flag:
            tokens.append('?')
        else:
            tokens.append('$')
        return tokens
