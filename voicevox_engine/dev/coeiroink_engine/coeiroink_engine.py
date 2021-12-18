from typing import List, NamedTuple

import librosa
import numpy as np
import pyopenjtalk
import resampy
import sklearn.neighbors._partition_nodes
import torch
import yaml
from espnet2.text.token_id_converter import TokenIDConverter
from espnet2.bin.tts_inference import Text2Speech
from parallel_wavegan.utils import load_model
from parallel_wavegan.utils import read_hdf5
from sklearn.preprocessing import StandardScaler

from voicevox_engine.model import AccentPhrase, AudioQuery


class EspnetSettings(NamedTuple):
    acoustic_model_config_path: str
    acoustic_model_path: str
    vocoder_model_path: str
    vocoder_stats_path: str


class EspnetModel:
    def __init__(self, settings: EspnetSettings, use_gpu=False, use_scaler=False):
        # init run for pyopenjtalk
        pyopenjtalk.g2p('a')

        self.device = 'cuda' if use_gpu else 'cpu'
        self.acoustic_model = Text2Speech(
            settings.acoustic_model_config_path,
            settings.acoustic_model_path,
            device=self.device,
            threshold=0.5,
            minlenratio=0.0,
            maxlenratio=10.0,
            use_att_constraint=False,
            backward_window=1,
            forward_window=3
        )
        self.acoustic_model.spc2wav = None
        self.vocoder = load_model(settings.vocoder_model_path).to(self.device).eval()

        self.use_scaler = use_scaler
        self.scaler = StandardScaler()
        self.scaler.mean_ = read_hdf5(settings.vocoder_stats_path, "mean")
        self.scaler.scale_ = read_hdf5(settings.vocoder_stats_path, "scale")
        self.scaler.n_features_in_ = self.scaler.mean_.shape[0]

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
            output = self.acoustic_model(text_ints)
            if self.use_scaler:
                if self.device == 'cuda':
                    mel = self.scaler.transform(output['feat_gen_denorm'].cpu())
                    mel = torch.tensor(mel, dtype=torch.float32, device='cuda')
                else:
                    mel = self.scaler.transform(output['feat_gen_denorm'])
                wave = self.vocoder.inference(mel)
            else:
                wave = self.vocoder.inference(output['feat_gen'])
        wave = wave.view(-1).cpu().numpy()
        return wave

    @classmethod
    def get_espnet_model(cls, acoustic_model_path, acoustic_model_config_path, use_gpu, use_scaler):
        vocoder_model_folder_path = "./models/VOCODER"
        vocoder_model_path = f"{vocoder_model_folder_path}/checkpoint-2500000steps.pkl"
        vocoder_stats_path = f"{vocoder_model_folder_path}/stats.h5"

        settings = EspnetSettings(
            acoustic_model_config_path=acoustic_model_config_path,
            acoustic_model_path=acoustic_model_path,
            vocoder_model_path=vocoder_model_path,
            vocoder_stats_path=vocoder_stats_path
        )
        return cls(settings, use_gpu=use_gpu, use_scaler=use_scaler)

    @classmethod
    def get_tsukuyomichan_model(cls, use_gpu):
        acoustic_model_folder_path = "./models/TSUKUYOMICHAN_COEIROINK_MODEL_v.2.0.0/ACOUSTIC_MODEL"
        return cls.get_espnet_model(
            acoustic_model_path=f"{acoustic_model_folder_path}/100epoch.pth",
            acoustic_model_config_path=f"{acoustic_model_folder_path}/config.yaml",
            use_gpu=use_gpu,
            use_scaler=True
        )

    @classmethod
    def get_harumachi_model(cls, use_gpu):
        acoustic_model_folder_path = "./models/HARUMACHI_COEIROINK_MODEL_v.2.0.0/ACOUSTIC_MODEL"
        return cls.get_espnet_model(
            acoustic_model_path=f"{acoustic_model_folder_path}/100epoch.pth",
            acoustic_model_config_path=f"{acoustic_model_folder_path}/config.yaml",
            use_gpu=use_gpu,
            use_scaler=True
        )


class SynthesisEngine:
    def __init__(self, **kwargs):
        self.speakers = kwargs["speakers"]

        self.default_sampling_rate = 24000
        self.use_gpu = False

        self.speaker_models: List[EspnetModel] = []
        self.speaker_models.append(EspnetModel.get_tsukuyomichan_model(use_gpu=self.use_gpu))
        self.speaker_models.append(EspnetModel.get_harumachi_model(use_gpu=self.use_gpu))

    @staticmethod
    def replace_phoneme_length(accent_phrases: List[AccentPhrase], speaker_id: int) -> List[AccentPhrase]:
        return accent_phrases

    @staticmethod
    def replace_mora_pitch(accent_phrases: List[AccentPhrase], speaker_id: int) -> List[AccentPhrase]:
        return accent_phrases

    def synthesis(self, query: AudioQuery, speaker_id: int, text: str = '') -> np.ndarray:
        # make_wave
        tokens = self.query2tokens_prosody(query, text)
        wave = self.speaker_models[speaker_id].make_voice(tokens)

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
