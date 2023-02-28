from typing import List, Optional

import numpy as np
from coeirocore.coeiro_manager import AudioManager
from coeirocore.query_manager import query2tokens_prosody

from ...model import AccentPhrase, AudioQuery
from ...synthesis_engine import SynthesisEngineBase


class MockSynthesisEngine(SynthesisEngineBase):
    def __init__(
        self,
        speakers: str,
        supported_devices: Optional[str] = None,
    ):
        super().__init__()

        self._speakers = speakers
        self._supported_devices = supported_devices
        self.default_sampling_rate = 44100

        self.audio_manager = AudioManager(
            fs=self.default_sampling_rate,
            use_gpu=True
        )

    @property
    def speakers(self) -> str:
        return self._speakers

    @property
    def supported_devices(self) -> Optional[str]:
        return self._supported_devices

    def replace_phoneme_length(
        self, accent_phrases: List[AccentPhrase], speaker_id: int
    ) -> List[AccentPhrase]:
        return accent_phrases

    def replace_mora_pitch(
        self, accent_phrases: List[AccentPhrase], speaker_id: int
    ) -> List[AccentPhrase]:
        return accent_phrases

    def _synthesis_impl(self, query: AudioQuery, speaker_id: int) -> np.ndarray:
        tokens = query2tokens_prosody(query)
        return self.audio_manager.synthesis(
            text=tokens,
            style_id=speaker_id,
            speed_scale=query.speedScale,
            volume_scale=query.volumeScale,
            pitch_scale=query.pitchScale,
            intonation_scale=query.intonationScale,
            pre_phoneme_length=query.prePhonemeLength,
            post_phoneme_length=query.postPhonemeLength,
            output_sampling_rate=query.outputSamplingRate
        )
