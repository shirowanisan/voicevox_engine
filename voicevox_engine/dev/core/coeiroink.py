import glob
import json
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from pyopenjtalk import tts
from scipy.signal import resample

DUMMY_TEXT = "これはダミーのテキストです"


def initialize(path: str, use_gpu: bool, *args: List[Any]) -> None:
    pass


def yukarin_s_forward(length: int, **kwargs: Dict[str, Any]) -> np.ndarray:
    logger = getLogger("uvicorn")  # FastAPI / Uvicorn 内からの利用のため
    logger.info(
        "Sorry, yukarin_s_forward() is a mock. Return values are incorrect.",
    )
    return np.ones(length) / 5


def yukarin_sa_forward(length: int, **kwargs: Dict[str, Any]) -> np.ndarray:
    logger = getLogger("uvicorn")  # FastAPI / Uvicorn 内からの利用のため
    logger.info(
        "Sorry, yukarin_sa_forward() is a mock. Return values are incorrect.",
    )
    return np.ones((1, length)) * 5


def decode_forward(length: int, **kwargs: Dict[str, Any]) -> np.ndarray:
    """
    合成音声の波形データをNumPy配列で返します。ただし、常に固定の文言を読み上げます（DUMMY_TEXT）
    参照→SynthesisEngine のdocstring [Mock]

    Parameters
    ----------
    length : int
        フレームの長さ

    Returns
    -------
    wave : np.ndarray
        音声合成した波形データ

    Note
    -------
        ここで行う音声合成では、調声（ピッチ等）を反映しない
        また、入力内容によらず常に固定の文言を読み上げる

        # pyopenjtalk.tts()の出力仕様
        dtype=np.float64, 16 bit, mono 48000 Hz

        # resampleの説明
        非モックdecode_forwardと合わせるために、出力を24kHzに変換した。
    """
    logger = getLogger("uvicorn")  # FastAPI / Uvicorn 内からの利用のため
    logger.info(
        "Sorry, decode_forward() is a mock. Return values are incorrect.",
    )
    wave, sr = tts(DUMMY_TEXT)
    wave = resample(
        wave.astype("int16"),
        24000 * len(wave) // 48000,
    )
    return wave


def get_metas_dict() -> List[dict]:
    paths: List[str] = sorted(glob.glob(str(Path(__file__).parents[3]) + '/speaker_info/**/'))

    speaker_infos = []
    for path in paths:
        with open(path + 'metas.json', encoding='utf-8') as f:
            meta = json.load(f)
        styles = [{'name': s['styleName'], 'id': s['styleId']} for s in meta['styles']]
        version = meta['version'] if 'version' in meta.keys() else '0.0.1'
        speaker_info = {
            'name': meta['speakerName'],
            'speaker_uuid': meta['speakerUuid'],
            'styles': styles,
            'version': version
        }
        speaker_infos.append(speaker_info)

    speaker_infos = sorted(speaker_infos, key=lambda x: x['styles'][0]['id'])
    return speaker_infos


def metas() -> str:
    return json.dumps(get_metas_dict())


def supported_devices() -> str:
    return json.dumps(
        {
            "cpu": True,
            "cuda": False,
        }
    )
