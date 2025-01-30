import numpy as np
from .text import text_to_sequence
from .models import SynthesizerTrn
from . import utils
from . import commons
import sys
import re
import os
from torch import no_grad, LongTensor

def get_text(text, hps, cleaned=False):
    if cleaned:
        text_norm = text_to_sequence(text, hps.symbols, [])
    else:
        text_norm = text_to_sequence(text, hps.symbols, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm

def get_label_value(text, label, default, warning_name='value'):
    value = re.search(rf'\[{label}=(.+?)\]', text)
    if value:
        try:
            text = re.sub(rf'\[{label}=(.+?)\]', '', text, 1)
            value = float(value.group(1))
        except:
            print(f'Invalid {warning_name}!')
            sys.exit(1)
    else:
        value = default
    return value, text

def get_label(text, label):
    if f'[{label}]' in text:
        return True, text.replace(f'[{label}]', '')
    else:
        return False, text

class Model:
    def __init__(self, path: str):
        hps_ms = utils.get_hparams_from_file(os.path.join(path, 'moegoe_config.json'))
        n_symbols = len(hps_ms['symbols']) if 'symbols' in hps_ms.keys() else 0
        emotion_embedding = hps_ms['data'].emotion_embedding if 'emotion_embedding' in hps_ms['data'].keys() else False

        net_g_ms = SynthesizerTrn(
            n_symbols,
            hps_ms['data'].filter_length // 2 + 1,
            hps_ms['train'].segment_size // hps_ms['data'].hop_length,
            n_speakers=1,
            emotion_embedding=emotion_embedding,
            **hps_ms['model'])
        _ = net_g_ms.eval()
        utils.load_checkpoint(os.path.join(path, 'G_latest.pth'), net_g_ms)

        assert(n_symbols != 0)
        assert(not emotion_embedding)

        self.hps_ms = hps_ms
        self.net_g_ms = net_g_ms

    @no_grad()
    def inference(self, speaker_id: int, text: str) -> tuple[int, np.ndarray]:
        length_scale, text = get_label_value(
            text, 'LENGTH', 1, 'length scale')
        noise_scale, text = get_label_value(
            text, 'NOISE', 0.667, 'noise scale')
        noise_scale_w, text = get_label_value(
            text, 'NOISEW', 0.8, 'deviation of noise')
        cleaned, text = get_label(text, 'CLEANED')

        stn_tst = get_text(text, self.hps_ms, cleaned=cleaned)

        x_tst = stn_tst.unsqueeze(0)
        x_tst_lengths = LongTensor([stn_tst.size(0)])
        sid = LongTensor([speaker_id])
        audio = self.net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=int(noise_scale),
                                noise_scale_w=noise_scale_w, length_scale=int(length_scale))[0][0, 0].data.cpu().float().numpy()

        return self.hps_ms['data'].sampling_rate, audio
