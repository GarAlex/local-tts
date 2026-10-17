"""CSS10 architecture assembled from the project's MIT VITS components."""
import math
import json
from pathlib import Path
import re
import sys
import weakref

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent / 'vendor/vits'))
import commons
from models import SynthesizerTrn, TextEncoder

PROFILE = 'css10-de-es-v1'
FRONTEND = 'css10-graphemes-v1'
MODEL = dict(inter_channels=192, hidden_channels=192, filter_channels=768,
             n_heads=2, n_layers=6, kernel_size=3, p_dropout=0.1,
             resblock='2', resblock_kernel_sizes=[3, 5, 7],
             resblock_dilation_sizes=[[1, 2], [2, 6], [3, 12]],
             upsample_rates=[8, 8, 4], upsample_initial_channel=256,
             upsample_kernel_sizes=[16, 16, 8], n_layers_q=3,
             use_spectral_norm=False, use_sdp=True, n_speakers=1, gin_channels=256)


def symbols():
    return json.loads((Path(__file__).resolve().parents[1] / 'assets/css10-symbols.json').read_text())


def validate_config(config):
    if (config['language'] not in ['de', 'es'] or config.get('frontend') != FRONTEND
            or config['symbols'] != symbols() or type(config.get('conditioning_speaker')) is not int
            or config['conditioning_speaker'] != 0):
        raise ValueError('invalid CSS10 language/frontend/inventory/conditioning speaker')


class LanguageEncoder(TextEncoder):
    def __init__(self, vocabulary_size=129):
        super().__init__(vocabulary_size, 192, 196, 768, 2, 6, 3, 0.1)
        self.emb = nn.Embedding(vocabulary_size, 192)
        self.language = nn.Embedding(1, 4)

    def forward(self, tokens, lengths):
        letters = self.emb(tokens) * math.sqrt(192)
        language = self.language.weight.unsqueeze(0).expand(tokens.shape[0], tokens.shape[1], 4)
        hidden = torch.cat((letters, language), dim=-1).transpose(1, 2)
        mask = commons.sequence_mask(lengths, tokens.shape[1]).unsqueeze(1).to(hidden.dtype)
        hidden = self.encoder(hidden * mask, mask)
        mean, logs = (self.proj(hidden) * mask).split(192, dim=1)
        return hidden, mean, logs, mask


class LanguageProjection(nn.Conv1d):
    def __init__(self, language):
        super().__init__(196, 192, 1)
        self.language_condition = nn.Conv1d(4, 192, 1)
        # The encoder owns the one shared trainable language embedding. A weak
        # reference avoids registering duplicate parameter/checkpoint entries.
        self.language = weakref.ref(language)

    def forward(self, hidden):
        language = self.language().weight.transpose(0, 1).unsqueeze(0)
        return super().forward(hidden) + self.language_condition(language)


def build_model(vocabulary_size=129):
    model = SynthesizerTrn(vocabulary_size, 513, 32, **MODEL)
    model.enc_p = LanguageEncoder(vocabulary_size)
    model.dp.pre = LanguageProjection(model.enc_p.language)
    model.emb_g = nn.Embedding(1, 256)
    return model


def convert_state(source, vocabulary_size=129):
    """Map every source tensor once; strict load rejects missing/extra shapes."""
    converted = {}
    for key, value in source.items():
        if key == 'emb_l.weight':
            name = 'enc_p.language.weight'
        else:
            name = key
            for original, replacement in [('text_encoder.', 'enc_p.'),
                                           ('posterior_encoder.', 'enc_q.'),
                                           ('waveform_decoder.', 'dec.'),
                                           ('duration_predictor.', 'dp.')]:
                if name.startswith(original):
                    name = replacement + name[len(original):]
                    break
            if name.startswith('flow.flows.'):
                name = re.sub(r'^flow\.flows\.(\d+)\.', lambda m: f'flow.flows.{2 * int(m[1])}.', name)
            if name.startswith(('dp.flows.', 'dp.post_flows.')):
                name = re.sub(r'^(dp\.(?:post_)?flows)\.(\d+)\.',
                              lambda m: f'{m[1]}.{0 if int(m[2]) == 0 else 2 * int(m[2]) - 1}.', name)
                name = name.removesuffix('.translation') + '.m' if name.endswith('.translation') else name
                name = name.removesuffix('.log_scale') + '.logs' if name.endswith('.log_scale') else name
            if name.startswith('dp.cond_lang.'):
                name = name.replace('dp.cond_lang.', 'dp.pre.language_condition.', 1)
            if name.startswith('dec.cond_layer.'):
                name = name.replace('dec.cond_layer.', 'dec.cond.', 1)
        if name in converted:
            raise ValueError('source tensor mapping collision: ' + name)
        converted[name] = value
    model = build_model(vocabulary_size)
    model.load_state_dict(converted, strict=True)
    return converted
