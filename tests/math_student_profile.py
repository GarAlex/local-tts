"""Small student checkpoint contract checks; synthetic weights are not speech."""
import copy
from pathlib import Path
import sys
import unittest
import hashlib
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
import torch
from train import MATH_MODEL, SynthesizerTrn
from checkpoint_tools import build

class StudentProfile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model = SynthesizerTrn(3, 513, 32, **MATH_MODEL)
        cls.saved = dict(model_profile='math-student-v1', model_config=MATH_MODEL,
                         training_config=dict(schema_version=1, language='en',
                                              frontend='grapheme-v2', sample_rate=22050,
                                              symbols=['\0', 'a', 'b']),
                         generator=cls.model.state_dict())

    def test_valid_profile_builds_small_inference_network(self):
        model = build(self.saved)
        parameters = sum(p.numel() for name in ['enc_p','dec','flow','dp']
                         for p in getattr(model, name).parameters())
        self.assertLess(parameters, 600000)

    def test_rejects_wrong_frontend_language_and_architecture(self):
        for field, value in [('language', 'ru'), ('frontend', 'grapheme-v1')]:
            saved = copy.deepcopy(self.saved)
            saved['training_config'][field] = value
            with self.assertRaises(ValueError): build(saved)
        saved = copy.deepcopy(self.saved)
        saved['model_config']['hidden_channels'] = 48
        with self.assertRaises(ValueError): build(saved)

    def test_rejects_nonfinite_weights(self):
        saved = copy.deepcopy(self.saved)
        saved['generator']['enc_p.emb.weight'][0,0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'non-finite'): build(saved)

    def test_phoneme_profile_requires_its_dictionary_and_frontend(self):
        saved = copy.deepcopy(self.saved)
        saved['model_profile'] = 'math-phoneme-v1'
        dictionary = Path(__file__).resolve().parents[1] / 'voices/kokoro-en/cmudict.dict'
        saved['training_config'].update(frontend='cmu-ipa-v1', dictionary=dict(
            path=str(dictionary), sha256=hashlib.sha256(dictionary.read_bytes()).hexdigest()))
        build(saved)
        saved['training_config']['frontend'] = 'grapheme-v2'
        with self.assertRaisesRegex(ValueError, 'frontend'): build(saved)
        saved['training_config']['frontend'] = 'cmu-ipa-v1'
        saved['training_config']['dictionary']['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'checksum'): build(saved)

if __name__ == '__main__': unittest.main()
