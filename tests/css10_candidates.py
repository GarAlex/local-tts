"""Candidate inspection failures; --installed also inspects both pinned checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))
import torch
from inspect_css10 import inspect_candidate, load_weights, token_inventory


class UnsafeMetadata:
    def __reduce__(self):
        return eval, ('1 / 0',)


def record(path):
    data = path.read_bytes()
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


class CandidateInspection(unittest.TestCase):
    def test_tampered_bytes_rejected_before_deserialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'weights.pt'
            torch.save({'model': {'weight': torch.ones(2)}}, path)
            pin = record(path)
            data = bytearray(path.read_bytes()); data[-1] ^= 1; path.write_bytes(data)
            with patch('inspect_css10.torch.load') as mocked:
                with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                    load_weights(path, pin)
                mocked.assert_not_called()

    def test_wrong_size_rejected_before_deserialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'weights.pt'; path.write_bytes(b'incomplete')
            pin = record(path); pin['bytes'] += 1
            with patch('inspect_css10.torch.load') as mocked:
                with self.assertRaisesRegex(ValueError, 'size/type mismatch'):
                    load_weights(path, pin)
                mocked.assert_not_called()

    def test_no_unsafe_pickle_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'weights.pt'
            torch.save({'metadata': UnsafeMetadata()}, path)
            # If eval were executed, this would raise ZeroDivisionError instead.
            with self.assertRaises(pickle.UnpicklingError):
                load_weights(path, record(path))

    def test_weight_only_checkpoint_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'weights.pt'
            torch.save({'model': {'weight': torch.tensor([1.0, 2.0])}}, path)
            loaded = load_weights(path, record(path))
            self.assertEqual(loaded['model']['weight'].tolist(), [1.0, 2.0])

    def test_in_place_change_during_loading_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'weights.pt'
            torch.save({'model': {'weight': torch.ones(2)}}, path)
            pin = record(path)
            original_load = torch.load

            def changed_load(stream, **kwargs):
                result = original_load(stream, **kwargs)
                with path.open('r+b') as changed:
                    changed.seek(-1, 2)
                    value = changed.read(1)[0]
                    changed.seek(-1, 2)
                    changed.write(bytes([value ^ 1]))
                return result

            with patch('inspect_css10.torch.load', side_effect=changed_load):
                with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                    load_weights(path, pin)


def installed_checks():
    results = []
    for language in ['de', 'es']:
        source = ROOT / 'voices/css10-candidates' / language
        report = inspect_candidate(language, source)
        assert report['source_verified'] and not report['training_ready']
        assert report['tensor_count'] == 696 and report['sample_rate'] == 22050
        assert not report['commercial_clearance_established']
        inventory = report['frontend']['symbols']
        assert inventory[:4] == ['<PAD>', '<EOS>', '<BOS>', '<BLNK>']
        assert all(c in inventory for c in 'äöüßáéíóúñ')
        assert report['frontend']['unsupported_punctuation'] == ['¡', '¿']
        config = json.loads((source / 'config.json').read_text())
        config['characters']['blank'] = '<PAD>'
        try:
            token_inventory(config)
        except ValueError:
            pass
        else:
            raise AssertionError('changed blank contract accepted')
        results.append(report)
    assert results[0]['weights_sha256'] != results[1]['weights_sha256']
    assert results[0]['tensor_schema_sha256'] == results[1]['tensor_schema_sha256']
    (ROOT / 'target/css10-candidate-inspection.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
    print('Both pinned sources verified; raw files require project import and provenance review.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--installed', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CandidateInspection))
    if not result.wasSuccessful():
        raise SystemExit(1)
    if args.installed:
        installed_checks()
