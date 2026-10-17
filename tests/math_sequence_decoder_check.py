"""Decoder diagnostics must reject changed or incomplete native package inputs."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
from math_sequence_decoder_check import validate_package

class Integrity(unittest.TestCase):
    def test_checked_components_and_containment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'voice'
            root.mkdir()
            for name in ['encoder.onnx', 'latent.onnx', 'decoder.onnx']:
                (root / name).write_bytes(name.encode())
            (root / 'sequence.json').write_text('{"language":"ru"}')
            hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}
            (root / 'checksums.json').write_text(json.dumps(hashes))
            self.assertEqual(validate_package(root)['language'], 'ru')
            (root / 'latent.onnx').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'integrity'):
                validate_package(root)
            (root / 'latent.onnx').write_bytes(b'latent.onnx')
            del hashes['decoder.onnx']
            (root / 'checksums.json').write_text(json.dumps(hashes))
            with self.assertRaisesRegex(ValueError, 'missing package checksum'):
                validate_package(root)
            hashes['decoder.onnx'] = hashlib.sha256(b'decoder.onnx').hexdigest()
            (Path(tmp) / 'outside').write_bytes(b'outside')
            hashes['../outside'] = hashlib.sha256(b'outside').hexdigest()
            (root / 'checksums.json').write_text(json.dumps(hashes))
            with self.assertRaisesRegex(ValueError, 'integrity'):
                validate_package(root)

if __name__ == '__main__':
    unittest.main()
