"""Teacher labels never admit held-out clips and generate exact duration paths."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training/vendor/vits'))
import torch
from math_alignment import load_alignment, alignment_batch


class TeacherAlignment(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'alignment.json'
        self.config = dict(utterances=[dict(audio='train.wav', sha256='audio', tokens=[0,1], split='train'),
                                       dict(audio='val.wav', sha256='heldout', tokens=[0,1], split='validation')])
        self.raw = json.dumps(self.config).encode()
        self.saved = dict(schema_version=1, teacher_profile='ljs-sdp-v1', teacher_sha256='teacher',
                          prepared_sha256=hashlib.sha256(self.raw).hexdigest(),
                          records=[dict(audio='train.wav', audio_sha256='audio',
                                        tokens_sha256=hashlib.sha256(json.dumps([0,1]).encode()).hexdigest(),
                                        durations=[2,3])])

    def test_exact_monotonic_path_and_frame_count(self):
        self.path.write_text(json.dumps(self.saved))
        labels, provenance = load_alignment(self.path, self.raw, self.config)
        self.assertEqual(provenance['teacher_sha256'], 'teacher')
        path = alignment_batch(self.config['utterances'][:1], labels, torch.tensor([2]), torch.tensor([5]))
        self.assertEqual(path.shape, (1,1,5,2))
        self.assertEqual(path[0,0].argmax(1).tolist(), [0,0,1,1,1])
        self.assertEqual(path[0,0].sum(0).tolist(), [2,3])
        with self.assertRaisesRegex(ValueError, 'frame sum'):
            alignment_batch(self.config['utterances'][:1], labels, torch.tensor([2]), torch.tensor([6]))

    def test_rejects_heldout_changed_tokens_and_invalid_labels(self):
        for edit in [lambda r: r['records'][0].update(audio='val.wav'),
                     lambda r: r['records'][0].update(tokens_sha256='changed'),
                     lambda r: r['records'][0].update(durations=[0,5]),
                     lambda r: r['records'][0].update(durations=[True,4]),
                     lambda r: r.update(records=[]),
                     lambda r: r.update(prepared_sha256='changed')]:
            saved = copy.deepcopy(self.saved); edit(saved)
            self.path.write_text(json.dumps(saved))
            with self.assertRaises(ValueError): load_alignment(self.path, self.raw, self.config)


if __name__ == '__main__': unittest.main()
