"""Offline listening artifact integrity and audit checks."""
import copy
from html.parser import HTMLParser
import json
from pathlib import Path
import sys
import tempfile
import unittest
from urllib.parse import unquote
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
from math_listening import build


class AudioParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.sources = []
    def handle_starttag(self, tag, attrs):
        if tag == 'audio':
            self.sources.append(dict(attrs)['src'])


class Listening(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paths = [self.root / name / 'comparison.json' for name in ('old', 'new')]
        self.report = dict(split='validation', dataset_sha256='dataset',
                           audited_training_lineage=[dict(dataset_sha256='training')],
                           examples=[dict(id='a', expression='1 < 2', spoken='one </script>',
                                          teacher_audio_sha256='audio',
                                          files={name: f'{name}.wav' for name in ('student', 'teacher', 'fragments')})])
        for path in self.paths:
            path.parent.mkdir()
            path.write_text(json.dumps(self.report))
            for name in ('student', 'teacher', 'fragments'):
                (path.parent / f'{name}.wav').write_bytes(b'test fixture')

    def test_safe_html_audio_paths_and_no_overwrite(self):
        output = self.root / 'review.html'
        self.assertEqual(build(*self.paths, output), 1)
        text = output.read_text()
        self.assertIn('one &lt;/script&gt;', text)
        self.assertIn('one \\u003c/script>', text)
        parser = AudioParser()
        parser.feed(text)
        self.assertEqual(len(parser.sources), 4)
        for source in parser.sources:
            self.assertTrue((output.parent / unquote(source)).is_file())
        with self.assertRaisesRegex(ValueError, 'already exists'):
            build(*self.paths, output)

    def test_requires_matching_audited_validation_and_safe_files(self):
        for edit, expected in [
            (lambda r: r.update(split='test'), 'validation'),
            (lambda r: r.update(audited_training_lineage=[]), 'auditing'),
            (lambda r: r.update(dataset_sha256='other'), 'same dataset'),
            (lambda r: r['examples'][0]['files'].update(student='../outside.wav'), 'audio path'),
        ]:
            report = copy.deepcopy(self.report)
            edit(report)
            self.paths[1].write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, expected):
                build(*self.paths, self.root / 'review.html')
            self.assertFalse((self.root / 'review.html').exists())


if __name__ == '__main__':
    unittest.main()
