"""ASR language guards must fail before loading/downloading a model."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class LanguageGuards(unittest.TestCase):
    def test_english_only_models_reject_cyrillic_before_model_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            for language in ('ru','uk'):
                for model in ('tiny.en','base.en'):
                    output=Path(tmp)/f'{language}-{model}.json'
                    cache=Path(tmp)/'cache'
                    result=subprocess.run([sys.executable,str(ROOT/'training/math_asr_check.py'),'--model',model,'--language',language,'--cache',str(cache),'--output',str(output),'missing.wav'],capture_output=True,text=True)
                    self.assertEqual(result.returncode,2)
                    self.assertIn('require a multilingual model',result.stderr)
                    self.assertFalse(output.exists())
                    self.assertFalse(cache.exists())
if __name__=='__main__':unittest.main()
