"""Reject invalid probe lists before model loading or audio output creation."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
BINARY=ROOT/'target/debug/examples/math_numeric_compare'
@unittest.skipUnless(BINARY.exists(),'build the math_numeric_compare example first')
class ProbeGuards(unittest.TestCase):
    def test_invalid_lists_leave_no_output(self):
        cases=[([], 'expected 1..32'),(['1','1'],'duplicate expression'),(['1000000000'],'integer exceeds'),(['0']*33,'expected 1..32'),(['1 +'],'incomplete expression')]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);full=root/'full';compact=root/'compact';full.mkdir();compact.mkdir()
            (full/'exported-checkpoint.json').write_text(json.dumps({'sha256':'fixture'}))
            (compact/'arithmetic-neural.json').write_text(json.dumps({'source_checkpoint_sha256':'fixture'}))
            for i,(items,message) in enumerate(cases):
                source=root/f'{i}.json';source.write_text(json.dumps(items));output=root/f'output-{i}'
                result=subprocess.run([str(BINARY),'ru',str(full),str(compact),str(output),'--expressions',str(source)],capture_output=True,text=True)
                self.assertNotEqual(result.returncode,0)
                self.assertIn(message,result.stderr)
                self.assertFalse(output.exists())
if __name__=='__main__':unittest.main()
