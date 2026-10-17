"""Evaluation snapshots remain immutable and are not resumable training checkpoints."""
from pathlib import Path
import subprocess
import sys
import tempfile
import shutil
import torch
root = Path(__file__).resolve().parents[1]
snapshot = root/'checkpoints/math-student-en-v2/evaluation-early.pt'
saved = torch.load(snapshot, map_location='cpu', weights_only=True)
assert saved['evaluation_snapshot']['resumable'] is False
assert 'optimizer' not in saved and 'discriminator' not in saved
assert snapshot.stat().st_size < 4_000_000
command = [sys.executable,str(root/'training/math_snapshot.py'),'--checkpoint',str(snapshot),'--output',str(snapshot)]
result = subprocess.run(command, capture_output=True, text=True)
assert result.returncode != 0 and 'output already exists' in result.stderr
with tempfile.TemporaryDirectory(prefix='math-snapshot-test-') as temp:
    shutil.copy2(snapshot,Path(temp)/'latest.pt')
    result = subprocess.run([sys.executable,str(root/'training/train.py'),
        '--prepared',str(root/'checkpoints/math-student-en-v2/prepared.json'),
        '--output',temp,'--resume','--math-student','--steps','1'],capture_output=True,text=True)
    assert result.returncode != 0 and 'evaluation snapshots cannot resume' in result.stderr
print('Snapshot immutability and resume rejection passed')
