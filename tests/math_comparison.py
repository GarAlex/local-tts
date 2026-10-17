"""Exercise held-out audit guards using a generated corpus; no model mutation."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
root = Path(__file__).resolve().parents[1]
dataset = json.loads((root/'datasets/math-teacher-en-v2/dataset.json').read_text())
with tempfile.TemporaryDirectory(prefix='math-lineage-') as temp:
    folder = Path(temp)
    earlier = copy.deepcopy(dataset)
    for item in earlier['utterances']:
        if item['split'] == 'test':
            item['split'] = 'train'
            break
    (folder/'dataset.json').write_text(json.dumps(earlier))
    command = [str(root/'target/release/examples/math_student_compare'),
               str(root/'datasets/math-teacher-en-v2'),str(root/'voices/math-en-v1'),
               str(root/'checkpoints/math-student-en-v2/voice-step200'),str(folder/'output'),str(folder)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0 and 'overlaps training/validation lineage' in result.stderr
    assert not (folder/'output').exists()
print('Held-out lineage overlap rejected before output/model loading')
