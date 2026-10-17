"""Staged checkpoints retain objective metadata and reject incompatible exports."""
from pathlib import Path
import subprocess
import sys
import tempfile
import torch

root=Path(__file__).resolve().parents[1]
source=root/'checkpoints/math-phoneme-wide-en-v1/prior-fit2000/prior.pt'
with tempfile.TemporaryDirectory(prefix='math-stage-test-') as temporary:
    folder=Path(temporary)
    snapshot=folder/'snapshot.pt'
    subprocess.run([sys.executable,str(root/'training/math_snapshot.py'),
                    '--checkpoint',str(source),'--output',str(snapshot)],check=True,capture_output=True,text=True)
    saved=torch.load(snapshot,map_location='cpu',weights_only=True)
    assert saved['model_profile']=='math-phoneme-wide-v1'
    assert saved['training_objective']=='fixed-acoustic-mean-prior-fit'
    assert saved['preferred_latent_noise_scale']==0
    assert saved['acoustic_fit_steps']==3000 and saved['prior_fit_steps']==2000
    assert saved['teacher_alignment']['sha256']
    assert 'optimizer' not in saved and 'discriminator' not in saved
    result=subprocess.run([sys.executable,str(root/'training/export.py'),
                           '--checkpoint',str(snapshot),'--output',str(folder/'wrong'),
                           '--latent-noise-scale','0.667','--new-output'],capture_output=True,text=True)
    assert result.returncode!=0 and 'trained on posterior means' in result.stderr
    assert not (folder/'wrong').exists()
print('Staged metadata survives snapshots; incompatible sampled export rejected')
