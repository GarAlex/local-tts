"""Pinned Vosk MB-iSTFT VITS2 architecture and deterministic export conversion."""
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from vendor.vosk.models import SynthesizerTrn, DurationDiscriminator2
from vendor.vosk.stft import OnnxSTFT

ROOT = Path(__file__).resolve().parents[1]
PROFILE = 'vosk-ru-multi-v1'
FRONTEND = 'vosk-ru-phones-v1'
MODEL = json.loads((ROOT/'assets/vosk-ru-training-model.json').read_text())

def symbols():
    return ['\0' if i == 0 else ' ' if i == 3 else chr(0xe000+i) for i in range(62)]

def build_model():
    return SynthesizerTrn(62, 80, 32, **MODEL)

class NormalizedOnnxSTFT(OnnxSTFT):
    """Use real-valued ONNX operators with torch.istft's overlap normalization."""
    def __init__(self):
        super().__init__(16, 4, 16)
        self.register_buffer('window_squared', torch.hann_window(16).square().reshape(1,1,16))

    def inverse(self, magnitude, phase):
        combined = torch.cat([magnitude*torch.cos(phase), magnitude*torch.sin(phase)], dim=1)
        audio = F.conv_transpose1d(combined, self.inverse_basis, stride=4) * 4
        envelope = F.conv_transpose1d(torch.ones_like(magnitude[:, :1, :]), self.window_squared, stride=4)
        return audio[:,:,8:-8] / envelope[:,:,8:-8].clamp_min(1e-8)

def prepare_export(net):
    # Source checkpoint has no STFT tensors. Replace only this derived operation;
    # all learned parameters remain intact. Training retains complex torch.istft.
    net.dec.stft = NormalizedOnnxSTFT().to(next(net.parameters()).device)

def duration_discriminator():
    return DurationDiscriminator2(192, 256, 3, 0.1, gin_channels=256)

def subband_loss(actual, predicted):
    """Publisher-config multi-resolution spectral convergence and log magnitude."""
    loss = actual.new_zeros(())
    for fft, hop, win in [(384,30,150), (683,60,300), (171,10,60)]:
        window = torch.hann_window(win, device=actual.device)
        def spectrum(audio):
            value = torch.stft(audio.reshape(-1,audio.shape[-1]), fft, hop, win,
                               window=window, return_complex=True)
            return value.abs().clamp_min(1e-7)
        real, fake = spectrum(actual), spectrum(predicted)
        loss = loss + torch.linalg.vector_norm(real-fake)/torch.linalg.vector_norm(real).clamp_min(1e-7)
        loss = loss + F.l1_loss(real.log(),fake.log())
    return loss/3
