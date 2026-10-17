"""Local, unprompted ASR diagnostics. A transcript is not a human quality verdict."""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', choices=['tiny.en', 'base.en', 'tiny', 'base'], default='tiny.en')
    parser.add_argument('--language', choices=['en', 'ru', 'uk'], default='en')
    parser.add_argument('--cache', default='target/asr-models')
    parser.add_argument('files', nargs='+')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists(): parser.error('output already exists')
    if args.model.endswith('.en') and args.language != 'en':
        parser.error('Russian/Ukrainian diagnostics require a multilingual model: tiny or base')
    # Optional development dependency; never imported by inference or training.
    import whisper
    torch.set_num_threads(2)
    model = whisper.load_model(args.model, device='cpu', download_root=args.cache)
    model_path = Path(args.cache) / (args.model + '.pt')
    rows = []
    for name in args.files:
        path = Path(name)
        rate, audio = wavfile.read(path)
        if audio.ndim != 1 or audio.dtype != np.int16 or not 1 <= len(audio) <= rate * 30:
            raise ValueError('ASR diagnostic requires mono PCM16 WAV up to 30 seconds')
        samples = audio.astype(np.float32) / 32768
        divisor = math.gcd(rate, 16000)
        samples = resample_poly(samples, 16000 // divisor, rate // divisor).astype(np.float32)
        mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(samples), n_mels=model.dims.n_mels)
        result = whisper.decode(model, mel, whisper.DecodingOptions(
            language=args.language, task='transcribe', temperature=0.0, fp16=False,
            without_timestamps=True))
        row = dict(file=str(path.resolve()), audio_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   text=result.text, no_speech_probability=result.no_speech_prob,
                   average_log_probability=result.avg_logprob, compression_ratio=result.compression_ratio)
        rows.append(row); print(json.dumps(row), flush=True)
    report = dict(model=args.model, language=args.language, model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                  package_version=importlib.metadata.version('openai-whisper'), clips=rows,
                  prompt=None, note='Unprompted local ASR can hallucinate or misrecognize numbers/noise. Compare teacher and negative controls; listening still determines speech acceptance.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream: json.dump(report, stream, indent=2)


if __name__ == '__main__': main()
