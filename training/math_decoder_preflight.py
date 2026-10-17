"""Export and verify fixed-speaker decoder components, without a text frontend.

These artifacts are not playable numeric voice packages. Random latent probes
check shapes, dynamic axes and compression error, not speech quality.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import torch
from checkpoint_tools import load, build
from math_cyrillic_export import Decoder
from math_pretrained_compress import compress

BASES = {'de': 'css10', 'es': 'css10', 'fr': 'neon', 'it': 'neon',
         'pt': 'neon', 'pl': 'neon', 'nl': 'neon'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--voices', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists')
    # Validate every input before creating an output directory.
    sources = []
    for language, base in BASES.items():
        checkpoint = args.voices / f'vits-{base}-{language}-base' / 'base.pt'
        saved, digest = load(checkpoint)
        config = saved['training_config']
        expected = 'css10-de-es-v1' if base == 'css10' else 'neon-european-v1'
        if saved['model_profile'] != expected or config['language'] != language or config['sample_rate'] != 22050:
            raise ValueError('source profile/language/rate mismatch')
        sources.append((language, checkpoint, digest))
        del saved
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=False)
    reports = []
    for language, checkpoint, digest in sources:
        torch.manual_seed(1234)
        saved, actual = load(checkpoint)
        if actual != digest:
            raise ValueError('checkpoint changed since validation')
        net = build(saved).eval()
        net.dec.remove_weight_norm()
        decoder = Decoder(net, saved['training_config']['conditioning_speaker'], False).eval()
        folder = args.output / language
        folder.mkdir()
        with torch.no_grad():
            torch.onnx.export(decoder, torch.zeros(1, 192, 32), folder / 'decoder.onnx',
                              input_names=['latent'], output_names=['audio'],
                              dynamic_axes={'latent': {2: 'frames'}, 'audio': {2: 'samples'}},
                              opset_version=17, dynamo=False)
        model = onnx.load(folder / 'decoder.onnx')
        tensors = compress(model, 8)
        onnx.checker.check_model(model)
        onnx.save(model, folder / 'decoder.onnx')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        session = ort.InferenceSession(str(folder / 'decoder.onnx'), options,
                                       providers=['CPUExecutionProvider'])
        probes = []
        with torch.no_grad():
            for frames in (17, 32, 73):
                latent = torch.randn(1, 192, frames) * 0.2
                expected = decoder(latent)[0, 0].numpy()
                audio = session.run(None, {'latent': latent.numpy()})[0].reshape(-1)
                if audio.shape != (frames * 256,) or not np.isfinite(audio).all():
                    raise ValueError('invalid exported waveform shape or samples')
                probes.append(dict(frames=frames, samples=len(audio),
                                   max_abs_error=float(np.max(np.abs(audio - expected))),
                                   rms_error=float(np.sqrt(np.mean((audio - expected) ** 2)))))
        report = dict(language=language, source_checkpoint_sha256=digest,
                      decoder_bytes=(folder / 'decoder.onnx').stat().st_size,
                      stored_weight_bits=8, quantized_tensors=tensors, probes=probes,
                      status='decoder component only; no numeric conditioning or frontend',
                      evaluation='random latent probes; not a speech-quality evaluation')
        (folder / 'report.json').write_text(json.dumps(report, indent=2))
        (folder / 'training-origin.json').write_text(json.dumps(saved['initialization'], indent=2))
        (folder / 'rights.json').write_text(json.dumps(saved['training_config']['rights'], indent=2))
        for index, notice in enumerate(saved['initialization'].get('notices', {}).values()):
            (folder / f'BASE-NOTICE-{index:02}.txt').write_text(notice)
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in folder.iterdir() if p.is_file()}
        (folder / 'checksums.json').write_text(json.dumps(hashes, indent=2))
        reports.append(report)
        print(json.dumps(dict(language=language, decoder_bytes=report['decoder_bytes'],
                              dynamic_shapes_passed=len(probes))), flush=True)
        del decoder, net, saved, session
    (args.output / 'report.json').write_text(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
