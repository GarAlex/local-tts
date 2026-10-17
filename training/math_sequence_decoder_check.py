"""Controlled validation diagnostic: identical latents through float and packed decoders.

Does not train, alter model weights, or consume reserved test utterances.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch
from scipy.io import wavfile
from checkpoint_tools import load, build
from math_cyrillic_export import Decoder
from math_sequence_evaluate import spectral_error
from math_sequence_export import features
from math_sequence_student import SequenceStudent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_package(root):
    hashes = json.loads((root / 'checksums.json').read_text())
    for name in ['sequence.json', 'encoder.onnx', 'latent.onnx', 'decoder.onnx']:
        if name not in hashes:
            raise ValueError('missing package checksum')
    for name, digest in hashes.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path) != digest:
            raise ValueError('package integrity differs')
    return json.loads((root / 'sequence.json').read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['student', 'teacher', 'dataset', 'package', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('output already exists')
    torch.set_num_threads(2)
    package = validate_package(args.package)
    checkpoint = torch.load(args.student, weights_only=True, map_location='cpu')
    manifest_path = args.dataset / 'dataset.json'
    manifest = json.loads(manifest_path.read_text())
    saved, teacher_sha = load(str(args.teacher))
    if (checkpoint['dataset_sha256'] != sha(manifest_path)
        or teacher_sha != manifest['teacher_checkpoint_sha256']
        or teacher_sha != checkpoint['teacher_checkpoint_sha256']
        or teacher_sha != package['source_checkpoint_sha256']
        or sha(args.student) != package['student_sha256']
        or manifest['language'] != package['language']):
        raise ValueError('experiment lineage differs')
    language = manifest['language']
    net = build(saved).eval()
    net.requires_grad_(False)
    if language == 'ru':
        from vosk_adapter import prepare_export
        prepare_export(net)
    elif language != 'uk':
        raise ValueError('unsupported language')
    decoder = Decoder(net, saved['training_config']['conditioning_speaker'], language == 'ru').eval()
    decoder.requires_grad_(False)
    student = SequenceStudent(checkpoint['vocabulary']).eval()
    student.load_state_dict(checkpoint['model'])
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    sessions = {name: ort.InferenceSession(str(args.package / f'{name}.onnx'), options,
                providers=['CPUExecutionProvider']) for name in ['encoder', 'latent', 'decoder']}
    wanted = ['785 + 17 = 802', '14 × 7', '6.56 + 7.17', '0.009 × 164']
    rows = [next(r for r in manifest['records'] if r['split'] == 'validation' and r['expression'] == text)
            for text in wanted]
    args.output.mkdir(parents=True, exist_ok=False)
    cases = []
    with torch.no_grad():
        for i, row in enumerate(rows):
            path = (args.dataset / row['file']).resolve()
            if not path.is_relative_to(args.dataset.resolve()) or sha(path) != row['sha256']:
                raise ValueError('target integrity differs')
            data = torch.load(path, weights_only=True, map_location='cpu')
            ids = np.asarray([row['tokens']], dtype=np.int64)
            if ids[0].tolist() != data['tokens'].tolist():
                raise ValueError('target tokens differ')
            expanded = features(*sessions['encoder'].run(None, {'tokens': ids}), 1.0)
            predicted = sessions['latent'].run(None, {'features': expanded})[0]
            truth = data['latent'][None].numpy()
            fitted, _, _ = student(torch.from_numpy(ids), torch.tensor([ids.shape[1]]), data['durations'][None])
            forced = (fitted * checkpoint['scale'][None] + checkpoint['mean'][None]).numpy()
            recordings = []
            decoded = {}
            for name, z, packed in [('teacher-float', truth, False), ('teacher-packed', truth, True),
                                    ('student-float', predicted, False), ('student-packed', predicted, True),
                                    ('student-teacher-timing-float', forced, False)]:
                value = sessions['decoder'].run(None, {'latent': z})[0] if packed else decoder(torch.from_numpy(z)).numpy()
                audio = np.asarray(value, dtype=np.float32).reshape(-1)
                if audio.size != z.shape[-1] * 256 or not np.isfinite(audio).all():
                    raise ValueError('invalid decoded audio')
                decoded[name] = torch.from_numpy(audio)
                file = args.output / f'{i:02}-{name}.wav'
                wavfile.write(file, 22050, np.clip(np.rint(audio * 32767), -32768, 32767).astype('<i2'))
                recordings.append(dict(kind=name, file=file.name, sha256=sha(file),
                    frames=z.shape[-1], latent_sha256=hashlib.sha256(z.astype('<f4').tobytes()).hexdigest(),
                    peak=float(np.abs(audio).max()), clipped_samples=int((np.abs(audio)>1).sum())))
            metrics = {name + '_packed_log_spectral_error': spectral_error(decoded[name + '-packed'], decoded[name + '-float'])
                       for name in ['teacher', 'student']}
            metrics['student_teacher_timing_log_spectral_error'] = spectral_error(decoded['student-teacher-timing-float'], decoded['teacher-float'])
            cases.append(dict(expression=row['expression'], spoken=row['spoken'], recordings=recordings, metrics=metrics))
            print(json.dumps(dict(case=i, expression=row['expression'], metrics=metrics)), flush=True)
    report = dict(language=language, student_sha256=sha(args.student), teacher_sha256=teacher_sha,
        dataset_sha256=sha(manifest_path), package_checksums_sha256=sha(args.package / 'checksums.json'),
        tool_sha256=sha(Path(__file__)), decoder_bits=package['decoder_bits'],
        split='validation', test_split_used=False, decoder_updated=bool(package.get('decoder_adaptation')),
        decoder_adaptation=package.get('decoder_adaptation'),
        design='identical native predicted latents through float and packed decoders; identical teacher latents as control; teacher timing isolates duration errors',
        cases=cases, status='diagnostic only; no student listening acceptance')
    (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
