"""Export a locally trained VITS checkpoint to the native ONNX contract."""
import argparse
import json
import hashlib
import math
import shutil
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent / 'vendor/vits'))
import torch
import onnx
from checkpoint_tools import load, build
from models import SynthesizerTrn

class FixedLatentNoise(torch.nn.Module):
    """Repeatable Gaussian samples for dynamic-length ONNX inference/parity.

    The 4096-frame period is about 47.6 seconds at the current hop/rate.
    This avoids a backend-specific random operator while retaining latent variance.
    """
    def __init__(self, channels):
        super().__init__()
        generator = torch.Generator().manual_seed(1234)
        self.register_buffer('bank', torch.randn(1, channels, 4096, generator=generator))

    def forward(self, reference):
        indices = torch.arange(reference.shape[2], device=reference.device) % 4096
        return torch.index_select(self.bank, 2, indices)

class Export(torch.nn.Module):
    def __init__(self, model, speaker_index=None, latent_noise_scale=0.0):
        super().__init__(); self.model = model; self.speaker_index = speaker_index
        self.latent_noise_scale = latent_noise_scale
        self.noise = FixedLatentNoise(model.inter_channels) if latent_noise_scale else None
    def forward(self, tokens, lengths, duration_scale):
        sid = None if self.speaker_index is None else torch.full_like(lengths, self.speaker_index)
        if self.noise is None:
            return self.model.infer(tokens, lengths, sid=sid, noise_scale=0.0,
                                    length_scale=duration_scale[0], noise_scale_w=0.0)[0]
        return self.model.infer(tokens, lengths, sid=sid, noise_scale=self.latent_noise_scale,
                                length_scale=duration_scale[0], noise_scale_w=0.0,
                                noise_source=self.noise)[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--expected-sha256')
    parser.add_argument('--new-output', action='store_true')
    parser.add_argument('--latent-noise-scale', type=float, default=0.0,
                        help='experimental math student sampled latent inference (0..1)')
    args = parser.parse_args()
    torch.set_num_threads(2)
    checkpoint, checkpoint_sha256 = load(args.checkpoint, args.expected_sha256)
    if not math.isfinite(args.latent_noise_scale) or not 0 <= args.latent_noise_scale <= 1:
        parser.error('latent noise scale must be finite and in [0,1]')
    if args.latent_noise_scale and checkpoint.get('model_profile') not in ('math-student-v1', 'math-phoneme-v1', 'math-phoneme-wide-v1'):
        parser.error('sampled export is currently restricted to math student profiles')
    if args.latent_noise_scale and checkpoint.get('preferred_latent_noise_scale') == 0.0:
        parser.error('this staged student was trained on posterior means; export with latent noise scale 0')
    config = checkpoint['training_config']
    model = build(checkpoint)
    if checkpoint.get('model_profile') == 'vosk-ru-multi-v1':
        from vosk_adapter import prepare_export
        prepare_export(model)
    model.dec.remove_weight_norm()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=not args.new_output)
    tokens = torch.tensor([[0, 1, 0, 1, 0]], dtype=torch.long)
    if config.get('frontend') in ['css10-graphemes-v1', 'neon-graphemes-v1']:
        tokens = torch.tensor([[3, 4, 3, 5, 3]], dtype=torch.long)
    lengths = torch.tensor([5], dtype=torch.long)
    scale = torch.tensor([1.0])
    temp = output / 'model.partial.onnx'
    wrapper = Export(model, config.get('conditioning_speaker'), args.latent_noise_scale).eval()
    with torch.no_grad():
        torch.onnx.export(wrapper, (tokens, lengths, scale), temp,
                          input_names=['tokens', 'lengths', 'duration_scale'], output_names=['audio'],
                          dynamic_axes={'tokens': {1: 'text_length'}, 'audio': {2: 'audio_length'}},
                          opset_version=17, dynamo=False)
    onnx.checker.check_model(onnx.load(temp))
    temp.replace(output / 'model.onnx')
    (output / 'tokens.json').write_text(json.dumps(dict(schema_version=1, symbols=config['symbols']), ensure_ascii=False))
    origin = checkpoint.get('initialization', dict(mode='from-scratch'))
    (output / 'exported-checkpoint.json').write_text(json.dumps(dict(checkpoint=str(Path(args.checkpoint).resolve()), sha256=checkpoint_sha256, step=checkpoint['step'], model_profile=checkpoint.get('model_profile', 'project-v1'), evaluation_snapshot=checkpoint.get('evaluation_snapshot'), math_prior_mel_weight=checkpoint.get('math_prior_mel_weight', 0.0), teacher_alignment=checkpoint.get('teacher_alignment'), acoustic_fit_steps=checkpoint.get('acoustic_fit_steps'), prior_fit_steps=checkpoint.get('prior_fit_steps'), training_objective=checkpoint.get('training_objective'), latent_noise_scale=args.latent_noise_scale, latent_noise_source=dict(kind='fixed-gaussian-bank', seed=1234, frames=4096) if args.latent_noise_scale else None), indent=2))
    (output / 'training-origin.json').write_text(json.dumps(origin, indent=2, ensure_ascii=False))
    manifest = dict(schema_version=1, name=Path(args.checkpoint).parent.name, language=config['language'],
                    sample_rate=config['sample_rate'], model='model.onnx', frontend=config.get('frontend', 'grapheme-v1'),
                    speakers=[config['speaker']], license='Custom weights — verify target data and base-model terms in training-origin.json',
                    provenance=f"Training mode {origin['mode']}; step {checkpoint['step']}; {checkpoint['source_revision']}; dataset {checkpoint['dataset_digest']}")
    if config.get('normalization') is not None:
        manifest['normalization'] = config['normalization']
    (output / 'voice.json').write_text(json.dumps(manifest, indent=2))
    (output / 'rights.json').write_text(json.dumps(config['rights'], indent=2))
    # Numerical fixture is consumed by the Rust ONNX verification command.
    extra_files = []
    if checkpoint.get('teacher_alignment'):
        alignment = checkpoint['teacher_alignment']
        raw = Path(alignment['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != alignment['sha256']:
            raise ValueError('teacher alignment cache changed before export')
        source = json.loads(raw)
        if source['teacher_sha256'] != alignment['teacher_sha256']:
            raise ValueError('alignment teacher provenance differs')
        provenance = dict(teacher_sha256=source['teacher_sha256'],
                          teacher_profile=source['teacher_profile'],
                          teacher_initialization=source.get('teacher_initialization'),
                          prepared_sha256=source['prepared_sha256'],
                          alignment_sha256=alignment['sha256'])
        (output / 'ALIGNMENT-TEACHER-provenance.json').write_text(json.dumps(provenance, indent=2))
        extra_files.append('ALIGNMENT-TEACHER-provenance.json')
        teacher_origin = source.get('teacher_initialization') or {}
        for name, content in teacher_origin.get('notices', {}).items():
            if Path(name).name != name: raise ValueError('unsafe alignment teacher notice path')
            target = 'ALIGNMENT-TEACHER-' + name
            (output / target).write_text(content)
            extra_files.append(target)
    if config.get('frontend') == 'cmu-ipa-v1':
        dictionary = config['dictionary']
        path = Path(dictionary['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != dictionary['sha256']:
            raise ValueError('frontend dictionary checksum mismatch')
        shutil.copyfile(path, output / 'cmudict.dict')
        root = Path(__file__).resolve().parents[1]
        notice = root / 'assets/licenses/CMUDICT.txt'
        records = json.loads((root / 'assets/english-assets.json').read_text())
        if hashlib.sha256(notice.read_bytes()).hexdigest() != records['CMUDICT-LICENSE.txt']['sha256']:
            raise ValueError('CMU license checksum mismatch')
        shutil.copyfile(notice, output / 'CMU-LICENSE.txt')
        extra_files += ['cmudict.dict', 'CMU-LICENSE.txt']
    if config.get('frontend') == 'vosk-ru-phones-v1':
        dictionary = config['dictionary']
        if hashlib.sha256(Path(dictionary['path']).read_bytes()).hexdigest() != dictionary['sha256']:
            raise ValueError('Russian dictionary checksum mismatch')
        shutil.copyfile(dictionary['path'], output/'dictionary')
        extra_files.append('dictionary')
    # Imported base notices survive every adaptation and subsequent export.
    lineage = origin
    while lineage:
        if lineage.get('mode') == 'public-import':
            for name, content in lineage['notices'].items():
                if Path(name).name != name: raise ValueError('unsafe base notice path')
                target = 'BASE-' + name
                (output / target).write_text(content)
                extra_files.append(target)
            break
        lineage = lineage.get('base_initialization')
    references = []
    cases = [([0, 1, 0], 0.8), ([0, 1, 0, 1, 0], 1.0), ([0, 1, 0, 1, 0, 1, 0], 1.3)]
    if config.get('frontend') == 'cmu-ipa-v1':
        indices = {c: i for i, c in enumerate(config['symbols'])}
        def encode(text):
            return [0] + [v for c in text for v in [indices[c], 0]]
        cases = [(encode(text), scale) for text, scale in [('həlˈoʊ', 0.8), ('həlˈoʊ wˈɜɹld', 1.0), ('ðə kˈæt sˈæt.', 1.3)]]
    if config.get('frontend') == 'piper-uk-text-v1':
        indices = {c: i for i,c in enumerate(config['symbols'])}
        import unicodedata
        def encode_piper(text):
            return [1,0] + [v for c in unicodedata.normalize('NFD',text) for v in [indices[c],0]] + [2]
        cases = [(encode_piper(text),scale) for text,scale in [('приві́т',0.8),('приві́т, сві́те!',1.0),('це украї́нська мо́ва.',1.3)]]
    if config.get('frontend') == 'vosk-ru-phones-v1':
        # Original phone IDs for explicitly stressed Russian words and punctuation.
        def encode_vosk(phones): return [1,0]+[v for phone in phones for v in [phone,0]]+[2]
        cases = [(encode_vosk(phones),scale) for phones,scale in [([43,46,31,16,23,51],0.8),([37,31,45,10],1.0),([14,15,22,23,41,42],1.3)]]
    if config.get('frontend') in ['css10-graphemes-v1', 'neon-graphemes-v1']:
        indices = {c: i for i, c in enumerate(config['symbols'])}
        phrases = {
            'de': ['hallo', 'guten morgen.', 'grüße aus köln.'],
            'es': ['hola', 'buenos días.', 'el niño está aquí.'],
            'fr': ['bonjour', 'la forêt.', 'le garçon est près du café.'],
            'it': ['ciao', 'buongiorno.', 'la città è già sveglia.'],
            'pt': ['olá', 'bom dia.', 'o coração está tranquilo.'],
            'pl': ['cześć', 'dzień dobry.', 'łódź płynie przez rzekę.'],
            'nl': ['hallo', 'goedemorgen.', 'het meisje fietst naar huis.'],
        }[config['language']]
        cases = [([3] + [v for c in phrase for v in [indices[c], 3]], scale)
                 for phrase, scale in zip(phrases, [0.8, 1.0, 1.3])]
    with torch.no_grad():
        for ids, duration in cases:
            reference = wrapper(torch.tensor([ids]), torch.tensor([len(ids)]), torch.tensor([duration])).numpy().reshape(-1)
            references.append(dict(tokens=ids, duration_scale=duration, samples=reference.tolist()))
    (output / 'reference.json').write_text(json.dumps(references))
    (output / 'VITS-LICENSE.txt').write_text((Path(__file__).parent / 'vendor/vits/LICENSE').read_text())
    hashes = {}
    for name in ['model.onnx', 'tokens.json', 'voice.json', 'rights.json', 'reference.json', 'VITS-LICENSE.txt', 'training-origin.json', 'exported-checkpoint.json'] + extra_files:
        digest = hashlib.sha256()
        with (output / name).open('rb') as stream:
            while chunk := stream.read(65536):
                digest.update(chunk)
        hashes[name] = digest.hexdigest()
    (output / 'checksums.json').write_text(json.dumps(hashes, indent=2))
    print(f'Exported {output}. Run local-tts vits-verify {output} before distribution.', flush=True)

if __name__ == '__main__': main()
