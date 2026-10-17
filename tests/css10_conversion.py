"""Compare conversion with pinned publisher code, used only as a test reference."""
import argparse
import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import torch
from torch import nn
from torch.package import PackageImporter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))
from css10_adapter import build_model, convert_state
from inspect_css10 import inspect_candidate, load_weights, verify_stream


def reference_model(path, state):
    package = PackageImporter(str(path))
    networks = package.import_module('TTS.tts.layers.vits.networks')
    duration = package.import_module('TTS.tts.layers.vits.stochastic_duration_predictor')
    vocoder = package.import_module('TTS.vocoder.models.hifigan_generator')
    helpers = package.import_module('TTS.tts.utils.helpers')
    model = nn.Module()
    model.text_encoder = networks.TextEncoder(state['text_encoder.emb.weight'].shape[0], 192, 192, 768, 2, 6, 3, 0.1, language_emb_dim=4)
    model.posterior_encoder = networks.PosteriorEncoder(513, 192, 192, 5, 1, 16, cond_channels=256)
    model.flow = networks.ResidualCouplingBlocks(192, 192, 5, 1, 4, cond_channels=256)
    model.duration_predictor = duration.StochasticDurationPredictor(192, 192, 3, 0.5, 4, cond_channels=256, language_emb_dim=4)
    model.waveform_decoder = vocoder.HifiganGenerator(192, 1, '2', [[1, 2], [2, 6], [3, 12]],
        [3, 5, 7], [16, 16, 8], 256, [8, 8, 4], inference_padding=0, cond_channels=256,
        conv_pre_weight_norm=False, conv_post_weight_norm=False, conv_post_bias=False)
    model.emb_g = nn.Embedding(1, 256)
    model.emb_l = nn.Embedding(1, 4)
    model.load_state_dict(state, strict=True)
    model.args = SimpleNamespace(use_speaker_embedding=True, use_language_embedding=True,
                                 use_sdp=True, condition_dp_on_speaker=True)
    model.inference_noise_scale = model.inference_noise_scale_dp = 0.0
    model.length_scale = 1.0
    model.max_inference_len = None
    model._set_cond_input = lambda aux: (aux['speaker_ids'], None, aux['language_ids'])
    model._set_x_lengths = lambda tokens, aux: aux['x_lengths']
    # Execute only the original inference method from the hash-pinned reference,
    # avoiding the publisher's application, trainer and pickle deserialization.
    with zipfile.ZipFile(path) as archive:
        tree = ast.parse(archive.read('model/TTS/tts/models/vits.py').decode())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Vits')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'inference')
    namespace = dict(torch=torch, sequence_mask=helpers.sequence_mask, generate_path=helpers.generate_path)
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<pinned-reference-inference>', 'exec'), namespace)
    return model.eval(), namespace['inference']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('language', choices=['de', 'es', 'fr', 'it', 'pt', 'pl', 'nl'])
    parser.add_argument('--reference', required=True, type=Path, help='pinned model.pt; executes original network code for comparison')
    args = parser.parse_args()
    torch.set_num_threads(2)
    language = args.language
    neon = language not in ['de', 'es']
    family = 'neon' if neon else 'css10'
    source = ROOT / f'voices/{family}-candidates' / language
    catalogue = 'vits-neon-european.json' if neon else 'vits-css10-candidates.json'
    record = json.loads((ROOT / 'assets' / catalogue).read_text())['candidates'][language]
    report = inspect_candidate(language, source, record)
    # Snapshot the verified archive before executing reference code so concurrent
    # path replacement cannot swap the program after verification.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        reference_path = Path(tmp) / 'reference.pt'
        with args.reference.open('rb', buffering=0) as stream:
            verify_stream(stream, record['test_reference'], 'reference.pt')
            reference_path.write_bytes(stream.read())
        with reference_path.open('rb', buffering=0) as stream:
            verify_stream(stream, record['test_reference'], 'reference.pt')
        state = load_weights(source / record['weights'], record['files'][record['weights']])['model']
        converted = convert_state(state, len(report['frontend']['symbols']))
        assert len(converted) == len(state) == 696
        model = build_model(len(report['frontend']['symbols'])).eval(); model.load_state_dict(converted, strict=True)
        reference, infer = reference_model(reference_path, state)
        double_reference, double_infer = reference_model(reference_path, state)
        double_reference = double_reference.double()
        double_model = build_model(len(report['frontend']['symbols'])).double().eval()
        double_model.load_state_dict(converted, strict=True)
        sid = torch.tensor([0]); g = reference.emb_g(sid).unsqueeze(-1)
        lang = reference.emb_l(sid).unsqueeze(-1)
        symbols = report['frontend']['symbols']
        phrases = {'de': ['guten morgen.', 'die straße führt über eine brücke.'], 'es': ['buenos días.', 'el niño está cerca del río.'], 'fr': ['bonjour.', 'le garçon est près du café.'], 'it': ['buongiorno.', 'la città è già sveglia.'], 'pt': ['bom dia.', 'o coração está tranquilo.'], 'pl': ['dzień dobry.', 'łódź płynie przez rzekę.'], 'nl': ['goedemorgen.', 'het meisje fietst naar het café.']}[language]
        results = []
        with torch.no_grad():
            for phrase in phrases:
                ids = [3]
                for c in phrase:
                    ids.extend([symbols.index(c), 3])
                tokens = torch.tensor([ids]); lengths = torch.tensor([len(ids)])
                original_text = reference.text_encoder(tokens, lengths, lang_emb=lang)
                adapted_text = model.enc_p(tokens, lengths)
                for actual, expected in zip(adapted_text, original_text):
                    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
                # Inference skips one duration flow. Check the training objective
                # as well so mappings used only by adaptation are covered.
                mask = original_text[3]
                durations = torch.full_like(mask, 3.0)
                torch.manual_seed(2345)
                expected_loss = reference.duration_predictor(original_text[0], mask, durations, g=g, lang_emb=lang)
                torch.manual_seed(2345)
                actual_loss = model.dp(adapted_text[0], mask, durations, g=g)
                torch.testing.assert_close(actual_loss, expected_loss, atol=2e-3, rtol=2e-5)
                for reverse in [False, True]:
                    x = torch.randn(1, 192, 17); mask = torch.ones(1, 1, 17)
                    torch.testing.assert_close(model.flow(x, mask, g=g, reverse=reverse),
                                               reference.flow(x, mask, g=g, reverse=reverse), atol=2e-5, rtol=2e-5)
                expected = infer(reference, tokens, dict(x_lengths=lengths, speaker_ids=sid, language_ids=sid))['model_outputs']
                actual = model.infer(tokens, lengths, sid=sid, noise_scale=0.0, noise_scale_w=0.0)[0]
                # Reordered equivalent operations accumulate FP32 rounding in
                # the flow/decoder. Bound waveform error, then independently
                # require near-exact FP64 inference from the same source weights.
                error = (actual - expected).abs()
                assert error.max().item() <= 1e-3
                assert error.square().mean().sqrt().item() <= 1e-5
                double_expected = double_infer(double_reference, tokens,
                    dict(x_lengths=lengths, speaker_ids=sid, language_ids=sid))['model_outputs']
                double_actual = double_model.infer(tokens, lengths, sid=sid, noise_scale=0.0, noise_scale_w=0.0)[0]
                torch.testing.assert_close(double_actual, double_expected, atol=1e-9, rtol=1e-9)
                assert actual.numel() > 22050 // 2 and actual.abs().max() > 0.001
                results.append(dict(text=phrase, samples=actual.numel(), max_absolute_error=error.max().item(), rms_error=error.square().mean().sqrt().item(), double_max_absolute_error=(double_actual - double_expected).abs().max().item()))
            spectrum = torch.randn(1, 513, 25); lengths = torch.tensor([25])
            torch.manual_seed(1234); expected = reference.posterior_encoder(spectrum, lengths, g=g)
            torch.manual_seed(1234); actual = model.enc_q(spectrum, lengths, g=g)
            for a, e in zip(actual, expected):
                torch.testing.assert_close(a, e, atol=2e-5, rtol=2e-5)
        path = ROOT / f'target/{family}-{language}-conversion.json'
        path.write_text(json.dumps(dict(language=language, source_tensors=696, converted_tensors=696,
                                      original_inference_parity=True, float64_inference_parity=True, duration_training_objective_parity=True,
                                      samples=results), indent=2) + '\n')
        print(f'{language}: original network and converted inference match: {path}')


if __name__ == '__main__':
    main()
