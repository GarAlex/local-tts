"""Single-speaker VITS fine-tuning and explicit scratch training; native token contract."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import signal
import sys

sys.path.insert(0, str(Path(__file__).parent / 'vendor' / 'vits'))
import numpy as np
import torch
from torch.nn import functional as F
from scipy.io import wavfile
from librosa.filters import mel as mel_filter
from models import SynthesizerTrn, MultiPeriodDiscriminator
from losses import discriminator_loss, generator_loss, feature_loss, kl_loss
import commons

MODEL = dict(inter_channels=192, hidden_channels=192, filter_channels=768,
             n_heads=2, n_layers=6, kernel_size=3, p_dropout=0.1,
             resblock='1', resblock_kernel_sizes=[3, 7, 11],
             resblock_dilation_sizes=[[1, 3, 5]] * 3, upsample_rates=[8, 8, 2, 2],
             upsample_initial_channel=512, upsample_kernel_sizes=[16, 16, 4, 4],
             n_layers_q=3, use_spectral_norm=False, use_sdp=False)

# Experimental student, deliberately separate from --tiny engineering fixtures.
MATH_MODEL = dict(MODEL, inter_channels=32, hidden_channels=32,
                  filter_channels=64, n_layers=2, upsample_initial_channel=64,
                  resblock='2', resblock_kernel_sizes=[3],
                  resblock_dilation_sizes=[[1, 2]])

MATH_WIDE_MODEL = dict(MATH_MODEL, upsample_initial_channel=256)
MATH_PROFILES = ('math-student-v1', 'math-phoneme-v1', 'math-phoneme-wide-v1')

PIPER_MODEL = dict(MODEL, resblock='2', resblock_kernel_sizes=[3,5,7],
                   resblock_dilation_sizes=[[1,2],[2,6],[3,12]],
                   upsample_rates=[8,8,4], upsample_initial_channel=256,
                   upsample_kernel_sizes=[16,16,8], n_speakers=3, gin_channels=512, use_sdp=True)

def conditioning_ids(config, x):
    sid = config.get('conditioning_speaker')
    return None if sid is None else torch.full((x.shape[0],), sid, dtype=torch.long, device=x.device)

class Acoustics:
    def __init__(self, device):
        self.window = torch.hann_window(1024, device=device)
        self.mel = torch.from_numpy(mel_filter(sr=22050, n_fft=1024, n_mels=80)).to(device)

    def spec(self, audio):
        audio = F.pad(audio.unsqueeze(1), (384, 384), mode='reflect').squeeze(1)
        spectrum = torch.stft(audio, 1024, hop_length=256, win_length=1024,
                              window=self.window, center=False, return_complex=True)
        return torch.sqrt(spectrum.real.square() + spectrum.imag.square() + 1e-6)

    def to_mel(self, spec):
        return torch.log(torch.clamp(self.mel @ spec, min=1e-5))


def load_batch(records, device, acoustics):
    items = []
    for record in records:
        rate, raw = wavfile.read(record['audio'])
        if rate != 22050 or raw.ndim != 1:
            raise ValueError('expected 22050 Hz mono WAV')
        if np.issubdtype(raw.dtype, np.integer):
            if raw.dtype == np.uint8:
                raw = (raw.astype(np.float32) - 128) / 128
            else:
                raw = raw.astype(np.float32) / (2 ** (np.iinfo(raw.dtype).bits - 1))
        audio = torch.tensor(raw.copy(), dtype=torch.float32, device=device)
        if len(audio) < 8192 or len(audio) > 22050 * 20:
            raise ValueError('training clips must be between 8192 samples and twenty seconds')
        tokens = torch.tensor(record['tokens'], dtype=torch.long, device=device)
        spec = acoustics.spec(audio[None])[0]
        if spec.shape[-1] < len(tokens):
            raise ValueError('too many tokens for audio length')
        items.append((tokens, spec, audio))
    b = len(items)
    x = torch.zeros(b, max(len(i[0]) for i in items), dtype=torch.long, device=device)
    spec = torch.zeros(b, 513, max(i[1].shape[-1] for i in items), device=device)
    y = torch.zeros(b, 1, max(len(i[2]) for i in items), device=device)
    xl, sl = [], []
    for n, (tokens, spectrum, audio) in enumerate(items):
        x[n, :len(tokens)] = tokens
        spec[n, :, :spectrum.shape[-1]] = spectrum
        y[n, 0, :len(audio)] = audio
        xl.append(len(tokens)); sl.append(spectrum.shape[-1])
    return x, torch.tensor(xl, device=device), spec, torch.tensor(sl, device=device), y


def math_prior_mel_loss(net, m_p, z_mask, slices, target_mel, acoustics):
    """Supervise the aligned mean text prior used by deterministic inference."""
    prior_z = net.flow(m_p, z_mask, reverse=True)
    prior_segment = commons.slice_segments(prior_z, slices, 32)
    prior_audio = net.dec(prior_segment)
    return F.l1_loss(target_mel, acoustics.to_mel(acoustics.spec(prior_audio.squeeze(1))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=10000, help='total target step count, including resumed steps')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--init-checkpoint', help='trusted local project VITS checkpoint for a new voice')
    mode.add_argument('--resume', action='store_true')
    mode.add_argument('--from-scratch', action='store_true')
    parser.add_argument('--learning-rate', type=float, help='default: 5e-5 for adaptation, 2e-4 for scratch')
    parser.add_argument('--tiny', action='store_true', help='engineering smoke tests only; not a production voice')
    parser.add_argument('--math-student', action='store_true', help='experimental compact English arithmetic student')
    parser.add_argument('--math-prior-mel-weight', type=float, default=None,
                        help='experimental direct supervision of aligned math text-prior audio')
    parser.add_argument('--teacher-alignment', help='training-only phoneme teacher duration labels')
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        parser.error('steps and batch-size must be positive')
    if args.learning_rate is not None and (not math.isfinite(args.learning_rate) or not 0 < args.learning_rate <= 0.01):
        parser.error('learning-rate must be finite and in (0, 0.01]')
    torch.set_num_threads(2)
    torch.manual_seed(1234); random.seed(1234); np.random.seed(1234)
    raw_config = Path(args.prepared).read_bytes()
    config = json.loads(raw_config)
    if config.get('normalization') not in [None, 'text-v1']:
        raise ValueError('unsupported text normalization version')
    if config.get('normalization') is not None and config.get('frontend', 'grapheme-v1') == 'grapheme-v1':
        raise ValueError('legacy grapheme frontend cannot use text-v1 normalization')
    digest = hashlib.sha256(raw_config).hexdigest()
    records = [u for u in config['utterances'] if u['split'] == 'train']
    validation = [u for u in config['utterances'] if u['split'] == 'validation']
    if not records or not validation:
        raise ValueError('train and validation splits are required')
    for record in config['utterances']:
        if hashlib.sha256(Path(record['audio']).read_bytes()).hexdigest() != record['sha256']:
            raise ValueError(f"recording changed after preparation: {record['audio']}")
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / 'latest.pt'
    if checkpoint.exists() and not args.resume:
        raise ValueError('output already contains a checkpoint; select Resume or a new directory')
    device = torch.device(args.device)
    if device.type == 'mps':
        if not torch.backends.mps.is_available(): raise ValueError('MPS is unavailable on this machine')
        torch.mps.manual_seed(1234)
    base = saved = None
    base_digest = None
    if args.init_checkpoint:
        with Path(args.init_checkpoint).open('rb') as stream:
            digestor = hashlib.sha256()
            while chunk := stream.read(65536): digestor.update(chunk)
            base_digest = digestor.hexdigest()
            stream.seek(0)
            base = torch.load(stream, map_location='cpu', weights_only=True)
    elif args.resume:
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if saved.get('evaluation_snapshot'):
            raise ValueError('evaluation snapshots cannot resume; use the full latest.pt training checkpoint')
    profile = (base or saved or {}).get('model_profile', ('math-phoneme-v1' if config.get('frontend') == 'cmu-ipa-v1' else 'math-student-v1') if args.math_student else 'project-v1')
    if args.math_student and (args.tiny or profile not in MATH_PROFILES):
        raise ValueError('math-student cannot mix with tiny or another architecture')
    if profile not in ['project-v1', 'ljs-sdp-v1', 'piper-uk-medium-v1', 'vosk-ru-multi-v1', 'css10-de-es-v1', 'neon-european-v1', 'math-student-v1', 'math-phoneme-v1', 'math-phoneme-wide-v1']:
        raise ValueError('unsupported model architecture profile')
    cfg = dict(MATH_WIDE_MODEL if profile == 'math-phoneme-wide-v1' else MATH_MODEL if profile in MATH_PROFILES else MODEL)
    prior_weight = args.math_prior_mel_weight
    if prior_weight is None:
        prior_weight = saved.get('math_prior_mel_weight', 0.0) if saved else 0.0
    if not math.isfinite(prior_weight) or not 0 <= prior_weight <= 100:
        raise ValueError('math prior mel weight must be finite and in [0,100]')
    if prior_weight and profile not in MATH_PROFILES:
        raise ValueError('prior mel supervision requires math student profile')
    if args.resume and prior_weight != saved.get('math_prior_mel_weight', 0.0):
        raise ValueError('resume math prior objective differs from checkpoint')
    alignment_labels = alignment_provenance = None
    alignment_path = args.teacher_alignment or ((saved or {}).get('teacher_alignment') or {}).get('path')
    if alignment_path:
        if profile not in ('math-phoneme-v1', 'math-phoneme-wide-v1'): raise ValueError('teacher alignment requires phoneme math profile')
        from math_alignment import load_alignment
        alignment_labels, alignment_provenance = load_alignment(alignment_path, raw_config, config)
    if args.resume and (alignment_provenance or {}).get('sha256') != ((saved or {}).get('teacher_alignment') or {}).get('sha256'):
        raise ValueError('resume teacher alignment objective differs from checkpoint')
    if profile in MATH_PROFILES and (config['language'] != 'en' or config.get('frontend') != ('cmu-ipa-v1' if profile != 'math-student-v1' else 'grapheme-v2') or args.tiny):
        raise ValueError('math student requires its English grapheme or phoneme frontend')
    if profile == 'ljs-sdp-v1': cfg['use_sdp'] = True
    if profile == 'piper-uk-medium-v1':
        cfg = dict(PIPER_MODEL)
        if args.tiny or config['language'] != 'uk' or config['frontend'] != 'piper-uk-text-v1' or (type(config.get('conditioning_speaker')) is not int or config['conditioning_speaker'] not in [0,1,2]):
            raise ValueError('invalid Ukrainian profile/frontend/conditioning speaker')
    russian = profile == 'vosk-ru-multi-v1'
    if russian:
        from vosk_adapter import MODEL as VOSK_MODEL, build_model, duration_discriminator, subband_loss
        cfg = dict(VOSK_MODEL)
        if args.tiny or config['language'] != 'ru' or config['frontend'] != 'vosk-ru-phones-v1' or type(config.get('conditioning_speaker')) is not int or not 0 <= config['conditioning_speaker'] < 5:
            raise ValueError('invalid Russian profile/frontend/conditioning speaker')
    css10 = profile == 'css10-de-es-v1'
    if css10:
        from css10_adapter import MODEL as CSS10_MODEL, build_model as build_css10, validate_config
        cfg = dict(CSS10_MODEL)
        validate_config(config)
        if args.tiny:
            raise ValueError('CSS10 profile cannot use tiny architecture')
    neon = profile == 'neon-european-v1'
    if neon:
        from neon_adapter import MODEL as NEON_MODEL, build_model as build_neon, validate_config
        cfg = dict(NEON_MODEL)
        validate_config(config)
        if args.tiny:
            raise ValueError('Neon profile cannot use tiny architecture')
    if args.tiny:
        cfg.update(inter_channels=16, hidden_channels=16, filter_channels=32, n_layers=2,
                   upsample_initial_channel=32, resblock_kernel_sizes=[3], resblock_dilation_sizes=[[1, 3, 5]])
    net = (build_model() if russian else build_css10() if css10 else build_neon(config['language']) if neon else SynthesizerTrn(len(config['symbols']), 513, 32, **cfg)).to(device)
    disc = MultiPeriodDiscriminator(False).to(device)
    learning_rate = args.learning_rate if args.learning_rate is not None else (5e-5 if args.init_checkpoint else 2e-4)
    optim = torch.optim.AdamW(net.parameters(), lr=learning_rate, betas=(0.8, 0.99))
    optim_d = torch.optim.AdamW(disc.parameters(), lr=learning_rate, betas=(0.8, 0.99))
    duration_disc = duration_discriminator().to(device) if russian else None
    duration_optim = torch.optim.AdamW(duration_disc.parameters(), lr=learning_rate, betas=(0.8, 0.99)) if russian else None
    step = 0
    initialization = dict(mode='from-scratch')
    if args.init_checkpoint:
        source = base['training_config']
        for field in ['schema_version', 'language', 'frontend', 'sample_rate', 'symbols', 'dictionary', 'conditioning_speaker']:
            source_value = source.get(field, 'grapheme-v1' if field == 'frontend' else None)
            target_value = config.get(field, 'grapheme-v1' if field == 'frontend' else None)
            if source_value != target_value:
                raise ValueError(f'fine-tuning base {field} differs from prepared target')
        if base['model_config'] != cfg:
            raise ValueError('fine-tuning model architecture differs from base checkpoint')
        net.load_state_dict(base['generator'], strict=True)
        if base.get('discriminator') is not None:
            disc.load_state_dict(base['discriminator'], strict=True)
        initialization = dict(mode='fine-tune', base_sha256=base_digest, base_step=base['step'],
                              base_source_revision=base['source_revision'], base_rights=source['rights'],
                              base_initialization=base.get('initialization', dict(mode='from-scratch')),
                              discriminator='pretrained' if base.get('discriminator') is not None else 'fresh')
        if profile in MATH_PROFILES:
            initialization['base_math_prior_mel_weight'] = base.get('math_prior_mel_weight', 0.0)
        # New target dataset, optimizer, RNG and step count: adaptation is not resume.
        del base
    if args.resume:
        if saved['dataset_digest'] != digest or saved['model_config'] != cfg:
            raise ValueError('resume dataset/config differs from checkpoint')
        net.load_state_dict(saved['generator']); disc.load_state_dict(saved['discriminator'])
        optim.load_state_dict(saved['optimizer']); optim_d.load_state_dict(saved['optimizer_d'])
        if russian:
            duration_disc.load_state_dict(saved['duration_discriminator'],strict=True)
            duration_optim.load_state_dict(saved['duration_optimizer'])
        step = saved['step']
        initialization = saved.get('initialization', dict(mode='from-scratch'))
        learning_rate = saved.get('learning_rate', optim.param_groups[0]['lr'])
        if args.learning_rate is not None and args.learning_rate != learning_rate:
            raise ValueError('resume learning-rate differs from checkpoint')
        # map_location may move checkpoint RNG tensors to CUDA; RNG APIs need CPU bytes.
        torch.set_rng_state(saved['rng'].cpu()); random.setstate(saved['python_rng'])
        if device.type == 'cuda' and 'cuda_rng' in saved:
            torch.cuda.set_rng_state_all([state.cpu() for state in saved['cuda_rng']])
        if device.type == 'mps' and 'mps_rng' in saved:
            torch.mps.set_rng_state(saved['mps_rng'].cpu())
    if russian:
        net.current_mas_noise_scale = saved.get('mas_noise_scale',0.01) if saved else 0.01
        from vendor.vosk.pqmf import PQMF
        pqmf = PQMF(device)
    acoustics = Acoustics(device)
    stopping = False
    def stop(_sig, _frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    def save():
        payload = dict(generator=net.state_dict(), discriminator=disc.state_dict(),
                       optimizer=optim.state_dict(), optimizer_d=optim_d.state_dict(), step=step,
                       dataset_digest=digest, training_config=config, model_config=cfg, model_profile=profile,
                       initialization=initialization, learning_rate=learning_rate,
                       rng=torch.get_rng_state(), python_rng=random.getstate(),
                       source_revision=('vosk:b492223c8747a24499e4c1ae905800e85ab887b4' if russian else 'vits:2e561ba58618d021b5b8323d3765880f7e0ecfdb'))
        if profile in MATH_PROFILES:
            payload['math_prior_mel_weight'] = prior_weight
            payload['teacher_alignment'] = alignment_provenance
        if russian:
            payload.update(duration_discriminator=duration_disc.state_dict(), duration_optimizer=duration_optim.state_dict(), mas_noise_scale=net.current_mas_noise_scale, training_objective='project waveform GAN + duration GAN + mel/KL/SDP + multiresolution subband STFT')
        if device.type == 'cuda': payload['cuda_rng'] = torch.cuda.get_rng_state_all()
        if device.type == 'mps': payload['mps_rng'] = torch.mps.get_rng_state().cpu()
        payload['training_device'] = str(device)
        temp = output / 'latest.tmp'; torch.save(payload, temp); temp.replace(checkpoint)
    print(json.dumps(dict(event='started', device=str(device), step=step, target=args.steps, mode=initialization['mode'], learning_rate=learning_rate,
                          math_prior_mel_weight=prior_weight,
                          discriminator=initialization.get('discriminator', 'scratch'))), flush=True)
    while step < args.steps and not stopping:
        net.train(); disc.train()
        batch = random.choices(records, k=args.batch_size)
        x, xl, spec, sl, y = load_batch(batch, device, acoustics)
        if russian:
            net.current_mas_noise_scale = max(0.01 - step*2e-6,0.0)
            y_hat, y_subbands, length_loss, _, slices, x_mask, z_mask, latent, durations = net(x, xl, acoustics.to_mel(spec), sl, sid=conditioning_ids(config,x))
        else:
            if alignment_labels is None:
                y_hat, length_loss, _, slices, _, z_mask, latent = net(x, xl, spec, sl, sid=conditioning_ids(config,x))
            else:
                from math_alignment import alignment_batch
                path = alignment_batch(batch, alignment_labels, xl, sl)
                y_hat, length_loss, _, slices, _, z_mask, latent = net(x, xl, spec, sl, sid=conditioning_ids(config,x), attn_override=path)
        z, z_p, m_p, logs_p, m_q, logs_q = latent
        y_segment = commons.slice_segments(y, slices * 256, 8192)
        real, fake, _, _ = disc(y_segment, y_hat.detach())
        d_loss = discriminator_loss(real, fake)[0]
        optim_d.zero_grad(); d_loss.backward(); torch.nn.utils.clip_grad_norm_(disc.parameters(), 1000); optim_d.step()
        if russian:
            duration_disc.train()
            hidden, predicted, aligned = durations
            real_dur, fake_dur = duration_disc(hidden.detach(), x_mask.detach(), aligned.detach(), predicted.detach())
            duration_d_loss = discriminator_loss(real_dur,fake_dur)[0]
            if not torch.isfinite(duration_d_loss): raise ValueError('non-finite duration discriminator loss')
            duration_optim.zero_grad(); duration_d_loss.backward()
            torch.nn.utils.clip_grad_norm_(duration_disc.parameters(),1000); duration_optim.step()
            for p in duration_disc.parameters(): p.requires_grad_(False)
        for p in disc.parameters(): p.requires_grad_(False)
        real, fake, real_maps, fake_maps = disc(y_segment, y_hat)
        target_mel = commons.slice_segments(acoustics.to_mel(spec), slices, 32)
        mel_loss = F.l1_loss(target_mel, acoustics.to_mel(acoustics.spec(y_hat.squeeze(1))))
        latent_kl_loss = kl_loss(z_p, logs_q, m_p, logs_p, z_mask)
        loss = generator_loss(fake)[0] + feature_loss(real_maps, fake_maps) + 45 * mel_loss + length_loss.sum() + latent_kl_loss
        prior_mel_loss = None
        if prior_weight:
            # Train the mean text prior used by deterministic export, rather than
            # supervising only the posterior waveform conditioned on real audio.
            prior_mel_loss = math_prior_mel_loss(net, m_p, z_mask, slices, target_mel, acoustics)
            loss = loss + prior_weight * prior_mel_loss
        if russian:
            duration_scores = duration_disc(hidden,x_mask,aligned,predicted)[1]
            loss = loss + generator_loss(duration_scores)[0] + subband_loss(pqmf.analysis(y_segment),y_subbands)
        if not torch.isfinite(loss) or not torch.isfinite(d_loss):
            raise ValueError('non-finite training loss; checkpoint not advanced')
        optim.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1000); optim.step()
        for p in disc.parameters(): p.requires_grad_(True)
        if russian:
            for p in duration_disc.parameters(): p.requires_grad_(True)
        step += 1
        metrics = dict(event='step', step=step, generator=float(loss.detach()), discriminator=float(d_loss.detach()), mel=float(mel_loss.detach()))
        if prior_mel_loss is not None:
            metrics['prior_mel'] = float(prior_mel_loss.detach())
        if profile in MATH_PROFILES:
            metrics['latent_kl'] = float(latent_kl_loss.detach())
            metrics['duration_loss'] = float(length_loss.sum().detach())
        print(json.dumps(metrics), flush=True)
        if step % 100 == 0 or step == args.steps or stopping:
            net.eval()
            with torch.no_grad():
                vx, vxl, vspec, vsl, teacher_audio = load_batch(validation[:1], device, acoustics)
                preview = net.infer(vx, vxl, sid=conditioning_ids(config,vx), noise_scale=0.0, noise_scale_w=0.0)[0][0, 0].cpu().numpy()
                if not np.isfinite(preview).all(): raise ValueError('invalid validation preview')
                wavfile.write(output / 'preview.wav', 22050, np.clip(preview * 32767, -32768, 32767).astype(np.int16))
                if profile in MATH_PROFILES:
                    print(json.dumps(dict(event='validation-preview', step=step,
                                          text_rms=float(np.sqrt(np.mean(preview ** 2))),
                                          teacher_rms=float(teacher_audio.square().mean().sqrt()),
                                          duration_ratio=len(preview) / teacher_audio.shape[-1],
                                          perceptual_evaluation='pending')), flush=True)
                # A preview is held-out synthesis, not an objective quality score.
            save()
    save()
    print(json.dumps(dict(event='stopped' if stopping else 'completed', step=step, checkpoint=str(checkpoint))), flush=True)

if __name__ == '__main__':
    main()
