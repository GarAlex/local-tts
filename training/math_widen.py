"""Function-preserving decoder widening; an initialization, not a quality claim."""
import argparse
from pathlib import Path
import torch
from checkpoint_tools import load, build
from train import MATH_WIDE_MODEL, SynthesizerTrn


def widen(source, target):
    originals = dict(source.named_modules())
    with torch.no_grad():
        for name, layer in target.named_modules():
            if not isinstance(layer, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)): continue
            old = originals[name]
            if type(layer) is not type(old) or layer.groups != 1 or old.groups != 1:
                raise ValueError('unsupported convolution widening')
            for field in ('kernel_size', 'stride', 'padding', 'dilation'):
                if getattr(layer, field) != getattr(old, field): raise ValueError('convolution geometry differs')
            if layer.in_channels % old.in_channels or layer.out_channels % old.out_channels:
                raise ValueError('channel widths must be integer multiples')
            inputs = layer.in_channels // old.in_channels
            outputs = layer.out_channels // old.out_channels
            if hasattr(old, 'weight_v'):
                v = old.weight_v
                weight = v * old.weight_g / v.norm(dim=(1,2), keepdim=True).clamp_min(1e-12)
            else:
                weight = old.weight
            repeat = (inputs, outputs, 1) if isinstance(layer, torch.nn.ConvTranspose1d) else (outputs, inputs, 1)
            weight = weight.repeat(*repeat) / inputs
            if inputs > 1:
                # Cancel across duplicated input channels to retain the forward
                # function while allowing different gradients to break symmetry.
                if isinstance(layer, torch.nn.ConvTranspose1d):
                    view = weight.reshape(inputs, old.in_channels, layer.out_channels, weight.shape[-1])
                    noise = torch.randn_like(view)
                    noise = noise - noise.mean(0, keepdim=True)
                else:
                    view = weight.reshape(layer.out_channels, inputs, old.in_channels, weight.shape[-1])
                    noise = torch.randn_like(view)
                    noise = noise - noise.mean(1, keepdim=True)
                weight = (view + noise * weight.std() * 0.01).reshape_as(weight)
            if hasattr(layer, 'weight_v'):
                layer.weight_v.copy_(weight)
                layer.weight_g.copy_(weight.norm(dim=(1,2), keepdim=True))
            else:
                layer.weight.copy_(weight)
            if layer.bias is not None: layer.bias.copy_(old.bias.repeat(outputs))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists(): parser.error('output already exists')
    torch.set_num_threads(2)
    torch.manual_seed(1234)
    saved, digest = load(args.checkpoint)
    if saved.get('model_profile') != 'math-phoneme-v1': raise ValueError('requires narrow phoneme student')
    source = build(saved).eval()
    target = SynthesizerTrn(len(saved['training_config']['symbols']), 513, 32, **MATH_WIDE_MODEL).eval()
    for name in ('enc_p', 'enc_q', 'flow', 'dp'):
        getattr(target, name).load_state_dict(getattr(source, name).state_dict(), strict=True)
    widen(source.dec, target.dec)
    with torch.no_grad():
        latent = torch.randn(1, 32, 32)
        error = float((source.dec(latent) - target.dec(latent)).abs().max())
    if error > 5e-6: raise ValueError(f'widening changed decoder function: {error}')
    saved.update(generator=target.state_dict(), model_config=MATH_WIDE_MODEL,
                 model_profile='math-phoneme-wide-v1',
                 initialization=dict(mode='function-preserving-widening', base_sha256=digest,
                                     base_initialization=saved.get('initialization'),
                                     width=256, decoder_max_error=error),
                 evaluation_snapshot=dict(source_sha256=digest, resumable=False))
    for name in ('optimizer', 'optimizer_d', 'discriminator', 'rng', 'python_rng', 'mps_rng', 'cuda_rng'):
        saved.pop(name, None)
    build(saved)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream: torch.save(saved, stream)
    print(f'Widened decoder with maximum function difference {error}; initialize a new run')


if __name__ == '__main__': main()
