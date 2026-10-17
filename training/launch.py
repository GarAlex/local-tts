"""Validate and tokenize through Rust, then run the local VITS trainer."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from runtime import cli

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--output', required=True)
mode = parser.add_mutually_exclusive_group(required=True)
mode.add_argument('--init-checkpoint')
mode.add_argument('--resume', action='store_true')
mode.add_argument('--from-scratch', action='store_true')
args, rest = parser.parse_known_args()
root = Path(__file__).resolve().parents[1]
output = Path(args.output).resolve()
output.mkdir(parents=True, exist_ok=True)
prepared = output / 'prepared.json'
command = cli()
prepare = command + ['training-prepare', str(Path(args.dataset).resolve()), str(prepared)]
if args.init_checkpoint or args.resume:
    # Restricted deserialization; only project checkpoint structures are supported.
    import torch
    checkpoint_path = Path(args.init_checkpoint).resolve() if args.init_checkpoint else output / 'latest.pt'
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    base_config = output / 'base-training.json'
    base_config.write_text(json.dumps(saved['training_config'], ensure_ascii=False))
    prepare.append(str(base_config))
    if args.init_checkpoint and saved['training_config'].get('frontend', 'grapheme-v1') != 'grapheme-v1':
        prepare.append('--text-v1')
subprocess.run(prepare, cwd=root, check=True)
training_mode = ['--init-checkpoint', str(Path(args.init_checkpoint).resolve())] if args.init_checkpoint else ['--resume'] if args.resume else ['--from-scratch']
os.execv(sys.executable, [sys.executable, '-u', str(root / 'training/train.py'), '--prepared', str(prepared), '--output', str(output)] + training_mode + rest)
