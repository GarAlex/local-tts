import argparse
import json
from pathlib import Path
import subprocess
import sys
from runtime import cli

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--expected-sha256')
parser.add_argument('--new-output', action='store_true')
args = parser.parse_args()
if args.expected_sha256 and not args.new_output:
    parser.error('evaluated exports require --new-output')
root = Path(__file__).resolve().parents[1]
command = [sys.executable, str(root / 'training/export.py'), '--checkpoint', args.checkpoint, '--output', args.output]
if args.expected_sha256: command += ['--expected-sha256', args.expected_sha256]
if args.new_output: command += ['--new-output']
subprocess.run(command, check=True)
subprocess.run(cli() + ['vits-verify', args.output], cwd=root, check=True)

if args.expected_sha256:
    print(json.dumps(dict(event='evaluated_export', voice=str(Path(args.output).resolve()), checkpoint_sha256=args.expected_sha256)), flush=True)
