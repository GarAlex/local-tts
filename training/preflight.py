"""Inspect a base and tokenize a target dataset without creating a training run."""
import argparse
import json
import tempfile
from pathlib import Path
import torch
from checkpoint_tools import load, build, prepare, summary, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    saved, digest = load(args.checkpoint)
    net = build(saved, discriminator=True)
    del net
    with tempfile.TemporaryDirectory(prefix='tts-preflight-') as directory:
        prepared, prepared_digest = prepare(args.dataset, saved['training_config'], directory, current=True)
    report = summary(saved, digest, args.checkpoint)
    report.update(event='preflight', dataset=str(Path(args.dataset).resolve()),
                  dataset_sha256=sha(args.dataset), prepared_sha256=prepared_digest,
                  target_speaker=prepared['speaker'],
                  target_normalization=prepared.get('normalization'),
                  clips={split:sum(u['split']==split for u in prepared['utterances']) for split in ['train','validation','test']})
    print(json.dumps(report, ensure_ascii=False), flush=True)

if __name__ == '__main__': main()
