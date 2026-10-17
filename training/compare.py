"""Create held-out before/after listening samples through the native VITS library."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import torch
from checkpoint_tools import ROOT, load, build, prepare, summary, sha, cli

# Forward Stop/interrupt to the current exporter or native child, avoiding orphan jobs.
child = None
def stop(signum, frame):
    if child is not None and child.poll() is None:
        child.terminate()
        child.wait()
    raise KeyboardInterrupt

def run(command):
    global child
    child = subprocess.Popen(list(map(str, command)), cwd=ROOT)
    try:
        code = child.wait()
        if code: raise subprocess.CalledProcessError(code, command)
    finally: child = None

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True, help='parent folder for new comparison jobs')
    parser.add_argument('--split', choices=['validation','test'], default='validation')
    parser.add_argument('--limit', type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.limit <= 12: parser.error('limit must be 1–12')
    import signal
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    torch.set_num_threads(2)
    base, base_digest = load(args.base)
    target, target_digest = load(args.checkpoint)
    for field in ['schema_version','language','frontend','sample_rate','symbols','conditioning_speaker']:
        if base['training_config'].get(field, 'grapheme-v1' if field=='frontend' else None) != target['training_config'].get(field, 'grapheme-v1' if field=='frontend' else None):
            raise ValueError('comparison base/target ' + field + ' differs')
    if base['model_config'] != target['model_config']:
        raise ValueError('comparison base/target architecture differs')
    if base['training_config'].get('dictionary',{}).get('sha256') != target['training_config'].get('dictionary',{}).get('sha256'):
        raise ValueError('comparison base/target dictionary differs')
    origin = target.get('initialization', {})
    if origin.get('mode') != 'fine-tune' or origin.get('base_sha256') != base_digest:
        raise ValueError('selected base does not match target fine-tuning lineage')
    # Do not relabel unrelated or changed data as this checkpoint's held-out set.
    dataset_digest = sha(args.dataset)
    with tempfile.TemporaryDirectory(prefix='tts-compare-preflight-') as directory:
        prepared, prepared_digest = prepare(args.dataset, target['training_config'], directory)
    if prepared_digest != target['dataset_digest']:
        raise ValueError('evaluation dataset differs from target checkpoint prepared dataset')
    if not any(u['split']==args.split for u in prepared['utterances']):
        raise ValueError('no recordings in selected held-out split')
    for saved in [base,target]:
        net = build(saved)
        del net
    metadata = dict(schema_version=1, base=summary(base,base_digest,args.base),
                    target=summary(target,target_digest,args.checkpoint),
                    dataset=str(Path(args.dataset).resolve()),dataset_sha256=dataset_digest,
                    prepared_sha256=prepared_digest,engine='native ONNX; deterministic noise settings; duration 1.0',
                    metrics_note='Duration, RMS, peak and clipping are audio diagnostics, not intelligibility or speaker similarity scores.',
                    report='samples/report.json')
    del base,target,prepared,saved
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=True)
    job = Path(tempfile.mkdtemp(prefix='comparison-',dir=output))
    # Temporary exported graphs are removed after success/failure; keep only audio/reports.
    with tempfile.TemporaryDirectory(prefix='.work-',dir=job) as directory:
        work = Path(directory)
        for checkpoint, name in [(args.base,'before'),(args.checkpoint,'after')]:
            run([sys.executable, ROOT/'training/export.py', '--checkpoint', checkpoint, '--output', work/name])
            run(cli()+['vits-verify',str(work/name)])
        # Bind exports to the inspected checkpoints despite external edits during a long export.
        if sha(args.base)!=base_digest or sha(args.checkpoint)!=target_digest:
            raise ValueError('checkpoint changed during comparison')
        run(cli()+['vits-compare',str(work/'before'),str(work/'after'),str(Path(args.dataset).resolve()),str(job/'samples'),args.split,str(args.limit)])
    report_path = job/'samples/report.json'
    report=json.loads(report_path.read_text())
    if report['dataset_sha256'] != dataset_digest or sha(args.dataset) != dataset_digest:
        raise ValueError('dataset changed during comparison')
    metadata['report_sha256'] = sha(report_path)
    (job/'comparison.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False))
    print(json.dumps(dict(event='comparison',directory=str(job/'samples'),report=report,
                          base=metadata['base'],target=metadata['target']),ensure_ascii=False),flush=True)

if __name__ == '__main__': main()
