"""Compare paired sequence diagnostics; metrics are not voice acceptance."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def compare(before_path,after_path,allow_shared_cases=False):
    before=json.loads(before_path.read_text());after=json.loads(after_path.read_text())
    keys=['language','teacher_sha256','split','decoder_precision']
    if not allow_shared_cases:keys.append('dataset_sha256')
    for key in keys:
        if key not in before or before[key]!=after.get(key):raise ValueError(f'comparison contract differs: {key}')
    if before.get('decoder_updated') or after.get('decoder_updated'):raise ValueError('decoder was updated')
    if not before['cases'] or len(before['cases'])!=len(after['cases']):raise ValueError('case counts differ')
    rows=[]
    for a,b in zip(before['cases'],after['cases']):
        if (a['expression'],a['spoken'])!=(b['expression'],b['spoken']):raise ValueError('paired utterances differ')
        for root,case in [(before_path.parent,a),(after_path.parent,b)]:
            for audio in case['recordings']:
                path=(root/audio['file']).resolve()
                if not path.is_relative_to(root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=audio['sha256']:raise ValueError('audio hash/path differs')
        teacher_a=[r for r in a['recordings'] if r['kind']=='teacher']
        teacher_b=[r for r in b['recordings'] if r['kind']=='teacher']
        if len(teacher_a)!=1 or len(teacher_b)!=1 or teacher_a[0]['sha256']!=teacher_b[0]['sha256']:raise ValueError('teacher control audio differs')
        metrics={}
        for name in ('teacher_timed_log_spectral_error','duration_frame_mae'):
            values=[a[name],b[name]]
            if any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in values):raise ValueError('invalid comparison metric')
            metrics[name]=dict(before=values[0],after=values[1],delta=values[1]-values[0])
        rows.append(dict(expression=a['expression'],metrics=metrics))
    summary={}
    for name in rows[0]['metrics']:
        summary[name]=dict(before_mean=statistics.mean(r['metrics'][name]['before'] for r in rows),after_mean=statistics.mean(r['metrics'][name]['after'] for r in rows),improved_cases=sum(r['metrics'][name]['delta']<0 for r in rows),cases=len(rows))
    return dict(language=before['language'],before_step=before['student_step'],after_step=after['student_step'],before_student_sha256=before['student_sha256'],after_student_sha256=after['student_sha256'],dataset_sha256=before['dataset_sha256'],after_dataset_sha256=after['dataset_sha256'],comparison_scope='shared identical utterances across corpora' if allow_shared_cases else 'same dataset',teacher_sha256=before['teacher_sha256'],summary=summary,cases=rows,teacher_controls_identical=True,note='Paired validation diagnostics only. Better reconstruction metrics do not establish naturalness or numeric accuracy.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before',type=Path,required=True);p.add_argument('--after',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--shared-cases',action='store_true',help='allow different dataset hashes only for identical paired utterances/teacher audio')
    a=p.parse_args()
    if a.output.exists():p.error('output already exists')
    result=compare(a.before,a.after,a.shared_cases)
    with a.output.open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result['summary']),flush=True)
if __name__=='__main__':main()
