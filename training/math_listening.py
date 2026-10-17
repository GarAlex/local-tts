"""Create an offline listening review of two lineage-audited student exports."""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import random


def build(before, after, output, before_label='previous student', after_label='new student'):
    if any(not isinstance(label,str) or not 1<=len(label)<=120 for label in (before_label,after_label)):
        raise ValueError('listening labels must be 1–120 characters')
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('output already exists')
    paths = [Path(before).resolve(), Path(after).resolve()]
    reports = [json.loads(p.read_text()) for p in paths]
    if any(not r.get('audited_training_lineage') for r in reports):
        raise ValueError('comparisons must include training-lineage auditing')
    if any(r.get('split') != 'validation' for r in reports):
        raise ValueError('listening development uses validation comparisons')
    if reports[0]['dataset_sha256'] != reports[1]['dataset_sha256']:
        raise ValueError('comparisons must use the same dataset')
    mappings = [{e['id']: e for e in r['examples']} for r in reports]
    if any(len(m) != len(r['examples']) for m, r in zip(mappings, reports)):
        raise ValueError('duplicate comparison IDs')
    if mappings[0].keys() != mappings[1].keys():
        raise ValueError('comparison examples differ')
    provenance = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    rows, audio_hashes = [], {}
    rng = random.Random(1234)
    for ident, old in mappings[0].items():
        new = mappings[1][ident]
        for key in ('spoken', 'expression', 'teacher_audio_sha256'):
            if old[key] != new[key]:
                raise ValueError('comparison transcripts/teacher differ')
        tracks = []
        for name, index, source in [(before_label, 0, 'student'),
                                    (after_label, 1, 'student'),
                                    ('teacher', 1, 'teacher'),
                                    ('word bank', 1, 'fragments')]:
            root = paths[index].parent
            audio = (root / mappings[index][ident]['files'][source]).resolve()
            if not audio.is_relative_to(root) or not audio.is_file():
                raise ValueError('missing or escaping audio path')
            audio_hashes[str(audio)] = hashlib.sha256(audio.read_bytes()).hexdigest()
            tracks.append(dict(source=name, url=os.path.relpath(audio, output.parent)))
        rng.shuffle(tracks)
        rows.append(dict(id=ident, expression=old['expression'], spoken=old['spoken'], tracks=tracks))
    payload = dict(comparisons=provenance, audio_sha256=audio_hashes, examples=rows)
    raw = json.dumps(payload, ensure_ascii=False).replace('<', '\\u003c')
    cards = []
    for row in rows:
        controls = []
        for i, track in enumerate(row['tracks']):
            key = html.escape(f"{row['id']}-{i}", quote=True)
            url = html.escape(track['url'], quote=True)
            controls.append(f'''<div class="track"><b>Sample {chr(65+i)}</b>
<audio controls preload="none" src="{url}"></audio>
<label>Speech <select data-key="{key}-speech"><option value="">Unrated</option>
<option>No recognizable voice</option><option>Voice, unintelligible</option>
<option>Partly intelligible</option><option>Intelligible</option></select></label>
<label>Words heard <input data-key="{key}-heard" placeholder="Write what you hear"></label>
<label>Number/operator errors <input data-key="{key}-errors" placeholder="Errors or none"></label>
</div>''')
        cards.append(f'''<section><h2>{html.escape(row['expression'])}</h2>
<details><summary>Expected reading</summary>{html.escape(row['spoken'])}</details>
{''.join(controls)}</section>''')
    doc = '''<!doctype html><html lang="en"><meta charset="utf-8">
<title>Arithmetic speech listening review</title><style>
body{font:16px system-ui;max-width:960px;margin:32px auto;padding:0 20px;background:#f7f8fa;color:#17212b}
section{background:white;border:1px solid #ccd3db;padding:20px;margin:20px 0;border-radius:10px}
.track{display:grid;gap:10px;margin:18px 0;padding:12px;border-top:1px solid #ddd}
audio{width:100%}input{width:60%;padding:6px}button,select{padding:8px}details{margin:12px 0}
</style><h1>Arithmetic speech listening review</h1>
<p>Listen at a comfortable volume. These are raw outputs with no loudness adjustment.
Samples are shuffled; rate recognizable speech first, then words and arithmetic accuracy.
Automatic measurements cannot replace these ratings.</p>
<p>Ratings are kept in this browser when local storage is available. Export ratings to preserve them.</p>
<button id="export">Export ratings JSON</button> <button id="reveal">Reveal model identities</button>
<span id="status"></span>''' + ''.join(cards) + '''<script>
const manifest=PAYLOAD;
const storageKey='math-listening-'+JSON.stringify(manifest.comparisons);
let ratings={};
try{ratings=JSON.parse(localStorage.getItem(storageKey)||'{}')}catch(e){}
for(const el of document.querySelectorAll('[data-key]')){
 el.value=ratings[el.dataset.key]||'';
 el.addEventListener('input',()=>{ratings[el.dataset.key]=el.value;
  try{localStorage.setItem(storageKey,JSON.stringify(ratings))}catch(e){
   document.querySelector('#status').textContent='Use Export to save ratings; browser storage is unavailable.'}
 });
}
document.querySelector('#export').onclick=()=>{
 const blob=new Blob([JSON.stringify({manifest,ratings},null,2)],{type:'application/json'});
 const url=URL.createObjectURL(blob);const a=document.createElement('a');
 a.href=url;a.download='math-listening-ratings.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};
document.querySelector('#reveal').onclick=()=>{
 document.querySelectorAll('section').forEach((section,j)=>{
  section.querySelectorAll('.track b').forEach((b,i)=>b.textContent=
   'Sample '+String.fromCharCode(65+i)+' — '+manifest.examples[j].tracks[i].source);
 });
};
</script></html>'''.replace('PAYLOAD', raw)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        stream.write(doc)
    return len(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', required=True)
    parser.add_argument('--after', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--before-label',default='previous student')
    parser.add_argument('--after-label',default='new student')
    args = parser.parse_args()
    print(f'Created {build(args.before, args.after, args.output,args.before_label,args.after_label)} listening comparisons')
