"""Compact numeric phoneme table preflight; no cached acoustic fragments."""
import argparse
import hashlib
import json
from pathlib import Path


def encode_words(words,inventory,language):
    if language not in ('ru','uk') or not words:raise ValueError('expected RU/UK numeric words')
    suffix=2 if language=='ru' else 1
    tokens=[1,0]
    for index,word in enumerate(words):
        ids=inventory.get(word)
        if ids is None:raise ValueError('missing numeric pronunciation')
        if len(ids)<=2+suffix or ids[:2]!=[1,0] or ids[-suffix:]!=([0,2] if language=='ru' else [2]):raise ValueError('invalid native word boundaries')
        if any(type(t)!=int or t<0 for t in ids):raise ValueError('invalid native word token')
        if index:tokens.extend([0,3,0] if language=='ru' else [3,0])
        tokens.extend(ids[2:-suffix])
    tokens.extend([0,2] if language=='ru' else [2])
    if len(tokens)>1024:raise ValueError('numeric utterance exceeds 1024 tokens')
    return tokens


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus',type=Path,required=True);p.add_argument('--words',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('output already exists')
    corpus=json.loads(args.corpus.read_text());records=json.loads(args.words.read_text())
    inventory={r['word']:r['tokens'] for r in records}
    if len(inventory)!=len(records):raise ValueError('duplicate pronunciation keys')
    for row in corpus['records']:
        if encode_words(row['spoken'].split(),inventory,corpus['language'])!=row['tokens']:raise ValueError('compact phonemes differ from native whole-utterance frontend')
    payload=dict(schema_version=1,language=corpus['language'],purpose='numeric phoneme token lookup only; no audio/latent word fragments',words=inventory,
                 verified_utterances=len(corpus['records']),corpus_sha256=hashlib.sha256(args.corpus.read_bytes()).hexdigest(),source_words_sha256=hashlib.sha256(args.words.read_bytes()).hexdigest())
    with args.output.open('x') as stream:json.dump(payload,stream,ensure_ascii=False,indent=2)
    print(json.dumps(dict(language=corpus['language'],utterances=len(corpus['records']),bytes=args.output.stat().st_size)),flush=True)
if __name__=='__main__':main()
