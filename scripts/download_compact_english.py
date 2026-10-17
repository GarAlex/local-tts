#!/usr/bin/env python3
"""Acquire pinned small English models. No publisher code is executed."""
import argparse, hashlib, json, pathlib, urllib.request, zipfile, struct, io
ROOT=pathlib.Path(__file__).resolve().parents[1]
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('model',choices=['inflect','kitten']);p.add_argument('--output',type=pathlib.Path);a=p.parse_args()
 pins=json.loads((ROOT/'assets/compact-english-assets.json').read_text())[a.model];out=a.output or ROOT/'voices'/pins['folder'];out.mkdir(parents=True,exist_ok=True)
 archive=None
 for name,r in pins['files'].items():
  if 'derived_from' in r:continue
  if 'copy_from' in r:
   source=ROOT/'voices'/r['copy_from']
   if not source.exists():raise ValueError('Install Kokoro Q8 once for the pinned BSD dictionary: '+str(source))
   b=source.read_bytes()
  else:b=urllib.request.urlopen(r['url'],timeout=120).read()
  if len(b)!=r['bytes'] or sha(b)!=r['sha256']:raise ValueError('checksum mismatch: '+name)
  if name=='voices.npz':archive=b;continue
  f=out/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(b)
 if archive:
  with zipfile.ZipFile(io.BytesIO(archive)) as z:
   for name,r in pins['files'].items():
    if r.get('derived_from')!='voices.npz':continue
    b=z.read(pathlib.Path(name).stem+'.npy');end=10+struct.unpack('<H',b[8:10])[0]
    if b[:8]!=b'\x93NUMPY\x01\x00' or b[10:end].strip()!=b"{'descr': '<f4', 'fortran_order': False, 'shape': (400, 256), }":raise ValueError('unexpected style format')
    b=b[end:]
    if len(b)!=r['bytes'] or sha(b)!=r['sha256']:raise ValueError('style checksum mismatch')
    f=out/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(b)
 print('Pinned candidate installed: '+str(out))
if __name__=='__main__':main()
