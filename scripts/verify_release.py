#!/usr/bin/env python3
"""Verify a candidate's file hashes and optionally extract/test its archives."""
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from release_notices import sha


def relative(name):
    path=PurePosixPath(name)
    if not name or path.is_absolute() or any(part in ('..','.') for part in path.parts) or '\\' in name:
        raise ValueError(f'Unsafe archive/manifest path: {name}')
    return path

def safe_tar(archive, destination):
    with tarfile.open(archive) as stream:
        for member in stream.getmembers():
            relative(member.name)
            if not (member.isdir() or member.isfile()): raise ValueError('Archive links/devices are not supported')
        stream.extractall(destination)

def safe_zip(archive, destination):
    with zipfile.ZipFile(archive) as stream:
        for member in stream.infolist():
            relative(member.filename)
            if stat.S_ISLNK(member.external_attr >> 16): raise ValueError('Archive symlinks are not supported')
    # ditto preserves executable modes and macOS app resource metadata.
    subprocess.run(['ditto','-x','-k',str(archive),str(destination)],check=True)

def verify(directory):
    directory=Path(directory)
    manifest=json.loads((directory/'release-manifest.json').read_text())
    if manifest.get('schema_version')!=1: raise ValueError('Unsupported release manifest')
    expected=manifest['files']
    actual={str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}
    if actual != set(expected)|{'release-manifest.json','SHA256SUMS'}:
        raise ValueError('Release file list differs from manifest')
    for name,digest in expected.items():
        path=directory.joinpath(*relative(name).parts)
        if path.is_symlink() or sha(path)!=digest: raise ValueError(f'Checksum mismatch: {name}')
    listed=set()
    for line in (directory/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split('  ',1);relative(name)
        if '/' in name or name in listed or sha(directory/name)!=digest: raise ValueError(f'Archive/report checksum mismatch: {name}')
        listed.add(name)
    top={p.name for p in directory.iterdir() if p.is_file()}-{'SHA256SUMS'}
    if listed!=top: raise ValueError('SHA256SUMS does not cover top-level files')
    return manifest

def extract_check(directory, manifest):
    directory=Path(directory)
    with tempfile.TemporaryDirectory(prefix='TTS archive check ') as temporary:
        root=Path(temporary)
        for archive in directory.glob('*.zip'): safe_zip(archive,root)
        for archive in directory.glob('*.tar.gz'): safe_tar(archive,root)
        # Every expanded app/SDK file must be exactly the staged file that was checked.
        prefixes=[f'local-tts-sdk-{manifest["version"]}-{manifest["target"]}']
        if 'studio' in manifest.get('products',['studio']): prefixes.append('Local TTS Studio.app')
        for prefix in prefixes:
            expected={name:digest for name,digest in manifest['files'].items() if name.startswith(prefix+'/')}
            actual={str(p.relative_to(root)) for p in (root/prefix).rglob('*') if p.is_file()}
            if actual!=set(expected): raise ValueError(f'Extracted contents differ: {prefix}')
            for name,digest in expected.items():
                if sha(root/name)!=digest: raise ValueError(f'Extracted checksum mismatch: {name}')
        source=root/f'local-tts-source-{manifest["version"]}'
        actual={str(p.relative_to(source)) for p in source.rglob('*') if p.is_file()}
        if actual!=set(manifest['source_files']): raise ValueError('Source archive differs from source manifest')
        for name,digest in manifest['source_files'].items():
            relative(name)
            if sha(source/name)!=digest: raise ValueError(f'Archived source mismatch: {name}')
        env={k:v for k,v in os.environ.items() if not k.startswith(('DYLD_','ORT_','LOCAL_TTS_','PYTHON'))}
        env.update(PATH='/usr/bin:/bin:/usr/sbin:/sbin',LOCAL_TTS_HOME=str(root/'User data'))
        app=root/'Local TTS Studio.app'
        if 'studio' in manifest.get('products',['studio']):
            subprocess.run(['codesign','--verify','--deep','--strict',str(app)],check=True,env=env)
            subprocess.run([str(app/'Contents/MacOS/tts-studio'),'--check-runtime'],check=True,env=env,cwd=root,stdout=subprocess.DEVNULL)
        sdk=root/f'local-tts-sdk-{manifest["version"]}-{manifest["target"]}'
        output=subprocess.check_output([str(sdk/'bin/local-tts'),'text-prepare','en','2 kg'],env=env,cwd=root,text=True)
        if output.strip()!='two kilograms': raise ValueError('Extracted CLI smoke failed')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--extract',action='store_true')
    args=parser.parse_args();manifest=verify(args.directory)
    if args.extract: extract_check(args.directory,manifest)
    print('Release hashes'+(' and extracted archives' if args.extract else '')+' verified')

if __name__=='__main__': main()
