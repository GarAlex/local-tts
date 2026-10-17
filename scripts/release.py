#!/usr/bin/env python3
"""Build, audit, relocate-test and archive a native macOS release candidate."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import sys
import tarfile
import tempfile

from release_notices import collect, dependency_tree, sha

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ['crates', 'include', 'examples', 'training', 'scripts', 'assets', 'docs', 'release', 'tests']
SOURCE_FILES = ['Cargo.toml', 'Cargo.lock', 'LICENSE', 'README.md', '.gitignore']
EXCLUDE = shutil.ignore_patterns('__pycache__', '*.pyc', '.DS_Store')

def run(*args, **kwargs):
    print('+', ' '.join(map(str,args)), flush=True)
    return subprocess.run(list(map(str,args)), check=True, **kwargs)

def json_file(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')

def files(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*')) if p.is_file()}

def copy_tree(source, target):
    # Source kits never follow links outside the allowlisted tree.
    if any(p.is_symlink() for p in source.rglob('*')): raise ValueError(f'Symlink in source tree: {source}')
    shutil.copytree(source, target, ignore=EXCLUDE)

def runtime_kit(destination, binary):
    destination.mkdir(parents=True)
    for name in ['training','assets']:
        copy_tree(ROOT/name, destination/name)
    scripts = destination/'scripts'; scripts.mkdir()
    for pattern in ['download_*.py','prepare_*.py']:
        for path in (ROOT/'scripts').glob(pattern): shutil.copy2(path, scripts/path.name)
    (destination/'bin').mkdir(); shutil.copy2(binary, destination/'bin/local-tts')
    shutil.copy2(ROOT/'LICENSE', destination/'LICENSE')

def sign(path, identity, entitlements=None):
    command=['codesign','--force','--sign',identity]
    if identity != '-': command += ['--timestamp','--options','runtime']
    if entitlements: command += ['--entitlements',entitlements]
    run(*command,path)

def audit_binary(path):
    text=subprocess.check_output(['otool','-L',str(path)],text=True)
    lines=text.splitlines()[1:]
    for line in lines:
        dependency=line.strip().split(' (')[0]
        if path.suffix == '.dylib' and dependency == '@rpath/liblocal_tts.dylib': continue
        if not dependency.startswith(('/usr/lib/','/System/Library/')):
            raise ValueError(f'Non-system runtime dependency in {path}: {dependency}')
    return text

def archive_tar(source, target):
    with tarfile.open(target,'w:gz') as archive: archive.add(source,arcname=source.name)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', choices=['all','sdk'], default='all', help='Build both products, or the independent library/CLI SDK')
    parser.add_argument('--output', type=Path, default=ROOT/'dist')
    parser.add_argument('--identity', default='-', help='Developer ID identity; default is local ad-hoc signing')
    parser.add_argument('--notary-profile', help='Existing notarytool keychain profile; uploads only when explicitly supplied')
    parser.add_argument('--training-python', type=Path, help='Optional existing training interpreter for relocated fine-tune/export checks')
    parser.add_argument('--voice', type=Path, help='Optional local Kokoro package for relocated synthesis checks; never included')
    parser.add_argument('--multilingual-voice', type=Path, help='Optional pinned Chatterbox folder for relocated French CLI/C++ tests; never included')
    parser.add_argument('--compact-voices', type=Path, help='Optional parent folder containing inflect-nano-en and kitten-nano-en for relocated C++ checks')
    parser.add_argument('--design-voice', type=Path, help='Optional pinned Qwen VoiceDesign folder for relocated CLI/C++ checks; never included')
    args=parser.parse_args()
    if sys.platform != 'darwin': parser.error('This release recipe currently supports native macOS builds only.')
    if args.notary_profile and args.identity == '-': parser.error('Notarization requires a Developer ID identity.')
    if any(os.environ.get(key) for key in ['ORT_LIB_LOCATION','ORT_PREFER_DYNAMIC_LINK','ORT_SKIP_DOWNLOAD','RUSTFLAGS','CARGO_ENCODED_RUSTFLAGS','CARGO_TARGET_DIR','CARGO_BUILD_TARGET']):
        parser.error('Use the reviewed default ONNX Runtime/link configuration; unset ORT overrides and custom Rust flags.')
    target=next(line.split(': ',1)[1] for line in subprocess.check_output(['rustc','-vV'],text=True).splitlines() if line.startswith('host:'))
    if target not in ['aarch64-apple-darwin','x86_64-apple-darwin']: parser.error('Unsupported native target')
    metadata=json.loads(subprocess.check_output(['cargo','metadata','--locked','--format-version','1','--filter-platform',target],cwd=ROOT,text=True))
    version=next(p['version'] for p in metadata['packages'] if p['name']=='local-tts-core')
    source_hashes={}
    for name in SOURCE_DIRS+SOURCE_FILES:
        path=ROOT/name
        entries=[path] if path.is_file() else sorted(path.rglob('*'))
        for p in entries:
            if p.is_file() and '__pycache__' not in p.parts and p.name != '.DS_Store' and p.suffix != '.pyc': source_hashes[str(p.relative_to(ROOT))]=sha(p)
    source_id=hashlib.sha256(json.dumps(source_hashes,sort_keys=True).encode()).hexdigest()
    label=f'local-tts-{version}-{target}-{source_id[:12]}' + ('-sdk' if args.package=='sdk' else '')
    output=args.output.resolve(); output.mkdir(parents=True,exist_ok=True)
    final=output/label
    if final.exists(): raise ValueError(f'Refusing to replace release: {final}')
    run(sys.executable,ROOT/'tests/release_packaging.py',cwd=ROOT)
    run('cargo','fmt','--all','--check',cwd=ROOT)
    run('cargo','test','--locked','--workspace',cwd=ROOT)
    run('cargo','clippy','--locked','--workspace','--all-targets','--','-D','warnings',cwd=ROOT)
    binaries=ROOT/'target'/target/'release'
    with tempfile.TemporaryDirectory(prefix='.release-',dir=output) as temporary:
        stage=Path(temporary)
        binary=binaries/'local-tts'
        products=['sdk','source']
        app=None
        app_notices=None
        if args.package=='all':
            run('cargo','build','--locked','--release','--target',target,
                '-p','tts-studio','-p','local-tts-cli',cwd=ROOT)
            products.insert(0,'studio')
            app=stage/'Local TTS Studio.app'; resources=app/'Contents/Resources'; macos=app/'Contents/MacOS'
            macos.mkdir(parents=True); resources.mkdir()
            shutil.copy2(binaries/'tts-studio',macos/'tts-studio')
            kit=resources/'runtime'; runtime_kit(kit,binary)
            selected,tree=dependency_tree(metadata,target,['tts-studio','local-tts-cli'])
            app_notices=collect(metadata,resources/'licenses',selected,['tts-studio','local-tts-cli'])
            (resources/'licenses/dependency-tree.txt').write_text(tree)
            with (app/'Contents/Info.plist').open('wb') as f:
                plistlib.dump(dict(CFBundleName='Local TTS Studio',CFBundleDisplayName='Local TTS Studio',
                    CFBundleIdentifier='local.tts.studio',CFBundleExecutable='tts-studio',CFBundlePackageType='APPL',
                    CFBundleVersion=version,CFBundleShortVersionString=version,LSMinimumSystemVersion='14.0',
                    NSHighResolutionCapable=True,NSMicrophoneUsageDescription='Record voice training clips locally when you press Record microphone.'),f)
        # Build each product with exactly its inventoried roots. A workspace build
        # can unify GUI features into dependencies shared with the SDK.
        run('cargo','build','--locked','--release','--target',target,
            '-p','local-tts-cli','-p','local-tts-ffi',cwd=ROOT)
        sdk=stage/f'local-tts-sdk-{version}-{target}'; runtime_kit(sdk,binary)
        (sdk/'lib').mkdir(); shutil.copy2(binaries/'liblocal_tts.dylib',sdk/'lib/liblocal_tts.dylib')
        run('install_name_tool','-id','@rpath/liblocal_tts.dylib',sdk/'lib/liblocal_tts.dylib')
        for name in ['include','examples','docs']: copy_tree(ROOT/name,sdk/name)
        selected,tree=dependency_tree(metadata,target,['local-tts-cli','local-tts-ffi'])
        sdk_notices=collect(metadata,sdk/'licenses',selected,['local-tts-cli','local-tts-ffi'])
        (sdk/'licenses/dependency-tree.txt').write_text(tree)
        notice_gaps=sdk_notices['unresolved']+(app_notices['unresolved'] if app_notices else [])
        if args.notary_profile and notice_gaps:
            raise ValueError('Selected products contain unresolved upstream notices; see license inventory.')
        shutil.copy2(ROOT/'release/SDK-CMakeLists.txt',sdk/'CMakeLists.txt')
        binaries_to_check=[sdk/'lib/liblocal_tts.dylib',sdk/'bin/local-tts']
        for path in binaries_to_check: sign(path,args.identity)
        if app:
            sign(kit/'bin/local-tts',args.identity)
            json_file(kit/'release-runtime.json',dict(version=version,files=files(kit)))
            sign(app,args.identity,ROOT/'release/entitlements.plist')
            run('codesign','--verify','--deep','--strict',app)
            binaries_to_check += [macos/'tts-studio',kit/'bin/local-tts']
        for path in binaries_to_check:
            run('codesign','--verify','--strict',path)
            expected='arm64' if target.startswith('aarch64') else 'x86_64'
            if subprocess.check_output(['lipo','-archs',str(path)],text=True).strip()!=expected:
                raise ValueError(f'Unexpected architecture: {path}')
        dependency_audit={str(p.relative_to(stage)):audit_binary(p) for p in binaries_to_check}
        json_file(stage/'binary-dependencies.json',dependency_audit)
        test_args=[sys.executable,ROOT/'scripts/test_release.py','--sdk',sdk,'--report',stage/'smoke-tests.json']
        if app: test_args += ['--app',app]
        if args.voice: test_args += ['--voice',args.voice.resolve()]
        if args.multilingual_voice: test_args += ['--multilingual-voice',args.multilingual_voice.resolve()]
        if args.compact_voices: test_args += ['--compact-voices',args.compact_voices.resolve()]
        if args.design_voice: test_args += ['--design-voice',args.design_voice.resolve()]
        if args.training_python: test_args += ['--training-python',args.training_python.absolute()]
        run(*test_args,cwd=ROOT)
        if args.notary_profile:
            upload=stage/'notary-upload.zip'
            submission=stage/'notary-submission';submission.mkdir()
            if app: shutil.copytree(app,submission/app.name)
            shutil.copytree(sdk,submission/sdk.name)
            run('ditto','-c','-k','--keepParent',submission,upload)
            result=subprocess.check_output(['xcrun','notarytool','submit',str(upload),'--keychain-profile',args.notary_profile,'--wait','--output-format','json'],text=True)
            reply=json.loads(result);json_file(stage/'notarization.json',reply)
            if reply.get('status')!='Accepted': raise ValueError('Apple notarization was not accepted')
            if app:
                run('xcrun','stapler','staple',app);run('xcrun','stapler','validate',app)
                run('spctl','--assess','--type','execute','--verbose=2',app)
            upload.unlink();shutil.rmtree(submission)
        if app:
            app_archive=stage/f'local-tts-studio-{version}-{target}.zip'
            run('ditto','-c','-k','--keepParent',app,app_archive)
        archive_tar(sdk,stage/(sdk.name+'.tar.gz'))
        source=stage/f'local-tts-source-{version}';source.mkdir()
        for name in SOURCE_DIRS: copy_tree(ROOT/name,source/name)
        for name in SOURCE_FILES: shutil.copy2(ROOT/name,source/name)
        archive_tar(source,stage/(source.name+'.tar.gz'));shutil.rmtree(source)
        if any(sha(ROOT/name)!=expected for name,expected in source_hashes.items()):
            raise ValueError('Source changed during packaging; rerun from a stable tree.')
        manifest=dict(schema_version=1,version=version,products=products,target=target,minimum_macos='14.0',
            created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=source_id,source_files=source_hashes,
            rustc=subprocess.check_output(['rustc','-vV'],text=True),cargo_lock_sha256=sha(ROOT/'Cargo.lock'),
            signing='ad-hoc' if args.identity=='-' else 'Developer ID',notarized=bool(args.notary_profile),
            public_release_ready=False,release_status='notarized-candidate' if args.notary_profile else 'local-candidate',
            manual_release_checks=(['clean-machine GUI/audio'] if app else [])+['minimum OS and target hardware','chosen model provenance and speech quality'],notice_gaps=notice_gaps,
            bundled_models=False,python_runtime_bundled=False,files=files(stage))
        json_file(stage/'release-manifest.json',manifest)
        (stage/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.name}\n' for p in sorted(stage.iterdir()) if p.is_file()))
        run(sys.executable,ROOT/'scripts/verify_release.py',stage,'--extract',cwd=ROOT)
        # Only successful complete stages become visible release directories.
        stage.rename(final)
    print(f'Release candidate: {final}',flush=True)

if __name__=='__main__': main()
