"""Collect notices from the exact Cargo resolution; never guess missing grants."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def notice(path):
    name = path.name.lower()
    return any(x in name for x in ('license', 'licence', 'copying', 'copyright', 'notice')) or name in ('ofl.txt', 'ufl.txt', 'unlicense', 'hack-regular.txt')

def build_selection(metadata, tree):
    """Map Cargo's enabled normal/build tree to exact metadata package IDs."""
    by_key = {}
    for package in metadata['packages']:
        by_key.setdefault((package['name'], package['version']), []).append(package['id'])
    selected = set()
    for line in tree.splitlines():
        if not line.strip(): continue
        fields = line.split()
        if len(fields) < 2 or not fields[1].startswith('v'):
            raise ValueError(f'Unexpected Cargo tree line: {line}')
        key = (fields[0], fields[1][1:])
        candidates = by_key.get(key, [])
        if len(candidates) != 1:
            raise ValueError(f'Ambiguous or missing Cargo package: {key}')
        selected.add(candidates[0])
    if not selected:
        raise ValueError('Empty release dependency tree')
    return selected

def dependency_tree(metadata, target, roots):
    command = ['cargo', 'tree', '--locked', '--target', target, '--edges', 'normal,build',
               '--prefix', 'none', '--format', '{p}']
    for root in roots: command += ['-p', root]
    tree = subprocess.check_output(command, cwd=ROOT, text=True)
    return build_selection(metadata, tree), tree

def collect(metadata, output, selected=None, roots=None):
    output = Path(output); output.mkdir(parents=True)
    policy = json.loads((ROOT / 'release/license-policy.json').read_text())
    sources = json.loads((ROOT / 'assets/release-notices/sources.json').read_text())['packages']
    if selected is None:
        raise ValueError('Notice collection requires an explicit enabled release dependency tree')
    packages, gaps = [], []
    for package in sorted(metadata['packages'], key=lambda p: (p['name'], p['version'])):
        if package['id'] not in selected or package['source'] is None: continue
        key = package['name'] + '@' + package['version']
        if package['license'] not in policy['allowed_expressions']:
            raise ValueError(f'Unreviewed license expression for {key}: {package["license"]}')
        directory = Path(package['manifest_path']).parent
        destination = output / 'crates' / (package['name'] + '-' + package['version'])
        destination.mkdir(parents=True)
        shutil.copy2(package['manifest_path'], destination / 'Cargo.toml')
        files = []
        for path in sorted(directory.rglob('*')):
            if path.is_file() and notice(path):
                target = destination / path.relative_to(directory)
                target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
                files.append(str(target.relative_to(output)))
        for record in sources.get(key, {}).get('files', []):
            source = ROOT / 'assets/release-notices' / record['path']
            if sha(source) != record['sha256']: raise ValueError(f'Notice checksum mismatch: {source}')
            target = destination / 'upstream' / Path(record['path']).name
            target.parent.mkdir(exist_ok=True); shutil.copy2(source, target)
            files.append(str(target.relative_to(output)))
        if not files:
            if key not in policy['known_missing_texts']:
                raise ValueError(f'New missing upstream notice: {key}')
            gaps.append(dict(package=key, reason=policy['known_missing_texts'][key]))
        packages.append(dict(name=package['name'], version=package['version'], license=package['license'],
                             authors=package['authors'], repository=package['repository'], source=package['source'], notices=files))
    for record in sources['onnxruntime@1.22.0']['files']:
        source = ROOT / 'assets/release-notices' / record['path']
        if sha(source) != record['sha256']: raise ValueError('ONNX Runtime notice checksum mismatch')
        target = output / record['path']; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
    for source, name in [(ROOT/'LICENSE', 'PROJECT-MIT.txt'), (ROOT/'assets/compact-notices/Inflect-symbols-MIT.txt','Inflect-symbols-MIT.txt'), (ROOT/'assets/compact-notices/Kitten-Apache-2.0.txt','Kitten-Apache-2.0.txt'), (ROOT/'crates/dispatch-compat/README.md','Dispatch-compat-README.txt'), (ROOT/'assets/VOSK-LICENSE.txt','VOSK-Apache-2.0.txt'),
                         (ROOT/'assets/fonts/OFL.txt','Noto-OFL.txt'),
                         (ROOT/'crates/tts-core/src/qwen_design/WAVEKAT-LICENSE.txt','WaveKat-NOTICE.txt'),
                         (ROOT/'crates/tts-core/src/qwen_design/APACHE-2.0.txt','WaveKat-Apache-2.0.txt'),
                         (ROOT/'crates/tts-core/native/vendor/signalsmith-stretch/LICENSE.txt','Signalsmith-MIT.txt'),
                         (ROOT/'crates/tts-core/native/vendor/linear/LICENSE.txt','Linear-MIT.txt')]:
        shutil.copy2(source, output/name)
    # The standard library's own attribution record is supplied by the exact toolchain.
    rustdoc = Path(subprocess.check_output(['rustc','--print','sysroot'],text=True).strip())/'share/doc/rust'
    shutil.copy2(rustdoc/'COPYRIGHT-library.html', output/'Rust-standard-library-COPYRIGHT.html')
    shutil.copytree(rustdoc/'licenses', output/'rust-toolchain-license-texts')
    shutil.copy2(ROOT/'assets/release-notices/sources.json', output/'upstream-sources.json')
    report = dict(schema_version=1, scope='Enabled normal/build dependency tree for the shipped Cargo package roots; excludes dev and inactive optional dependencies. Python wheels and models are not bundled.', roots=roots,
                  packages=packages, unresolved=gaps, complete=not gaps)
    (output/'inventory.json').write_text(json.dumps(report, indent=2)+'\n')
    (output/'README.txt').write_text('Keep this directory with redistributed binaries. inventory.json lists the exact resolved crates and any unresolved notice gaps. Rust toolchain notices are a conservative superset; alternative permissive licenses remain available where declared. Training source notices are retained beside their sources. Model downloads retain their own notices. This inventory is not model/data clearance.\n')
    return report
