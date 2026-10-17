"""Native CLI selection for source checkouts and installed training kits."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def cli():
    override = os.environ.get('LOCAL_TTS_CLI')
    if override:
        binary = Path(override).expanduser().resolve()
        if not binary.is_file():
            raise ValueError(f'LOCAL_TTS_CLI does not identify a file: {binary}')
        return [str(binary)]
    name = 'local-tts.exe' if os.name == 'nt' else 'local-tts'
    for binary in [ROOT / 'bin' / name, ROOT / 'target/debug' / name]:
        if binary.is_file():
            return [str(binary)]
    if (ROOT / 'Cargo.toml').is_file():
        return ['cargo', 'run', '--locked', '-p', 'local-tts-cli', '--']
    raise ValueError('Native local-tts executable is missing; reinstall the training kit.')
