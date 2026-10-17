"""Create the project-only Python training environment."""
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
if sys.version_info < (3, 10):
    raise SystemExit('Training requires Python 3.10 or later; select a newer executable.')
environment = root / '.venv'
if Path(sys.prefix).resolve() != environment.resolve():
    subprocess.run([sys.executable, '-m', 'venv', str(environment)], check=True)
python = environment / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
subprocess.run([str(python), '-m', 'pip', 'install', '-r', str(root / 'training/requirements-lock.txt')], check=True)
print(f'Ready. Select Use project environment: {python}', flush=True)
