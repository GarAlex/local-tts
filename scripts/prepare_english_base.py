"""Acquire and import the pinned English training base; never replace an existing base."""
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
subprocess.run([sys.executable, str(ROOT / 'scripts/download_vits_base.py')], check=True)
subprocess.run([sys.executable, str(ROOT / 'training/import_ljs.py')], check=True)
