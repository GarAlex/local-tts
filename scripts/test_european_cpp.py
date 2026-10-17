#!/usr/bin/env python3
"""Compile the C ABI consumer and exercise French, German and Spanish."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
subprocess.run(['cargo','build','--release','-p','local-tts-ffi'],cwd=ROOT,check=True)
out=ROOT/'target/european-cpp'
out.mkdir(exist_ok=True)
consumer=out/'european-test'
subprocess.run(['c++','-std=c++17','-Wall','-Wextra','-Werror','-I',str(ROOT/'include'),
                str(ROOT/'examples/cpp/european_test.cpp'),'-L',str(ROOT/'target/release'),
                '-llocal_tts',f'-Wl,-rpath,{ROOT/"target/release"}','-o',str(consumer)],check=True)
subprocess.run([str(consumer),str(ROOT/'voices/chatterbox-multilingual'),str(out)],check=True,timeout=300)
print('French/German/Spanish C++ inference passed')
