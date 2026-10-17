#!/usr/bin/env python3
"""Verify relocated packaged binaries, runtime installation and SDK linkage."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import wave


def run(command, env, cwd):
    return subprocess.check_output(list(map(str,command)),env=env,cwd=cwd,text=True,stderr=subprocess.STDOUT)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--app',type=Path)
    parser.add_argument('--sdk',required=True,type=Path)
    parser.add_argument('--report',required=True,type=Path)
    parser.add_argument('--voice',type=Path)
    parser.add_argument('--training-python',type=Path)
    parser.add_argument('--multilingual-voice',type=Path)
    parser.add_argument('--design-voice',type=Path)
    parser.add_argument('--compact-voices',type=Path)
    args=parser.parse_args()
    checks=[]
    cmake=shutil.which('cmake')
    with tempfile.TemporaryDirectory(prefix='Local TTS relocation ') as temporary:
        root=Path(temporary); app=root/'Moved Studio.app'; sdk=root/'Moved SDK'
        if args.app: shutil.copytree(args.app,app)
        shutil.copytree(args.sdk,sdk)
        # No Cargo, Homebrew, repository working directory or dynamic-library overrides.
        env={k:v for k,v in os.environ.items() if not k.startswith(('DYLD_','ORT_','LOCAL_TTS_','PYTHON'))}
        env.update(PATH='/usr/bin:/bin:/usr/sbin:/sbin',LOCAL_TTS_HOME=str(root/'User data'),PYTHONDONTWRITEBYTECODE='1')
        cwd=root/'Empty working directory';cwd.mkdir()
        cli=sdk/'bin/local-tts';studio=app/'Contents/MacOS/tts-studio'
        assert run([cli,'--version'],env,cwd).startswith('local-tts ')
        for language,text,expected in [('en','2 kg','two kilograms'),('ru','2 кг','два килогра́мма'),('uk','2 кг','два кілогра́ми'),('fr','ÉCOLE ET CŒUR','école et cœur'),('de','GRÜẞE AUS KÖLN','grüße aus köln'),('es','¡BUENOS DÍAS, NIÑO!','¡buenos días, niño!')]:
            assert run([cli,'text-prepare',language,text],env,cwd).strip()==expected
        checks.append('relocated CLI version and EN/RU/UK/FR/DE/ES text preparation without Cargo')
        workspace=sdk
        if args.app:
            runtime=json.loads(run([studio,'--check-runtime'],env,cwd));workspace=Path(runtime['root'])
            assert workspace.is_relative_to(root/'User data')
            (workspace/'user-recording.txt').write_text('preserve')
            assert json.loads(run([studio,'--check-runtime'],env,cwd))==runtime
            assert (workspace/'user-recording.txt').read_text()=='preserve'
            checks.append('Studio resources install outside bundle and preserve user files on relaunch')
        for home in set([workspace,sdk]):
            code=f"import sys; sys.path.insert(0,{str(home/'training')!r}); from runtime import cli; import subprocess; print(subprocess.check_output(cli()+['text-prepare','en','21st'],text=True).strip())"
            assert run([sys.executable,'-c',code],env,cwd).strip()=='twenty first'
        checks.append('packaged training kits resolve bundled CLI without Cargo')
        consumer=sdk/'bin/text-test'
        run(['/usr/bin/clang++','-std=c++17','-pthread','-I',sdk/'include',sdk/'examples/cpp/text_test.cpp',
             '-L',sdk/'lib','-llocal_tts','-Wl,-rpath,@loader_path/../lib','-o',consumer],env,cwd)
        run([consumer],env,cwd)
        # Move once more after linking: neither install name nor rpath may depend on staging.
        moved=root/'SDK moved again';sdk.rename(moved);sdk=moved;cli=sdk/'bin/local-tts'
        if not args.app: workspace=sdk
        run([sdk/'bin/text-test'],env,cwd)
        checks.append('C++ UTF-8, ownership, errors and concurrent calls after second relocation')
        cmake_consumer=None
        if cmake:
            build=root/'CMake build';installed=root/'CMake install'
            run([cmake,'-S',sdk,'-B',build,'-DCMAKE_BUILD_TYPE=Release'],env,cwd)
            run([cmake,'--build',build],env,cwd)
            run([cmake,'--install',build,'--prefix',installed],env,cwd)
            final_install=root/'CMake relocated';installed.rename(final_install)
            cmake_consumer=final_install/'bin/tts-speak'
            # Without a voice the usage exit still proves the installed dylib loads.
            usage=subprocess.run([cmake_consumer],env=env,cwd=cwd,capture_output=True,text=True)
            assert usage.returncode==2 and 'usage:' in usage.stderr
            checks.append('CMake consumer configures/builds/installs and loads after relocation')
        if args.voice:
            voice=root/'External voice';shutil.copytree(args.voice.resolve(),voice)
            wav=root/'speech.wav'
            run([cli,'synthesize',voice,wav,'Hello world. The price is $5.20.'],env,cwd)
            with wave.open(str(wav)) as f:
                samples=struct.unpack('<'+'h'*f.getnframes(),f.readframes(f.getnframes()))
                assert f.getnchannels()==1 and len(samples)>f.getframerate()//10 and max(map(abs,samples))>32
            checks.append('relocated native Kokoro inference produces finite non-silent PCM')
            consumer=sdk/'bin/speak-test'
            run(['/usr/bin/clang++','-std=c++17','-I',sdk/'include',sdk/'examples/cpp/speak.cpp',
                 '-L',sdk/'lib','-llocal_tts','-Wl,-rpath,@loader_path/../lib','-o',consumer],env,cwd)
            for index,executable in enumerate([consumer]+([cmake_consumer] if cmake_consumer else [])):
                wav=root/f'cpp-{index}.wav';run([executable,voice,wav],env,cwd)
                with wave.open(str(wav)) as f:
                    samples=struct.unpack('<'+'h'*f.getnframes(),f.readframes(f.getnframes()))
                    assert f.getnchannels()==1 and len(samples)>f.getframerate()//10 and max(map(abs,samples))>32
            checks.append('relocated C++ SDK consumers synthesize through the shipped shared library')
        if args.compact_voices:
            consumer=sdk/'bin/compact-test'
            run(['/usr/bin/clang++','-std=c++17','-I',sdk/'include',sdk/'examples/cpp/compact_test.cpp',
                 '-L',sdk/'lib','-llocal_tts','-Wl,-rpath,@loader_path/../lib','-o',consumer],env,cwd)
            for name,backend in [('inflect-nano-en','6'),('kitten-nano-en','7')]:
                run([consumer,args.compact_voices.resolve()/name,backend,root/(name+'.wav')],env,cwd)
            checks.append('relocated C++ Inflect Nano and Kitten Nano English native inference; external models not bundled')
        if args.design_voice:
            voice=args.design_voice.resolve()
            wav=root/'Designed English.wav'
            run([cli,'voice-design',voice,wav,'en','A calm adult male voice.','Hello, welcome.'],env,cwd)
            consumer=sdk/'bin/design-test'
            run(['/usr/bin/clang++','-std=c++17','-I',sdk/'include',sdk/'examples/cpp/design_test.cpp',
                 '-L',sdk/'lib','-llocal_tts','-Wl,-rpath,@loader_path/../lib','-o',consumer],env,cwd)
            russian=root/'Designed Russian.wav'
            run([consumer,voice,russian],env,cwd)
            for path in [wav,russian]:
                with wave.open(str(path)) as f:
                    assert f.getnchannels()==1 and f.getframerate()==24000
                    samples=struct.unpack('<'+'h'*f.getnframes(),f.readframes(f.getnframes()))
                    assert len(samples)>2400 and max(map(abs,samples))>32
            checks.append('relocated Qwen VoiceDesign CLI English and C++ Russian inference, cancellation and request validation; external models not bundled')
        if args.multilingual_voice:
            # Models are external resources, never baked into the native release.
            voice=args.multilingual_voice.resolve()
            phrase='Bonjour, le soleil entre par la fenêtre.'
            wav=root/'french-cli.wav'
            run([cli,'chatterbox-synthesize',voice,wav,'fr',phrase],env,cwd)
            consumer=sdk/'bin/european-test'
            run(['/usr/bin/clang++','-std=c++17','-I',sdk/'include',sdk/'examples/cpp/european_test.cpp',
                 '-L',sdk/'lib','-llocal_tts','-Wl,-rpath,@loader_path/../lib','-o',consumer],env,cwd)
            cpp_output=root/'European C++ audio'
            run([consumer,voice,cpp_output],env,cwd)
            for path in [wav]+[cpp_output/(code+'.wav') for code in ['fr','de','es']]:
                with wave.open(str(path)) as f:
                    samples=struct.unpack('<'+'h'*f.getnframes(),f.readframes(f.getnframes()))
                    assert f.getnchannels()==1 and f.getframerate()==24000 and len(samples)>24000 and max(map(abs,samples))>32
            checks.append('relocated French CLI plus French/German/Spanish C++ shared-library inference (external model)')
        if args.training_python:
            python=args.training_python.absolute(); dataset=root/'fixture';dataset.mkdir()
            rows=[]
            for split,frequency in [('train',220),('validation',330)]:
                with wave.open(str(dataset/(split+'.wav')),'wb') as f:
                    f.setparams((1,2,22050,0,'NONE','not compressed'))
                    f.writeframes(b''.join(struct.pack('<h',int(8000*math.sin(i*2*math.pi*frequency/22050))) for i in range(22050*4)))
                rows.append(dict(id=split,audio=split+'.wav',text='2.5 kg',speaker='fixture',split=split))
            manifest=dataset/'dataset.json';manifest.write_text(json.dumps(dict(schema_version=1,language='en',sample_rate=22050,
                rights=dict(audio='Synthetic tones',transcripts='Original fixture',speaker_consent='No human speaker'),utterances=rows)))
            base=root/'base';adapt=root/'adapt';voice=root/'exported voice'
            launch=workspace/'training/launch.py'
            common=['--dataset',manifest,'--tiny','--steps','1','--batch-size','1']
            run([python,launch,'--output',base,'--from-scratch']+common,env,cwd)
            run([python,launch,'--output',adapt,'--init-checkpoint',base/'latest.pt']+common,env,cwd)
            run([python,workspace/'training/export_and_verify.py','--checkpoint',adapt/'latest.pt','--output',voice],env,cwd)
            run([cli,'vits-synthesize',voice,root/'fixture.wav','2.5 kg'],env,cwd)
            checks.append('installed training kit scratch/fine-tune/export/native parity without Cargo (synthetic tones)')
        if args.app:
            # Corrupt only disposable installed code: app must refuse it, not overwrite it.
            installed_script=workspace/'training/runtime.py';installed_script.write_text('modified')
            result=subprocess.run([studio,'--check-runtime'],env=env,cwd=cwd,capture_output=True,text=True)
            assert result.returncode!=0 and 'runtime resource changed' in result.stderr
            checks.append('changed installed code is rejected without overwriting user data')
    args.report.write_text(json.dumps(dict(passed=True,checks=checks,gui_and_audio_hardware_tested=False),indent=2)+'\n')
    print('Release relocation checks passed',flush=True)

if __name__=='__main__': main()
