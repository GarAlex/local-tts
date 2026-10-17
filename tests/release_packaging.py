"""Failure-path checks for release integrity and native CLI discovery."""
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from verify_release import relative, safe_tar, verify
from release_notices import sha, collect, build_selection
spec=importlib.util.spec_from_file_location('training_runtime',ROOT/'training/runtime.py')
runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)

class Packaging(unittest.TestCase):
    def test_archive_escape_and_links_rejected_before_extraction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for name,kind in [('../escape',tarfile.REGTYPE),('safe/link',tarfile.SYMTYPE),('/tmp/escape',tarfile.REGTYPE)]:
                archive=root/'bad.tar.gz'
                with tarfile.open(archive,'w:gz') as f:
                    entry=tarfile.TarInfo(name);entry.type=kind;entry.linkname='/tmp/escape'
                    f.addfile(entry,io.BytesIO(b''))
                with self.assertRaises(ValueError): safe_tar(archive,root/'extract')
                self.assertFalse((root/'extract').exists())
    def test_changed_unlisted_and_unsafe_files_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'cli').write_bytes(b'original')
            manifest=dict(schema_version=1,files={'cli':sha(root/'cli')})
            (root/'release-manifest.json').write_text(json.dumps(manifest))
            (root/'SHA256SUMS').write_text(''.join(f'{sha(root/n)}  {n}\n' for n in ['cli','release-manifest.json']))
            verify(root)
            (root/'extra').write_text('unlisted')
            with self.assertRaises(ValueError):verify(root)
            (root/'extra').unlink();(root/'cli').write_bytes(b'changed')
            with self.assertRaises(ValueError):verify(root)
            with self.assertRaises(ValueError):relative('../outside')
    def test_packaged_cli_does_not_fall_back_to_cargo(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ,{},clear=True):
            root=Path(temporary)
            with patch.object(runtime,'ROOT',root):
                with self.assertRaises(ValueError):runtime.cli()
                (root/'bin').mkdir();binary=root/'bin'/('local-tts.exe' if os.name=='nt' else 'local-tts');binary.touch()
                self.assertEqual(runtime.cli(),[str(binary)])
                with patch.dict(os.environ,{'LOCAL_TTS_CLI':str(root/'missing')}):
                    with self.assertRaises(ValueError):runtime.cli()
    def test_dependency_selection_excludes_unbuilt_optional_packages(self):
        metadata={'packages':[dict(name='built',version='1.0.0',id='built-id'),
                              dict(name='optional',version='1.0.0',id='optional-id')]}
        self.assertEqual(build_selection(metadata,'built v1.0.0\n\nbuilt v1.0.0 (*)\n'),{'built-id'})
        for tree in ['', 'unknown v1.0.0', 'malformed']:
            with self.assertRaises(ValueError):build_selection(metadata,tree)
        metadata['packages'].append(dict(name='built',version='1.0.0',id='different-source'))
        with self.assertRaises(ValueError):build_selection(metadata,'built v1.0.0')
    def test_new_license_expression_blocks_packaging(self):
        with tempfile.TemporaryDirectory() as temporary:
            package=dict(id='test',name='test',version='1',source='registry',license='UNREVIEWED',manifest_path='unused')
            metadata=dict(resolve=dict(nodes=[dict(id='test')]),packages=[package])
            with self.assertRaisesRegex(ValueError,'Unreviewed license'):
                collect(metadata,Path(temporary)/'notices',{'test'},['test'])

if __name__=='__main__':unittest.main()
