"""Prevent metrics from comparing different utterances or altered audio controls."""
import sys,json,tempfile,hashlib,copy
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from math_sequence_compare import compare
class PairedChecks(unittest.TestCase):
    def test_pairing_and_hash_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            base=dict(language='ru',teacher_sha256='teacher',dataset_sha256='dataset',split='validation',decoder_precision='float32',decoder_updated=False,student_step=1000,student_sha256='student',cases=[dict(expression='1 + 2',spoken='fixture',recordings=[dict(kind='teacher',file='teacher.wav',sha256=hashlib.sha256(b'fixture').hexdigest())],teacher_timed_log_spectral_error=1.,duration_frame_mae=.5)])
            for name in ['before','after']:
                folder=root/name;folder.mkdir();(folder/'teacher.wav').write_bytes(b'fixture');(folder/'report.json').write_text(json.dumps(base))
            before=root/'before/report.json';after=root/'after/report.json'
            self.assertTrue(compare(before,after)['teacher_controls_identical'])
            for key,value,message in [('expression','1 + 3','utterances differ'),('teacher_timed_log_spectral_error',float('nan'),'invalid comparison metric')]:
                changed=copy.deepcopy(base);changed['cases'][0][key]=value;after.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError,message):compare(before,after)
            changed=copy.deepcopy(base);changed['dataset_sha256']='expanded';after.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,'dataset_sha256'):compare(before,after)
            result=compare(before,after,allow_shared_cases=True)
            self.assertEqual(result['after_dataset_sha256'],'expanded')
            self.assertIn('shared identical',result['comparison_scope'])
            after.write_text(json.dumps(base));(root/'after/teacher.wav').write_bytes(b'altered')
            with self.assertRaisesRegex(ValueError,'audio hash/path'):compare(before,after)
if __name__=='__main__':unittest.main()
