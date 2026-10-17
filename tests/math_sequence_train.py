"""Verify held-out isolation, checkpoint resume and teacher-cache integrity."""
import sys,json,hashlib,tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from math_sequence_train import fit,spoken_hash
class Pipeline(unittest.TestCase):
    def test_test_targets_do_not_train_student_and_resume_keeps_lineage(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);states=[]
            for label,test_offset in [('a',10),('b',100000)]:
                dataset=root/label;dataset.mkdir();rows=[]
                for i,split in enumerate(['train','validation','test']):
                    latent=torch.arange(768,dtype=torch.float32).reshape(192,4)/192+(test_offset if split=='test' else i*.1)
                    file=dataset/f'{i}.pt';torch.save(dict(tokens=torch.tensor([1,2]),durations=torch.tensor([2,2]),latent=latent),file)
                    rows.append(dict(file=file.name,sha256=hashlib.sha256(file.read_bytes()).hexdigest(),spoken=split,split=split,frames=4))
                (dataset/'dataset.json').write_text(json.dumps(dict(schema_version=1,language='ru',channels=192,vocabulary=62,records=rows,teacher_checkpoint_sha256='0'*64,source_origin={'mode':'synthetic fixture'},rights={'audio':'generated tensors'})))
                output=root/f'fit-{label}';fit(SimpleNamespace(dataset=dataset,output=output,steps=1,batch=1,resume=None))
                checkpoint=torch.load(output/'student.pt',weights_only=True);states.append(checkpoint['model'])
                self.assertEqual(checkpoint['step'],1)
                self.assertFalse(json.loads((output/'report.json').read_text())['decoder_updated'])
            for key in states[0]:torch.testing.assert_close(states[0][key],states[1][key],rtol=0,atol=0)
            fit(SimpleNamespace(dataset=root/'a',output=root/'resume',steps=1,batch=1,resume=root/'fit-a/student.pt'))
            self.assertEqual(torch.load(root/'resume/student.pt',weights_only=True)['step'],2)
            with self.assertRaisesRegex(ValueError,'resume dataset differs'):
                fit(SimpleNamespace(dataset=root/'b',output=root/'bad-resume',steps=1,batch=1,resume=root/'fit-a/student.pt'))
            self.assertFalse((root/'bad-resume').exists())
            weighted=torch.load(root/'fit-a/student.pt',weights_only=True);weighted['waveform_weight']=0.1
            torch.save(weighted,root/'weighted.pt')
            with self.assertRaisesRegex(ValueError,'requires source teacher'):
                fit(SimpleNamespace(dataset=root/'a',output=root/'missing-teacher',steps=1,batch=1,resume=root/'weighted.pt'))
            self.assertFalse((root/'missing-teacher').exists())
            fit(SimpleNamespace(dataset=root/'b',output=root/'transfer',steps=1,batch=1,resume=None,initialize=root/'fit-a/student.pt',initial_dataset=root/'a'))
            transferred=torch.load(root/'transfer/student.pt',weights_only=True)
            self.assertTrue(transferred['initialization']['optimizer_reset'])
            self.assertEqual(transferred['initialization']['source_step'],1)
            torch.testing.assert_close(transferred['mean'],torch.load(root/'fit-a/student.pt',weights_only=True)['mean'])
            historical=torch.load(root/'fit-a/student.pt',weights_only=True);historical['training_spoken_sha256']=[spoken_hash('test')]
            torch.save(historical,root/'historical.pt')
            with self.assertRaisesRegex(ValueError,'prior training overlaps'):
                fit(SimpleNamespace(dataset=root/'b',output=root/'history-leak',steps=1,batch=1,resume=None,initialize=root/'historical.pt',initial_dataset=root/'a'))
            self.assertFalse((root/'history-leak').exists())
            changed=json.loads((root/'b/dataset.json').read_text());changed['records'][0]['spoken']='new training';changed['records'][1]['spoken']='train'
            (root/'b/dataset.json').write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,'prior training overlaps'):
                fit(SimpleNamespace(dataset=root/'b',output=root/'direct-leak',steps=1,batch=1,resume=None,initialize=root/'fit-a/student.pt',initial_dataset=root/'a'))
            self.assertFalse((root/'direct-leak').exists())
            self.assertEqual(torch.load(root/'resume/student.pt',weights_only=True)['duration_weight'],0.2)
            fit(SimpleNamespace(dataset=root/'a',output=root/'duration-fit',steps=1,batch=1,resume=root/'fit-a/student.pt',duration_weight=1.0))
            duration_checkpoint=torch.load(root/'duration-fit/student.pt',weights_only=True)
            self.assertEqual(duration_checkpoint['duration_weight'],1.0)
            self.assertTrue(any(not torch.equal(duration_checkpoint['model'][key],torch.load(root/'resume/student.pt',weights_only=True)['model'][key]) for key in duration_checkpoint['model']))
            fit(SimpleNamespace(dataset=root/'a',output=root/'duration-resume',steps=1,batch=1,resume=root/'duration-fit/student.pt'))
            self.assertEqual(torch.load(root/'duration-resume/student.pt',weights_only=True)['duration_weight'],1.0)
            for i,weight in enumerate([float('nan'),float('inf'),-1,11]):
                with self.assertRaisesRegex(ValueError,'duration weight'):
                    fit(SimpleNamespace(dataset=root/'a',output=root/f'invalid-duration-{i}',steps=1,batch=1,resume=None,duration_weight=weight))
                self.assertFalse((root/f'invalid-duration-{i}').exists())
            (root/'a/0.pt').write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError,'checksum differs'):
                fit(SimpleNamespace(dataset=root/'a',output=root/'tampered',steps=1,batch=1,resume=None))
            self.assertFalse((root/'tampered').exists())
if __name__=='__main__':unittest.main()
