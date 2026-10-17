"""Quantization-aware training must match packed inference and isolate test data."""
import sys,tempfile,hashlib,copy
from pathlib import Path
import unittest
import numpy as np
import onnxruntime as ort
from onnx import helper,TensorProto,numpy_helper
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from math_sequence_decoder_qat import fake_quantize,quantized_forward,read_latents
from math_pretrained_compress import packed_four_bit,compress

class Qat(unittest.TestCase):
    def test_rounding_matches_native_unpack_and_keeps_gradients(self):
        torch.manual_seed(1)
        weight=torch.randn(32,37,requires_grad=True)
        with torch.no_grad():weight[0].zero_()
        actual=fake_quantize(weight)
        constants,nodes=packed_four_bit(weight.detach().numpy(),'w')
        model=helper.make_model(helper.make_graph(nodes,'weights',[],[helper.make_tensor_value_info('w',TensorProto.FLOAT,[32,37])],constants),opset_imports=[helper.make_opsetid('',17)],ir_version=10)
        native=ort.InferenceSession(model.SerializeToString(),providers=['CPUExecutionProvider']).run(None,{})[0]
        np.testing.assert_allclose(actual.detach().numpy(),native,atol=3e-7,rtol=1e-6)
        actual.sum().backward()
        torch.testing.assert_close(weight.grad,torch.ones_like(weight))
        net=torch.nn.Linear(37,32);teacher=copy.deepcopy(net).requires_grad_(False)
        before={k:v.clone() for k,v in teacher.state_dict().items()}
        x=torch.randn(2,37);optimizer=torch.optim.AdamW(net.parameters(),lr=1e-4)
        loss=(quantized_forward(net,x)-teacher(x)).square().mean();loss.backward();optimizer.step()
        self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters()))
        for k,v in teacher.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))

    def test_int8_qat_matches_export_per_axis(self):
        weight=torch.randn(32,37,requires_grad=True)
        with torch.no_grad():weight[0].zero_()
        model=helper.make_model(helper.make_graph([helper.make_node('Identity',['w'],['out'])],'weights',[],[helper.make_tensor_value_info('out',TensorProto.FLOAT,[32,37])],[numpy_helper.from_array(weight.detach().numpy(),'w')]),opset_imports=[helper.make_opsetid('',17)],ir_version=10)
        compress(model,8)
        native=ort.InferenceSession(model.SerializeToString(),providers=['CPUExecutionProvider']).run(None,{})[0]
        actual=fake_quantize(weight,bits=8)
        np.testing.assert_allclose(actual.detach().numpy(),native,atol=3e-7,rtol=1e-6)
        actual.sum().backward();torch.testing.assert_close(weight.grad,torch.ones_like(weight))
        with self.assertRaisesRegex(ValueError,'bits'):fake_quantize(weight,bits=3)

    def test_reserved_test_tensor_never_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);records=[]
            for i,split in enumerate(['train','validation','test']):
                path=root/f'{i}.pt';torch.save({'latent':torch.full((192,4),float(i))},path)
                records.append(dict(spoken=split,split=split,file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            (root/'2.pt').write_bytes(b'not a tensor, intentionally unreadable')
            train,val=read_latents(root,{'records':records})
            self.assertEqual(len(train),1);self.assertEqual(len(val),1)
            self.assertTrue((train[0]==0).all());self.assertTrue((val[0]==1).all())
            (root/'1.pt').write_bytes(b'bad validation')
            with self.assertRaisesRegex(ValueError,'integrity'):read_latents(root,{'records':records})

if __name__=='__main__':unittest.main()
