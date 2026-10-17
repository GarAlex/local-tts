"""Check stored-weight compression on native-runtime compatible convolutions."""
import sys
from pathlib import Path
import unittest
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import onnxruntime as ort
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
from math_pretrained_compress import compress, packed_four_bit, packed_five_bit


class Compression(unittest.TestCase):
    def test_dynamic_convolution_zero_channel_and_error_bound(self):
        rng = np.random.default_rng(8)
        weights = rng.normal(size=(16, 32, 3)).astype(np.float32)
        weights[0] = 0
        graph = helper.make_graph([helper.make_node('Conv', ['x', 'w'], ['y'])], 'test',
                                 [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1,32,'n'])],
                                 [helper.make_tensor_value_info('y', TensorProto.FLOAT, [1,16,'m'])],
                                 [numpy_helper.from_array(weights, 'w')])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('',17)], ir_version=10)
        before = ort.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])
        for bits,block,overrides in [(4,64,[]),(4,32,[]),(4,16,[]),(5,64,[]),(8,64,[]),(4,64,["w"])]:
            candidate=onnx.ModelProto();candidate.CopyFrom(model)
            self.assertEqual(compress(candidate,bits,block,overrides), 1)
            after = ort.InferenceSession(candidate.SerializeToString(), providers=['CPUExecutionProvider'])
            for length in (4,17):
                x = rng.normal(size=(1,32,length)).astype(np.float32)
                a, b = before.run(None, {'x':x})[0], after.run(None, {'x':x})[0]
                self.assertTrue(np.isfinite(b).all())
                np.testing.assert_array_equal(b[:,0], np.zeros_like(b[:,0]))
                # Per-block quantization cannot exceed this global rounding bound.
                scale = np.max(np.abs(weights))/(127 if bits==8 or overrides else 15 if bits==5 else 7)
                bound = scale/2 * 96 * np.abs(x).max() + 1e-4
                self.assertLess(float(np.max(np.abs(a-b))), bound)

    def test_invalid_mixed_override(self):
        model=helper.make_model(helper.make_graph([], 'empty', [], []),opset_imports=[helper.make_opsetid('',17)],ir_version=10)
        with self.assertRaisesRegex(ValueError,'ineligible int8'):
            compress(model,4,64,['missing'])

    def test_block_padding_and_rounding_bound(self):
        values=np.arange(1045,dtype=np.float32).reshape(5,209)/500-1
        for bits,block in [(4,16),(4,32),(4,64),(5,16),(5,64)]:
            pack=packed_four_bit if bits==4 else packed_five_bit
            constants,nodes=pack(values,'weights',block)
            model=helper.make_model(helper.make_graph(nodes,'unpack',[],[helper.make_tensor_value_info('weights',TensorProto.FLOAT,list(values.shape))],constants),opset_imports=[helper.make_opsetid('',17)],ir_version=10)
            session=ort.InferenceSession(model.SerializeToString(),providers=['CPUExecutionProvider'])
            restored=session.run(None,{})[0]
            self.assertEqual(restored.shape,values.shape)
            flat=values.reshape(-1)
            padded=np.pad(flat,(0,(-flat.size)%block)).reshape(-1,block)
            scales=np.maximum(np.abs(padded).max(1)/(7 if bits==4 else 15),np.finfo(np.float32).tiny)
            bound=np.repeat(scales/2,block)[:flat.size]+1e-6
            self.assertTrue((np.abs(restored.reshape(-1)-flat)<=bound).all())
        for bad in (0,4,9,1026,16.0):
            with self.assertRaisesRegex(ValueError,'block size'):packed_five_bit(values,'w',bad)
        for bad in (0,1,3,-2,1026,16.0):
            with self.assertRaisesRegex(ValueError,'block size'):packed_four_bit(values,'w',bad)


if __name__ == '__main__': unittest.main()
