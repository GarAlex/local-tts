"""Keep fixed STFT synthesis nodes out of learned-upsample stage selection."""
import sys
from pathlib import Path
import unittest
import numpy as np
from onnx import helper, numpy_helper
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from math_sequence_repack import stage_weights, middle_weights

class Stages(unittest.TestCase):
    def test_uses_learned_weights_not_later_stft_transpose(self):
        weights=[numpy_helper.from_array(np.ones((32,32),dtype=np.float32),name) for name in ['pre','up1','middle','up2','post']]
        nodes=[helper.make_node(kind,['x',weight],['x'+str(i)]) for i,(kind,weight) in enumerate([('Conv','pre'),('ConvTranspose','up1'),('Conv','middle'),('ConvTranspose','up2'),('Conv','post'),('ConvTranspose','fixed-filter-from-constant')])]
        model=helper.make_model(helper.make_graph(nodes,'fixture',[],[],weights))
        self.assertEqual(stage_weights(model,False),['pre','up1'])
        self.assertEqual(stage_weights(model,True),['post','up2'])
        empty=helper.make_model(helper.make_graph(nodes[-1:],'fixed',[],[]))
        with self.assertRaisesRegex(ValueError,'no learned upsampling'):stage_weights(empty,True)

    def test_middle_kernel_groups_exclude_input_and_final_stage(self):
        spec=[('pre',7),('up1',16),('middle3',3),('middle7',7),('up2',16),('post',7)]
        weights=[numpy_helper.from_array(np.ones((32,32,k),dtype=np.float32),name) for name,k in spec]
        nodes=[helper.make_node('ConvTranspose' if name.startswith('up') else 'Conv',['x',name],['out'+name]) for name,_ in spec]
        model=helper.make_model(helper.make_graph(nodes,'stages',[],[],weights))
        self.assertEqual(middle_weights(model),{3:['middle3'],7:['middle7']})

if __name__=='__main__':unittest.main()
