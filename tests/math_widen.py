"""Channel widening retains function and breaks replicated-weight symmetry."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
import torch
from train import MATH_MODEL,MATH_WIDE_MODEL,SynthesizerTrn
from math_widen import widen


class Widening(unittest.TestCase):
    def test_preserves_dynamic_length_decoder_output(self):
        torch.set_num_threads(2);torch.manual_seed(1234)
        old=SynthesizerTrn(3,513,32,**MATH_MODEL).eval()
        new=SynthesizerTrn(3,513,32,**MATH_WIDE_MODEL).eval()
        widen(old.dec,new.dec)
        with torch.no_grad():
            for frames in (8,17,32):
                z=torch.randn(1,32,frames)
                self.assertLess(float((old.dec(z)-new.dec(z)).abs().max()),5e-6)
        # Asymmetric incoming connections preserve summed activations but allow
        # replicated channels to receive different gradients during training.
        weights=new.dec.ups[0].weight_v
        self.assertFalse(torch.equal(weights[:64],weights[64:128]))
        self.assertGreater(sum(p.numel() for p in new.dec.parameters()),10*sum(p.numel() for p in old.dec.parameters()))


if __name__=='__main__':unittest.main()
