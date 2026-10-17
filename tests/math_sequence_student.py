"""Sequence boundaries and supervision must not leak padded acoustics into speech."""
import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from math_sequence_student import SequenceStudent,supervised_loss
class SequenceContract(unittest.TestCase):
    def test_padding_invariance_and_masked_gradients(self):
        torch.set_num_threads(2);torch.manual_seed(10)
        model=SequenceStudent(62).eval()
        tokens=torch.tensor([[1,2,3,4],[5,6,7,8]])
        lengths=torch.tensor([2,4]);durations=torch.tensor([[2,3,0,0],[1,2,2,2]])
        predicted,log_duration,mask=model(tokens,lengths,durations)
        changed=tokens.clone();changed[0,2:]=torch.tensor([40,50])
        again,_,_=model(changed,lengths,durations)
        torch.testing.assert_close(predicted[0,:,:5],again[0,:,:5])
        self.assertEqual(predicted[0,:,5:].count_nonzero(),0)
        target=torch.randn_like(predicted);token_mask=torch.arange(4)[None,:]<lengths[:,None]
        losses=supervised_loss(predicted,target,log_duration,durations,token_mask,mask)
        alternate=target.clone();alternate[0,:,5:]=100000
        other=supervised_loss(predicted,alternate,log_duration,durations,token_mask,mask)
        torch.testing.assert_close(losses[0],other[0])
        sum(losses).backward()
        self.assertTrue(torch.isfinite(model.embedding.weight.grad).all())
        self.assertGreater(model.embedding.weight.grad.abs().sum(),0)
    def test_invalid_alignment_and_parameter_budget(self):
        model=SequenceStudent(62)
        self.assertLess(sum(p.numel() for p in model.parameters())*4,1500000)
        for durations in [torch.tensor([[1,-1]]),torch.tensor([[0,0]]),torch.tensor([[7741,0]]),torch.tensor([[1,1]])]:
            with self.assertRaises(ValueError):model(torch.tensor([[1,2]]),torch.tensor([1]),durations)
    def test_predicted_timing_is_bounded_masked_and_reproducible(self):
        torch.set_num_threads(2);torch.manual_seed(12)
        model=SequenceStudent(62).eval()
        tokens=torch.tensor([[1,2,3],[4,5,6]]);lengths=torch.tensor([2,3])
        with torch.no_grad():
            model.duration.weight.zero_();model.duration.bias.fill_(torch.log(torch.tensor(4.0)))
        latent,duration,mask=model.infer(tokens,lengths)
        self.assertEqual(duration.tolist(),[[3,3,0],[3,3,3]])
        expected,_,expected_mask=model(tokens,lengths,duration)
        torch.testing.assert_close(latent,expected)
        self.assertTrue(torch.equal(mask,expected_mask))
        self.assertEqual(latent[0,:,6:].count_nonzero(),0)
        _,slow,_=model.infer(tokens,lengths,2.0)
        self.assertEqual(slow.tolist(),[[6,6,0],[6,6,6]])
        for invalid in [0.1,2.1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):model.infer(tokens,lengths,invalid)
        model.train()
        with self.assertRaisesRegex(ValueError,'eval mode'):model.infer(tokens,lengths)
        model.eval()
        with torch.no_grad():model.duration.bias.fill_(100.0)
        with self.assertRaisesRegex(ValueError,'frame limit'):
            model.infer(torch.ones(1,100,dtype=torch.long),torch.tensor([100]))
        with torch.no_grad():model.duration.bias.fill_(float('nan'))
        with self.assertRaisesRegex(ValueError,'nonfinite'):model.infer(tokens,lengths)

if __name__=='__main__':unittest.main()
