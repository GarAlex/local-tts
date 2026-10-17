"""Staged prior fitting updates text/durations while preserving acoustic weights."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
import torch
from train import MATH_WIDE_MODEL,SynthesizerTrn
from math_prior_fit import prior_loss
from math_alignment import alignment_batch


class PriorFit(unittest.TestCase):
    def test_frozen_acoustics_and_trainable_text_duration_gradients(self):
        torch.set_num_threads(2);torch.manual_seed(1234)
        net=SynthesizerTrn(3,513,32,**MATH_WIDE_MODEL).eval()
        for p in net.parameters():p.requires_grad_(False)
        parameters=list(net.enc_p.parameters())+list(net.dp.parameters())
        for p in parameters:p.requires_grad_(True)
        original={k:v.clone() for k,v in net.state_dict().items() if k.startswith(('enc_q.','flow.','dec.'))}
        records=[dict(audio='train')];labels={'train':[8,8,8,8,8]}
        x=torch.tensor([[0,1,0,2,0]]);xl=torch.tensor([5]);sl=torch.tensor([40])
        path=alignment_batch(records,labels,xl,sl)
        mean,latent,duration=prior_loss(net,x,xl,path,torch.ones(1,1,40),torch.randn(1,32,40),path.sum(2))
        loss=mean.square().mean()+latent+duration
        self.assertTrue(torch.isfinite(loss))
        optimizer=torch.optim.AdamW(parameters,lr=0.0002)
        loss.backward()
        for module in (net.enc_p,net.dp):
            gradients=[p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(float(g.abs().sum()) for g in gradients),0)
        optimizer.step()
        for key,value in original.items():self.assertTrue(torch.equal(value,net.state_dict()[key]),key)
        for module in (net.enc_q,net.flow,net.dec):self.assertTrue(all(p.grad is None for p in module.parameters()))


if __name__=='__main__':unittest.main()
