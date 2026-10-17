"""Waveform loss must train latent inputs without changing a frozen decoder."""
import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from math_sequence_train import frozen_waveform_loss
class FrozenDecoder(unittest.TestCase):
    def test_gradient_reaches_inputs_and_never_decoder_weights(self):
        torch.set_num_threads(2);torch.manual_seed(8)
        decoder=torch.nn.Conv1d(3,1,5,padding=2).eval().requires_grad_(False)
        before={k:v.clone() for k,v in decoder.state_dict().items()}
        latent=torch.randn(1,3,4096,requires_grad=True)
        with torch.no_grad():teacher=decoder(torch.randn_like(latent))
        actual=decoder(latent)
        self.assertEqual(float(frozen_waveform_loss(teacher,teacher)),0.0)
        loss=frozen_waveform_loss(actual,teacher);loss.backward()
        self.assertTrue(torch.isfinite(latent.grad).all())
        self.assertGreater(float(latent.grad.abs().sum()),0)
        for parameter in decoder.parameters():self.assertIsNone(parameter.grad)
        for key,value in decoder.state_dict().items():torch.testing.assert_close(value,before[key],rtol=0,atol=0)
if __name__=='__main__':unittest.main()
