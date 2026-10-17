"""Ensure direct text-prior supervision reaches the inference network."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
import torch
from train import Acoustics, MATH_MODEL, SynthesizerTrn, math_prior_mel_loss


class PriorObjective(unittest.TestCase):
    def test_prior_loss_reaches_text_flow_and_decoder(self):
        torch.set_num_threads(2)
        torch.manual_seed(1234)
        net = SynthesizerTrn(3, 513, 32, **MATH_MODEL)
        tokens = torch.tensor([[0, 1, 0, 2, 0]])
        _, means, _, _ = net.enc_p(tokens, torch.tensor([5]))
        aligned = means.repeat_interleave(8, dim=2)
        mask = torch.ones(1, 1, 40)
        acoustics = Acoustics('cpu')
        target = acoustics.to_mel(acoustics.spec(torch.randn(1, 8192) * 0.05))
        loss = math_prior_mel_loss(net, aligned, mask, torch.tensor([0]), target, acoustics)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name in ('enc_p', 'flow', 'dec'):
            grads = [p.grad for p in getattr(net, name).parameters() if p.grad is not None]
            self.assertTrue(grads, name)
            self.assertTrue(all(torch.isfinite(g).all() for g in grads), name)
            self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0, name)
        self.assertTrue(all(p.grad is None for p in net.enc_q.parameters()))


if __name__ == '__main__':
    unittest.main()
