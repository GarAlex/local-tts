"""Sampled latent export preserves variance, deterministic parity and old contracts."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
import torch
from export import Export, FixedLatentNoise
from train import MATH_MODEL, SynthesizerTrn


class OldContract(torch.nn.Module):
    def infer(self, tokens, lengths, sid, noise_scale, length_scale, noise_scale_w):
        assert noise_scale == 0 and noise_scale_w == 0
        return (tokens.float().unsqueeze(1),)


class SampledExport(unittest.TestCase):
    def test_zero_noise_preserves_old_inference_contract(self):
        output = Export(OldContract())(torch.tensor([[1, 2]]), torch.tensor([2]), torch.tensor([1.0]))
        self.assertEqual(output.tolist(), [[[1.0, 2.0]]])

    def test_dynamic_noise_has_variance_and_is_reproducible(self):
        noise = FixedLatentNoise(32)
        first = noise(torch.zeros(1, 32, 20))
        other = FixedLatentNoise(32)(torch.zeros(1, 32, 20))
        self.assertTrue(torch.equal(first, other))
        self.assertLess(abs(float(noise.bank.mean())), 0.02)
        self.assertLess(abs(float(noise.bank.std()) - 1), 0.02)
        long = noise(torch.zeros(1, 32, 4100))
        self.assertEqual(long.shape, (1, 32, 4100))
        self.assertTrue(torch.equal(long[:, :, :4], long[:, :, 4096:]))

    def test_sampled_inference_is_repeatable_and_differs_from_mean(self):
        torch.set_num_threads(2)
        torch.manual_seed(1234)
        model = SynthesizerTrn(3, 513, 32, **MATH_MODEL).eval()
        sampled = Export(model, latent_noise_scale=0.667).eval()
        deterministic = Export(model).eval()
        args = (torch.tensor([[0, 1, 0, 2, 0]]), torch.tensor([5]), torch.tensor([1.0]))
        with torch.no_grad():
            first = sampled(*args)
            self.assertTrue(torch.equal(first, sampled(*args)))
            self.assertFalse(torch.equal(first, deterministic(*args)))
            self.assertTrue(torch.isfinite(first).all())


if __name__ == '__main__':
    unittest.main()
