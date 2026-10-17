"""Compare portable monotonic alignment to exhaustive valid paths."""
import itertools
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training/vendor/vits'))
import numpy as np
import torch
from monotonic_align import maximum_path

rng = np.random.default_rng(42)
for ty in range(2, 9):
    for tx in range(1, min(ty, 4) + 1):
        for _ in range(5):
            scores = rng.normal(size=(ty, tx)).astype(np.float32)
            result = maximum_path(torch.tensor(scores[None]), torch.ones(1, ty, tx))[0].numpy()
            np.testing.assert_array_equal(result.sum(axis=1), np.ones(ty))
            indices = result.argmax(axis=1)
            assert indices[0] == 0 and indices[-1] == tx - 1
            assert np.all(np.isin(np.diff(indices), [0, 1]))
            best = -np.inf
            for moves in itertools.combinations(range(1, ty), tx - 1):
                path = np.zeros(ty, dtype=int)
                for move in moves: path[move:] += 1
                best = max(best, scores[np.arange(ty), path].sum())
            np.testing.assert_allclose((result * scores).sum(), best, atol=1e-5)
# Padded batch respects both dimensions of the mask.
mask = torch.zeros(2, 8, 4); mask[0, :8, :4] = 1; mask[1, :5, :2] = 1
result = maximum_path(torch.randn(2, 8, 4), mask)
assert result[1, 5:].sum() == 0 and result[1, :, 2:].sum() == 0
print('Exhaustive monotonic alignment comparisons passed')
