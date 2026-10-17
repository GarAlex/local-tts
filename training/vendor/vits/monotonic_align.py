"""Portable maximum-path alignment replacing upstream Cython build dependency.

VITS-compatible dynamic programming on detached scores. CPU implementation;
correctness takes priority over throughput for the initial training tool.
"""
import numpy as np
import torch

def maximum_path(value, mask):
    scores = value.detach().cpu().numpy()
    masks = mask.detach().cpu().numpy()
    paths = np.zeros_like(scores)
    for b in range(scores.shape[0]):
        ty = int(masks[b].sum(axis=1).astype(bool).sum())
        tx = int(masks[b].sum(axis=0).astype(bool).sum())
        if ty < tx or tx == 0:
            raise ValueError("audio has fewer frames than text tokens")
        dp = np.full((ty, tx), -np.inf, dtype=np.float32)
        move = np.zeros((ty, tx), dtype=bool)
        dp[0, 0] = scores[b, 0, 0]
        for y in range(1, ty):
            for x in range(max(0, tx - (ty - y)), min(tx, y + 1)):
                stay = dp[y - 1, x]
                advance = dp[y - 1, x - 1] if x else -np.inf
                move[y, x] = advance > stay
                dp[y, x] = max(stay, advance) + scores[b, y, x]
        x = tx - 1
        for y in range(ty - 1, -1, -1):
            paths[b, y, x] = 1
            if move[y, x]:
                x -= 1
    return torch.from_numpy(paths).to(value.device, value.dtype)
