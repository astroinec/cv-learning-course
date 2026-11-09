"""参考实现（学员完成本章前不要查看）。"""

from __future__ import annotations

import numpy as np


def two_layer_forward(x, w1, b1, w2, b2):
    hidden = np.maximum(x @ w1 + b1, 0.0)
    return hidden, hidden @ w2 + b2


def cross_entropy_from_logits(logits, labels):
    z = logits - logits.max(axis=1, keepdims=True)
    log_probs = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return float(-log_probs[np.arange(len(labels)), labels].mean())
