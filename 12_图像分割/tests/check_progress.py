"""第 12 章进度检查。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location("chapter12_hw", ROOT / "作业_Dice与跳跃连接.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def check(name, fn):
    try:
        fn()
    except NotImplementedError as exc:
        print(f"[TODO] {name} — {exc}")
        return False
    except Exception as exc:
        print(f"[FAIL] {name} — {type(exc).__name__}: {exc}")
        return False
    print(f"[PASS] {name}")
    return True


def main() -> int:
    m = load()

    def dice_case():
        logits = torch.full((2, 1, 4, 4), -12.0, requires_grad=True)
        target = torch.zeros_like(logits)
        loss = m.binary_dice_loss(logits, target)
        assert loss.ndim == 0 and torch.isfinite(loss)
        loss.backward()
        assert logits.grad is not None and torch.isfinite(logits.grad).all()
        ones = torch.full((1, 1, 2, 2), 12.0)
        assert float(m.binary_dice_loss(ones, torch.ones_like(ones))) < 1e-4

    def concat_case():
        decoder = torch.randn(2, 3, 9, 11)
        skip = torch.randn(2, 5, 8, 14)
        out = m.align_and_concat(decoder, skip)
        assert out.shape == (2, 8, 8, 14)
        assert torch.allclose(out[:, 3:], skip)

    results = [check("二分类 Dice loss", dice_case), check("skip 对齐与拼接", concat_case)]
    print(f"通过 {sum(results)}/{len(results)}；TODO 不应导致检查器崩溃。")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
