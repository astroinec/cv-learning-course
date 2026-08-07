"""第 9 章进度检查。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location("chapter09_hw", ROOT / "作业_卷积shape与残差块.py")
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

    def size_case():
        assert m.conv_output_size(32, 3, 1, 1) == 32
        assert m.conv_output_size(31, 3, 2, 1) == 16
        assert m.conv_output_size(15, 3, 1, 2, 2) == 15

    def block_case():
        torch.manual_seed(9)
        block = m.BasicResidualBlock(4).eval()
        x = torch.randn(2, 4, 11, 13, requires_grad=True)
        y = block(x)
        assert y.shape == x.shape and torch.isfinite(y).all()
        y.mean().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()

    results = [check("卷积输出尺寸", size_case), check("残差块 shape 与梯度", block_case)]
    print(f"通过 {sum(results)}/{len(results)}；未完成项应显示 TODO。")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
