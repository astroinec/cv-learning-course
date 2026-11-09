"""第 8 章进度检查：TODO 是正常未完成状态。"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_IMPL = Path(os.environ["COURSE_IMPL_DIR"]) / ROOT.name if os.environ.get("COURSE_IMPL_DIR") else ROOT


def load():
    spec = importlib.util.spec_from_file_location("chapter08_hw", _IMPL / "作业_手写两层网络.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def run_check(name, fn):
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

    def forward_check():
        rng = np.random.default_rng(8)
        x = rng.normal(size=(5, 3))
        w1 = rng.normal(size=(3, 4))
        b1 = rng.normal(size=4)
        w2 = rng.normal(size=(4, 2))
        b2 = rng.normal(size=2)
        h, logits = m.two_layer_forward(x, w1, b1, w2, b2)
        assert h.shape == (5, 4) and logits.shape == (5, 2)
        assert np.all(h >= 0) and np.all(np.isfinite(logits))
        np.testing.assert_allclose(logits, np.maximum(x @ w1 + b1, 0) @ w2 + b2)

    def loss_check():
        logits = np.array([[1000.0, 1001.0], [-1000.0, -999.0]])
        labels = np.array([1, 0])
        loss = m.cross_entropy_from_logits(logits, labels)
        expected = (np.log1p(np.exp(-1.0)) + np.log1p(np.exp(1.0))) / 2
        assert isinstance(loss, (float, np.floating)) and np.isfinite(loss)
        assert abs(float(loss) - expected) < 1e-8

    results = [run_check("两层前向 shape 与数值", forward_check), run_check("稳定交叉熵", loss_check)]
    print(f"通过 {sum(results)}/{len(results)}；TODO 不代表程序崩溃。")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
