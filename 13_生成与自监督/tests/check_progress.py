"""第 13 章进度检查。"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
_IMPL = Path(os.environ["COURSE_IMPL_DIR"]) / ROOT.name if os.environ.get("COURSE_IMPL_DIR") else ROOT


def load():
    spec = importlib.util.spec_from_file_location("chapter13_hw", _IMPL / "作业_重参数与对比损失.py")
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

    def reparam_case():
        mu = torch.tensor([[1., 2.]], requires_grad=True)
        logvar = torch.zeros_like(mu, requires_grad=True)
        eps = torch.tensor([[0.5, -1.]])
        z = m.reparameterize(mu, logvar, eps)
        assert z.shape == mu.shape and torch.allclose(z, mu + eps)
        z.sum().backward()
        assert mu.grad is not None and logvar.grad is not None

    def contrast_case():
        torch.manual_seed(13)
        z1 = torch.randn(4, 6, requires_grad=True)
        z2 = z1.detach().clone() + 0.01 * torch.randn(4, 6)
        loss = m.nt_xent_loss(z1, z2, 0.2)
        swapped = m.nt_xent_loss(z2, z1, 0.2)
        assert loss.ndim == 0 and torch.isfinite(loss)
        assert torch.allclose(loss, swapped, atol=1e-6)
        loss.backward()
        assert z1.grad is not None and torch.isfinite(z1.grad).all()

    results = [check("VAE 重参数", reparam_case), check("对称 NT-Xent", contrast_case)]
    print(f"通过 {sum(results)}/{len(results)}；初始 TODO 是预期结果。")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
