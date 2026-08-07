"""第 10 章进度检查。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location("chapter10_hw", ROOT / "作业_冻结策略与热图归一化.py")
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

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 4))
            self.head = nn.Linear(4, 2)

    def freeze_case():
        model = Toy()
        names = m.configure_trainable_parameters(model, ("backbone.2", "head"))
        expected = [n for n, _ in model.named_parameters() if n.startswith(("backbone.2", "head"))]
        assert names == expected
        assert all(p.requires_grad == (n in expected) for n, p in model.named_parameters())

    def heatmap_case():
        cam = torch.tensor([[[[-2.0, 1.0], [3.0, 0.0]]], [[[5.0, 5.0], [5.0, 5.0]]]])
        out = m.normalize_heatmap(cam)
        assert out.shape == cam.shape and torch.isfinite(out).all()
        assert out.min() >= 0 and out.max() <= 1
        assert torch.allclose(out[1], torch.zeros_like(out[1]))
        assert torch.isclose(out[0].max(), torch.tensor(1.0))

    results = [check("参数冻结策略", freeze_case), check("Grad-CAM 热图归一化", heatmap_case)]
    print(f"通过 {sum(results)}/{len(results)}；TODO 是正常学习状态。")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
