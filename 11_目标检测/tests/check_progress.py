"""第 11 章进度检查。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location("chapter11_hw", ROOT / "作业_批量IoU与类别NMS.py")
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

    def iou_case():
        a = torch.tensor([[0., 0., 2., 2.], [3., 3., 4., 4.]])
        b = torch.tensor([[0., 0., 2., 2.], [1., 1., 3., 3.]])
        out = m.pairwise_iou(a, b)
        expected = torch.tensor([[1., 1 / 7], [0., 0.]])
        assert out.shape == (2, 2)
        assert torch.allclose(out, expected, atol=1e-6)
        assert m.pairwise_iou(a[:0], b).shape == (0, 2)

    def nms_case():
        boxes = torch.tensor([[0., 0., 2., 2.], [0.1, 0.1, 2.1, 2.1], [0., 0., 2., 2.]])
        scores = torch.tensor([0.9, 0.8, 0.7])
        labels = torch.tensor([0, 0, 1])
        keep = m.class_aware_nms(boxes, scores, labels, 0.5)
        assert keep.dtype == torch.long
        assert keep.tolist() == [0, 2]
        empty = m.class_aware_nms(boxes[:0], scores[:0], labels[:0], 0.5)
        assert empty.shape == (0,) and empty.dtype == torch.long

    results = [check("批量 IoU", iou_case), check("类别感知 NMS", nms_case)]
    print(f"通过 {sum(results)}/{len(results)}；未完成实现应显示 TODO。")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
