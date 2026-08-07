"""第 14 章进度检查。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location("chapter14_hw", ROOT / "作业_预处理与部署契约.py")
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

    def metadata_case():
        meta = m.letterbox_metadata((100, 200), (320, 320))
        assert abs(float(meta["scale"]) - 1.6) < 1e-9
        assert meta["resized_hw"] == (160, 320)
        assert meta["pad_xy"] == (0, 80)

    def inverse_case():
        meta = {"scale": 2.0, "resized_hw": (200, 300), "pad_xy": (10, 20)}
        boxes = np.array([[10., 20., 310., 220.], [-5., 0., 400., 300.]])
        before = boxes.copy()
        out = m.boxes_to_original(boxes, meta, (100, 150))
        expected = np.array([[0., 0., 150., 100.], [0., 0., 150., 100.]])
        assert out.shape == boxes.shape and np.all(np.isfinite(out))
        np.testing.assert_allclose(out, expected)
        np.testing.assert_array_equal(boxes, before)

    results = [check("letterbox 元数据", metadata_case), check("检测框回映射", inverse_case)]
    print(f"通过 {sum(results)}/{len(results)}；TODO 允许非零退出。")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
