"""第 14 章进度检查。"""

from __future__ import annotations

import copy
import importlib.util
import numbers
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_IMPL = Path(os.environ["COURSE_IMPL_DIR"]) / ROOT.name if os.environ.get("COURSE_IMPL_DIR") else ROOT


class Wrong(Exception):
    """检查没有通过，消息里写哪里不对、该查什么。"""


def need(cond, msg: str) -> None:
    if not bool(cond):
        raise Wrong(msg)


def load():
    spec = importlib.util.spec_from_file_location("chapter14_hw", _IMPL / "作业_预处理与部署契约.py")
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
    except Wrong as exc:
        print(f"[FAIL] {name} — {exc}")
        return False
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] {name} — 运行时出错 {type(exc).__name__}: {exc}")
        return False
    print(f"[PASS] {name}")
    return True


def is_int(v) -> bool:
    return isinstance(v, numbers.Integral) and not isinstance(v, bool)


def main() -> int:
    m = load()

    def metadata(src_hw, dst_hw):
        """调用作业函数并检查返回值的结构，返回 (scale, (new_h, new_w), (pad_left, pad_top))。"""
        what = f"{src_hw[0]}x{src_hw[1]}（高x宽）放进 {dst_hw[0]}x{dst_hw[1]}"
        meta = m.letterbox_metadata(src_hw, dst_hw)
        need(isinstance(meta, dict), f"{what}：返回值应是字典，现在是 {type(meta).__name__}")
        for key in ("scale", "resized_hw", "pad_xy"):
            need(key in meta, f"{what}：返回的字典里缺少 {key}")
        scale, resized, pad = meta["scale"], meta["resized_hw"], meta["pad_xy"]
        need(isinstance(scale, (float, np.floating)) and scale > 0, f"{what}：scale 应是正的 float，现在是 {scale!r}（{type(scale).__name__}）")
        for key, value in (("resized_hw", resized), ("pad_xy", pad)):
            need(isinstance(value, (tuple, list)) and len(value) == 2, f"{what}：{key} 应是两个数组成的元组，现在是 {value!r}")
            need(all(is_int(v) for v in value), f"{what}：{key} 里应是整数（像素数），现在是 {value!r}。检查取整之后有没有转成 int，加边有没有用整数除法")
        need(pad[0] >= 0 and pad[1] >= 0, f"{what}：pad_xy 不能是负数，现在是 {tuple(pad)}")
        need(resized[0] <= dst_hw[0] and resized[1] <= dst_hw[1], f"{what}：resized_hw 不能超过目标画布，现在是 {tuple(resized)}")
        return float(scale), (int(resized[0]), int(resized[1])), (int(pad[0]), int(pad[1])), what

    def expect(src_hw, dst_hw, scale, resized, pad, hint: str):
        got_scale, got_resized, got_pad, what = metadata(src_hw, dst_hw)
        need(abs(got_scale - scale) < 1e-9, f"{what}：scale 应是 {scale:.6g}，现在是 {got_scale:.6g}。两个方向各有一个缩放系数，要取能让整张图放进画布的那个")
        need(got_resized == resized, f"{what}：resized_hw 应是（高, 宽）= {resized}，现在是 {got_resized}。{hint}")
        need(got_pad == pad, f"{what}：缩放后是 {resized}，pad_xy 应是（左, 上）= {pad}，现在是 {got_pad}。{hint}")

    def metadata_basic():
        expect((100, 200), (320, 320), 1.6, (160, 320), (0, 80), "横图：宽正好放满，上下加边")
        expect((200, 100), (320, 320), 1.6, (320, 160), (80, 0), "竖图：高正好放满，左右加边。检查两个方向是不是都算了，以及 pad_xy 的顺序是（左, 上）")
        expect((50, 50), (64, 64), 1.28, (64, 64), (0, 0), "小图要放大到正好放满")
        expect((480, 640), (480, 640), 1.0, (480, 640), (0, 0), "尺寸相同时不缩放、不加边")
        expect((400, 400), (100, 300), 0.25, (100, 100), (100, 0), "画布不是正方形：高、宽分别和画布的高、宽比")

    def metadata_rounding():
        expect((100, 300), (326, 320), 320 / 300, (107, 320), (0, 109),
               "缩放后的高是 106.67，要取到最近的整数，不能截断；上下一共要加 219 个像素，是奇数，多出来的 1 个像素放在下侧")
        expect((300, 100), (320, 326), 320 / 300, (320, 107), (109, 0),
               "缩放后的宽是 106.67，要取到最近的整数；左右一共要加 219 个像素，是奇数，多出来的 1 个像素放在右侧")
        expect((10, 30), (33, 30), 1.0, (10, 30), (0, 11), "上下一共要加 23 个像素，上侧 11、下侧 12")
        expect((721, 1280), (640, 640), 0.5, (360, 640), (0, 140),
               "缩放后的高正好是 360.5。文档字符串规定了正好落在 .5 时往哪边取，按它的规定这里是 360")
        expect((723, 1280), (640, 640), 0.5, (362, 640), (0, 139),
               "缩放后的高正好是 361.5。文档字符串规定了正好落在 .5 时往哪边取，按它的规定这里是 362")
        expect((1280, 725), (640, 640), 0.5, (640, 362), (139, 0),
               "缩放后的宽正好是 362.5。文档字符串规定了正好落在 .5 时往哪边取，按它的规定这里是 362")

    def to_original(boxes, meta, src_hw, what: str) -> np.ndarray:
        before, meta_before = boxes.copy(), copy.deepcopy(meta)
        out = m.boxes_to_original(boxes, meta, src_hw)
        need(isinstance(out, np.ndarray), f"{what}：返回值应是 numpy 数组，现在是 {type(out).__name__}")
        need(out.shape == boxes.shape, f"{what}：输出形状应和输入相同 {boxes.shape}，现在是 {out.shape}")
        need(np.issubdtype(out.dtype, np.floating), f"{what}：输出应是浮点数组，现在是 {out.dtype}。框是连续坐标，不要取整")
        need(np.array_equal(boxes, before), f"{what}：输入的框被原地改动了，先复制一份再算")
        need(meta == meta_before, f"{what}：传进来的 metadata 被改动了")
        need(np.all(np.isfinite(out)), f"{what}：结果里有 NaN 或 inf")
        return out

    def inverse_values():
        meta = {"scale": 2.0, "resized_hw": (200, 300), "pad_xy": (10, 20)}
        boxes = np.array([[10., 20., 310., 220.], [-5., 0., 400., 300.]])
        out = to_original(boxes, meta, (100, 150), "放大 2 倍、加边（左 10, 上 20）")
        want = np.array([[0., 0., 150., 100.], [0., 0., 150., 100.]])
        need(np.allclose(out, want, atol=1e-9),
             f"scale 2、pad_xy (10, 20)、原图高 100 宽 150：画布上内容区域的四个角应正好映射回原图的四个角，超出内容区域的框要裁剪到原图范围。期望 {want.tolist()}，现在是 {np.round(out, 3).tolist()}")
        meta = {"scale": 0.25, "resized_hw": (100, 100), "pad_xy": (100, 0)}
        boxes = np.array([[110.5, 20.25, 150., 60.5], [101., 1., 102., 3.]])
        out = to_original(boxes, meta, (400, 400), "缩小到 1/4、只有左边加边 100")
        want = np.array([[42., 81., 200., 242.], [4., 4., 8., 12.]])
        need(np.allclose(out, want, atol=1e-9),
             f"scale 0.25、pad_xy (100, 0)、原图 400x400：框 {boxes[0].tolist()} 应映射到 {want[0].tolist()}，现在是 {np.round(out[0], 4).tolist()}。"
             "检查三件事的先后和方向：减加边、除以缩放系数、x 用左边的加边而 y 用上边的加边")
        meta = {"scale": 1.6, "resized_hw": (160, 320), "pad_xy": (0, 80)}
        boxes = np.array([[1., 81., 3., 84.]])
        out = to_original(boxes, meta, (100, 200), "放大 1.6 倍、只有上边加边 80")
        want = np.array([[0.625, 0.625, 1.875, 2.5]])
        need(np.allclose(out, want, atol=1e-9), f"scale 1.6、pad_xy (0, 80)：框 {boxes[0].tolist()} 应映射到 {want[0].tolist()}（结果带小数，不要取整），现在是 {np.round(out[0], 4).tolist()}")

    def inverse_clip_and_types():
        meta = {"scale": 0.5, "resized_hw": (360, 640), "pad_xy": (0, 140)}
        boxes = np.array([[-20., 100., 700., 520.], [600., 130., 660., 560.], [0., 140., 640., 500.]])
        out = to_original(boxes, meta, (721, 1280), "缩小一半、上边加边 140、原图 721x1280")
        want = np.array([[0., 0., 1280., 721.], [1200., 0., 1280., 721.], [0., 0., 1280., 720.]])
        need(np.allclose(out, want, atol=1e-9),
             f"scale 0.5、pad_xy (0, 140)、原图高 721 宽 1280：期望 {want.tolist()}，现在是 {np.round(out, 3).tolist()}。"
             "裁剪的范围是 x 在 [0, 原图宽]、y 在 [0, 原图高]，上下界都要裁，用的是原图的尺寸，不减 1")
        empty = to_original(np.zeros((0, 4)), meta, (721, 1280), "没有框（N=0）")
        need(empty.shape == (0, 4), "没有框时应返回形状 (0, 4) 的数组")
        boxes32 = np.array([[10., 150., 20., 160.]], dtype=np.float32)
        out = to_original(boxes32, meta, (721, 1280), "float32 的输入")
        need(np.allclose(out, [[20., 20., 40., 40.]], atol=1e-4), f"float32 的输入也应正确映射，期望 [[20, 20, 40, 40]]，现在是 {out.tolist()}")
        meta_list = {"scale": 2.0, "resized_hw": [200, 300], "pad_xy": [10, 20], "extra": "别的字段不影响"}
        out = to_original(np.array([[30., 40., 50., 80.]]), meta_list, (100, 150), "metadata 里多了别的字段")
        need(np.allclose(out, [[10., 10., 20., 30.]], atol=1e-9), f"metadata 里多了别的字段、pad_xy 是列表时也应正确映射，期望 [[10, 10, 20, 30]]，现在是 {out.tolist()}")

    results = [
        check("letterbox 元数据：横图、竖图、放大、非正方形画布", metadata_basic),
        check("letterbox 元数据：取整规则、奇数加边", metadata_rounding),
        check("检测框回映射：数值", inverse_values),
        check("检测框回映射：裁剪、不改输入、输出类型", inverse_clip_and_types),
    ]
    print(f"通过 {sum(results)}/{len(results)}；TODO 允许非零退出。")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
