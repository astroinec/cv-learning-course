"""第 27 章自动验收：TODO / FAIL / PASS。"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

CHAPTER = Path(__file__).resolve().parents[1]
ASSIGNMENT = "27-1_提示与掩码后处理.py"


def load_assignment():
    override = os.environ.get("COURSE_IMPL_DIR")
    base = Path(override) / CHAPTER.name if override else CHAPTER
    spec = importlib.util.spec_from_file_location("chapter27_assignment", base / ASSIGNMENT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def report(status, name, detail):
    print(f"[{status:4}] {name} - {detail}")


def run(name, func):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ok, detail = func()
    except NotImplementedError as exc:
        report("TODO", name, str(exc))
        return False
    except Exception as exc:  # noqa: BLE001
        report("FAIL", name, f"运行异常：{type(exc).__name__}: {exc}")
        return False
    report("PASS" if ok else "FAIL", name, detail)
    return ok


def raises_value_error(func) -> bool:
    try:
        func()
    except ValueError:
        return True
    except Exception:  # noqa: BLE001
        return False
    return False


# ---------------------------------------------------------------- 1. 点提示的位置编码
def check_encoding(mod):
    f64 = torch.float64
    only_x = torch.tensor([[1.0], [0.0]], dtype=f64)
    only_y = torch.tensor([[0.0], [1.0]], dtype=f64)
    size = (4, 8)  # 高 4，宽 8
    # 只看 x：期望值是手算的。x=3.5 正好在图像正中，角度为 0；x=4.5 的角度是 π/2；x=0 的角度是 -1.75π
    xs = torch.tensor([[3.5, 0.0], [4.5, 0.0], [1.5, 0.0], [0.0, 0.0]], dtype=f64)
    got = mod.point_position_encoding(xs, size, only_x)
    if not isinstance(got, torch.Tensor) or tuple(got.shape) != (4, 2):
        return False, "points 是 (4, 2)、freqs 是 (2, 1) 时，返回值应是形状 (4, 2) 的张量"
    r = math.sqrt(0.5)
    want = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, -1.0], [r, r]], dtype=f64)
    if not torch.allclose(got, want, atol=1e-9):
        if torch.allclose(got, want.flip(-1), atol=1e-9):
            return False, "正弦和余弦的先后放反了：前一半应是 sin，后一半是 cos"
        return False, f"只含 x 的频率下结果不对，逐步核对文档字符串里的 5 步；得到 {got.tolist()}"
    # 只看 y：高是 4，y=0 的角度是 -1.5π；如果拿宽去除就会得到别的值
    ys = torch.tensor([[0.0, 0.0], [5.0, 1.5], [2.0, 3.0]], dtype=f64)
    got = mod.point_position_encoding(ys, size, only_y)
    want = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=f64)
    if not torch.allclose(got, want, atol=1e-9):
        return False, "只含 y 的频率下结果不对：检查 x、y 各自除以的是宽还是高（image_size 是先高后宽）"
    # 一般情况：两个频率，期望值由独立的逐点计算预先得到
    freqs = torch.tensor([[1.0, 0.5], [0.25, -1.0]], dtype=f64)
    pts = torch.tensor([[0.0, 0.0], [5.0, 2.0], [7.0, 3.0]], dtype=f64)
    want = torch.tensor([
        [-0.3826834324, 0.9238795325, 0.9238795325, -0.3826834324],
        [0.3826834324, -0.3826834324, -0.9238795325, 0.9238795325],
        [0.3826834324, -0.9238795325, 0.9238795325, -0.3826834324],
    ], dtype=f64)
    keep = pts.clone()
    got = mod.point_position_encoding(pts, size, freqs)
    if tuple(got.shape) != (3, 4):
        return False, "F=2 时输出最后一维应是 4（前 2 个 sin，后 2 个 cos）"
    if not torch.allclose(got, want, atol=1e-8):
        if torch.allclose(got[:, [0, 2, 1, 3]], want, atol=1e-8):
            return False, "sin 和 cos 被交错排列了：应先放全部 sin，再放全部 cos"
        return False, "两个频率同时作用时结果不对：检查 freqs 的哪一行乘 x、哪一行乘 y，以及矩阵乘法的方向"
    if not torch.equal(pts, keep):
        return False, "传入的 points 被修改了"
    # 形状和数据类型
    f32 = torch.tensor([[0.3, -1.2, 2.0], [1.1, 0.4, -0.7]], dtype=torch.float32)
    batch = torch.tensor([[[1.0, 2.0], [6.0, 0.0]], [[3.0, 3.0], [0.0, 1.0]]])
    out = mod.point_position_encoding(batch, size, f32)
    if tuple(out.shape) != (2, 2, 6) or out.dtype != torch.float32:
        return False, f"points 是 (2, 2, 2)、freqs 是 float32 的 (2, 3) 时，应返回 float32 的 (2, 2, 6)，得到 {tuple(out.shape)} {out.dtype}"
    one = mod.point_position_encoding(batch[1, 0], size, f32)
    if tuple(one.shape) != (6,) or not torch.allclose(one, out[1, 0], atol=1e-6):
        return False, "points 只有一维 (2,) 时应返回 (2F,)，并且和批量计算里对应的那一行相同"
    if not raises_value_error(lambda: mod.point_position_encoding(torch.zeros(3, 3), size, f32)):
        return False, "points 的最后一维不是 2 时应抛出 ValueError"
    if not raises_value_error(lambda: mod.point_position_encoding(torch.zeros(3, 2), size, torch.zeros(3, 2))):
        return False, "freqs 的第 0 维不是 2 时应抛出 ValueError"
    return True, "手算的特殊点、两个频率的一般情况、形状和数据类型都正确"


# ---------------------------------------------------------------- 2. 按预测 IoU 选掩码
def check_select(mod):
    b, m, h, w = 4, 3, 2, 5
    masks = np.arange(b * m * h * w, dtype=np.float32).reshape(b, m, h, w)
    # 每一行最高分的位置都不同；第 2 行全是负数；第 3 行有并列
    pred = np.array([[0.2, 0.9, 0.5], [0.8, 0.1, 0.3], [-0.6, -0.4, -0.1], [0.7, 0.7, 0.2]])
    keep_m, keep_p = masks.copy(), pred.copy()
    out = mod.select_by_predicted_iou(masks, pred)
    if not isinstance(out, tuple) or len(out) != 3:
        return False, "应返回三个数组：选中的掩码、下标、分数"
    chosen, index, score = out
    if not isinstance(index, np.ndarray) or index.dtype != np.int64 or index.shape != (b,):
        return False, "index 应是形状 (B,) 的 int64 数组"
    if index.tolist() != [1, 0, 2, 0]:
        hint = "分数全是负数的那一行选错了，预测的 IoU 可以是负数" if index.tolist()[:2] == [1, 0] and index[2] != 2 else \
               "分数并列时应取下标小的" if index.tolist()[:3] == [1, 0, 2] else "每个提示应各自在自己的 M 个候选里选分数最高的"
        return False, f"选中的下标应为 [1, 0, 2, 0]，得到 {index.tolist()}；{hint}"
    if not isinstance(chosen, np.ndarray) or chosen.shape != (b, h, w) or chosen.dtype != masks.dtype:
        return False, "chosen 应是 (B, H, W)，数据类型和 masks 相同"
    for i, j in enumerate([1, 0, 2, 0]):
        if not np.array_equal(chosen[i], keep_m[i, j]):
            return False, f"第 {i} 个提示返回的掩码不是它自己的第 {j} 个候选"
    if not isinstance(score, np.ndarray) or score.dtype != np.float64 or score.shape != (b,) or not np.allclose(score, [0.9, 0.8, -0.1, 0.7]):
        return False, "score 应是 (B,) 的 float64，等于各提示选中候选的预测 IoU"
    if not (np.array_equal(masks, keep_m) and np.array_equal(pred, keep_p)):
        return False, "传入的数组被修改了"
    bools = np.zeros((2, 4, 3, 3), dtype=bool)
    bools[0, 3, 1, 1] = bools[1, 2, 0, 0] = True
    chosen, index, _ = mod.select_by_predicted_iou(bools, np.array([[0.1, 0.2, 0.3, 1.2], [0.0, 0.1, 0.5, 0.4]], dtype=np.float32))
    if chosen.dtype != np.bool_ or index.tolist() != [3, 2] or not (chosen[0, 1, 1] and chosen[1, 0, 0]) or chosen.sum() != 2:
        return False, "布尔掩码、4 个候选、float32 分数的例子不对"
    if not raises_value_error(lambda: mod.select_by_predicted_iou(masks, pred[:, :2])):
        return False, "iou_pred 的形状和 masks 的前两维不一致时应抛出 ValueError"
    if not raises_value_error(lambda: mod.select_by_predicted_iou(masks[0], pred[0])):
        return False, "masks 不是四维时应抛出 ValueError"
    return True, "各提示独立选择，负分、并列、数据类型和非法输入都正确"


# ---------------------------------------------------------------- 3. 稳定性分数
def check_stability(mod):
    x = np.full((6, 6), -3.0)
    x.flat[:5] = 3.0      # 5 个像素很确定是前景
    x.flat[5:7] = 1.0     # 2 个像素正好等于高阈值
    x.flat[7:10] = 0.5
    x.flat[10:12] = -0.5
    x.flat[12:14] = -1.0  # 2 个像素正好等于低阈值
    # 默认参数：大于 1 的有 5 个，大于 -1 的有 12 个
    got = float(mod.stability_score(x))
    if not math.isfinite(got) or abs(got - 5 / 12) > 1e-6:
        if abs(got - 12 / 5) < 1e-9:
            return False, "分子分母放反了：分数应不大于 1"
        if abs(got - 7 / 14) < 1e-9 or abs(got - 7 / 12) < 1e-9 or abs(got - 5 / 14) < 1e-9:
            return False, "正好等于阈值的像素不算前景，比较要用严格大于"
        return False, f"默认参数下应为 5/12，得到 {got}"
    # 改阈值和偏移：阈值 0.75、偏移 0.5，大于 1.25 的有 5 个，大于 0.25 的有 10 个
    got = float(mod.stability_score(x, threshold=0.75, offset=0.5))
    if abs(got - 0.5) > 1e-6:
        return False, f"threshold=0.75、offset=0.5 时应为 0.5，得到 {got}；检查两个阈值是不是 threshold 加减 offset"
    # 多个前置维度，每张掩码各算各的
    stack = np.stack([x, np.full((6, 6), -5.0), np.full((6, 6), 4.0), np.where(np.arange(36).reshape(6, 6) < 9, 0.2, -0.2)])
    out = mod.stability_score(stack.reshape(2, 2, 6, 6))
    if not isinstance(out, np.ndarray) or out.shape != (2, 2) or out.dtype != np.float64:
        return False, "输入是 (2, 2, H, W) 时应返回 (2, 2) 的 float64 数组，每张掩码一个分数"
    if not np.all(np.isfinite(out)):
        return False, "低阈值掩码为空时出现了 nan 或 inf，这种情况分数应记 0"
    if not np.allclose(out, [[5 / 12, 0.0], [1.0, 0.0]], atol=1e-12):
        return False, f"四张掩码的分数应为 [[5/12, 0], [1, 0]]，得到 {out.tolist()}"
    # 像素数超过 255，查计数时的整数溢出
    big = np.full((40, 40), -4.0, dtype=np.float32)
    big[:30, :30] = 0.5
    big[:20, :30] = 2.0
    got = float(mod.stability_score(big))
    if abs(got - 600 / 900) > 1e-6:
        return False, f"40x40 的例子应为 600/900，得到 {got}；检查计数用的数据类型"
    return True, "手算的例子、阈值边界、空掩码和多维输入都正确"


# ---------------------------------------------------------------- 4. 由掩码求外接框
def check_box(mod):
    h, w = 5, 9
    one = np.zeros((h, w), bool)
    one[2, 6] = True
    box = mod.mask_to_box(one)
    if not isinstance(box, np.ndarray) or box.shape != (4,) or box.dtype != np.int64:
        return False, "单张掩码应返回形状 (4,) 的 int64 数组"
    if box.tolist() != [6, 2, 6, 2]:
        hint = "x 和 y 放反了：x 是列号" if box.tolist() == [2, 6, 2, 6] else "四个数都是包含在内的像素下标，不要加 1" if box.tolist() == [6, 2, 7, 3] else ""
        return False, f"只有 (行 2, 列 6) 一个像素时应为 [6, 2, 6, 2]，得到 {box.tolist()}；{hint}"
    rect = np.zeros((h, w), bool)
    rect[1:4, 2:8] = True
    lshape = np.zeros((h, w), bool)
    lshape[0, 3] = lshape[4, 3] = lshape[4, 8] = True      # 三个分开的像素，贴着图像边
    full = np.ones((h, w), bool)
    empty = np.zeros((h, w), bool)
    left = np.zeros((h, w), bool)
    left[3, 0] = True                                       # 贴着第 0 列和非 0 行，查空掩码的判断有没有误伤
    stack = np.stack([rect, lshape, full, empty, left, one]).reshape(2, 3, h, w)
    keep = stack.copy()
    out = mod.mask_to_box(stack)
    if not isinstance(out, np.ndarray) or out.shape != (2, 3, 4) or out.dtype != np.int64:
        return False, "输入是 (2, 3, H, W) 时应返回 (2, 3, 4) 的 int64 数组"
    want = [[2, 1, 7, 3], [3, 0, 8, 4], [0, 0, 8, 4], [0, 0, 0, 0], [0, 3, 0, 3], [6, 2, 6, 2]]
    names = ["矩形", "三个分开的像素", "全是前景", "空掩码", "只有 (行 3, 列 0) 一个像素", "单个像素"]
    for got, exp, name in zip(out.reshape(-1, 4).tolist(), want, names):
        if got != exp:
            return False, f"{name}：应为 {exp}，得到 {got}"
    if not np.array_equal(stack, keep):
        return False, "传入的掩码被修改了"
    num = np.zeros((h, w), dtype=np.float32)
    num[1, 4] = 0.3
    num[3, 5] = 2.0
    if mod.mask_to_box(num).tolist() != [4, 1, 5, 3]:
        return False, "数值数组里非零的像素都算前景（0.3 也算）"
    return True, "单像素、贴边、空掩码、多维输入和数值掩码都正确"


# ---------------------------------------------------------------- 5. 掩码的非极大值抑制
def check_nms(mod):
    h, w = 8, 12

    def rect(r0, r1, c0, c1):
        m = np.zeros((h, w), bool)
        m[r0:r1, c0:c1] = True
        return m

    # 链式重叠：a 和 b 的 IoU 是 0.6，b 和 c 的 IoU 是 0.6，a 和 c 的 IoU 是 1/3
    a, bb, c = rect(0, 4, 0, 4), rect(0, 4, 1, 5), rect(0, 4, 2, 6)
    far = rect(5, 8, 8, 12)
    masks = np.stack([c, far, a, bb])                      # 故意打乱顺序
    scores = np.array([0.7, 0.4, 0.9, 0.8])
    keep_m, keep_s = masks.copy(), scores.copy()
    out = mod.mask_nms(masks, scores, iou_threshold=0.5)
    if not isinstance(out, np.ndarray) or out.dtype != np.int64 or out.ndim != 1:
        return False, "应返回一维的 int64 数组"
    if out.tolist() != [2, 0, 1]:
        got = out.tolist()
        hint = ("被丢掉的掩码不应该再去抑制别人（这里 b 被 a 抑制后，c 应该保留）" if got == [2, 1] else
                "返回的应是输入里的下标，不是排序后的位置" if got == [0, 2, 3] else
                "结果要按分数从高到低排列" if sorted(got) == [0, 1, 2] else
                "应从分数最高的开始处理" if 3 in got or 2 not in got else "")
        return False, f"链式重叠的例子应保留下标 [2, 0, 1]，得到 {got}；{hint}"
    if not (np.array_equal(masks, keep_m) and np.array_equal(scores, keep_s)):
        return False, "传入的数组被修改了"
    # 嵌套：部件完全落在整体里面，IoU 只有 16/96，应都保留
    whole, part = rect(0, 8, 0, 12), rect(2, 6, 3, 7)
    if mod.mask_nms(np.stack([part, whole]), np.array([0.5, 0.9]), iou_threshold=0.7).tolist() != [1, 0]:
        return False, "整体和它里面的部件 IoU 只有 1/6，应都保留，并按分数从高到低返回 [1, 0]；检查分母是不是并集"
    # 中间隔着一张不相干的掩码：a 的复制品要和所有已保留的掩码比，不能只和最近保留的那张比
    if mod.mask_nms(np.stack([a, far, a]), np.array([0.9, 0.8, 0.7]), iou_threshold=0.5).tolist() != [0, 1]:
        return False, "当前掩码要和每一张已经保留的掩码比较：这里第三张和第一张完全重合，应被丢掉"
    # IoU 正好等于阈值：不抑制
    p, q = rect(0, 2, 0, 3), rect(0, 2, 1, 4)              # 交 4、并 8，IoU 正好 0.5
    if mod.mask_nms(np.stack([p, q]), np.array([0.6, 0.5]), iou_threshold=0.5).tolist() != [0, 1]:
        return False, "IoU 正好等于阈值时不应抑制，只有严格大于才丢掉"
    if mod.mask_nms(np.stack([p, q]), np.array([0.6, 0.5]), iou_threshold=0.49).tolist() != [0]:
        return False, "IoU 0.5 大于阈值 0.49，分数低的那张应被丢掉"
    # 外接框相同但像素不重叠：按掩码的 IoU 两张都要保留
    board = (np.indices((h, w)).sum(0) % 2).astype(bool)
    board[0, 0] = board[-1, -1] = True
    other = ~board
    other[0, 0] = other[-1, -1] = True                     # 两张的外接框都是整张图，IoU 只有 2/96
    if mod.mask_nms(np.stack([board, other]), np.array([0.9, 0.8]), iou_threshold=0.5).tolist() != [0, 1]:
        return False, "两张掩码外接框相同、像素几乎不重叠时应都保留：重叠要按掩码的像素算，不是按外接框"
    # 完全相同的掩码和并列的分数
    same = np.stack([a, a, far])
    if mod.mask_nms(same, np.array([0.5, 0.5, 0.5]), iou_threshold=0.7).tolist() != [0, 2]:
        return False, "分数并列时下标小的先处理：三张分数相同、前两张完全重合，应保留 [0, 2]"
    # 空掩码之间并集为空，IoU 记 0
    blank = np.zeros((3, h, w), bool)
    blank[2] = a
    out = mod.mask_nms(blank, np.array([0.3, 0.2, 0.1]), iou_threshold=0.5)
    if out.tolist() != [0, 1, 2]:
        return False, "两张空掩码的并集为空，IoU 记 0，不能出现除以零或者互相抑制"
    out = mod.mask_nms(np.zeros((0, h, w), bool), np.zeros(0))
    if not isinstance(out, np.ndarray) or out.dtype != np.int64 or out.shape != (0,):
        return False, "没有掩码时应返回长度为 0 的 int64 数组"
    if not raises_value_error(lambda: mod.mask_nms(masks, scores[:3])):
        return False, "masks 和 scores 个数不同时应抛出 ValueError"
    return True, "链式重叠、阈值边界、同框不同掩码、嵌套、并列和空输入都正确"


def main() -> int:
    print("第 27 章进度检查\n" + "=" * 52)
    try:
        mod = load_assignment()
    except Exception as exc:  # noqa: BLE001
        report("FAIL", "加载作业", f"{type(exc).__name__}: {exc}")
        return 1
    results = [
        run("点提示的位置编码", lambda: check_encoding(mod)),
        run("按预测 IoU 选掩码", lambda: check_select(mod)),
        run("稳定性分数", lambda: check_stability(mod)),
        run("由掩码求外接框", lambda: check_box(mod)),
        run("掩码的非极大值抑制", lambda: check_nms(mod)),
    ]
    print("=" * 52)
    print(f"通过 {sum(results)}/{len(results)}")
    if not all(results):
        print("提示：TODO 是正常学习状态；实现后重新运行。")
        return 1
    print("零件通过；下一步在 Notebook 里用它们训练小模型，再做一遍全图自动分割。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
