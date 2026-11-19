"""第 11 章进度检查。"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
_IMPL = Path(os.environ["COURSE_IMPL_DIR"]) / ROOT.name if os.environ.get("COURSE_IMPL_DIR") else ROOT


class Wrong(Exception):
    """检查没有通过，消息里写哪里不对、该查什么。"""


def need(cond, msg: str) -> None:
    if not bool(cond):
        raise Wrong(msg)


def load():
    spec = importlib.util.spec_from_file_location("chapter11_hw", _IMPL / "作业_批量IoU与类别NMS.py")
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


def pixel_masks(boxes: torch.Tensor, size: int) -> torch.Tensor:
    """把整数坐标的框画成 size x size 的像素掩膜：像素中心落在框内记 True。只给本文件的检查用。"""
    centers = torch.arange(size, dtype=torch.float32) + 0.5
    ys, xs = torch.meshgrid(centers, centers, indexing="ij")
    b = boxes.float()[:, :, None, None]
    return (xs > b[:, 0]) & (xs < b[:, 2]) & (ys > b[:, 1]) & (ys < b[:, 3])


def pixel_iou(boxes1: torch.Tensor, boxes2: torch.Tensor, size: int) -> torch.Tensor:
    """数像素得到的 IoU：重合的像素数除以至少被一个框盖住的像素数。"""
    m1, m2 = pixel_masks(boxes1, size), pixel_masks(boxes2, size)
    inter = (m1[:, None] & m2[None]).flatten(2).sum(-1).double()
    union = (m1[:, None] | m2[None]).flatten(2).sum(-1).double()
    return inter / union


def iou_matrix(m, a, b, what: str) -> torch.Tensor:
    before = (a.clone(), b.clone())
    out = m.pairwise_iou(a, b)
    need(isinstance(out, torch.Tensor), f"{what}：返回值应是 torch.Tensor，现在是 {type(out).__name__}")
    need(tuple(out.shape) == (len(a), len(b)), f"{what}：输出形状应是 (N, M) = ({len(a)}, {len(b)})，现在是 {tuple(out.shape)}。检查是不是只算了一一对应的框，或者行列反了")
    need(torch.equal(a, before[0]) and torch.equal(b, before[1]), f"{what}：输入的框被改动了，函数不应修改输入")
    need(out.is_floating_point(), f"{what}：输出应是浮点张量，现在是 {out.dtype}")
    need(torch.isfinite(out).all(), f"{what}：结果里有 NaN 或 inf")
    if out.numel():
        need(float(out.min()) >= -1e-6 and float(out.max()) <= 1 + 1e-6,
             f"{what}：IoU 应在 0 到 1 之间，现在最小 {float(out.min()):.4g}、最大 {float(out.max()):.4g}。检查不相交时交集的宽和高")
    return out


def short(items: list) -> str:
    """列表太长时只显示前 8 个。"""
    return str(items) if len(items) <= 8 else f"{items[:8]} 等 {len(items)} 个"


def run_nms(m, boxes, scores, labels, thr, what: str) -> list[int]:
    before = (boxes.clone(), scores.clone(), labels.clone())
    keep = m.class_aware_nms(boxes, scores, labels, thr)
    need(isinstance(keep, torch.Tensor), f"{what}：返回值应是 torch.Tensor，现在是 {type(keep).__name__}")
    need(keep.dtype == torch.long, f"{what}：返回的下标应是 long 张量，现在是 {keep.dtype}（空输入时也一样）")
    need(keep.ndim == 1, f"{what}：返回值应是一维张量 (K,)，现在的形状是 {tuple(keep.shape)}")
    need(torch.equal(boxes, before[0]) and torch.equal(scores, before[1]) and torch.equal(labels, before[2]),
         f"{what}：输入的 boxes、scores 或 labels 被改动了。要改坐标或排序时先复制一份")
    return keep.tolist()


def explain_nms(keep: list[int], boxes, scores, labels, thr, size: int, what: str) -> None:
    """按定义逐条核对保留结果（IoU 用数像素的办法另算，不依赖作业里的 pairwise_iou）。

    IoU 和阈值相差不到 1e-4 的框对两种判定都算对，避免浮点舍入不同造成误判。
    """
    margin = 1e-4
    n = len(boxes)
    need(all(0 <= i < n for i in keep) and len(set(keep)) == len(keep), f"{what}：返回的下标应在 0 到 {n - 1} 之间且不重复，现在是 {short(keep)}")
    kept_scores = [float(scores[i]) for i in keep]
    need(kept_scores == sorted(kept_scores, reverse=True),
         f"{what}：返回 {short(keep)}，对应的分数是 {short([round(s, 2) for s in kept_scores])}，没有按分数从高到低排。"
         "如果分数顺序不对、而你确认算法没错，检查返回的是输入里的原始下标，还是排序之后的位置")
    iou = pixel_iou(boxes, boxes, size)
    kept = set(keep)
    for i in keep:
        for j in keep:
            if i < j and int(labels[i]) == int(labels[j]) and float(iou[i, j]) > thr + margin:
                raise Wrong(f"{what}：下标 {i} 和 {j} 的框同类、IoU 是 {float(iou[i, j]):.3f}，超过阈值 {thr}，却都被保留了")
    for i in range(n):
        if i in kept:
            continue
        killers = [j for j in keep if int(labels[j]) == int(labels[i]) and float(scores[j]) > float(scores[i]) and float(iou[i, j]) > thr - margin]
        need(killers, f"{what}：下标 {i} 的框被删掉了，但保留下来的框里没有一个和它同类、分数比它高、并且和它的 IoU 超过阈值 {thr}。"
                      "检查三件事：是不是把不同类别的框也拿来比了；已经被删掉的框是不是还在删别的框；IoU 正好等于阈值时是不是也删了")


def main() -> int:
    m = load()

    def iou_basic():
        a = torch.tensor([[0., 0., 2., 2.], [3., 3., 4., 4.]])
        b = torch.tensor([[0., 0., 2., 2.], [1., 1., 3., 3.]])
        c = torch.tensor([[0., 0., 2., 2.], [0., 0., 4., 4.], [10., 10., 12., 12.]])
        iou_matrix(m, c, b, "3 个框对 2 个框")
        out = iou_matrix(m, a, b, "2 个框对 2 个框")
        expected = torch.tensor([[1., 1 / 7], [0., 0.]])
        need(torch.allclose(out, expected, atol=1e-6),
             f"2 个框对 2 个框：结果应是 [[1, 1/7], [0, 0]]，现在是 {out.tolist()}。第 [i, j] 项是 boxes1[i] 和 boxes2[j] 的 IoU；再检查交集、并集和面积的定义（面积不加 1）")
        out = iou_matrix(m, c, b, "3 个框对 2 个框")
        expected = torch.tensor([[1., 1 / 7], [0.25, 0.25], [0., 0.]])
        need(torch.allclose(out, expected, atol=1e-6), f"3 个框对 2 个框：期望 [[1, 1/7], [0.25, 0.25], [0, 0]]，现在是 {out.tolist()}")
        touch = iou_matrix(m, torch.tensor([[0., 0., 1., 1.]]), torch.tensor([[1., 0., 2., 1.], [1., 1., 2., 2.], [5., 0., 6., 1.]]), "相接和分开的框")
        need(torch.allclose(touch, torch.zeros(1, 3), atol=1e-6), f"只有一条边相接、只有一个角相接、完全分开的框，IoU 都应是 0，现在是 {touch.tolist()}")
        for n1, n2 in ((0, 2), (2, 0), (0, 0)):
            iou_matrix(m, a[:n1], b[:n2], f"N={n1}、M={n2} 的空输入")
        g = torch.Generator().manual_seed(11)
        lo = torch.randint(0, 9, (2, 7, 2), generator=g)
        wh = torch.randint(1, 7, (2, 7, 2), generator=g)
        boxes = torch.cat([lo, lo + wh], dim=-1).float()
        out = iou_matrix(m, boxes[0], boxes[1], "7 个框对 7 个框（整数坐标）")
        expected = pixel_iou(boxes[0], boxes[1], 16)
        worst = int((out.double() - expected).abs().argmax())
        i, j = divmod(worst, 7)
        need(torch.allclose(out.double(), expected, atol=1e-5),
             f"整数坐标的随机框：和数像素得到的 IoU 对不上。例如 {boxes[0, i].tolist()} 和 {boxes[1, j].tolist()}，"
             f"数像素是 {float(expected[i, j]):.4f}，你的结果是 {float(out[i, j]):.4f}")

    def iou_edge():
        zero = torch.tensor([[5., 5., 5., 5.], [2., 2., 2., 6.]])
        normal = torch.tensor([[0., 0., 10., 10.]])
        out = m.pairwise_iou(zero, zero)
        need(isinstance(out, torch.Tensor) and tuple(out.shape) == (2, 2), "面积为 0 的框：输出应是形状 (2, 2) 的张量")
        need(torch.isfinite(out).all(), f"面积为 0 的框和它自己的 IoU 出现了 NaN 或 inf：{out.tolist()}。并集面积为 0 时要单独处理，约定返回 0")
        need(torch.allclose(out, torch.zeros(2, 2), atol=1e-6), f"面积为 0 的框之间的 IoU 约定为 0，现在是 {out.tolist()}")
        out = iou_matrix(m, zero, normal, "面积为 0 的框对正常的框")
        need(torch.allclose(out, torch.zeros(2, 1), atol=1e-6), f"面积为 0 的框和正常框的 IoU 应是 0，现在是 {out.tolist()}")
        s = 1.0 / 512
        tiny = torch.tensor([[0.25, 0.25, 0.25 + s, 0.25 + s], [0.25, 0.25, 0.25 + 2 * s, 0.25 + s]])
        out = iou_matrix(m, tiny, tiny, "归一化坐标下的小框")
        need(abs(float(out[0, 0]) - 1) <= 0.01 and abs(float(out[1, 1]) - 1) <= 0.01,
             f"坐标归一化到 0 到 1 时，边长 1/512 的框和它自己的 IoU 应是 1，现在是 {float(out[0, 0]):.4f}。"
             "检查防止除零的办法有没有改变正常框的结果")
        need(abs(float(out[0, 1]) - 0.5) <= 0.01, f"小框和把它加宽一倍的框，IoU 应是 0.5，现在是 {float(out[0, 1]):.4f}")
        big = torch.tensor([[1000., 2000., 1100., 2050.], [1050., 2000., 1150., 2050.]])
        out = iou_matrix(m, big, big, "坐标上千的框")
        need(abs(float(out[0, 1]) - 1 / 3) <= 1e-5, f"两个 100x50 的框错开一半，IoU 应是 1/3，现在是 {float(out[0, 1]):.5f}")

    def nms_basic():
        boxes = torch.tensor([[0., 0., 2., 2.], [0.1, 0.1, 2.1, 2.1], [0., 0., 2., 2.]])
        scores = torch.tensor([0.9, 0.8, 0.7])
        labels = torch.tensor([0, 0, 1])
        keep = run_nms(m, boxes, scores, labels, 0.5, "三个框")
        need(keep == [0, 2], f"三个框：下标 0、1 同类且高度重合，只留分高的；下标 2 和它们位置相同但类别不同，要保留。期望 [0, 2]，现在是 {keep}")
        empty = m.class_aware_nms(boxes[:0], scores[:0], labels[:0], 0.5)
        need(isinstance(empty, torch.Tensor) and tuple(empty.shape) == (0,) and empty.dtype == torch.long,
             "空输入：应返回形状 (0,) 的 long 张量")
        one = run_nms(m, boxes[:1], scores[:1], labels[:1], 0.5, "只有一个框")
        need(one == [0], f"只有一个框时应原样保留，返回 [0]，现在是 {one}")
        apart = torch.tensor([[0., 0., 2., 2.], [5., 5., 7., 7.], [10., 0., 12., 2.]])
        keep = run_nms(m, apart, torch.tensor([0.3, 0.9, 0.6]), torch.tensor([0, 0, 0]), 0.5, "三个互不重叠的同类框")
        need(keep == [1, 2, 0], f"三个互不重叠的同类框都要保留，并按分数从高到低给出原始下标，期望 [1, 2, 0]，现在是 {keep}")

    def nms_order():
        boxes = torch.tensor([[0., 0., 10., 10.], [20., 20., 30., 30.], [1., 1., 11., 11.],
                              [21., 21., 31., 31.], [40., 40., 50., 50.], [1., 1., 11., 11.]])
        scores = torch.tensor([0.60, 0.90, 0.80, 0.70, 0.95, 0.75])
        labels = torch.tensor([0, 0, 0, 0, 0, 1])
        what = "分数没有排好序的 6 个框"
        keep = run_nms(m, boxes, scores, labels, 0.5, what)
        explain_nms(keep, boxes, scores, labels, 0.5, 64, what)
        need(keep == [4, 1, 2, 5], f"{what}：期望保留 [4, 1, 2, 5]，现在是 {keep}")

    def nms_greedy():
        chain = torch.tensor([[0., 0., 10., 10.], [4., 0., 14., 10.], [8., 0., 18., 10.]])
        same = torch.tensor([0, 0, 0])
        for scores, expected in ((torch.tensor([0.9, 0.8, 0.7]), [0, 2]), (torch.tensor([0.8, 0.9, 0.7]), [1])):
            what = f"链式重叠的三个框（分数 {[round(float(s), 1) for s in scores]}，阈值 0.3）"
            keep = run_nms(m, chain, scores, same, 0.3, what)
            explain_nms(keep, chain, scores, same, 0.3, 32, what)
            need(keep == expected, f"{what}：期望 {expected}，现在是 {keep}")
        pair = torch.tensor([[0., 0., 2., 2.], [0., 0., 2., 1.]])
        ps, pl = torch.tensor([0.9, 0.8]), torch.tensor([0, 0])
        keep = run_nms(m, pair, ps, pl, 0.5, "IoU 正好等于阈值")
        need(keep == [0, 1], f"两个框的 IoU 正好是 0.5，阈值也是 0.5：只有严格大于阈值才抑制，两个都应保留，现在是 {keep}")
        keep = run_nms(m, pair, ps, pl, 0.4, "IoU 0.5、阈值 0.4")
        need(keep == [0], f"两个框的 IoU 是 0.5，阈值 0.4：应只保留分高的那个，现在是 {keep}")
        g = torch.Generator().manual_seed(5)
        lo = torch.randint(0, 36, (40, 2), generator=g)
        wh = torch.randint(6, 16, (40, 2), generator=g)
        boxes = torch.cat([lo, lo + wh], dim=-1).float()
        scores = (torch.randperm(40, generator=g).float() + 1) / 41
        labels = torch.randint(0, 3, (40,), generator=g)
        for thr in (0.2, 0.4):
            what = f"40 个随机框、3 个类别（阈值 {thr}）"
            keep = run_nms(m, boxes, scores, labels, thr, what)
            explain_nms(keep, boxes, scores, labels, thr, 64, what)

    results = [
        check("批量 IoU：形状、基本数值、和数像素对照", iou_basic),
        check("批量 IoU：面积为 0 的框、很小的框", iou_edge),
        check("类别感知 NMS：同类抑制、跨类保留、空输入", nms_basic),
        check("类别感知 NMS：分数乱序时返回原始下标并按分数排序", nms_order),
        check("类别感知 NMS：链式重叠、阈值边界、随机框", nms_greedy),
    ]
    print(f"通过 {sum(results)}/{len(results)}；未完成实现应显示 TODO。")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
