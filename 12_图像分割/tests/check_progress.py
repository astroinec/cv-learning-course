"""第 12 章进度检查。"""

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
    spec = importlib.util.spec_from_file_location("chapter12_hw", _IMPL / "作业_Dice与跳跃连接.py")
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


def dice(m, logits, target, what: str, **kwargs) -> float:
    """调用作业里的损失，检查返回的是有限的标量，并且没有改动输入。"""
    before = (logits.detach().clone(), target.clone())
    loss = m.binary_dice_loss(logits, target, **kwargs)
    need(isinstance(loss, torch.Tensor), f"{what}：返回值应是 torch.Tensor，现在是 {type(loss).__name__}")
    need(loss.ndim == 0, f"{what}：返回值应是标量（0 维），现在的形状是 {tuple(loss.shape)}。逐样本的 Dice 要再对 batch 取平均")
    need(torch.equal(logits.detach(), before[0]) and torch.equal(target, before[1]), f"{what}：输入的 logits 或 target 被改动了，不要用原地操作")
    need(torch.isfinite(loss), f"{what}：损失不是有限值（{float(loss.detach())}）。检查分母为 0 的情况和 eps 加在了哪里")
    return float(loss.detach())


def main() -> int:
    m = load()

    def dice_values():
        logits = torch.zeros(1, 1, 2, 2)
        target = torch.tensor([[[[1., 0.], [0., 0.]]]])
        got = dice(m, logits, target, "logit 全 0、目标 1 个像素")
        need(abs(got - 2 / 3) < 1e-4,
             f"2x2 的图，logit 全为 0（概率都是 0.5），目标只有 1 个像素：按文档字符串的定义手算，Dice 是 1/3，损失应是 0.6667，现在是 {got:.4f}。"
             "逐项核对：有没有做 sigmoid、分子的系数、分母是不是概率之和加目标之和、最后是不是 1 减去 Dice")
        got = dice(m, torch.zeros(1, 1, 2, 2), torch.full((1, 1, 2, 2), 0.5), "logit 全 0、目标全是 0.5")
        need(abs(got - 0.5) < 1e-4,
             f"2x2 的图，logit 全为 0，目标的每个像素都是 0.5（target 可以是 0 到 1 之间的小数）：按定义手算损失应是 0.5，现在是 {got:.4f}。检查有没有把 target 也拿去和阈值比")
        ones = torch.full((1, 1, 2, 2), 12.0)
        got = dice(m, ones, torch.ones_like(ones), "预测完全正确")
        need(-1e-6 <= got < 1e-4, f"logit 全为 12、目标全 1：损失应在 0 到 1e-4 之间，现在是 {got:.6f}。Dice 不会超过 1，损失是负数说明拿去算的不是 0 到 1 之间的概率")
        logits = torch.full((2, 1, 4, 4), -20.0)
        logits[0] = 20.0
        target = torch.zeros(2, 1, 4, 4)
        target[0] = 1.0
        target[1, 0, 1, 2] = 1.0
        got = dice(m, logits, target, "一张全对、一张全漏")
        need(abs(got - 0.5) < 1e-3,
             f"一批两张图：第一张 16 个像素的目标全部预测对，第二张 1 个像素的目标完全漏掉。两张图的 Dice 分别约为 1 和 0，损失应是 0.5，现在是 {got:.4f}。"
             "检查求和是在每张图自己的像素上做，还是把整批的像素合在了一起；以及对 batch 是取平均还是求和")

    def dice_empty():
        logits = torch.full((2, 1, 4, 4), -12.0)
        target = torch.zeros_like(logits)
        expected = 0.9899  # 手算：每张图 16 个像素，每个像素的概率是 1/(1+e^12)，按文档字符串里空 target 的式子
        got = dice(m, logits, target, "空掩膜、logit 全为 -12")
        need(abs(got - expected) < 2e-3,
             f"空掩膜、logit 全为 -12、默认 eps：按文档字符串里“空 target 的约定”，损失约为 {expected:.4f}（接近 1，不是 0），现在是 {got:.4f}。检查 eps 是不是分子和分母都加了")
        got = dice(m, torch.full((1, 1, 4, 4), -200.0), torch.zeros(1, 1, 4, 4), "空掩膜、logit 全为 -200")
        need(abs(got) < 1e-6,
             f"空掩膜、logit 全为 -200（sigmoid 的结果正好是 0）：这时分子和分母都只剩 eps，损失应是 0，现在是 {got:.6f}")
        got = dice(m, logits[:1], target[:1], "空掩膜、logit 全为 -12、eps=1", eps=1.0)
        need(0 <= got < 1e-3, f"空掩膜、logit 全为 -12、eps 取 1：损失应接近 0（小于 0.001），现在是 {got:.4f}。检查函数里用的是不是传进来的 eps")
        got = dice(m, torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 2, 2), "空掩膜、logit 全为 0、eps=2", eps=2.0)
        need(abs(got - 0.5) < 1e-4, f"2x2 的空掩膜、logit 全为 0、eps 取 2：概率之和是 2，Dice 是 2/(2+2)，损失应是 0.5，现在是 {got:.4f}")

    def dice_grad():
        logits = torch.full((2, 1, 4, 4), -12.0, requires_grad=True)
        loss = m.binary_dice_loss(logits, torch.zeros(2, 1, 4, 4))
        need(isinstance(loss, torch.Tensor) and loss.ndim == 0 and torch.isfinite(loss), "空掩膜、logit 全为 -12：损失应是有限的标量")
        need(loss.requires_grad, "损失没有连着计算图，无法反向传播。检查有没有先做硬判定（大于 0.5）、有没有用 .detach() 或 .item()")
        loss.backward()
        need(logits.grad is not None and torch.isfinite(logits.grad).all(), "空掩膜时对 logits 的梯度应是有限值")
        logits = torch.zeros(1, 1, 2, 2, requires_grad=True)
        loss = m.binary_dice_loss(logits, torch.ones(1, 1, 2, 2))
        need(isinstance(loss, torch.Tensor) and loss.requires_grad, "损失没有连着计算图，无法反向传播")
        loss.backward()
        g = logits.grad
        need(g is not None and torch.isfinite(g).all(), "目标全 1、logit 全 0：对 logits 的梯度应是有限值")
        need((g < 0).all(), f"目标全 1 时把任何一个 logit 调大，损失都应变小，梯度应全为负，现在是 {g.flatten().tolist()}")
        need(torch.allclose(g, torch.full_like(g, -1 / 18), atol=1e-5),
             f"2x2 的图、目标全 1、logit 全 0：按定义手算，每个 logit 上的梯度是 -1/18（约 -0.0556），现在是 {g.flatten().tolist()}")

    def call_align(decoder, skip, what: str):
        before = (decoder.detach().clone(), skip.clone())
        out = m.align_and_concat(decoder, skip)
        need(isinstance(out, torch.Tensor), f"{what}：返回值应是 torch.Tensor")
        shape = (skip.shape[0], decoder.shape[1] + skip.shape[1], skip.shape[2], skip.shape[3])
        need(tuple(out.shape) == shape, f"{what}：输出形状应是 {shape}（空间尺寸跟 skip 走），现在是 {tuple(out.shape)}")
        need(torch.equal(decoder.detach(), before[0]) and torch.equal(skip, before[1]), f"{what}：输入被改动了")
        c1 = decoder.shape[1]
        need(torch.equal(out[:, c1:], skip), f"{what}：输出的后 {skip.shape[1]} 个通道应该就是 skip 本身。检查拼接的先后顺序：decoder 在前，skip 在后")
        return out[:, :c1]

    def filled(n, c, h, w):
        return torch.arange(1, n * c * h * w + 1, dtype=torch.float32).reshape(n, c, h, w)

    def align_even():
        dec, skip = filled(2, 1, 6, 8), torch.randn(2, 2, 4, 4)
        got = call_align(dec, skip, "decoder 6x8 裁到 4x4")
        need(torch.equal(got, dec[:, :, 1:5, 2:6]),
             "decoder 6x8、skip 4x4：居中裁剪应留下第 1 到 4 行、第 2 到 5 列（从 0 数），上下各裁 1、左右各裁 2。现在留下的不是这一块，检查裁剪的起点，以及有没有对内容做缩放")
        dec, skip = filled(1, 2, 2, 3), torch.randn(1, 1, 6, 5)
        got = call_align(dec, skip, "decoder 2x3 补到 6x5")
        want = torch.zeros(1, 2, 6, 5)
        want[:, :, 2:4, 1:4] = dec
        need(torch.equal(got[:, :, 2:4, 1:4], dec),
             "decoder 2x3、skip 6x5：对称补零后，原来的内容应落在第 2 到 3 行、第 1 到 3 列（从 0 数），上下各补 2、左右各补 1。现在不在这个位置，检查补零是不是两侧各一半、F.pad 的参数顺序，以及有没有用插值")
        need(torch.equal(got, want), "decoder 2x3、skip 6x5：内容的位置对了，但补出来的部分不全是 0。只能补 0，不能复制边缘或插值")
        dec, skip = filled(1, 1, 6, 3), torch.randn(1, 3, 4, 5)
        got = call_align(dec, skip, "decoder 6x3 对到 4x5")
        want = torch.zeros(1, 1, 4, 5)
        want[:, :, :, 1:4] = dec[:, :, 1:5, :]
        need(torch.equal(got, want), "decoder 6x3、skip 4x5：高要居中裁掉 2 行，宽要两侧各补 1 列 0，两个方向分别处理。现在的结果不对，检查高和宽的差有没有用反")
        dec, skip = filled(1, 2, 3, 3), torch.randn(1, 1, 3, 3)
        need(torch.equal(call_align(dec, skip, "尺寸相同"), dec), "decoder 和 skip 的空间尺寸相同时，decoder 应原样拼进去")

    def align_odd():
        dec, skip = filled(2, 3, 9, 11), torch.randn(2, 5, 8, 14)
        got = call_align(dec, skip, "decoder 9x11 对到 8x14")
        ok = False
        for top in (0, 1):
            for left in (1, 2):
                want = torch.zeros(2, 3, 8, 14)
                want[:, :, :, left:left + 11] = dec[:, :, top:top + 8, :]
                ok = ok or torch.equal(got, want)
        need(ok, "decoder 9x11、skip 8x14：高要裁掉 1 行（裁上面或下面都可以），宽要补 3 列 0（左 1 右 2 或左 2 右 1 都可以）。现在的结果和这四种都对不上，检查内容有没有被缩放、补的是不是 0、两侧相差是不是超过了 1 列")
        dec = filled(1, 2, 4, 4).requires_grad_(True)
        out = m.align_and_concat(dec, torch.randn(1, 1, 6, 2))
        need(isinstance(out, torch.Tensor) and out.requires_grad, "decoder 带梯度时，拼接结果也应连着计算图。检查有没有 .detach()，或者转成了 numpy 再转回来")
        try:
            m.align_and_concat(torch.zeros(1, 1, 4, 4), torch.zeros(2, 1, 4, 4))
        except ValueError:
            pass
        except Exception as exc:  # noqa: BLE001
            raise Wrong(f"decoder 和 skip 的 batch 不同时应抛出 ValueError，现在抛出的是 {type(exc).__name__}。在拼接之前先检查 batch") from exc
        else:
            raise Wrong("decoder 和 skip 的 batch 不同时应抛出 ValueError，现在没有报错")

    results = [
        check("Dice 损失：手算的数值、逐样本再平均", dice_values),
        check("Dice 损失：空掩膜和 eps 的约定", dice_empty),
        check("Dice 损失：可以反向传播、梯度方向和大小", dice_grad),
        check("skip 对齐：居中裁剪和对称补零的位置", align_even),
        check("skip 对齐：差为奇数、梯度、batch 不同", align_odd),
    ]
    print(f"通过 {sum(results)}/{len(results)}；TODO 不应导致检查器崩溃。")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
