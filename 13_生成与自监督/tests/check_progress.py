"""第 13 章进度检查。"""

from __future__ import annotations

import importlib.util
import math
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
    except Wrong as exc:
        print(f"[FAIL] {name} — {exc}")
        return False
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] {name} — 运行时出错 {type(exc).__name__}: {exc}")
        return False
    print(f"[PASS] {name}")
    return True


def main() -> int:
    m = load()

    def sample(mu, logvar, eps, what: str) -> torch.Tensor:
        before = (mu.detach().clone(), logvar.detach().clone(), None if eps is None else eps.clone())
        z = m.reparameterize(mu, logvar) if eps is None else m.reparameterize(mu, logvar, eps)
        need(isinstance(z, torch.Tensor), f"{what}：返回值应是 torch.Tensor，现在是 {type(z).__name__}")
        need(z.shape == mu.shape, f"{what}：输出形状应和 mu 相同 {tuple(mu.shape)}，现在是 {tuple(z.shape)}")
        need(z.dtype == mu.dtype, f"{what}：输出的 dtype 应和 mu 相同（{mu.dtype}），现在是 {z.dtype}")
        need(torch.equal(mu.detach(), before[0]) and torch.equal(logvar.detach(), before[1]) and (eps is None or torch.equal(eps, before[2])),
             f"{what}：输入的 mu、logvar 或 eps 被改动了，不要用原地操作")
        need(torch.isfinite(z).all(), f"{what}：结果里有 NaN 或 inf")
        return z

    def reparam_values():
        mu = torch.tensor([[1., 2.]])
        eps = torch.tensor([[0.5, -1.]])
        z = sample(mu, torch.zeros_like(mu), eps, "logvar 为 0")
        need(torch.allclose(z, torch.tensor([[1.5, 1.]]), atol=1e-6), f"logvar 为 0 时标准差是 1，z 应等于 mu + eps = [[1.5, 1.0]]，现在是 {z.tolist()}。检查传进来的 eps 有没有用上")
        z = sample(mu, torch.full_like(mu, math.log(4.0)), eps, "logvar 为 ln 4")
        need(torch.allclose(z, torch.tensor([[2., 0.]]), atol=1e-5),
             f"logvar = ln 4 时方差是 4、标准差是 2，z 应等于 mu + 2 * eps = [[2.0, 0.0]]，现在是 {z.tolist()}。检查由 logvar 得到标准差的那一步：logvar 是方差的对数，不是标准差的对数")
        mu = torch.tensor([[1., 2.], [0., -1.]], dtype=torch.float64)
        logvar = torch.tensor([[math.log(4.0), 0.], [math.log(0.25), math.log(9.0)]], dtype=torch.float64)
        eps = torch.tensor([[0.5, -1.], [2., 1.]], dtype=torch.float64)
        z = sample(mu, logvar, eps, "每个元素的 logvar 不同（float64）")
        want = torch.tensor([[2., 1.], [1., 2.]], dtype=torch.float64)
        need(torch.allclose(z, want, atol=1e-9), f"每个元素的 logvar 不同时要逐元素计算：标准差依次是 2、1、0.5、3，z 应是 {want.tolist()}，现在是 {z.tolist()}")
        again = sample(mu, logvar, eps, "同一个 eps 调两次")
        need(torch.equal(z, again), "传入同一个 eps 调用两次，结果应逐元素相同。传了 eps 就不能再另外采样")

    def reparam_grad_and_sampling():
        mu = torch.tensor([[1., 2.]], requires_grad=True)
        logvar = torch.full((1, 2), math.log(4.0), requires_grad=True)
        eps = torch.tensor([[0.5, -1.]])
        z = sample(mu, logvar, eps, "带梯度的输入")
        need(z.requires_grad, "输出没有连着计算图。检查有没有 .detach()，或者用了不带梯度的采样函数")
        z.sum().backward()
        need(mu.grad is not None and torch.allclose(mu.grad, torch.ones(1, 2)), f"z 对 mu 的梯度应全为 1，现在是 {None if mu.grad is None else mu.grad.tolist()}")
        need(logvar.grad is not None, "梯度没有传到 logvar。标准差要由 logvar 可导地算出来，中间不能 detach")
        need(torch.allclose(logvar.grad, eps, atol=1e-5),
             f"logvar = ln 4、eps = [[0.5, -1]] 时，z 对 logvar 的梯度按链式法则手算是 [[0.5, -1.0]]，现在是 {logvar.grad.tolist()}")
        torch.manual_seed(13)
        n = 200000
        mu = torch.tensor([3., -1.]).expand(n, 2).clone()
        logvar = torch.tensor([math.log(4.0), math.log(0.25)]).expand(n, 2).clone()
        z1 = sample(mu, logvar, None, "不传 eps")
        z2 = sample(mu, logvar, None, "不传 eps")
        need(not torch.equal(z1, z2), "不传 eps 时每次调用都应重新采样，现在连续两次的结果完全相同")
        mean, std = z1.mean(0), z1.std(0)
        need(float(z1[:, 0].unique().numel()) > n / 2, "不传 eps 时每个元素都应有自己独立的噪声，现在一批里有大量重复的值")
        need(abs(float(mean[0]) - 3) < 0.03 and abs(float(mean[1]) + 1) < 0.03,
             f"不传 eps、采 {n} 个样本：均值应约为 mu = [3, -1]，现在是 {[round(float(v), 3) for v in mean]}。检查噪声是不是均值为 0 的标准高斯")
        need(abs(float(std[0]) - 2) < 0.04 and abs(float(std[1]) - 0.5) < 0.01,
             f"不传 eps、采 {n} 个样本：logvar 是 [ln 4, ln 0.25]，样本标准差应约为 [2, 0.5]，现在是 {[round(float(v), 3) for v in std]}。检查噪声是不是标准差为 1 的高斯，以及由 logvar 得到标准差的那一步")

    def loss_of(z1, z2, t, what: str) -> float:
        before = (z1.detach().clone(), z2.detach().clone())
        loss = m.nt_xent_loss(z1, z2, t)
        need(isinstance(loss, torch.Tensor), f"{what}：返回值应是 torch.Tensor，现在是 {type(loss).__name__}")
        need(loss.ndim == 0, f"{what}：返回值应是标量（0 维），现在的形状是 {tuple(loss.shape)}")
        need(torch.isfinite(loss), f"{what}：损失不是有限值（{float(loss.detach())}）")
        need(torch.equal(z1.detach(), before[0]) and torch.equal(z2.detach(), before[1]), f"{what}：输入的 z1 或 z2 被改动了，归一化不要用原地操作")
        return float(loss.detach())

    def contrast_values():
        eye = torch.eye(6)[:4]
        want = math.log(1 + 6 * math.exp(-5))
        got = loss_of(eye.clone(), eye.clone(), 0.2, "4 对互相垂直的单位向量")
        need(abs(got - want) < 1e-4,
             f"z1 = z2 = 6 维单位矩阵的前 4 行、温度 0.2：每个锚点和正样本的相似度是 1，和另外 6 个向量的相似度是 0，手算每个锚点的损失是 ln(1 + 6e^-5)，约 {want:.4f}，现在是 {got:.4f}。"
             "对照文档字符串的定义逐项检查：分母里有没有把自己排除掉、正样本是不是另一个视图的同一行、候选是不是除自己以外的全部 2B-1 个、最后是不是对 2B 个锚点取平均")
        want = math.log(1 + 6 * math.exp(-2))
        got = loss_of(eye.clone(), eye.clone(), 0.5, "同一组向量、温度 0.5")
        need(abs(got - want) < 1e-4, f"同一组向量、温度换成 0.5：手算是 ln(1 + 6e^-2)，约 {want:.4f}，现在是 {got:.4f}。检查相似度是不是除以了传进来的 temperature")
        a, b = [1., 0., 0.], [0., 1., 0.]
        z1, z2 = torch.tensor([a, a]), torch.tensor([a, b])
        part = math.log(2 + math.exp(-5))
        want = (3 * part + 5 + math.log(3)) / 4
        got = loss_of(z1, z2, 0.2, "两个视图不一样的 2 对向量")
        need(abs(got - want) < 1e-4,
             f"z1 = [a, a]、z2 = [a, b]（a、b 互相垂直）、温度 0.2：四个锚点的损失手算依次是 ln(2 + e^-5)、5 + ln(2 + e^-5)、ln(2 + e^-5)、ln 3，平均约 {want:.4f}，现在是 {got:.4f}。"
             "检查是不是 z1、z2 两边的向量都当了锚点，以及同一个视图里的其他向量有没有算进分母")
        got = loss_of(z2, z1, 0.2, "交换两个视图")
        need(abs(got - want) < 1e-4, f"上一组输入交换 z1 和 z2，损失应不变（约 {want:.4f}），现在是 {got:.4f}")

    def contrast_invariance():
        torch.manual_seed(13)
        z1 = torch.randn(4, 6, requires_grad=True)
        z2 = (z1.detach().clone() + 0.5 * torch.randn(4, 6)).requires_grad_(True)
        loss = m.nt_xent_loss(z1, z2, 0.2)
        need(isinstance(loss, torch.Tensor) and loss.ndim == 0 and torch.isfinite(loss), "随机输入：损失应是有限的标量")
        base = float(loss.detach())
        swapped = loss_of(z2, z1, 0.2, "交换两个视图")
        need(abs(base - swapped) < 1e-5, f"交换 z1 和 z2 后损失应不变：{base:.5f} 对 {swapped:.5f}。两个视图都要当锚点")
        s1 = torch.tensor([[10.], [0.1], [3.], [7.]])
        s2 = torch.tensor([[2.], [50.], [0.5], [1.]])
        scaled = loss_of(z1.detach() * s1, z2.detach() * s2, 0.2, "每一行乘上不同的正数")
        need(abs(base - scaled) < 1e-4,
             f"把每一行各乘一个正数（方向不变），损失应不变：{base:.5f} 对 {scaled:.5f}。检查有没有做 L2 归一化，以及归一化是不是沿着特征那一维、每一行各自做的")
        loss.backward()
        need(z1.grad is not None and torch.isfinite(z1.grad).all() and float(z1.grad.abs().sum()) > 0, "梯度应能传回 z1，并且是有限值、不全为 0")
        need(z2.grad is not None and torch.isfinite(z2.grad).all() and float(z2.grad.abs().sum()) > 0, "梯度也应能传回 z2。两个视图都参与计算，不要对其中一个 detach")
        big = loss_of(z1.detach() * 1e4, z2.detach() * 1e4, 0.2, "数值很大的输入")
        need(abs(base - big) < 1e-3, f"输入整体放大 10000 倍，损失应不变：{base:.5f} 对 {big:.5f}")

    results = [
        check("VAE 重参数：给定 eps 时的数值", reparam_values),
        check("VAE 重参数：梯度、不传 eps 时的采样", reparam_grad_and_sampling),
        check("NT-Xent：手算的数值、温度、对称", contrast_values),
        check("NT-Xent：归一化、梯度传到两个视图", contrast_invariance),
    ]
    print(f"通过 {sum(results)}/{len(results)}；初始 TODO 是预期结果。")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
