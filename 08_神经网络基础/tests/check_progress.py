"""第 8 章进度检查：TODO 是正常未完成状态。

期望值都来自手算的小例子和数学性质（平移不变、逐样本独立、极端输入），这个文件里没有作业的实现。
"""

from __future__ import annotations

import importlib.util
import math
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
    except AssertionError as exc:
        print(f"[FAIL] {name} — {exc}")
        return False
    except Exception as exc:
        print(f"[FAIL] {name} — {type(exc).__name__}: {exc}")
        return False
    print(f"[PASS] {name}")
    return True


def call(hint, fn, *args):
    """调用作业函数。还没实现时原样抛出（显示 TODO），其他异常补一句该查什么。"""
    try:
        return fn(*args)
    except NotImplementedError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(f"调用时抛出 {type(exc).__name__}: {exc}。{hint}") from None


def first_diff(got, want, tol):
    """返回第一个对不上的位置和两边的值，用来写提示。"""
    bad = np.argwhere(~np.isclose(got, want, rtol=0, atol=tol))
    idx = tuple(int(v) for v in bad[0])
    return idx, float(got[idx]), float(want[idx])


# 手算例子：B=5，D=3，H=4，C=2。数都取得很简单，可以拿纸笔按教程 8.2 节的形状约定核对。
X = np.array([[1.0, 2.0, -1.0], [0.0, -1.0, 3.0], [-2.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.5, 0.0, -0.5]])
W1 = np.array([[1.0, 0.0, -1.0, 2.0], [0.0, 1.0, 1.0, -1.0], [2.0, -1.0, 0.0, 1.0]])
B1 = np.array([0.0, 1.0, -1.0, 0.5])
W2 = np.array([[1.0, -1.0], [0.5, 1.0], [3.0, 3.0], [-1.0, 0.0]])
B2 = np.array([0.5, -1.0])
HIDDEN = np.array([[0.0, 4.0, 0.0, 0.0], [6.0, 0.0, 0.0, 4.5], [0.0, 2.0, 2.0, 0.0], [0.0, 1.0, 0.0, 0.5], [0.0, 1.5, 0.0, 1.0]])
LOGITS = np.array([[2.5, 3.0], [2.0, -7.0], [7.5, 7.0], [0.5, 0.0], [0.25, 0.5]])

# 交叉熵的手算例子：logits 取成整数的对数，每行的概率就是 整数/这一行的和。四行的和都是 8。
COUNTS = np.array([[1.0, 2.0, 5.0], [3.0, 1.0, 4.0], [2.0, 2.0, 4.0], [6.0, 1.0, 1.0]])
CE_LABELS = np.array([2, 0, 1, 1])
# 四个样本正确类别的概率是 5/8、3/8、2/8、1/8，平均交叉熵 = ln(8^4 / (5*3*2*1)) / 4
CE_EXPECTED = math.log(4096 / 30) / 4


def main() -> int:
    m = load()
    shape_hint = "作业里 w1 的形状是 (D, H)、w2 是 (H, C)，检查矩阵相乘的方向（教程 8.2 节的提示框）。"

    def unpack(result):
        assert isinstance(result, tuple) and len(result) == 2, "返回值应是 (hidden, logits) 两个数组组成的二元组"
        hidden, logits = result
        assert isinstance(hidden, np.ndarray) and isinstance(logits, np.ndarray), "hidden 和 logits 都应是 NumPy 数组"
        return hidden, logits

    def forward_hand_check():
        hidden, logits = unpack(call(shape_hint, m.two_layer_forward, X, W1, B1, W2, B2))
        assert hidden.shape == (5, 4) and logits.shape == (5, 2), (
            f"形状不对：hidden 应为 (5, 4)、logits 应为 (5, 2)，得到 {hidden.shape} 和 {logits.shape}。检查返回的先后顺序。")
        assert np.all(np.isfinite(hidden)) and np.all(np.isfinite(logits)), "输出里有 nan 或 inf"
        assert np.all(hidden >= 0), "hidden 里有负数。返回的应是经过激活之后的隐藏层，不是激活之前的值。"
        if not np.allclose(hidden, HIDDEN, rtol=0, atol=1e-9):
            idx, got, want = first_diff(hidden, HIDDEN, 1e-9)
            raise AssertionError(f"hidden 和手算结果不一致：第 {idx[0]} 个样本第 {idx[1]} 个隐藏单元应为 {want:g}，得到 {got:g}。"
                                 "检查第一层的偏置有没有加、加在激活之前还是之后，以及激活函数用的是哪一种。")
        if not np.allclose(logits, LOGITS, rtol=0, atol=1e-9):
            idx, got, want = first_diff(logits, LOGITS, 1e-9)
            raise AssertionError(f"hidden 正确，logits 和手算结果不一致：第 {idx[0]} 个样本第 {idx[1]} 类应为 {want:g}，得到 {got:g}。"
                                 "检查第二层的偏置有没有加，以及 logits 上有没有多做 softmax 或 ReLU。")

    def forward_property_check():
        rng = np.random.default_rng(8)
        x = rng.normal(size=(6, 5))
        w1 = rng.normal(size=(5, 7))
        b1 = rng.normal(size=7)
        w2 = rng.normal(size=(7, 3))
        b2 = rng.normal(size=3)
        args = (x, w1, b1, w2, b2)
        copies = [a.copy() for a in args]
        hidden, logits = unpack(call(shape_hint, m.two_layer_forward, *args))
        for name, a, c in zip(("x", "w1", "b1", "w2", "b2"), args, copies):
            assert np.array_equal(a, c), f"函数原地修改了输入 {name}。检查有没有用 +=、out= 这类原地运算，或者直接往输入数组里写值。"
        assert hidden.shape == (6, 7) and logits.shape == (6, 3), f"形状不对：应为 (6, 7) 和 (6, 3)，得到 {hidden.shape} 和 {logits.shape}"

        # 逐样本独立，B=1 也要能算
        for i in (0, 3, 5):
            h1, l1 = unpack(call("B=1 时应该也能计算。", m.two_layer_forward, x[i:i + 1], w1, b1, w2, b2))
            assert h1.shape == (1, 7) and l1.shape == (1, 3), f"B=1 时形状应为 (1, 7) 和 (1, 3)，得到 {h1.shape} 和 {l1.shape}"
            assert np.allclose(h1[0], hidden[i], atol=1e-10) and np.allclose(l1[0], logits[i], atol=1e-10), (
                "同一个样本单独计算和放在一批里计算，结果不同。每个样本应独立计算，检查有没有沿 batch 方向做求和或归一化。")

        # 极端输入一：第二层权重全 0，logits 只剩第二层的偏置
        _, l0 = unpack(m.two_layer_forward(x, w1, b1, np.zeros_like(w2), b2))
        assert np.allclose(l0, np.broadcast_to(b2, l0.shape), atol=1e-12), (
            "w2 全为 0 时，每一行 logits 都应正好等于 b2。检查第二层的偏置，以及 logits 之后有没有多余的运算。")
        # 极端输入二：第一层权重全 0、偏置全为负，隐藏层应全部关掉
        h0, l0 = unpack(m.two_layer_forward(x, np.zeros_like(w1), -np.abs(b1) - 1.0, w2, b2))
        assert np.all(h0 == 0), "w1 全为 0、b1 全为负时，hidden 应全为 0。检查激活函数。"
        assert np.allclose(l0, np.broadcast_to(b2, l0.shape), atol=1e-12), "hidden 全为 0 时，每一行 logits 都应正好等于 b2。"
        # 极端输入三：输入全 0、第一层偏置全为正，隐藏层的每一行都等于这个偏置
        pos = np.abs(b1) + 0.5
        h0, _ = unpack(m.two_layer_forward(np.zeros_like(x), w1, pos, w2, b2))
        assert np.allclose(h0, np.broadcast_to(pos, h0.shape), atol=1e-12), "x 全为 0、b1 全为正时，hidden 的每一行都应等于 b1。检查第一层的偏置和激活函数。"

        # 偏置为 0 时的两条性质：输入放大 3 倍，输出也正好放大 3 倍；输入取反，输出不是简单地取反或不变
        zb1, zb2 = np.zeros_like(b1), np.zeros_like(b2)
        ha, la = unpack(m.two_layer_forward(x, w1, zb1, w2, zb2))
        hb, lb = unpack(m.two_layer_forward(3.0 * x, w1, zb1, w2, zb2))
        assert np.allclose(hb, 3.0 * ha, atol=1e-9) and np.allclose(lb, 3.0 * la, atol=1e-9), (
            "偏置为 0 时把输入放大 3 倍，hidden 和 logits 都应正好放大 3 倍。作业规定的激活函数满足这一条，检查是不是换成了别的激活，或者在 logits 上多做了运算。")
        hc, lc = unpack(m.two_layer_forward(-x, w1, zb1, w2, zb2))
        assert not np.allclose(lc, -la, atol=1e-6), "偏置为 0 时输入取反，logits 也正好取反，说明两层之间没有非线性。检查隐藏层的激活。"
        assert not np.allclose(hc, ha, atol=1e-6), "输入取反后 hidden 完全不变。激活函数应该把负的部分截成 0，而不是取绝对值。"

    def ce_scalar(value):
        assert isinstance(value, (float, np.floating)), f"应返回一个标量（Python float），得到的类型是 {type(value).__name__}。检查有没有对 batch 取平均。"
        return float(value)

    def ce_hand_check():
        logits = np.log(COUNTS)
        loss = ce_scalar(call("logits 的形状是 (B, C)，labels 是 (B,) 的整数数组。", m.cross_entropy_from_logits, logits, CE_LABELS))
        assert math.isfinite(loss), "结果是 nan 或 inf"
        assert abs(loss - CE_EXPECTED) < 1e-9, (
            f"4 个样本、3 个类别的手算例子应得 {CE_EXPECTED:.6f}，得到 {loss:.6f}。"
            "依次检查：每个样本取的是不是它自己标签对应的那一类（这个例子里样本数和类别数不同、各行的概率也不同）；"
            "对样本做的是平均还是求和；用的是不是自然对数；符号。")
        uniform = ce_scalar(m.cross_entropy_from_logits(np.zeros((3, 10)), np.array([0, 4, 9])))
        assert abs(uniform - math.log(10)) < 1e-9, f"10 个类别的 logits 全相同时，损失应为 ln 10 = {math.log(10):.6f}，得到 {uniform:.6f}"
        single = ce_scalar(m.cross_entropy_from_logits(np.log(np.array([[1.0, 3.0]])), np.array([1])))
        assert abs(single - math.log(4 / 3)) < 1e-9, f"只有 1 个样本（B=1）时结果不对：应为 ln(4/3) = {math.log(4 / 3):.6f}，得到 {single:.6f}"

    def ce_property_check():
        rng = np.random.default_rng(88)
        logits = rng.normal(size=(6, 4)) * 2.0
        labels = np.array([3, 0, 1, 1, 2, 0])
        keep_logits, keep_labels = logits.copy(), labels.copy()
        base = ce_scalar(call("logits 的形状是 (B, C)，labels 是 (B,) 的整数数组。", m.cross_entropy_from_logits, logits, labels))
        assert np.array_equal(logits, keep_logits) and np.array_equal(labels, keep_labels), (
            "函数原地修改了输入的 logits 或 labels。减最大值时要得到一个新数组，不要用 -= 写回原数组。")
        assert math.isfinite(base) and base > 0, f"交叉熵应是正的有限值，得到 {base}。检查符号。"

        # 每一行各加一个常数，这一行的概率不变，损失也不变
        shift = np.array([[5.0], [-3.0], [40.0], [0.0], [-17.5], [2.0]])
        shifted = ce_scalar(m.cross_entropy_from_logits(logits + shift, labels))
        assert abs(shifted - base) < 1e-9, (
            f"给每一行的 logits 各加一个不同的常数，损失应该不变（{base:.6f}），得到 {shifted:.6f}。检查归一化是不是按行做的（轴的方向、keepdims）。")
        # 同一批样本重复一遍，平均值不变
        doubled = ce_scalar(m.cross_entropy_from_logits(np.concatenate([logits, logits]), np.concatenate([labels, labels])))
        assert abs(doubled - base) < 1e-9, f"把同一批样本重复一遍，平均损失应该不变（{base:.6f}），得到 {doubled:.6f}。检查是不是对 batch 求了和。"
        # 样本和标签一起换顺序，损失不变；只换标签，损失要变
        order = np.array([4, 2, 5, 0, 3, 1])
        permuted = ce_scalar(m.cross_entropy_from_logits(logits[order], labels[order]))
        assert abs(permuted - base) < 1e-9, "样本和标签一起换顺序，损失应该不变。"
        moved = ce_scalar(m.cross_entropy_from_logits(logits, labels[order]))
        assert abs(moved - base) > 1e-3, (
            "只把标签换了顺序、logits 不动，损失居然没变。说明没有做到“第 i 个样本取第 i 个标签对应的那一类”，检查取正确类别的索引写法。")
        # 正确类别的 logit 比别的类别大很多时，损失接近 0
        confident = np.zeros((3, 5))
        confident[np.arange(3), [1, 4, 0]] = 60.0
        tiny = ce_scalar(m.cross_entropy_from_logits(confident, np.array([1, 4, 0])))
        assert abs(tiny) < 1e-9, f"正确类别的 logit 比其他类别大 60 时，损失应非常接近 0，得到 {tiny}"

    def ce_stability_check():
        hint = "大 logits 不应导致报错。"
        # 原有用例：两行各自减掉最大值后是 [-1, 0]，正确类别分别是大的和小的那个
        loss = ce_scalar(call(hint, m.cross_entropy_from_logits, np.array([[1000.0, 1001.0], [-1000.0, -999.0]]), np.array([1, 0])))
        expected = (math.log1p(math.exp(-1.0)) + math.log1p(math.exp(1.0))) / 2
        assert math.isfinite(loss), (
            f"logits 在正负 1000 附近时得到 {loss}。直接对 logits 取指数会溢出；另外检查减掉的是每一行各自的最大值，还是整个 batch 的最大值。")
        assert abs(loss - expected) < 1e-8, f"logits 在正负 1000 附近的用例应得 {expected:.6f}，得到 {loss:.6f}"
        # 正确类别比这一行的最大值小几百：它的概率在浮点数里已经是 0，但交叉熵仍然是一个确定的有限值
        loss = ce_scalar(call(hint, m.cross_entropy_from_logits, np.array([[0.0, 800.0], [-300.0, 300.0]]), np.array([0, 0])))
        assert math.isfinite(loss), (
            f"正确类别的 logit 比这一行的最大值小几百时得到 {loss}。这时正确类别的概率下溢成 0，先算概率再取对数会得到 inf。"
            "检查能不能不经过概率、直接得到对数概率（教程 8.3 节）。")
        assert abs(loss - 700.0) < 1e-6, (
            f"两个样本的交叉熵应分别约为 800 和 600，平均 700，得到 {loss:.4f}。"
            "如果结果是几十，多半是给概率加了一个小常数或做了截断再取对数，这样会把很大的损失压小。")
        loss = ce_scalar(m.cross_entropy_from_logits(np.array([[10000.0, 0.0, -10000.0]]), np.array([0])))
        assert abs(loss) < 1e-9, f"正确类别的 logit 是 10000、其余小得多时，损失应为 0，得到 {loss}"

    results = [
        run_check("两层前向：手算例子的形状与数值", forward_hand_check),
        run_check("两层前向：逐样本独立、极端输入、不改输入", forward_property_check),
        run_check("交叉熵：手算例子", ce_hand_check),
        run_check("交叉熵：平移不变、取平均、标签对应、不改输入", ce_property_check),
        run_check("交叉熵：大 logits 的数值稳定性", ce_stability_check),
    ]
    print(f"通过 {sum(results)}/{len(results)}；TODO 不代表程序崩溃。")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
