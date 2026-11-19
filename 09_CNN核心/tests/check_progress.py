"""第 9 章进度检查。

尺寸函数的期望值来自手算和 PyTorch 卷积的实际输出；残差块用几条性质（关掉残差分支、关掉第一处激活）
和 torchvision 自带的 BasicBlock 对照。这个文件里没有作业的实现。
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
_IMPL = Path(os.environ["COURSE_IMPL_DIR"]) / ROOT.name if os.environ.get("COURSE_IMPL_DIR") else ROOT


def load():
    spec = importlib.util.spec_from_file_location("chapter09_hw", _IMPL / "作业_卷积shape与残差块.py")
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
    except AssertionError as exc:
        print(f"[FAIL] {name} — {exc}")
        return False
    except Exception as exc:
        print(f"[FAIL] {name} — {type(exc).__name__}: {exc}")
        return False
    print(f"[PASS] {name}")
    return True


def describe(args) -> str:
    names = ("输入", "核", "步长", "填充", "膨胀")
    full = tuple(args) + (1, 0, 1)[len(args) - 2:]
    return "、".join(f"{n} {v}" for n, v in zip(names, full))


def main() -> int:
    m = load()

    def size_of(args):
        """调用作业的尺寸函数；ValueError 交给调用方处理，其他异常说明是哪组参数。"""
        try:
            return m.conv_output_size(*args)
        except (NotImplementedError, ValueError):
            raise
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(f"{describe(args)} 时抛出 {type(exc).__name__}: {exc}") from None

    def expect(args, want, hint):
        try:
            got = size_of(args)
        except ValueError as exc:
            raise AssertionError(f"{describe(args)} 是放得下的，应得 {want}，却抛出了 ValueError（{exc}）。{hint}") from None
        assert isinstance(got, int) and not isinstance(got, bool), (
            f"{describe(args)} 时返回的类型是 {type(got).__name__}，应为 Python int。检查取整之后有没有留在浮点数或 NumPy 类型上。")
        assert got == want, f"{describe(args)} 时应得 {want}，得到 {got}。{hint}"

    def size_exact_case():
        hint = "这一组算例都能整除，检查公式里填充、核和膨胀各项。"
        expect((32, 3, 1, 1), 32, hint)
        expect((31, 3, 2, 1), 16, hint)
        expect((15, 3, 1, 2, 2), 15, "检查公式里有没有膨胀：带膨胀的核实际覆盖的范围比核本身大。")
        expect((28, 5, 1, 2), 28, hint)
        expect((64, 4, 4, 0), 16, hint)
        expect((32, 3), 30, "步长、填充、膨胀不传时用默认值 1、0、1。")
        assert m.conv_output_size(size=31, kernel=3, stride=2, padding=1, dilation=1) == 16, "按关键字传参时结果不对，不要改函数的参数名。"
        edge = "核正好和（补边后的）输入一样大时，输出长度是 1，这是合法的，不应拒绝。"
        expect((1, 1, 1, 0), 1, edge)
        expect((3, 3, 1, 0), 1, edge)
        expect((5, 3, 1, 0, 2), 1, edge)

    def size_floor_case():
        hint = "这一组除不尽，结果应和 PyTorch 卷积的实际输出一致。检查除以步长之后的取整方向。"
        expect((32, 3, 2, 0), 15, hint)
        expect((32, 3, 3, 0), 10, hint)
        expect((32, 3, 2, 1), 16, hint)
        expect((224, 7, 2, 3), 112, hint)
        expect((7, 2, 2, 0), 3, hint)
        expect((33, 2, 2, 0), 16, hint)
        expect((10, 3, 2, 0, 2), 3, hint)
        expect((5, 3, 4, 0), 1, hint)
        expect((9, 3, 2, 1, 3), 3, hint)

    def size_reject_case():
        for args in [(2, 3, 2, 0), (2, 3, 1, 0), (1, 7, 1, 0), (4, 3, 1, 0, 2), (3, 5, 2, 0), (6, 3, 2, 0, 3), (1, 2, 3, 0)]:
            try:
                got = size_of(args)
            except ValueError:
                continue
            raise AssertionError(
                f"{describe(args)} 时核比补边后的输入还大，放不下，应抛出 ValueError，却返回了 {got!r}。"
                "检查两处：按公式算出来不是正数时有没有拒绝；分子为负时的取整方向（负数向下取整和向零取整结果不同）。")

    def size_torch_case():
        # 拿一维卷积的实际输出长度当标准答案，逐个比。PyTorch 报错的组合，作业函数应抛出 ValueError。
        examples = []
        wrong = total = 0
        for size in range(1, 15):
            for kernel in (1, 2, 3, 4, 5):
                for stride in (1, 2, 3):
                    for padding in (0, 1, 2):
                        for dilation in (1, 2, 3):
                            total += 1
                            args = (size, kernel, stride, padding, dilation)
                            try:
                                want = F.conv1d(torch.zeros(1, 1, size), torch.zeros(1, 1, kernel), stride=stride, padding=padding, dilation=dilation).shape[-1]
                            except RuntimeError:
                                want = None
                            try:
                                got = size_of(args)
                            except ValueError:
                                got = None
                            if got != want:
                                wrong += 1
                                if len(examples) < 3:
                                    examples.append(f"{describe(args)}：PyTorch {'报错' if want is None else '输出 ' + str(want)}，作业函数{'抛出 ValueError' if got is None else '返回 ' + repr(got)}")
        assert not wrong, f"和 PyTorch 卷积的实际输出比较，{total} 种组合里有 {wrong} 种不一致。前几个：" + "；".join(examples)

    def run_block(block, x, what):
        try:
            return block(x)
        except NotImplementedError:
            raise
        except Exception as exc:  # noqa: BLE001
            extra = "检查有没有原地修改输入 x（例如 x += ...）。" if "in-place" in str(exc) or "inplace" in str(exc) else ""
            raise AssertionError(f"{what}时 forward 抛出 {type(exc).__name__}: {exc}。{extra}") from None

    def block_basic_case():
        torch.manual_seed(9)
        block = m.BasicResidualBlock(4).eval()
        x = torch.randn(2, 4, 11, 13)
        keep = x.clone()
        with torch.no_grad():
            y = run_block(block, x, "推理模式下前向")
        assert isinstance(y, torch.Tensor), "forward 应返回一个张量"
        assert y.shape == keep.shape, f"输出的 shape 应和输入相同 {tuple(keep.shape)}，得到 {tuple(y.shape)}"
        assert torch.isfinite(y).all(), "输出里有 nan 或 inf"
        assert torch.equal(x, keep), "forward 原地修改了输入 x。相加时不要写回 x（例如 x += ...），ReLU 也不要原地作用在 x 上。"
        assert (y >= 0).all(), f"输出里有 {int((y < 0).sum())} 个负数。按文档字符串，相加之后还有最后一步，输出不应有负数。"

        x2 = keep.clone().requires_grad_(True)
        y2 = run_block(block, x2, "输入需要梯度时前向")
        try:
            y2.mean().backward()
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(f"反向传播时抛出 {type(exc).__name__}: {exc}。检查 forward 里有没有原地修改输入或中间结果。") from None
        assert x2.grad is not None and torch.isfinite(x2.grad).all(), "梯度没有传回输入，或者梯度里有 nan"
        assert float(x2.grad.abs().sum()) > 0, "传回输入的梯度全是 0"
        for name in ("conv1.weight", "bn1.weight", "bn1.bias", "conv2.weight", "bn2.weight", "bn2.bias"):
            layer, attr = name.split(".")
            grad = getattr(getattr(block, layer), attr).grad
            assert grad is not None and float(grad.abs().sum()) > 0, (
                f"{name} 没有收到梯度，说明 forward 没有用到 {layer}。__init__ 里的四个层要各用一次，检查有没有把某一层写成了另一层的名字。")

        torch.manual_seed(19)
        block3 = m.BasicResidualBlock(3).train()
        x3 = torch.randn(5, 3, 6, 4)
        y3 = run_block(block3, x3, "训练模式下前向")
        assert y3.shape == x3.shape and torch.isfinite(y3).all() and (y3 >= 0).all(), "训练模式、3 个通道、输入 (5, 3, 6, 4) 时，输出的 shape、数值范围不对"

    def block_shortcut_case():
        # 把 bn2 的缩放系数置 0、偏移设成每个通道一个常数：推理模式下残差分支的输出就是这个常数，和输入无关。
        torch.manual_seed(90)
        block = m.BasicResidualBlock(4).eval()
        offset = torch.tensor([0.5, -1.0, 0.0, 2.0])
        with torch.no_grad():
            block.bn2.weight.zero_()
            block.bn2.bias.copy_(offset)
        x = torch.randn(3, 4, 7, 5, requires_grad=True)
        y = run_block(block, x, "前向")
        want = (x.detach() + offset.view(1, 4, 1, 1)).clamp(min=0)
        diff = float((y.detach() - want).abs().max())
        assert diff < 1e-5, (
            f"把 bn2 的缩放系数置 0、偏移设成逐通道的常数之后，残差分支的输出就是这个常数，块的输出应等于“输入加这个常数、再截掉负值”，现在最大相差 {diff:.3g}。"
            "检查：有没有把输入 x 加回来，加的是不是原样的 x（系数是 1、没有先经过别的运算）；bn2 是不是残差分支的最后一步（它之后、相加之前没有 ReLU）；相加之后有没有再过 ReLU。")
        y.sum().backward()
        want_grad = (x.detach() + offset.view(1, 4, 1, 1) > 0).float()
        assert x.grad is not None and torch.allclose(x.grad, want_grad, atol=1e-6), (
            "残差分支被关掉之后，输出对输入的梯度应沿捷径原样传回（输出为正的位置是 1，其余是 0），现在不是。"
            "检查相加时有没有把 x 从计算图里断开（detach、先转成 NumPy 再转回来之类）。")

    def block_first_relu_case():
        # 把 bn1 的输出固定成负的常数：它之后的激活应把残差分支整个关掉。
        torch.manual_seed(91)
        block = m.BasicResidualBlock(4).eval()
        with torch.no_grad():
            block.bn1.weight.zero_()
            block.bn1.bias.fill_(-1.0)
        x = torch.randn(2, 4, 6, 8)
        with torch.no_grad():
            y = run_block(block, x, "前向")
        diff = float((y - x.clamp(min=0)).abs().max())
        assert diff < 1e-5, (
            f"把 bn1 的输出固定成负的常数之后，残差分支应被 bn1 后面的激活截成 0，块的输出应等于输入截掉负值，现在最大相差 {diff:.3g}。"
            "检查 bn1 之后有没有 ReLU、conv2 接的是不是这个 ReLU 的输出，以及 bn1 有没有被跳过。")

    def block_library_case():
        torch.manual_seed(92)
        mine = m.BasicResidualBlock(6).eval()
        with torch.no_grad():
            for bn in (mine.bn1, mine.bn2):  # 让 BN 的四组数都不是默认值，每一层放错位置都会改变结果
                bn.weight.uniform_(0.5, 1.5)
                bn.bias.uniform_(-0.5, 0.5)
                bn.running_mean.uniform_(-0.5, 0.5)
                bn.running_var.uniform_(0.5, 1.5)
        x = torch.randn(4, 6, 9, 7)
        with torch.no_grad():
            got_eval = run_block(mine, x.clone(), "前向")
        try:
            from torchvision.models.resnet import BasicBlock
        except Exception:  # noqa: BLE001
            print("      （没有安装 torchvision，跳过和库实现的逐元素对照）")
            return
        ref = BasicBlock(6, 6).eval()
        try:
            ref.load_state_dict(mine.state_dict())
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(f"作业的块和 torchvision 的 BasicBlock 参数对不上（{str(exc).splitlines()[0][:120]}）。不要在 __init__ 里新增或改名带参数的层。") from None
        hint = "按文档字符串的顺序逐步核对：每一步用的是哪个层、它的输入是上一步的输出还是 x、两处 ReLU 各在什么位置。"
        with torch.no_grad():
            want_eval = ref(x.clone())
        diff = float((got_eval - want_eval).abs().max())
        assert diff < 1e-4, f"推理模式下，和 torchvision 的 BasicBlock（同样的权重，不带下采样）输出不一致，最大相差 {diff:.3g}。{hint}"
        mine.train()
        ref.train()
        with torch.no_grad():
            got_train = run_block(mine, x.clone(), "训练模式下前向")
            want_train = ref(x.clone())
        diff = float((got_train - want_train).abs().max())
        assert diff < 1e-4, f"训练模式下，和 torchvision 的 BasicBlock 输出不一致，最大相差 {diff:.3g}。{hint}"

    results = [
        check("卷积输出尺寸：能整除的算例与边界", size_exact_case),
        check("卷积输出尺寸：除不尽时的取整", size_floor_case),
        check("卷积输出尺寸：放不下时抛出 ValueError", size_reject_case),
        check("卷积输出尺寸：和 PyTorch 的实际输出逐个比", size_torch_case),
        check("残差块：shape、输出非负、梯度、不改输入", block_basic_case),
        check("残差块：关掉残差分支后只剩捷径", block_shortcut_case),
        check("残差块：bn1 之后的激活", block_first_relu_case),
        check("残差块：和 torchvision 的 BasicBlock 对照", block_library_case),
    ]
    print(f"通过 {sum(results)}/{len(results)}；未完成项应显示 TODO。")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
