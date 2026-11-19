"""第 10 章进度检查。

期望的参数名单和热图数值都是手写的，另外用几条性质（重复调用、逐样本独立、缩放和平移不变）检查。
这个文件里没有作业的实现。
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
_IMPL = Path(os.environ["COURSE_IMPL_DIR"]) / ROOT.name if os.environ.get("COURSE_IMPL_DIR") else ROOT


def load():
    spec = importlib.util.spec_from_file_location("chapter10_hw", _IMPL / "作业_冻结策略与热图归一化.py")
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


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 4))
        self.head = nn.Linear(4, 2)


class Heads(nn.Module):
    """几个名字互相包含的层，用来区分“以它开头”和“名字里包含”。"""

    def __init__(self):
        super().__init__()
        self.stem = nn.Linear(2, 2)
        self.prehead = nn.Linear(2, 2)
        self.head = nn.Linear(2, 2)
        self.head2 = nn.Linear(2, 2)


class WithBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, 1)
        self.bn = nn.BatchNorm2d(2)
        self.fc = nn.Linear(2, 2)


def wb(*layers: str) -> list[str]:
    """把层名展开成 层名.weight、层名.bias。"""
    return [f"{layer}.{kind}" for layer in layers for kind in ("weight", "bias")]


def main() -> int:
    m = load()

    def configure(model, prefixes, expected, hint):
        try:
            names = m.configure_trainable_parameters(model, prefixes)
        except NotImplementedError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(f"前缀为 {prefixes!r} 时抛出 {type(exc).__name__}: {exc}") from None
        assert isinstance(names, list), f"应返回参数名组成的列表，得到的类型是 {type(names).__name__}"
        flags = [n for n, p in model.named_parameters() if p.requires_grad]
        if names != expected:
            if sorted(names) == sorted(expected):
                raise AssertionError(f"前缀为 {prefixes!r} 时，返回的名字一个不少，但顺序不对：得到 {names}。顺序应和 model.named_parameters() 一致，和前缀怎么写无关。")
            extra = [n for n in names if n not in expected]
            missing = [n for n in expected if n not in names]
            detail = "，".join(text for text in (f"多了 {extra}" if extra else "", f"少了 {missing}" if missing else "") if text) or f"得到 {names}"
            raise AssertionError(f"前缀为 {prefixes!r} 时，返回的名单不对：{detail}。{hint}")
        assert flags == expected, (
            f"前缀为 {prefixes!r} 时，返回的名单是对的，但参数的 requires_grad 和它不一致：现在可训练的是 {flags}。{hint}")

    def freeze_basic_case():
        hint = "检查匹配规则，以及是不是每个参数的 requires_grad 都设置过。"
        configure(Toy(), ("backbone.2", "head"), wb("backbone.2", "head"), hint)
        configure(Toy(), ("head",), wb("head"), hint)
        configure(Toy(), ("backbone",), wb("backbone.0", "backbone.2"), hint)
        configure(Toy(), (), [], "空元组表示全部冻结，应返回空列表。")

    def freeze_repeat_case():
        # 模拟逐块解冻：同一个模型上先后调用几次，每次调用前模型都带着上一次留下的状态。
        hint = ("这个模型在调用之前已经有一部分参数被冻结或解冻过。函数不能假设所有参数一开始都是可训练的："
                "匹配的参数要明确设成可训练，不匹配的要明确设成冻结。")
        model = Toy()
        for p in model.parameters():
            p.requires_grad = False
        configure(model, ("head",), wb("head"), hint)
        configure(model, ("backbone.2", "head"), wb("backbone.2", "head"), hint)
        configure(model, ("backbone.0",), wb("backbone.0"), hint)
        configure(model, (), [], hint)
        configure(model, ("backbone", "head"), wb("backbone.0", "backbone.2", "head"), hint)

    def freeze_prefix_case():
        prefix_hint = "匹配规则是“参数名以前缀开头”，名字里包含这个前缀不算。"
        configure(Heads(), ("head",), wb("head", "head2"),
                  "前缀是普通的字符串前缀：\"head\" 不应匹配 prehead（只是包含），但会匹配 head2（见文档字符串）。")
        configure(Heads(), ("head.",), wb("head"), prefix_hint)
        configure(Toy(), ("2",), [], prefix_hint)
        configure(Toy(), ("weight",), [], prefix_hint)
        configure(Toy(), ("bias", "ead"), [], prefix_hint)
        seq = nn.Sequential(*[nn.Linear(2, 2) for _ in range(12)])
        configure(seq, ("1",), wb("1", "10", "11"), "12 层的 nn.Sequential 里，前缀 \"1\" 按字符串前缀会同时匹配第 1、10、11 层。")
        configure(seq, ("1.",), wb("1"), prefix_hint)
        # 顺序和去重：前缀倒着写、互相重叠，返回的顺序仍然按 named_parameters，每个名字只出现一次
        order_hint = "检查是不是按前缀的顺序去收集名字了。"
        configure(Toy(), ("head", "backbone.0"), wb("backbone.0", "head"), order_hint)
        configure(Toy(), ("backbone.2", "backbone", "head", "head.weight"), wb("backbone.0", "backbone.2", "head"), "前缀互相重叠时，每个参数名只应出现一次。")
        # 只管参数：BatchNorm 的 running_mean、running_var、num_batches_tracked 是缓冲区，不在返回值里
        configure(WithBN(), ("bn",), wb("bn"), "返回值只含 named_parameters() 里的名字，不含 BatchNorm 的 running_mean 这类缓冲区。")

    def normalize(cam):
        try:
            out = m.normalize_heatmap(cam)
        except NotImplementedError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(f"输入 shape 为 {tuple(cam.shape)} 时抛出 {type(exc).__name__}: {exc}") from None
        assert isinstance(out, torch.Tensor), "应返回一个张量"
        assert out.shape == cam.shape, f"输出的 shape 应和输入相同 {tuple(cam.shape)}，得到 {tuple(out.shape)}"
        assert out.is_floating_point() and out.device == cam.device, "输出应是和输入同一设备上的浮点张量"
        assert torch.isfinite(out).all(), "输出里有 nan 或 inf。检查最大值等于最小值时的除法。"
        assert out.min() >= 0 and out.max() <= 1 + 1e-6, f"输出应在 0 到 1 之间，得到的范围是 {float(out.min()):.4g} 到 {float(out.max()):.4g}"
        return out

    def heatmap_hand_case():
        # 第一张带负值：负值先变成 0，再按 0 到 3 拉伸。第二张是常量。
        cam = torch.tensor([[[[-2.0, 1.0], [3.0, 0.0]]], [[[5.0, 5.0], [5.0, 5.0]]]])
        out = normalize(cam)
        want0 = torch.tensor([[0.0, 1.0 / 3.0], [1.0, 0.0]])
        assert torch.allclose(out[0, 0], want0, atol=1e-5), (
            f"热图 [[-2, 1], [3, 0]] 归一化后应为 [[0, 1/3], [1, 0]]，得到 {[[round(v, 4) for v in row] for row in out[0, 0].tolist()]}。"
            "检查负值是在归一化之前就截成 0 的，还是留到了最小值里；以及最小最大值是不是在这一张图的全部像素上取的。")
        assert torch.allclose(out[1], torch.zeros_like(out[1])), "所有值都相同的热图应返回全 0。"
        # 全是正数、最小值不是 0：最小值要映到 0
        cam = torch.tensor([[[[2.0, 4.0, 6.0], [10.0, 2.0, 8.0]]]])
        out = normalize(cam)
        want = torch.tensor([[[[0.0, 0.25, 0.5], [1.0, 0.0, 0.75]]]])
        assert torch.allclose(out, want, atol=1e-5), (
            f"热图 [[2, 4, 6], [10, 2, 8]] 归一化后应为 [[0, 0.25, 0.5], [1, 0, 0.75]]，得到 {[[round(v, 4) for v in row] for row in out[0, 0].tolist()]}。"
            "检查有没有减掉最小值，是不是只除以了最大值。")

    def heatmap_constant_case():
        zero_hint = "按文档字符串，“常量”是按 ReLU 之后算的，这类热图应返回全 0，不能出现 nan，也不能返回全 1。"
        for label, cam in [
            ("全是 5 的热图", torch.full((1, 1, 3, 4), 5.0)),
            ("全是 0 的热图", torch.zeros(2, 1, 3, 3)),
            ("全是 -2 的热图", torch.full((1, 1, 2, 2), -2.0)),
            ("各处不同、但全为负的热图", torch.tensor([[[[-3.0, -1.0], [-2.0, -0.5]]]])),
        ]:
            out = normalize(cam)
            assert torch.count_nonzero(out) == 0, f"{label}应返回全 0，得到的最大值是 {float(out.max()):.4g}。{zero_hint}"
        # 很弱的热图不是常量：eps 只用来防止除以 0，不应把它压扁
        weak = 0.01 * torch.tensor([[[[0.0, 0.2, 0.5], [1.0, 0.7, 0.1]]]])
        out = normalize(weak)
        assert float(out.max()) > 0.999 and float(out.min()) == 0.0, (
            f"最大值只有 0.01 的弱热图，归一化后最大值仍应是 1，得到 {float(out.max()):.4g}。检查防止除以 0 的小常数是不是取得太大，或者用错了地方。")

    def heatmap_batch_case():
        torch.manual_seed(10)
        cam = torch.rand(3, 1, 4, 6)
        cam[0] *= 0.01
        cam[1] = cam[1] * 300.0 + 20.0
        cam[2] = cam[2] * 4.0 - 2.0
        keep = cam.clone()
        out = normalize(cam)
        assert torch.equal(cam, keep), "函数原地修改了输入 cam。检查有没有用 relu_、-=、/= 这类原地运算。"
        for i in range(3):
            hi, lo = float(out[i].max()), float(out[i].min())
            assert abs(hi - 1.0) < 1e-4 and lo == 0.0, (
                f"三张强度相差很大的热图一起归一化，第 {i} 张的范围应是 0 到 1，得到 {lo:.4g} 到 {hi:.4g}。"
                "检查最小值和最大值是不是按样本分别取的（不能在整个批上取），以及是不是把这一张的最小值映到 0、最大值映到 1。")
            alone = normalize(keep[i:i + 1].clone())
            assert torch.allclose(alone[0], out[i], atol=1e-6), "同一张热图单独归一化和放在一批里归一化，结果应该相同。"
        # 没有负值的热图整体乘一个正数、整体加一个常数，归一化的结果不变
        base = normalize(keep[1:2].clone())
        scaled = normalize(keep[1:2] * 1000.0)
        shifted = normalize(keep[1:2] + 7.0)
        assert torch.allclose(scaled, base, atol=1e-4), "热图整体乘以 1000，单张归一化的结果应该不变。"
        assert torch.allclose(shifted, base, atol=1e-4), "没有负值的热图整体加上一个常数，单张归一化的结果应该不变。检查有没有减掉最小值。"
        # 带负值的那张：原来不大于 0 的位置，输出都应是 0
        assert torch.count_nonzero(out[2][keep[2] <= 0]) == 0, "输入不大于 0 的位置，输出应为 0。检查负值是不是在归一化之前截掉的。"

    results = [
        check("参数冻结策略：基本用例", freeze_basic_case),
        check("参数冻结策略：在已经部分冻结的模型上重复调用", freeze_repeat_case),
        check("参数冻结策略：前缀匹配、返回顺序、只含参数", freeze_prefix_case),
        check("Grad-CAM 热图归一化：手算例子", heatmap_hand_case),
        check("Grad-CAM 热图归一化：常量、全负和很弱的热图", heatmap_constant_case),
        check("Grad-CAM 热图归一化：逐样本独立、不改输入", heatmap_batch_case),
    ]
    print(f"通过 {sum(results)}/{len(results)}；TODO 是正常学习状态。")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
