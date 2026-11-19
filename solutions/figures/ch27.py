"""第 27 章配图与实测数字。

用法（在 solutions 分支的工作目录运行）：
    COURSE_MAIN_DIR=../cv-learning-course python solutions/figures/ch27.py

输出：<主分支>/27_可提示分割SAM/figures/*.svg 和 实测数字.json。
讲义里引用的数字全部来自这个脚本，改讲义数字之前先重跑。

数据和模型都在课程目录的 `27_可提示分割SAM/迷你SAM.py` 里（那不是作业，学员可以读）；
点提示的位置编码、按预测 IoU 选掩码、稳定性分数、外接框、掩码 NMS 调用本章作业的参考实现。
全部在 CPU 上跑，固定种子，不下载任何数据和权重。六个模型（两个种子的主模型和四个对照）各在一个子进程里训练
（每个 2 线程），所以总耗时比训练一个模型多不了多少。耗时的测量是钉住的，设置 CH27_RETIME=1 才重测。

演示页 demos/点提示的歧义.html 里装着主模型（种子 0）的全部权重。平时运行本脚本只核对页面里的权重和这次训练的
结果是否一致，不一致会打印提醒；设置 CH27_WRITE_DEMO=1 才重新写入（同时把五步引导的参考结果写到临时目录，
供 node 核对页面里的 JavaScript 实现）。设置 CH27_CACHE=<目录> 可以把训练好的权重存下来，调试画图时不用每次重训。
"""

from __future__ import annotations

import base64
import importlib.util
import json
import multiprocessing as mp
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _style as S  # noqa: E402
from _style import plt  # noqa: E402

CH = "27_可提示分割SAM"
mod = S.load_solution(CH, "27-1_提示与掩码后处理.py")


def _load_mini():
    spec = importlib.util.spec_from_file_location("mini_sam", S.MAIN / CH / "迷你SAM.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ms = _load_mini()
NUM: dict = {}
N = ms.N
STEPS, N_TRAIN, N_TEST = 1500, 1500, 300
CACHE = os.environ.get("CH27_CACHE")       # 调试用：把训练好的权重存在这个目录里，下次直接读
MAIN0, MAIN1, CTRL_ALL, CTRL_IOU, LOWC = "主模型（种子 0）", "主模型（种子 1）", "对照：三个多输出都监督", "对照：IoU 分支只看输出向量", "训练时见过低对比度"
CTRL_PE = "对照：位置编码没加 0.5"


def pe_no_half(points, image_size, freqs):
    """故意写错的位置编码：像素下标没有加 0.5。只用来看“这种错误能不能从训练结果里看出来”。"""
    h, w = image_size
    unit = torch.as_tensor(points, dtype=freqs.dtype) / torch.tensor([w, h], dtype=freqs.dtype)
    proj = (2.0 * unit - 1.0) @ freqs * (2.0 * np.pi)
    return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


PE_FN = {"正确": mod.point_position_encoding, "没加 0.5": pe_no_half}
JOBS = {
    MAIN0: dict(seed=0),
    MAIN1: dict(seed=1),
    CTRL_ALL: dict(seed=0, multimask=False),
    CTRL_IOU: dict(seed=0, iou_sees_mask=False),
    LOWC: dict(seed=0, data=dict(contrast=(0.03, 0.4))),
    CTRL_PE: dict(seed=0, pe="没加 0.5"),
}
DATA_NOTE = ("64x64 的灰度图，背景灰度 0.3 到 0.7；每张图 2 到 3 个物体（圆、旋转的矩形或椭圆，半径 10 到 15 px），75% 的物体里套一个部件（半径 5 px 以上），"
             "65% 的部件里再套一个小部件（半径 2.5 px 以上）；相邻两级灰度差 0.15 到 0.4；叠加标准差 0.04 的平滑纹理和标准差 0.03 的白噪声；后画的物体会挡住先画的")
MODEL_NOTE = (f"小模型（迷你SAM.py）：卷积编码器加两层双向注意力的解码器，1 个单输出和 3 个多输出；AdamW，学习率 0.003，每步 8 张图各 4 个提示，训练 {STEPS} 步，CPU 2 线程；"
              f"训练时提示点落在被 1、2、3 个掩码包含的像素上的比例是 0.15、0.25、0.6，四分之一的提示是框；训练集 {N_TRAIN} 张，测试集 {N_TEST} 张（另一个种子生成）")


def r3(v):
    return round(float(v), 3)


def sig2(v):
    return float(f"{float(v):.2g}")


# ---------------------------------------------------------------- 训练（子进程）
def _train(job):
    name, kw = job
    kw = dict(kw)
    torch.set_num_threads(2)
    data = ms.SceneSet(0, N_TRAIN, **kw.pop("data", {}))
    t = time.perf_counter()
    model = ms.train_mini_sam(PE_FN[kw.pop("pe", "正确")], mod.mask_to_box, data, steps=STEPS, **kw)
    return name, {k: v.numpy() for k, v in model.state_dict().items()}, time.perf_counter() - t


TRAIN_SEC: list = []


def train_all():
    models, todo = {}, []
    for name, kw in JOBS.items():
        path = Path(CACHE) / f"{name}.pt" if CACHE else None
        if path and path.exists():
            models[name] = torch.load(path)
        else:
            todo.append((name, kw))
    if todo:
        with ProcessPoolExecutor(max_workers=len(todo), mp_context=mp.get_context("spawn")) as pool:
            for name, state, sec in pool.map(_train, todo):
                models[name] = {k: torch.from_numpy(v) for k, v in state.items()}
                TRAIN_SEC.append(sec)
                print(f"训练完成：{name}（{sec:.0f} 秒）", flush=True)
                if CACHE:
                    Path(CACHE).mkdir(parents=True, exist_ok=True)
                    torch.save(models[name], Path(CACHE) / f"{name}.pt")
    out = {}
    for name, kw in JOBS.items():
        m = ms.MiniSAM(PE_FN[kw.get("pe", "正确")], seed=kw.get("seed", 0), iou_sees_mask=kw.get("iou_sees_mask", True))
        m.load_state_dict(models[name])
        out[name] = m.eval()
    return out


# ---------------------------------------------------------------- 评估用的公共函数
@torch.no_grad()
def decode(model, feats, idx, pts, lab, chunk=512):
    """分块解码。idx 是每个提示属于第几张图。返回 logits (B, 4, H, W) 和预测 IoU (B, 4)。"""
    lg, ip = [], []
    for i in range(0, len(idx), chunk):
        a, b = model.decode(ms.repeat_feats(feats, idx[i:i + chunk]), pts[i:i + chunk], lab[i:i + chunk])
        lg.append(a)
        ip.append(b)
    return torch.cat(lg), torch.cat(ip)


def pick_by_pred(logits, iou_pred):
    """在三个多输出里按预测 IoU 选一个（调用作业的 select_by_predicted_iou）。返回布尔掩码 (B, H, W) 和下标 (B,)，下标从 0 到 2。"""
    chosen, index, _ = mod.select_by_predicted_iou((logits[:, 1:] > 0).numpy(), iou_pred[:, 1:].numpy())
    return torch.from_numpy(chosen), torch.from_numpy(index)


def first_clicks(data, n_prompt, seed):
    """每张图抽 n_prompt 个点，每个点从包含它的掩码里等概率抽一个当目标（抽法见 迷你SAM.py 的 sample_targets；评估时三类像素等概率）。"""
    gen = torch.Generator().manual_seed(seed)
    idx = torch.arange(len(data)).repeat_interleave(n_prompt)
    p0, target, pick, k = ms.sample_targets(data, torch.arange(len(data)), n_prompt, gen, depth_weights=(1.0, 1.0, 1.0))
    inside = data.masks[idx, :, p0[:, 1].long(), p0[:, 0].long()]
    return dict(idx=idx, p0=p0, target=target, pick=pick, K=k, inside=inside, level=data.level[idx, pick])


@torch.no_grad()
def run_first_click(model, feats, data, cs, chunk=256):
    """点一下之后的全部统计量。valid[b, m, j] 是第 m 个输出和第 j 个“合理答案”（包含该点的掩码）的 IoU，不是合理答案的位置填 -1。"""
    n = len(cs["idx"])
    pts, lab = ms.point_prompt(cs["p0"][:, None], torch.full((n, 1), ms.POS))
    iou_t, ipred, valid, stab, zone_sum, zone_cnt, mid_sum, in_cnt = [], [], [], [], [], [], [], []
    for i in range(0, n, chunk):
        sl = slice(i, i + chunk)
        logits, ip = model.decode(ms.repeat_feats(feats, cs["idx"][sl]), pts[sl], lab[sl])
        pred = logits > 0
        iou_t.append(ms.mask_iou(pred, cs["target"][sl][:, None]))
        ipred.append(ip)
        gt = data.masks[cs["idx"][sl]]                                               # (b, m_max, H, W)
        v = ms.mask_iou(pred[:, :, None], gt[:, None])
        valid.append(v.masked_fill(~cs["inside"][sl][:, None], -1.0))
        stab.append(torch.from_numpy(mod.stability_score(logits.numpy())))
        # 按“被几个合理答案包含”把像素分区，统计各输出在每个区里的平均概率
        depth = (gt & cs["inside"][sl][:, :, None, None]).sum(1)                     # (b, H, W)，0 到 K
        prob = logits.sigmoid()
        onehot = torch.stack([depth == c for c in range(4)], 1).float()             # (b, 4 个区, H, W)
        zone_sum.append(torch.einsum("bmhw,bzhw->bmz", prob, onehot))
        zone_cnt.append(onehot.flatten(2).sum(-1))
        fg = (depth > 0).float()
        mid = ((prob > 0.2) & (prob < 0.8)).float()
        mid_sum.append((mid * fg[:, None]).flatten(2).sum(-1))
        in_cnt.append(fg.flatten(1).sum(-1))
    return dict(iou=torch.cat(iou_t), pred=torch.cat(ipred), valid=torch.cat(valid), stab=torch.cat(stab), zone_sum=torch.cat(zone_sum),
                zone_cnt=torch.cat(zone_cnt), mid_sum=torch.cat(mid_sum), in_cnt=torch.cat(in_cnt))


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def inner_distance(mask):
    """每个前景像素到掩码外面最近像素的距离（图像边界以外也算外面）。"""
    return cv2.distanceTransform(np.pad(mask, 1).astype(np.uint8), cv2.DIST_L2, 5)[1:-1, 1:-1]


def next_click(pred, target):
    """模拟用户补点：在错得最大的那一块区域里，点离这块区域边缘最远的像素。返回 (x, y, 标签)，没有错误时返回 None。"""
    err = pred ^ target
    if not err.any():
        return None
    n, lab = cv2.connectedComponents(err.astype(np.uint8), connectivity=8)
    region = lab == 1 + int(np.bincount(lab.ravel())[1:].argmax())
    y, x = np.unravel_index(int(inner_distance(region).argmax()), region.shape)
    return int(x), int(y), int(target[y, x])


# ---------------------------------------------------------------- 图 1：提示的歧义，单输出和多输出
def fig_ambiguity(models, feats, test):
    cs = first_clicks(test, 8, seed=100)
    res = {name: run_first_click(models[name], feats[name], test, cs) for name in (MAIN0, MAIN1, CTRL_ALL, CTRL_IOU, CTRL_PE)}
    STATE["clicks"], STATE["first"] = cs, res
    k_of = cs["K"]

    def table(r):
        out = {}
        for k in (1, 2, 3):
            s = k_of == k
            iou, valid = r["iou"][s], r["valid"][s]
            ok = cs["inside"][s]
            cover = valid[:, 1:].max(1).values                                        # 每个合理答案对上的最好候选
            out[f"点被 {k} 个掩码包含"] = {
                "提示个数": int(s.sum()),
                "单输出和目标的 IoU": r3(iou[:, 0].mean()),
                "三个多输出里最好的和目标的 IoU": r3(iou[:, 1:].max(1).values.mean()),
                "单输出和离它最近的合理答案的 IoU": r3(valid[:, 0].max(1).values.mean()),
                "每个合理答案被最好的候选对上的 IoU（对所有合理答案取平均）": r3(cover[ok].mean()),
                "合理答案里有候选对到 IoU 0.7 以上的比例": r3((cover[ok] >= 0.7).float().mean()),
                "单输出对到 IoU 0.7 以上的合理答案的比例": r3((valid[:, 0][ok] >= 0.7).float().mean()),
            }
        return out

    NUM["提示的歧义"] = {
        "说明": DATA_NOTE + "。" + MODEL_NOTE + f"。测试时每张图抽 8 个正点（先在被 1、2、3 个掩码包含的三类像素里等概率选一类，再在这一类里均匀抽点），共 {len(k_of)} 个提示；"
              "每个点的目标从包含它的掩码里等概率抽一个，所以被 k 个掩码包含的点有 k 个合理答案，任何只给一个答案的做法对目标的 IoU 都有上限。"
              "对照模型的三个多输出各自都被监督（不取最小），其余完全相同",
        **{name: table(res[name]) for name in (MAIN0, MAIN1, CTRL_ALL)},
    }

    wrong_pe = table(res[CTRL_PE])
    NUM["位置编码写错的后果"] = {
        "说明": f"把点提示的位置编码里“像素下标加 0.5”这一步去掉，用同样的数据、种子和步数重新训练，和{MAIN0}、{MAIN1}比。模型里图像特征每个格子的位置也是用同一个函数编码的，"
              "所以点和格子一起偏了半个像素，训练出来的模型自己是自洽的。每项三个数对应点被 1、2、3 个掩码包含",
        **{name: {"单输出和目标的 IoU": [t_[f"点被 {k} 个掩码包含"]["单输出和目标的 IoU"] for k in (1, 2, 3)],
                  "三个多输出里最好的和目标的 IoU": [t_[f"点被 {k} 个掩码包含"]["三个多输出里最好的和目标的 IoU"] for k in (1, 2, 3)]}
           for name, t_ in ((CTRL_PE, wrong_pe), (MAIN0, NUM["提示的歧义"][MAIN0]), (MAIN1, NUM["提示的歧义"][MAIN1]))},
    }

    # 单输出是不是“几个答案的平均”：按被几个合理答案包含分区，看平均概率
    def zones(r, m):
        out = {}
        for k in (2, 3):
            s = k_of == k
            mean = r["zone_sum"][s][:, m].sum(0) / r["zone_cnt"][s].sum(0)
            out[f"点被 {k} 个掩码包含"] = {f"被 {c} 个合理答案包含的像素": r3(mean[c]) for c in range(k, -1, -1)}
        return out

    amb = k_of >= 2
    r0 = res[MAIN0]
    NUM["单输出是几个答案的平均"] = {
        "说明": "同上一组提示里有歧义的那些（点被 2 或 3 个掩码包含）。把像素按“被几个合理答案包含”分区，统计输出概率（sigmoid 之后）在每个区里的平均值。"
              "如果输出正好是几个合理答案的逐像素平均，被 c 个答案包含的区里概率应是 c/k。“半信半疑的像素”指概率在 0.2 到 0.8 之间、且至少属于一个合理答案的像素所占比例",
        "理论值（逐像素平均）": {"k=2": [1.0, 0.5, 0.0], "k=3": [1.0, r3(2 / 3), r3(1 / 3), 0.0]},
        f"{MAIN0}的单输出": zones(r0, 0),
        f"{MAIN0}的三个多输出（三个取平均）": {k: {c: r3(np.mean([zones(r0, m)[k][c] for m in (1, 2, 3)])) for c in v} for k, v in zones(r0, 1).items()},
        "半信半疑的像素所占比例": {
            "单输出": r3(r0["mid_sum"][amb][:, 0].sum() / r0["in_cnt"][amb].sum()),
            "三个多输出（平均）": r3(r0["mid_sum"][amb][:, 1:].sum() / (3 * r0["in_cnt"][amb].sum())),
            "对照模型的三个多输出（平均）": r3(res[CTRL_ALL]["mid_sum"][amb][:, 1:].sum() / (3 * res[CTRL_ALL]["in_cnt"][amb].sum())),
        },
    }

    # 三个多输出各自学成了什么
    pts, lab = ms.point_prompt(cs["p0"][:, None], torch.full((len(k_of), 1), ms.POS))
    division = {}
    for name in (MAIN0, MAIN1):
        area = torch.cat([(models[name].decode(ms.repeat_feats(feats[name], cs["idx"][i:i + 512]), pts[i:i + 512], lab[i:i + 512])[0] > 0).flatten(2).sum(-1)
                          for i in range(0, len(k_of), 512)]).float()
        win = res[name]["iou"][:, 1:].argmax(1)
        heads = {"单输出的平均面积（像素）": round(float(area[:, 0].mean()))}
        for m in range(3):
            w = win == m
            heads[f"第 {m + 1} 个多输出"] = {
                "输出掩码的平均面积（像素）": round(float(area[:, 1 + m].mean())),
                "是三个里最好的那个的比例": r3(w.float().mean()),
                "它最好时目标是整体、部件、小部件的次数": [int((cs["level"][w] == lv).sum()) for lv in range(3)],
            }
        division[name] = heads
    NUM["三个多输出的分工"] = {
        "说明": f"同一组 {len(k_of)} 个单点提示。面积是 logit 大于 0 的像素数；“最好”指和目标的 IoU 在三个多输出里最高。两个种子只差随机初始化和抽样顺序",
        **division,
        "目标是整体、部件、小部件的提示各有多少": [int((cs["level"] == lv).sum()) for lv in range(3)],
        "目标的平均面积（整体、部件、小部件）": [round(float(cs["target"][cs["level"] == lv].flatten(1).sum(1).float().mean())) for lv in range(3)],
    }
    model = models[MAIN0]

    # 例图：点在小部件里（3 个合理答案），三个多输出各对上一个答案的例子里取第一个
    k3 = torch.nonzero((k_of == 3) & (r0["valid"][:, 1:].max(1).values.masked_fill(~cs["inside"], 2).min(1).values > 0.75))[:, 0]
    ex = int(k3[0]) if len(k3) else int(torch.nonzero(k_of == 3)[0, 0])
    logits, ip = model.decode(ms.repeat_feats(feats[MAIN0], cs["idx"][ex:ex + 1]), pts[ex:ex + 1], lab[ex:ex + 1])
    img = test.images[cs["idx"][ex], 0].numpy()
    gts = test.masks[cs["idx"][ex]][cs["inside"][ex]].numpy()
    px, py = cs["p0"][ex].tolist()
    NUM["提示的歧义"]["例图"] = {"说明": "图 1 左边用的例子：点落在小部件里，有 3 个合理答案",
                           "单输出和三个合理答案的 IoU（从大到小的答案）": [r3(ms.mask_iou(logits[0, 0] > 0, torch.from_numpy(g))) for g in sorted(gts, key=lambda g: -g.sum())],
                           "三个多输出各自最像的合理答案的 IoU": [r3(max(ms.mask_iou(logits[0, m] > 0, torch.from_numpy(g)) for g in gts)) for m in (1, 2, 3)],
                           "三个多输出的预测 IoU": [r3(v) for v in ip[0, 1:]]}

    fig, ax = plt.subplots(1, 7, figsize=(17.5, 3.3), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1, 1, 2.1]}, layout="constrained")
    ax[0].imshow(img, cmap="gray", vmin=0, vmax=1)
    ax[0].plot(px, py, marker="*", color=S.RED, ms=13, mec="white", mew=0.8)
    ax[0].set_title("点一下（红星）")
    ax[1].imshow(img, cmap="gray", vmin=0, vmax=1, alpha=0.55)
    for g, c in zip(sorted(gts, key=lambda g: -g.sum()), (S.BLUE, S.ORANGE, S.GREEN)):
        ax[1].contour(g, levels=[0.5], colors=[c], linewidths=1.8)
    ax[1].set_title("3 个合理答案")
    ax[2].imshow(logits[0, 0].sigmoid(), cmap="viridis", vmin=0, vmax=1)
    ax[2].set_title("单输出的概率图")
    order = np.argsort([-(logits[0, m] > 0).sum().item() for m in (1, 2, 3)])
    for a, m in zip(ax[3:6], order):
        a.imshow(logits[0, 1 + m].sigmoid(), cmap="viridis", vmin=0, vmax=1)
        a.set_title(f"多输出 {m + 1}：预测 {ip[0, 1 + m]:.2f}")
    for a in ax[:6]:
        S.image_axes(a)
    x = np.arange(3)
    series = [("单输出", "单输出和目标的 IoU", MAIN0, S.MUTED), ("三个多输出里最好的", "三个多输出里最好的和目标的 IoU", MAIN0, S.BLUE),
              ("对照：三个都监督，取最好的", "三个多输出里最好的和目标的 IoU", CTRL_ALL, S.ORANGE)]
    for i, (label, key, name, color) in enumerate(series):
        vals = [NUM["提示的歧义"][name][f"点被 {k} 个掩码包含"][key] for k in (1, 2, 3)]
        bars = ax[6].bar(x + (i - 1) * 0.27, vals, 0.25, color=color, alpha=0.9, label=label)
        for b, v in zip(bars, vals):
            ax[6].text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}", ha="center", fontsize=8.5)
    ax[6].set_xticks(x, ["1 个合理答案", "2 个合理答案", "3 个合理答案"])
    ax[6].set_ylim(0, 1.02)
    ax[6].set_ylabel("和目标的 IoU")
    ax[6].set_title(TITLES["歧义"])
    ax[6].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, columnspacing=1.0, handlelength=1.2)
    S.clean(ax[6])
    ax[6].grid(axis="x", visible=False)
    S.save(fig, CH, "fig01_提示的歧义")


# ---------------------------------------------------------------- 图 2：结构和耗时
def measure_time():
    torch.manual_seed(0)
    x = torch.rand(1, 1, N, N)
    pts, lab = ms.point_prompt(torch.tensor([[[30.0, 30.0]]]), torch.tensor([[ms.POS]]))

    def med(fn, n=200):
        for _ in range(20):
            fn()
        ts = []
        for _ in range(n):
            t = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t)
        return float(np.median(ts) * 1e3)

    out = {}
    with torch.no_grad():
        dec = ms.MaskDecoder(mod.point_position_encoding).eval()
        for label, width, depth in (("本章的编码器（宽 48，4 层）", 48, 4), ("加宽加深（宽 128，8 层）", 128, 8), ("再加宽加深（宽 256，12 层）", 256, 12)):
            enc = ms.ImageEncoder(width=width, depth=depth).eval()
            f = enc(x)
            out[label] = {"编码一次（毫秒）": sig2(med(lambda: enc(x), 200 if width < 200 else 60)), "参数量": sum(p.numel() for p in enc.parameters())}
        out["解码一个提示（毫秒）"] = sig2(med(lambda: dec(f, pts, lab)))
        p64, l64 = ms.point_prompt(torch.rand(64, 1, 2) * 63, torch.full((64, 1), ms.POS))
        f64 = ms.repeat_feats(f, torch.zeros(64, dtype=torch.long))
        out["一次解码 64 个提示（毫秒）"] = sig2(med(lambda: dec(f64, p64, l64), 40))
    return out


def fig_structure(models):
    model = models[MAIN0]
    t = S.pinned(CH, "耗时", measure_time, "CH27_RETIME")
    enc_keys = [k for k in t if "编码器" in k or "加宽" in k]
    d = t["解码一个提示（毫秒）"]
    clicks = 10
    session = {k: {"只编码一次": sig2(t[k]["编码一次（毫秒）"] + clicks * d), "每次点击都重新编码": sig2(clicks * (t[k]["编码一次（毫秒）"] + d)),
                   "省下的比例": r3(1 - (t[k]["编码一次（毫秒）"] + clicks * d) / (clicks * (t[k]["编码一次（毫秒）"] + d)))} for k in enc_keys}
    NUM["耗时"] = t
    NUM["耗时说明"] = {
        "说明": "在一台 Mac 的 CPU 上测的一次结果（PyTorch 2 线程，批大小 1，输入 64x64，先预热 20 次，再取多次的中位数），只保留两位有效数字；"
              "机器忙的时候会变，只看量级和倍数。后两种编码器只是把本章编码器的卷积加宽加深（随机权重，没有训练），用来看编码器变重之后两部分耗时的比例怎么变；解码器不变",
        f"同一张图上点 {clicks} 下的总耗时（毫秒）": session,
        "一次解码 64 个提示相当于逐个解码的多少分之一": r3(t["一次解码 64 个提示（毫秒）"] / (64 * d)),
    }
    NUM["参数量"] = {
        "说明": "本章训练的小模型（迷你SAM.py 的默认结构）",
        "图像编码器": sum(p.numel() for p in model.encoder.parameters()), "提示编码加掩码解码器": sum(p.numel() for p in model.decoder.parameters()),
        "其中预测 IoU 的分支": sum(p.numel() for p in model.decoder.iou_head.parameters()),
        "特征图": "16x16，每个位置 32 维", "点提示的位置编码用的频率个数": int(model.decoder.freqs.shape[1]),
    }

    fig, ax = plt.subplots(1, 2, figsize=(15.5, 4.2), gridspec_kw={"width_ratios": [1.55, 1]}, layout="constrained")
    a = ax[0]
    a.set_xlim(0, 100)
    a.set_ylim(0, 52)
    a.axis("off")

    def box(x, y, w, h, text, color, fs=9.5):
        a.add_patch(plt.Rectangle((x, y), w, h, fc=color, ec="#8a909a", lw=1.0, alpha=0.9))
        a.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1):
        a.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": "-|>", "color": S.MUTED, "lw": 1.3})

    box(1, 30, 13, 14, "图像\n64×64", "#eef1f6")
    box(20, 28, 22, 18, "图像编码器\n（每张图只跑一次）", "#cfe0ff")
    box(48, 30, 17, 14, "图像特征\n16×16×32\n（存起来反复用）", "#eef1f6", 9)
    box(1, 6, 13, 14, "提示\n点 (x, y)+正负\n或框的两个角", "#eef1f6", 9)
    box(20, 6, 22, 14, "提示编码\n位置编码 + 类型向量", "#ffe2c2", 9)
    box(71, 14, 17, 26, "掩码解码器\n（每个提示跑一次）\n\n2 层双向注意力\n放大 4 倍", "#ffe2c2", 9)
    box(91, 28, 8.5, 16, "4 张\n掩码", "#d9f2e0", 9)
    box(91, 8, 8.5, 14, "4 个\n预测\nIoU", "#d9f2e0", 9)
    arrow(14, 37, 20, 37)
    arrow(42, 37, 48, 37)
    arrow(65, 37, 71, 33)
    arrow(14, 13, 20, 13)
    arrow(42, 13, 71, 22)
    arrow(88, 33, 91, 36)
    arrow(88, 20, 91, 16)
    a.text(50, 49.5, "蓝色只在换图时算一次，橙色每点一下算一次", ha="center", fontsize=10, color=S.MUTED)
    a.set_title("编码一次、解码多次的结构")
    b = ax[1]
    xs = np.arange(len(enc_keys))
    once = [session[k]["只编码一次"] for k in enc_keys]
    every = [session[k]["每次点击都重新编码"] for k in enc_keys]
    b1 = b.bar(xs - 0.2, once, 0.38, color=S.BLUE, alpha=0.9, label="只编码一次")
    b2 = b.bar(xs + 0.2, every, 0.38, color=S.MUTED, alpha=0.9, label="每次点击都重新编码")
    for bars in (b1, b2):
        for r in bars:
            b.text(r.get_x() + r.get_width() / 2, r.get_height() * 1.06, f"{r.get_height():g}", ha="center", fontsize=8.5)
    b.set_yscale("log")
    b.set_ylim(3, max(every) * 2.2)
    b.set_xticks(xs, [f"{k.split('（')[0]}\n编码 {t[k]['编码一次（毫秒）']:g} ms" for k in enc_keys], fontsize=9)
    b.set_ylabel(f"同一张图点 {clicks} 下的总耗时（毫秒，对数轴）")
    b.set_title(TITLES["耗时"])
    b.legend(fontsize=9, loc="upper left")
    S.clean(b)
    b.grid(axis="x", visible=False)
    S.save(fig, CH, "fig02_结构与耗时")


STATE: dict = {}
TITLES = {
    "歧义": "只回传最好的那个，多输出才有用；三个都监督和单输出一样",
    "耗时": "编码器越重，只编码一次省得越多",
}

# ---------------------------------------------------------------- 图 3：预测 IoU 的分支
def fig_pred_iou():
    cs, res = STATE["clicks"], STATE["first"]
    k_of = cs["K"]
    rng = np.random.default_rng(0)

    def stats(r):
        out = {}
        rand_col = torch.from_numpy(rng.integers(0, 3, len(k_of)))
        for k in (1, 2, 3):
            s = k_of == k
            tgt = r["iou"][s][:, 1:]                          # 和抽中的目标的 IoU
            near = r["valid"][s][:, 1:].max(2).values          # 和最近的合理答案的 IoU
            pred, stab = r["pred"][s][:, 1:], r["stab"][s][:, 1:]
            by_pred = pred.argmax(1, keepdim=True)
            by_stab = stab.argmax(1, keepdim=True)
            by_rand = rand_col[s][:, None]
            rand_v, pred_v, best_v = near.mean(), near.gather(1, by_pred).mean(), near.max(1).values.mean()
            out[f"点被 {k} 个掩码包含"] = {
                "预测 IoU 和“与目标的真实 IoU”的相关系数": r3(np.corrcoef(pred.flatten().numpy(), tgt.flatten().numpy())[0, 1]),
                "预测 IoU 和“与最近的合理答案的 IoU”的相关系数": r3(np.corrcoef(pred.flatten().numpy(), near.flatten().numpy())[0, 1]),
                "同上，秩相关": r3(spearman(pred.flatten().numpy(), near.flatten().numpy())),
                "预测 IoU 的平均值": r3(pred.mean()),
                "质量：随机选": r3(rand_v),
                "质量：按稳定性分数选": r3(near.gather(1, by_stab).mean()),
                "质量：按预测 IoU 选": r3(pred_v),
                "质量：三个里最好的": r3(best_v),
                "按预测 IoU 选补上了“随机选到最好的”这段差距的多少": r3((pred_v - rand_v) / (best_v - rand_v)),
                "按预测 IoU 选中的正好是质量最好的那个的比例": r3((by_pred[:, 0] == near.argmax(1)).float().mean()),
                "按稳定性分数选中的正好是质量最好的那个的比例": r3((by_stab[:, 0] == near.argmax(1)).float().mean()),
                "对目标：按预测 IoU 选": r3(tgt.gather(1, by_pred).mean()),
                "对目标：随机选一个（实际抽一次）": r3(tgt.gather(1, by_rand).mean()),
                "对目标：三个里最好的": r3(tgt.max(1).values.mean()),
            }
        return out

    NUM["预测 IoU 的分支"] = {
        "说明": "和“提示的歧义”同一组单点提示。每个提示有三个多输出候选，各有一个预测 IoU。“质量”指候选和“最近的合理答案”的 IoU（包含该点的掩码里和这个候选 IoU 最高的那个），"
              "它衡量这个候选本身是不是一张好掩码，和用户到底想要哪一个无关；“对目标”是和评估时从合理答案里随机抽中的那一个算的 IoU。"
              "稳定性分数用作业的 stability_score（阈值 0，偏移 1）。秩相关是 Spearman 相关系数。三选一时随机选中最好那个的概率是 0.333",
        **{name: stats(res[name]) for name in (MAIN0, MAIN1, CTRL_IOU)},
    }
    # 校准：预测 IoU 分箱后的真实 IoU（只看没有歧义的提示）
    r = res[MAIN0]
    s1 = k_of == 1
    pred1, true1 = r["pred"][s1][:, 1:].flatten().numpy(), r["iou"][s1][:, 1:].flatten().numpy()
    edges = [-1, 0.3, 0.5, 0.7, 0.85, 2]
    calib = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (pred1 >= lo) & (pred1 < hi)
        name = f"预测 IoU {'小于 ' + str(hi) if lo < 0 else ('不小于 ' + str(lo) if hi > 1 else str(lo) + ' 到 ' + str(hi))}"
        calib[name] = {"候选个数": int(m.sum()), "真实 IoU 的平均": r3(true1[m].mean()) if m.any() else None,
                       "真实 IoU 不到 0.5 的比例": r3((true1[m] < 0.5).mean()) if m.any() else None}
    NUM["预测 IoU 的校准"] = {"说明": f"{MAIN0}，只看没有歧义的提示（点只被 1 个掩码包含），每个提示 3 个候选。按预测 IoU 分段，看每段里候选的真实 IoU",
                          **calib, "预测 IoU 的平均绝对误差": r3(np.abs(pred1 - true1).mean())}
    s3 = k_of == 3
    pred3, near3 = r["pred"][s3][:, 1:].flatten().numpy(), r["valid"][s3][:, 1:].max(2).values.flatten().numpy()
    good1, good3 = true1 >= 0.9, near3 >= 0.9
    NUM["歧义压低预测 IoU"] = {
        "说明": f"{MAIN0}。只看本身很好的候选（和最近的合理答案的 IoU 不低于 0.9），比较它们的预测 IoU。训练时“真实 IoU”是和随机抽中的那个目标算的，"
              "有 3 个合理答案时一张完美的掩码平均只有三分之一的机会对上目标，所以预测值被压低",
        "没有歧义的提示：好候选的个数和预测 IoU 的平均": [int(good1.sum()), r3(pred1[good1].mean())],
        "3 个合理答案的提示：好候选的个数和预测 IoU 的平均": [int(good3.sum()), r3(pred3[good3].mean())],
        "没有歧义的提示：好候选里预测 IoU 不到 0.7 的比例": r3((pred1[good1] < 0.7).mean()),
        "3 个合理答案的提示：好候选里预测 IoU 不到 0.7 的比例": r3((pred3[good3] < 0.7).mean()),
    }

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.1), gridspec_kw={"width_ratios": [1, 1, 1.35]}, layout="constrained")
    for a, pred, true, title, ylabel in ((ax[0], pred1, true1, "没有歧义的提示", "和目标的真实 IoU"), (ax[1], pred3, near3, "有 3 个合理答案的提示", "和最近的合理答案的 IoU")):
        a.scatter(pred, true, s=5, color=S.BLUE, alpha=0.25, linewidths=0, rasterized=True)
        a.plot([0, 1], [0, 1], color=S.MUTED, lw=1, ls="--")
        bins = np.linspace(0, 1, 11)
        mid, mean = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (pred >= lo) & (pred < hi)
            if m.sum() >= 15:
                mid.append((lo + hi) / 2)
                mean.append(true[m].mean())
        a.plot(mid, mean, "o-", color=S.ORANGE, lw=1.8, ms=5, label="每段的平均")
        a.set_xlim(-0.05, 1.05)
        a.set_ylim(-0.03, 1.03)
        a.set_xlabel("模型预测的 IoU")
        a.set_ylabel(ylabel)
        a.set_title(title)
        a.legend(fontsize=8.5, loc="upper left")
        S.clean(a)
    x = np.arange(3)
    d = NUM["预测 IoU 的分支"][MAIN0]
    keys = [("随机选", "质量：随机选", S.MUTED), ("按稳定性分数选", "质量：按稳定性分数选", S.PURPLE),
            ("按预测 IoU 选", "质量：按预测 IoU 选", S.BLUE), ("三个里最好的", "质量：三个里最好的", S.GREEN)]
    for i, (label, key, color) in enumerate(keys):
        vals = [d[f"点被 {k} 个掩码包含"][key] for k in (1, 2, 3)]
        bars = ax[2].bar(x + (i - 1.5) * 0.2, vals, 0.19, color=color, alpha=0.9, label=label)
        for b, v in zip(bars, vals):
            ax[2].text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}", ha="center", fontsize=8)
    ax[2].set_xticks(x, ["1 个合理答案", "2 个合理答案", "3 个合理答案"])
    ax[2].set_ylim(0, 1.04)
    ax[2].set_ylabel("选出的掩码的质量（和最近的合理答案的 IoU）")
    ax[2].set_title(TITLES["预测IoU"])
    ax[2].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, columnspacing=1.0, handlelength=1.2)
    S.clean(ax[2])
    ax[2].grid(axis="x", visible=False)
    S.save(fig, CH, "fig03_预测IoU")


# ---------------------------------------------------------------- 图 4：交互，加正点和负点
def simulate(model, feats, cs, sel, n_clicks=8, policy="按预测 IoU 选", strategy="中心", seed=7):
    """模拟交互。policy：第 2 下以后给用户看哪张掩码（“按预测 IoU 选”或“单输出”）；strategy：补的点落在错误区域的“中心”还是“随机”位置。"""
    rng = np.random.default_rng(seed)
    idx, target = cs["idx"][sel], cs["target"][sel]
    n = len(idx)
    pts, lab = ms.point_prompt(cs["p0"][sel][:, None], torch.full((n, 1), ms.POS), capacity=n_clicks)
    tgt = target.numpy()
    ious, second = [], np.full(n, -1)
    snapshots = []
    for c in range(n_clicks):
        logits, ip = decode(model, feats, idx, pts, lab)
        shown = pick_by_pred(logits, ip)[0] if (policy == "按预测 IoU 选" or c == 0) else logits[:, 0] > 0
        ious.append(ms.mask_iou(shown, target).numpy())
        snapshots.append((shown[:40].numpy().copy(), pts[:40].numpy().copy(), lab[:40].numpy().copy()))
        if c == n_clicks - 1:
            break
        shown_np = shown.numpy()
        for i in range(n):
            if strategy == "中心":
                nc = next_click(shown_np[i], tgt[i])
            else:
                ys, xs = np.nonzero(shown_np[i] ^ tgt[i])
                j = int(rng.integers(len(ys))) if len(ys) else -1
                nc = (int(xs[j]), int(ys[j]), int(tgt[i][ys[j], xs[j]])) if j >= 0 else None
            if nc is not None:
                pts[i, c + 1, 0], pts[i, c + 1, 1], lab[i, c + 1] = nc[0], nc[1], nc[2]
                if c == 0:
                    second[i] = nc[2]
    return np.stack(ious), second, snapshots


def fig_interaction(models, feats, test):
    cs = STATE["clicks"]
    sel = torch.arange(len(cs["idx"])) < 1200                  # 前 150 张图的 1200 个提示
    model = models[MAIN0]
    main, second, snaps = simulate(model, feats[MAIN0], cs, sel)
    alt_single, _, _ = simulate(model, feats[MAIN0], cs, sel, policy="单输出")
    alt_random, _, _ = simulate(model, feats[MAIN0], cs, sel, strategy="随机")
    seed1, _, _ = simulate(models[MAIN1], feats[MAIN1], cs, sel)
    wrong = main[0] < 0.5
    k_of = cs["K"][sel].numpy()

    def reach(x, thr):
        return [r3((x[:c + 1].max(0) >= thr).mean()) for c in range(len(x))]

    gain = main[1] - main[0]
    never, lv_sel = main.max(0) < 0.8, cs["level"][sel].numpy()
    NUM["交互"] = {
        "说明": f"{MAIN0}，测试集前 150 张图的 {int(sel.sum())} 个提示。第 1 下是随机落在目标里的正点；之后每一下都点在“当前给用户看的掩码”错得最大的那一块连通区域里离区域边缘最远的像素，"
              "漏掉的地方补正点，多出来的地方补负点；没有错误就不再点。所有点一起重新送进解码器（图像特征不重算）。默认给用户看的是三个多输出里预测 IoU 最高的那张。"
              "“第一下之后选错了”指第 1 下之后选中的掩码和目标的 IoU 不到 0.5",
        "点击次数": list(range(1, 9)),
        "全部提示：和目标的 IoU": [r3(v) for v in main.mean(1)],
        "第一下之后选错了的提示：个数和占比": [int(wrong.sum()), r3(wrong.mean())],
        "第一下之后选错了的提示：和目标的 IoU": [r3(v) for v in main[:, wrong].mean(1)],
        "第一下就选对了的提示：和目标的 IoU": [r3(v) for v in main[:, ~wrong].mean(1)],
        "选错了的提示里，点被 1、2、3 个掩码包含的各占多少": [r3((k_of[wrong] == k).mean()) for k in (1, 2, 3)],
        "每多点一下带来的 IoU 提高（全部提示）": [r3(v) for v in np.diff(main.mean(1))],
        "到第 c 下为止 IoU 曾达到 0.8 的提示比例": reach(main, 0.8),
        "到第 c 下为止 IoU 曾达到 0.9 的提示比例": reach(main, 0.9),
        "第 2 下是负点的比例": r3((second == 0).mean()), "第 2 下是正点的比例": r3((second == 1).mean()), "不需要第 2 下的比例": r3((second == -1).mean()),
        "第 2 下是负点时 IoU 的提高": r3(gain[second == 0].mean()), "第 2 下是正点时 IoU 的提高": r3(gain[second == 1].mean()),
        "第 2 下之后 IoU 下降超过 0.02 的比例（在补了点的提示里）": r3((gain[second >= 0] < -0.02).mean()),
        "对照：第 2 下起给用户看单输出": [r3(v) for v in alt_single.mean(1)],
        "对照：补的点落在错误区域里的随机位置": [r3(v) for v in alt_random.mean(1)],
        "8 下之内 IoU 从没到过 0.8 的提示": {
            "占全部提示的比例": r3(never.mean()),
            "这些提示里目标是整体、部件、小部件的各占多少": [r3((lv_sel[never] == k).mean()) for k in range(3)],
            "整体、部件、小部件各自有多少比例从没到过 0.8": [r3(never[lv_sel == k].mean()) for k in range(3)],
            "全部提示里目标是整体、部件、小部件的各占多少": [r3((lv_sel == k).mean()) for k in range(3)],
        },
        f"{MAIN1}，同样的做法": [r3(v) for v in seed1.mean(1)],
        f"{MAIN1}：第一下之后选错了的提示": [r3(v) for v in seed1[:, seed1[0] < 0.5].mean(1)],
    }

    # 例图：第一下选成了整体、目标是部件，点两三下之后修正的例子
    cand = [i for i in range(40) if main[0, i] < 0.4 and main[2, i] > 0.85 and k_of[i] >= 2]
    ex = cand[0] if cand else int(np.argmin(main[0, :40]))
    img = test.images[cs["idx"][ex], 0].numpy()
    tgt = cs["target"][ex].numpy()
    NUM["交互"]["例图"] = {"说明": "图 4 左边三栏用的例子", "第 1、2、3 下之后和目标的 IoU": [r3(main[c, ex]) for c in range(3)]}
    fig, ax = plt.subplots(1, 5, figsize=(17, 3.6), gridspec_kw={"width_ratios": [1, 1, 1, 1.55, 1.55]}, layout="constrained")
    for c in range(3):
        shown, pts, lab = snaps[c]
        ax[c].imshow(img, cmap="gray", vmin=0, vmax=1)
        ax[c].contour(tgt, levels=[0.5], colors="cyan", linewidths=1.3)
        ax[c].contourf(shown[ex].astype(float), levels=[0.5, 1.5], colors=[S.ORANGE], alpha=0.38)
        for (x, y), l in zip(pts[ex], lab[ex]):
            if l == ms.POS:
                ax[c].plot(x, y, marker="o", mfc=S.GREEN, ms=7.5, mec="white", mew=1.2, ls="none")
            elif l == ms.NEG:
                ax[c].plot(x, y, marker="X", mfc=S.RED, ms=9, mec="white", mew=1.0, ls="none")
        ax[c].set_title(f"点 {c + 1} 下：IoU {main[c, ex]:.2f}")
        S.image_axes(ax[c])
    xs = np.arange(1, 9)
    ax[3].plot(xs, main.mean(1), "o-", color=S.BLUE, lw=1.9, ms=5, label="全部提示")
    ax[3].plot(xs, main[:, wrong].mean(1), "s-", color=S.RED, lw=1.9, ms=5, label=f"第一下之后选错了的（占 {wrong.mean():.0%}）")
    ax[3].plot(xs, main[:, ~wrong].mean(1), "^-", color=S.GREEN, lw=1.6, ms=5, label="第一下就选对了的")
    ax[3].set_xlabel("点击次数")
    ax[3].set_ylabel("和目标的 IoU")
    ax[3].set_ylim(0, 1.0)
    ax[3].set_title(TITLES["交互"])
    ax[3].legend(fontsize=8.5, loc="lower right")
    S.clean(ax[3])
    ax[4].plot(xs, [100 * v for v in NUM["交互"]["到第 c 下为止 IoU 曾达到 0.8 的提示比例"]], "o-", color=S.BLUE, lw=1.9, ms=5, label="IoU 达到 0.8")
    ax[4].plot(xs, [100 * v for v in NUM["交互"]["到第 c 下为止 IoU 曾达到 0.9 的提示比例"]], "s-", color=S.PURPLE, lw=1.9, ms=5, label="IoU 达到 0.9")
    ax[4].set_xlabel("点击次数")
    ax[4].set_ylabel("已经达到要求的提示所占比例（%）")
    ax[4].set_ylim(0, 100)
    ax[4].set_title(TITLES["饱和"])
    ax[4].legend(fontsize=8.5, loc="lower right")
    S.clean(ax[4])
    S.save(fig, CH, "fig04_交互")


# ---------------------------------------------------------------- 图 5：点的位置，框和点
def all_targets(data):
    idx, m = torch.nonzero(data.level >= 0, as_tuple=True)
    return idx, data.masks[idx, m], data.level[idx, m]


def score_prompts(model, feats, idx, pts, lab, target):
    logits, ip = decode(model, feats, idx, pts, lab)
    sel, _ = pick_by_pred(logits, ip)
    iou = ms.mask_iou(logits > 0, target[:, None])
    return dict(single=iou[:, 0].numpy(), best=iou[:, 1:].max(1).values.numpy(), sel=ms.mask_iou(sel, target).numpy())


def fig_position(models, feats, test):
    model, f = models[MAIN0], feats[MAIN0]
    idx, target, level = all_targets(test)
    n = len(idx)
    tgt = target.numpy()
    rng = np.random.default_rng(11)
    dist = [inner_distance(t) for t in tgt]
    depth = test.depth[idx].numpy()
    STATE["pos_scores"] = {}
    bands = [("贴着边缘（最外一圈像素）", 0.0, 1.2), ("离边缘 2 px 左右", 1.2, 2.6), ("离边缘 3 到 4 px", 2.6, 4.5), ("离边缘 5 px 以上", 4.5, 99.0)]
    rows = {}
    for name, lo, hi in bands + [("正中心（离边缘最远的像素）", None, None), ("点偏了：落在目标外面 1 px", -1, -1)]:
        pts, keep, deeper = np.zeros((n, 2), np.float32), np.zeros(n, bool), np.zeros(n, bool)
        for i in range(n):
            if lo is None:
                y, x = np.unravel_index(int(dist[i].argmax()), (N, N))
            elif lo == -1:
                ring = cv2.dilate(tgt[i].astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~tgt[i]
                ys, xs = np.nonzero(ring)
                if not len(ys):
                    continue
                j = int(rng.integers(len(ys)))
                y, x = ys[j], xs[j]
            else:
                ys, xs = np.nonzero((dist[i] > lo) & (dist[i] <= hi))
                if not len(ys):
                    continue
                j = int(rng.integers(len(ys)))
                y, x = ys[j], xs[j]
            pts[i], keep[i], deeper[i] = (x, y), True, depth[i][y, x] > level[i] + 1
        sel = torch.from_numpy(keep)
        p, l = ms.point_prompt(torch.from_numpy(pts)[sel][:, None], torch.full((int(sel.sum()), 1), ms.POS))
        sc = score_prompts(model, f, idx[sel], p, l, target[sel])
        full = {k: np.full(n, np.nan) for k in sc}
        for k in sc:
            full[k][keep] = sc[k]
        STATE["pos_scores"][name] = (sc, keep, full, deeper)
    common = np.all([~np.isnan(v[2]["best"]) for v in STATE["pos_scores"].values()], axis=0)
    lv_all = level.numpy()
    for name, (sc, keep, full, deeper) in STATE["pos_scores"].items():
        c = common
        rows[name] = {"这一段里有像素的目标个数": int(keep.sum()), "三个候选里最好的": r3(sc["best"].mean()),
                      "同一批大目标：三个候选里最好的": r3(full["best"][c].mean()), "同一批大目标：按预测 IoU 选": r3(full["sel"][c].mean()), "同一批大目标：单输出": r3(full["single"][c].mean()),
                      "同一批大目标：三个候选里最好的 IoU 不到 0.5 的比例": r3((full["best"][c] < 0.5).mean()),
                      "同一批大目标里的整体：三个候选里最好的": r3(full["best"][c & (lv_all == 0)].mean()),
                      "同一批大目标里的部件：三个候选里最好的": r3(full["best"][c & (lv_all == 1)].mean()),
                      "同一批大目标：点落在目标的下一级（部件或小部件）里面的比例": r3(deeper[c].mean()),
                      "同一批大目标：点没落在下一级里时，按预测 IoU 选": r3(full["sel"][c & ~deeper].mean()) if (c & ~deeper).any() else None,
                      "同一批大目标：点落在下一级里时，按预测 IoU 选": r3(full["sel"][c & deeper].mean()) if (c & deeper).any() else None}
    NUM["点的位置"] = {
        "说明": f"{MAIN0}，测试集全部 {n} 个掩码（整体 {int((level == 0).sum())} 个、部件 {int((level == 1).sum())} 个、小部件 {int((level == 2).sum())} 个）各当一次目标，只点一下。"
              "“离边缘的距离”是点到目标掩码外面最近像素的距离，每个目标在每个距离段里随机取一个像素（目标太小、没有那么靠里的像素时这一段就没有它）。"
              "各段里的目标不是同一批，直接比会把“目标大小”和“点的位置”混在一起，所以另外只取每一段都有像素的那批大目标来比。"
              "和目标的 IoU；“三个候选里最好的”排除了选哪个候选的问题，只看点的位置本身。“点落在下一级里面”指目标是整体而点在它的部件里，或者目标是部件而点在它的小部件里",
        "每一段都有像素的大目标个数（其中整体、部件各多少）": [int(common.sum()), int((common & (lv_all == 0)).sum()), int((common & (lv_all == 1)).sum())],
        **rows,
    }

    # 框和点
    np_box = mod.mask_to_box(tgt)
    box = torch.from_numpy(np_box).float()
    side = (box[:, 2:] - box[:, :2] + 1).repeat(1, 2)
    gen = torch.Generator().manual_seed(12)
    fill = np.zeros(n)
    for i, (x0, y0, x1, y1) in enumerate(np_box):
        fill[i] = tgt[i].sum() / ((x1 - x0 + 1) * (y1 - y0 + 1))
    box_rows = {}
    for name, jit, grow in (("紧贴目标的框", 0.0, 0.0), ("框的四条边各有 5% 边长的随机误差", 0.05, 0.0), ("各有 10% 的随机误差", 0.10, 0.0), ("各有 20% 的随机误差", 0.20, 0.0),
                            ("框每边外扩 10%", 0.0, 0.10), ("框每边外扩 25%", 0.0, 0.25)):
        b = box + torch.randn(n, 4, generator=gen) * jit * side + torch.tensor([-1.0, -1.0, 1.0, 1.0]) * grow * side
        p, l = ms.box_prompt(b.clamp(0, N - 1))
        sc = score_prompts(model, f, idx, p, l, target)
        box_rows[name] = {"单输出": r3(sc["single"].mean()), "三个候选里最好的": r3(sc["best"].mean()), "按预测 IoU 选": r3(sc["sel"].mean()),
                          "单输出 IoU 不到 0.5 的比例": r3((sc["single"] < 0.5).mean()), "单输出 IoU 的标准差": r3(sc["single"].std()),
                          "单输出，按整体、部件、小部件分": [r3(sc["single"][level.numpy() == lv].mean()) for lv in range(3)]}
        if jit == 0 and grow == 0:
            STATE["box_tight"] = sc
    rand_pts, _ = ms.sample_pixel(target, gen)
    p, l = ms.point_prompt(rand_pts[:, None], torch.full((n, 1), ms.POS))
    sc_rand = score_prompts(model, f, idx, p, l, target)
    sc_center = STATE["pos_scores"]["正中心（离边缘最远的像素）"][0]
    lv = level.numpy()
    point_rows = {}
    for name, sc in (("目标里随机一个点", sc_rand), ("目标正中心的点", sc_center)):
        point_rows[name] = {"按预测 IoU 选": r3(sc["sel"].mean()), "三个候选里最好的": r3(sc["best"].mean()), "单输出": r3(sc["single"].mean()),
                            "按预测 IoU 选的 IoU 不到 0.5 的比例": r3((sc["sel"] < 0.5).mean()), "按预测 IoU 选的 IoU 的标准差": r3(sc["sel"].std()),
                            "按预测 IoU 选，按整体、部件、小部件分": [r3(sc["sel"][lv == k].mean()) for k in range(3)],
                            "三个候选里最好的，按整体、部件、小部件分": [r3(sc["best"][lv == k].mean()) for k in range(3)]}
    NUM["框和点"] = {
        "说明": f"{MAIN0}，同样的 {n} 个目标。框提示是目标外接框的两个角（训练时框带 5% 边长的随机误差）。随机误差指四个坐标各加一个标准差为“比例乘对应边长”的高斯数；"
              "外扩指四条边各向外移“比例乘边长”。“把框填满”是把整个框当成掩码的 IoU，是框提示的一条基线：模型什么都没学会、只会把框涂满也能拿到这个分数",
        **box_rows, **point_rows,
        "把框填满的 IoU（基线）": {"全部": r3(fill.mean()), "按整体、部件、小部件分": [r3(fill[lv == k].mean()) for k in range(3)]},
    }

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.1), gridspec_kw={"width_ratios": [1, 1.25]}, layout="constrained")
    names = ["点偏了：落在目标外面 1 px"] + [b[0] for b in bands] + ["正中心（离边缘最远的像素）"]
    xs = np.arange(len(names))
    ax[0].plot(xs, [rows[k]["同一批大目标：三个候选里最好的"] for k in names], "o-", color=S.GREEN, lw=1.9, ms=5, label="三个候选里最好的")
    ax[0].plot(xs, [rows[k]["同一批大目标：按预测 IoU 选"] for k in names], "s-", color=S.BLUE, lw=1.9, ms=5, label="按预测 IoU 选")
    ax[0].plot(xs, [rows[k]["同一批大目标：单输出"] for k in names], "^--", color=S.MUTED, lw=1.5, ms=5, label="单输出")
    ax[0].axvspan(-0.5, 0.5, color=S.RED, alpha=0.06)
    ax[0].set_xlim(-0.5, len(names) - 0.5)
    ax[0].set_xticks(xs, ["点偏了：目标\n外面 1 px", "贴着边缘\n（最外一圈）", "离边缘\n约 2 px", "离边缘\n3 到 4 px", "离边缘\n5 px 以上", "正中心"], fontsize=9)
    ax[0].set_ylim(0, 1.0)
    ax[0].set_ylabel("和目标的 IoU（同一批大目标）")
    ax[0].set_title(TITLES["位置"])
    ax[0].legend(fontsize=8.5, loc="lower right")
    S.clean(ax[0])
    groups = ["整体", "部件", "小部件"]
    xg = np.arange(3)
    series = [("随机一个点，按预测 IoU 选", point_rows["目标里随机一个点"]["按预测 IoU 选，按整体、部件、小部件分"], S.MUTED),
              ("随机一个点，三个里最好的", point_rows["目标里随机一个点"]["三个候选里最好的，按整体、部件、小部件分"], S.GREEN),
              ("紧贴目标的框，单输出", box_rows["紧贴目标的框"]["单输出，按整体、部件、小部件分"], S.BLUE),
              ("框有 10% 的误差，单输出", box_rows["各有 10% 的随机误差"]["单输出，按整体、部件、小部件分"], S.PURPLE)]
    for i, (label, vals, color) in enumerate(series):
        bars = ax[1].bar(xg + (i - 1.5) * 0.2, vals, 0.19, color=color, alpha=0.9, label=label)
        for b, v in zip(bars, vals):
            ax[1].text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}", ha="center", fontsize=8)
    base = NUM["框和点"]["把框填满的 IoU（基线）"]["按整体、部件、小部件分"]
    for xg_i, v in zip(xg, base):
        ax[1].plot([xg_i - 0.42, xg_i + 0.42], [v, v], color=S.RED, lw=1.3, ls="--")
    ax[1].plot([], [], color=S.RED, lw=1.3, ls="--", label="基线：把框填满")
    ax[1].set_xticks(xg, groups)
    ax[1].set_ylim(0, 1.06)
    ax[1].set_ylabel("和目标的 IoU")
    ax[1].set_title(TITLES["框和点"])
    ax[1].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, columnspacing=1.0, handlelength=1.4)
    S.clean(ax[1])
    ax[1].grid(axis="x", visible=False)
    S.save(fig, CH, "fig05_点的位置与框")
    STATE["targets"] = (idx, target, level)


# ---------------------------------------------------------------- 图 6：全图自动分割
def grid_points(g):
    c = (np.arange(g) + 0.5) * N / g - 0.5
    xx, yy = np.meshgrid(c, c)
    return torch.from_numpy(np.stack([xx.ravel(), yy.ravel()], -1).astype(np.float32))


def box_nms(masks, scores, thr):
    """按外接框的 IoU 做 NMS（真 SAM 的自动分割用的是这种），只用来和作业的 mask_nms 对照。"""
    boxes = mod.mask_to_box(masks).astype(np.float64)
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    keep = []
    for i in np.argsort(-scores, kind="stable"):
        ok = True
        for j in keep:
            w = min(boxes[i, 2], boxes[j, 2]) - max(boxes[i, 0], boxes[j, 0]) + 1
            h = min(boxes[i, 3], boxes[j, 3]) - max(boxes[i, 1], boxes[j, 1]) + 1
            inter = max(w, 0) * max(h, 0)
            if inter / (area[i] + area[j] - inter) > thr:
                ok = False
                break
        if ok:
            keep.append(int(i))
    return np.asarray(keep, dtype=np.int64)


@torch.no_grad()
def auto_candidates(model, feats, n_img, g):
    """每张图在 g x g 的网格上撒正点，每个点取三个多输出。返回每张图的 (掩码, 预测 IoU, 稳定性分数)。"""
    pts = grid_points(g)
    p, l = ms.point_prompt(pts[:, None], torch.full((len(pts), 1), ms.POS))
    out = []
    for i in range(n_img):
        logits, ip = decode(model, feats, torch.full((len(pts),), i, dtype=torch.long), p, l)
        lg = logits[:, 1:].reshape(-1, N, N).numpy()
        out.append((lg > 0, ip[:, 1:].reshape(-1).numpy().astype(np.float64), mod.stability_score(lg)))
    return out


def match_stats(masks, gt, lv, thr=0.7):
    """masks (K, H, W)，gt (M, H, W)。返回：和某个真值对上的掩码个数、被对上的真值（布尔 M）、重复的个数（同一个真值被对上多次，多出来的那些）。"""
    if len(masks) == 0:
        return 0, np.zeros(len(gt), bool), 0
    a, b = masks.reshape(len(masks), -1).astype(np.float32), gt.reshape(len(gt), -1).astype(np.float32)
    inter = a @ b.T
    iou = inter / np.maximum(a.sum(1)[:, None] + b.sum(1)[None] - inter, 1)
    best = iou.argmax(1)
    good = iou.max(1) >= thr
    hit = np.zeros(len(gt), bool)
    hit[best[good]] = True
    return int(good.sum()), hit, int(good.sum() - hit.sum())


def fig_auto(models, feats, test):
    model, f = models[MAIN0], feats[MAIN0]
    n_img = 150
    gts = [(test.masks[i, :test.count[i]].numpy(), test.level[i, :test.count[i]].numpy()) for i in range(n_img)]
    T_IOU, T_STAB, T_NMS = 0.6, 0.85, 0.7

    def pipeline(cands, t_iou=T_IOU, t_stab=T_STAB, t_nms=T_NMS, nms="mask", order=("iou", "stab", "nms")):
        """返回每一步之后的统计：平均每张图的掩码数、其中和真值对上的比例、真值被找到的比例（按层级）、重复数。"""
        steps = [[] for _ in range(len(order) + 1)]
        for (masks, pred, stab), (gt, lv) in zip(cands, gts):
            keep = np.flatnonzero(masks.reshape(len(masks), -1).any(1))
            stage = [keep]
            for name in order:
                if name == "iou":
                    keep = keep[pred[keep] >= t_iou]
                elif name == "stab":
                    keep = keep[stab[keep] >= t_stab]
                else:
                    sub = mod.mask_nms(masks[keep], pred[keep], t_nms) if nms == "mask" else box_nms(masks[keep], pred[keep], t_nms)
                    keep = keep[sub]
                stage.append(keep)
            for s, k in zip(steps, stage):
                good, hit, dup = match_stats(masks[k], gt, lv)
                s.append((len(k), good, dup, hit, lv))
        out = []
        for s in steps:
            n = sum(x[0] for x in s)
            hit = np.concatenate([x[3] for x in s])
            lv = np.concatenate([x[4] for x in s])
            out.append({"平均每张图的掩码数": round(n / len(s), 1), "其中和某个真值对上的比例": r3(sum(x[1] for x in s) / max(n, 1)),
                        "平均每张图重复的掩码数": round(sum(x[2] for x in s) / len(s), 1), "真值被找到的比例": r3(hit.mean()),
                        "按整体、部件、小部件分": [r3(hit[lv == k].mean()) for k in range(3)]})
        return out

    cands = {g: auto_candidates(model, f, n_img if g == 8 else 60, g) for g in (4, 8, 16)}
    base = pipeline(cands[8])
    names = ["网格撒点得到的全部候选", f"去掉预测 IoU 低于 {T_IOU} 的", f"再去掉稳定性分数低于 {T_STAB} 的", f"再做掩码 NMS（IoU 阈值 {T_NMS}）"]
    n_gt = np.mean([len(g[0]) for g in gts])
    NUM["全图自动分割"] = {
        "说明": f"{MAIN0}，测试集前 {n_img} 张图（平均每张 {n_gt:.1f} 个真值掩码）。每张图在 8x8 的网格上撒 64 个正点，每个点取三个多输出，共 192 个候选；"
              "依次按预测 IoU、稳定性分数（作业的 stability_score，偏移 1）过滤，再做掩码 NMS（作业的 mask_nms，分数用预测 IoU）。"
              "“对上”指和某个真值掩码的 IoU 不低于 0.7；“重复”指同一个真值被对上多次时多出来的那些；“真值被找到”指至少有一张留下的掩码和它对上",
        **{name: row for name, row in zip(names, base)},
        "每一步去掉的掩码数（平均每张图）": [round(base[i]["平均每张图的掩码数"] - base[i + 1]["平均每张图的掩码数"], 1) for i in range(3)],
        "平均每张图的真值掩码数": round(float(n_gt), 1),
    }
    sweep = {}
    for label, kw in (("不按预测 IoU 过滤", dict(t_iou=-9)), ("预测 IoU 阈值 0.5", dict(t_iou=0.5)), ("预测 IoU 阈值 0.7", dict(t_iou=0.7)), ("预测 IoU 阈值 0.8", dict(t_iou=0.8)),
                      ("不按稳定性过滤", dict(t_stab=-9)), ("稳定性阈值 0.7", dict(t_stab=0.7)), ("稳定性阈值 0.95", dict(t_stab=0.95)),
                      ("NMS 阈值 0.5", dict(t_nms=0.5)), ("NMS 阈值 0.9", dict(t_nms=0.9)), ("只做 NMS，不做前两步过滤", dict(t_iou=-9, t_stab=-9)),
                      ("NMS 换成按外接框的 IoU", dict(nms="box"))):
        sweep[label] = pipeline(cands[8], **kw)[-1]
    NUM["自动分割的阈值"] = {"说明": f"同上，每次只改一个设置，其余保持默认（预测 IoU 阈值 {T_IOU}、稳定性阈值 {T_STAB}、NMS 阈值 {T_NMS}），看最后留下的掩码", "默认设置": base[-1], **sweep}
    grid = {}
    for g in (4, 8, 16):
        row = pipeline(cands[g])
        grid[f"{g}x{g} 的网格（{g * g} 个点）"] = {"候选数": row[0]["平均每张图的掩码数"], "最后留下的": row[-1],
                                                    "不做任何过滤时真值被找到的比例（候选里有没有）": row[0]["真值被找到的比例"],
                                                    "同上，按整体、部件、小部件分": row[0]["按整体、部件、小部件分"]}
    NUM["自动分割的网格密度"] = {"说明": "同样的流程和默认阈值，改变网格密度。8x8 用前 150 张图，4x4 和 16x16 用前 60 张图。网格间距分别是 16、8、4 px", **grid}
    # 被预测 IoU 过滤掉的好掩码：歧义压低预测 IoU 的后果
    lost = {"整体": [0, 0], "部件": [0, 0], "小部件": [0, 0]}
    for (masks, pred, stab), (gt, lv) in zip(cands[8], gts):
        a, b = masks.reshape(len(masks), -1).astype(np.float32), gt.reshape(len(gt), -1).astype(np.float32)
        inter = a @ b.T
        iou = inter / np.maximum(a.sum(1)[:, None] + b.sum(1)[None] - inter, 1)
        good = iou.max(1) >= 0.7
        for k, name in enumerate(lost):
            m = good & (lv[iou.argmax(1)] == k)
            lost[name][0] += int(m.sum())
            lost[name][1] += int((m & (pred < T_IOU)).sum())
    NUM["全图自动分割"]["和真值对上的候选里，被预测 IoU 阈值去掉的比例"] = {k: r3(v[1] / max(v[0], 1)) for k, v in lost.items()}

    ex = 3
    masks, pred, stab = cands[8][ex]
    keep = np.flatnonzero(masks.reshape(len(masks), -1).any(1))
    keep = keep[pred[keep] >= T_IOU]
    keep = keep[stab[keep] >= T_STAB]
    final = keep[mod.mask_nms(masks[keep], pred[keep], T_NMS)]
    img = test.images[ex, 0].numpy()
    fig, ax = plt.subplots(1, 5, figsize=(17.5, 3.7), gridspec_kw={"width_ratios": [1, 1, 1, 1.5, 1.5]}, layout="constrained")
    ax[0].imshow(img, cmap="gray", vmin=0, vmax=1)
    gp = grid_points(8).numpy()
    ax[0].scatter(gp[:, 0], gp[:, 1], s=9, color=S.ORANGE, linewidths=0)
    ax[0].set_title("8×8 的网格撒 64 个点")
    palette = [S.BLUE, S.ORANGE, S.GREEN, S.RED, S.PURPLE, "#00a3a3", "#b8860b", "#c2185b", "#5d4037", "#455a64"]
    ax[1].imshow(img, cmap="gray", vmin=0, vmax=1, alpha=0.6)
    for j, k in enumerate(final):
        ax[1].contour(masks[k], levels=[0.5], colors=[palette[j % len(palette)]], linewidths=1.6)
    ax[1].set_title(f"192 个候选最后留下 {len(final)} 个")
    ax[2].imshow(img, cmap="gray", vmin=0, vmax=1, alpha=0.6)
    for j, g in enumerate(gts[ex][0]):
        ax[2].contour(g, levels=[0.5], colors=[palette[j % len(palette)]], linewidths=1.6)
    ax[2].set_title(f"真值：{len(gts[ex][0])} 个掩码")
    for a in ax[:3]:
        S.image_axes(a)
    xs = np.arange(4)
    counts = [b["平均每张图的掩码数"] for b in base]
    good = [b["平均每张图的掩码数"] * b["其中和某个真值对上的比例"] for b in base]
    ax[3].bar(xs, counts, 0.6, color=S.MUTED, alpha=0.55, label="留下的掩码")
    ax[3].bar(xs, good, 0.6, color=S.BLUE, alpha=0.9, label="其中和某个真值对上的")
    for x, c, b in zip(xs, counts, base):
        ax[3].text(x, c * 1.12, f"{c:g} 个\n对上 {b['其中和某个真值对上的比例']:.0%}", ha="center", fontsize=8.5)
    ax[3].axhline(n_gt, color=S.RED, lw=1.2, ls="--", label=f"真值平均每张图 {n_gt:.1f} 个")
    ax[3].set_yscale("log")
    ax[3].set_ylim(1, max(counts) * 4)
    ax[3].set_xticks(xs, ["全部候选", "预测 IoU\n过滤", "稳定性\n过滤", "掩码 NMS"], fontsize=9)
    ax[3].set_ylabel("平均每张图的掩码数（对数轴）")
    ax[3].set_title(TITLES["自动"])
    ax[3].legend(fontsize=8.5, loc="upper right")
    S.clean(ax[3])
    ax[3].grid(axis="x", visible=False)
    xg = np.arange(3)
    step_names = ["全部候选", "预测 IoU 过滤后", "稳定性过滤后", "NMS 后"]
    for i, (label, color) in enumerate(zip(step_names, (S.MUTED, S.ORANGE, S.PURPLE, S.BLUE))):
        vals = base[i]["按整体、部件、小部件分"]
        bars = ax[4].bar(xg + (i - 1.5) * 0.2, vals, 0.19, color=color, alpha=0.9, label=label)
        for b, v in zip(bars, vals):
            ax[4].text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}" if v >= 0.005 else "0", ha="center", fontsize=7.5)
    ax[4].set_xticks(xg, ["整体", "部件", "小部件"])
    ax[4].set_ylim(0, 1.1)
    ax[4].set_ylabel("真值被找到的比例")
    ax[4].set_title(TITLES["自动召回"])
    ax[4].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, columnspacing=0.9, handlelength=1.2)
    S.clean(ax[4])
    ax[4].grid(axis="x", visible=False)
    S.save(fig, CH, "fig06_全图自动分割")


# ---------------------------------------------------------------- 图 7：帮不上的地方（低对比度、极小目标、只靠纹理）
BLIND_R = 12.0      # “盲猜一个圆”的半径：和整体的典型大小相当（整体的半径在 10 到 15 px 之间）


def center_points(tgt):
    pts = np.zeros((len(tgt), 2), np.float32)
    for i, t in enumerate(tgt):
        y, x = np.unravel_index(int(inner_distance(t).argmax()), (N, N))
        pts[i] = (x, y)
    return torch.from_numpy(pts)


def eval_set(model, data_or_tuple):
    """对一个数据集的全部目标各给一个框提示和一个正中心的点提示。返回各目标的 IoU 和把框填满的基线。"""
    if isinstance(data_or_tuple, tuple):
        images, idx, target, level = data_or_tuple
    else:
        images = data_or_tuple.images
        idx, target, level = all_targets(data_or_tuple)
    with torch.no_grad():
        f = model.encode(images)
    tgt = target.numpy()
    box = mod.mask_to_box(tgt)
    fill = np.array([t.sum() / ((b[2] - b[0] + 1) * (b[3] - b[1] + 1)) for t, b in zip(tgt, box)])
    bp, bl = ms.box_prompt(torch.from_numpy(box).float())
    sb = score_prompts(model, f, idx, bp, bl, target)
    centers = center_points(tgt)
    pp, pl = ms.point_prompt(centers[:, None], torch.full((len(idx), 1), ms.POS))
    sp = score_prompts(model, f, idx, pp, pl, target)
    yy, xx = np.mgrid[0:N, 0:N]
    disc = np.stack([(yy - c[1]) ** 2 + (xx - c[0]) ** 2 <= BLIND_R ** 2 for c in centers.numpy()])
    blind = ms.mask_iou(torch.from_numpy(disc), target).numpy()
    return dict(box=sb["single"], point=sp["best"], fill=fill, blind=blind, level=level.numpy(), area=tgt.reshape(len(tgt), -1).sum(1), f=f, idx=idx, target=target)


def texture_set(seed, n, amp):
    """目标只靠纹理区分：平均灰度和周围一样，里面每个像素加减 amp（随机正负）。每张图 2 个物体，没有部件。"""
    rng = np.random.default_rng(seed)
    images, idx, target = [], [], []
    for i in range(n):
        img, masks, level, _ = ms.make_scene(rng, contrast=(0.0, 0.0), n_obj=2)
        img = img.copy()
        for m in masks[level == 0]:
            img[m] += amp * rng.choice([-1.0, 1.0], size=int(m.sum())).astype(np.float32)
            idx.append(i)
            target.append(m)
        images.append(img)
    target = torch.from_numpy(np.stack(target))
    return torch.from_numpy(np.stack(images))[:, None], torch.tensor(idx), target, torch.zeros(len(idx), dtype=torch.long)


def outline_set(seed, n, amp):
    """模拟另一种成像方式：物体内部和背景一样亮，只有最外一圈像素比周围亮 amp（像只有边缘散射光的暗场图）。每张图 2 个物体，没有部件。"""
    rng = np.random.default_rng(seed)
    images, idx, target = [], [], []
    for i in range(n):
        img, masks, level, _ = ms.make_scene(rng, contrast=(0.0, 0.0), n_obj=2)
        img = img.copy()
        for m in masks[level == 0]:
            img[m & ~cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)] += amp
            idx.append(i)
            target.append(m)
        images.append(img)
    return torch.from_numpy(np.stack(images))[:, None], torch.tensor(idx), torch.from_numpy(np.stack(target)), torch.zeros(len(idx), dtype=torch.long)


def fig_limits(models, feats, test):
    contrasts = [0.03, 0.05, 0.08, 0.12, 0.2, 0.3]
    rows = {MAIN0: [], LOWC: []}
    example = {}
    for c in contrasts:
        data = ms.SceneSet(2, 120, contrast=(c, c))
        for name in rows:
            r = eval_set(models[name], data)
            rows[name].append([r["box"].mean(), r["point"].mean(), r["fill"].mean(), (r["point"] < 0.5).mean()])
            if c == 0.05:
                example[name] = (data, r)
    normal = {name: eval_set(models[name], test) for name in rows}
    NUM["低对比度"] = {
        "说明": f"测试图里相邻两级的灰度差固定为 c（其余和主设置相同，每个 c 120 张图），全部掩码各当一次目标。{MAIN0} 训练时灰度差在 0.15 到 0.4 之间，没见过更低的；"
              f"“{LOWC}”的训练图灰度差在 0.03 到 0.4 之间，其余完全相同。纹理的标准差 0.04、白噪声的标准差 0.03，合起来约 0.05，所以 c=0.05 时目标和周围只差 1 倍噪声标准差。"
              "点提示点在目标正中心、取三个候选里最好的；框提示用紧贴目标的框、取单输出",
        "灰度差 c": contrasts,
        **{f"{name}：{label}": [r3(v[i]) for v in rows[name]] for name in rows for i, label in ((0, "框提示"), (1, "点提示"), (3, "点提示 IoU 不到 0.5 的比例"))},
        "把框填满的 IoU（基线）": [r3(v[2]) for v in rows[MAIN0]],
        "在主设置的测试集上（灰度差 0.15 到 0.4）": {name: {"框提示": r3(normal[name]["box"].mean()), "点提示": r3(normal[name]["point"].mean())} for name in rows},
    }
    # 极小目标：按面积分组（主设置的测试集）
    r = normal[MAIN0]
    bins = [(12, 25), (25, 50), (50, 100), (100, 200), (200, 400), (400, 5000)]
    small = {}
    for lo, hi in bins:
        s = (r["area"] >= lo) & (r["area"] < hi)
        small[f"{lo} 到 {hi} 像素" if hi < 5000 else f"{lo} 像素以上"] = {"目标个数": int(s.sum()), "点提示": r3(r["point"][s].mean()), "框提示": r3(r["box"][s].mean()),
                                                                    "把框填满": r3(r["fill"][s].mean()), "点提示 IoU 不到 0.5 的比例": r3((r["point"][s] < 0.5).mean())}
    NUM["极小目标"] = {"说明": f"{MAIN0}，主设置的测试集，全部掩码按面积分组。点提示点在目标正中心、取三个候选里最好的；框提示取单输出。特征图一个格子对应 4x4 = 16 个像素",
                   **small}
    # 只靠纹理区分的目标
    tex = {}
    for amp in (0.05, 0.1, 0.2):
        ts = texture_set(3, 100, amp)
        row = {}
        for name in rows:
            r = eval_set(models[name], ts)
            row[name] = {"点提示": r3(r["point"].mean()), "框提示": r3(r["box"].mean())}
        row["基线：以点为圆心、半径 12 px 的圆（不看图）"] = r3(r["blind"].mean())
        row["基线：把框填满"] = r3(r["fill"].mean())
        tex[f"纹理幅度 {amp}"] = row
    plain = normal[MAIN0]
    whole = plain["level"] == 0
    NUM["只靠纹理区分的目标"] = {
        "说明": "目标的平均灰度和背景完全一样，只是里面每个像素随机加或减一个幅度（像一块更粗糙的区域）；每张图 2 个这样的物体（没有部件），100 张图。"
              "两种模型训练时都只见过靠灰度区分的目标。点提示点在目标正中心、取三个候选里最好的；框提示取单输出。"
              "两条基线都不看图像内容：一条是以点击位置为圆心画一个半径 12 px 的圆（整体的半径在 10 到 15 px 之间），一条是把框填满",
        **tex,
        "对照：主设置测试集里的整体（靠灰度区分）": {"点提示": r3(plain["point"][whole].mean()), "框提示": r3(plain["box"][whole].mean()),
                                    "基线：以点为圆心、半径 12 px 的圆（不看图）": r3(plain["blind"][whole].mean()), "基线：把框填满": r3(plain["fill"][whole].mean())},
    }
    out_rows = {}
    for amp in (0.15, 0.3):
        ts = outline_set(4, 100, amp)
        row = {}
        for name in rows:
            r = eval_set(models[name], ts)
            row[name] = {"点提示": r3(r["point"].mean()), "框提示": r3(r["box"].mean()), "点提示 IoU 不到 0.5 的比例": r3((r["point"] < 0.5).mean())}
        row["基线：以点为圆心、半径 12 px 的圆（不看图）"] = r3(r["blind"].mean())
        row["基线：把框填满"] = r3(r["fill"].mean())
        out_rows[f"轮廓比周围亮 {amp}"] = row
    NUM["只有轮廓亮的目标"] = {
        "说明": "模拟另一种成像方式（像只有边缘散射光的暗场图）：物体内部和背景一样亮，只有最外一圈像素比周围亮；每张图 2 个物体，100 张图。两种模型训练时都没见过这种图。"
              "点提示点在目标正中心、取三个候选里最好的；框提示取单输出；两条基线同“只靠纹理区分的目标”",
        **out_rows,
    }
    # 模拟的异常热图当提示
    idx, target, level = all_targets(test)
    sel = level >= 1
    idx, target = idx[sel], target[sel]
    tgt = target.numpy()
    area = tgt.reshape(len(tgt), -1).sum(1)
    model, f = models[MAIN0], feats[MAIN0]

    def split(v):
        return {"全部": r3(v.mean()), "面积不到 60 像素的": r3(v[area < 60].mean()), "面积 60 像素以上的": r3(v[area >= 60].mean())}

    heat_rows = {}
    for label, cells, sigma in (("细热图：每格 4 px，平滑 sigma=1.5", 16, 1.5), ("粗热图：每格 8 px，平滑 sigma=3", 8, 3.0)):
        rng = np.random.default_rng(21)
        heat = np.zeros((len(tgt), N, N), np.float32)
        for i, t in enumerate(tgt):
            low = cv2.resize(t.astype(np.float32), (cells, cells), interpolation=cv2.INTER_AREA)
            low = low / max(float(low.max()), 1e-6) + rng.normal(scale=0.08, size=(cells, cells)).astype(np.float32)
            heat[i] = cv2.GaussianBlur(cv2.resize(low, (N, N), interpolation=cv2.INTER_LINEAR), (0, 0), sigma)
        peak = np.stack([np.unravel_index(int(h.argmax()), (N, N))[::-1] for h in heat]).astype(np.float32)
        inside = tgt[np.arange(len(tgt)), peak[:, 1].astype(int), peak[:, 0].astype(int)]
        hot = heat >= 0.5 * heat.reshape(len(heat), -1).max(1)[:, None, None]
        hot_t = torch.from_numpy(hot)
        pp, pl = ms.point_prompt(torch.from_numpy(peak)[:, None], torch.full((len(idx), 1), ms.POS))
        logits, ip = decode(model, f, idx, pp, pl)
        cand = logits[:, 1:] > 0
        iou_c = ms.mask_iou(cand, target[:, None])
        by_pred = iou_c.gather(1, torch.from_numpy(mod.select_by_predicted_iou(cand.numpy(), ip[:, 1:].numpy())[1])[:, None])[:, 0].numpy()
        by_heat = iou_c.gather(1, ms.mask_iou(cand, hot_t[:, None]).argmax(1, keepdim=True))[:, 0].numpy()
        bp, bl = ms.box_prompt(torch.from_numpy(mod.mask_to_box(hot)).float())
        sb = score_prompts(model, f, idx, bp, bl, target)
        heat_rows[label] = {
            "热图峰值落在缺陷里面的比例": r3(inside.mean()),
            "直接把热图按峰值的一半二值化": split(ms.mask_iou(hot_t, target).numpy()),
            "峰值当点提示，按预测 IoU 选": split(by_pred),
            "峰值当点提示，挑和二值化热图最像的候选": split(by_heat),
            "峰值当点提示，三个候选里最好的（要人来挑）": split(iou_c.max(1).values.numpy()),
            "二值化热图的外接框当框提示，单输出": split(sb["single"]),
        }
    NUM["热图当提示"] = {
        "说明": f"{MAIN0}，主设置的测试集里全部 {len(idx)} 个部件和小部件各当一次“缺陷”。热图是模拟的，不是第 19 章的方法算出来的：把缺陷掩码缩成每格 4 px 或 8 px 的小图（每格取平均，再把最大值归一化到 1），"
              "加标准差 0.08 的噪声，双线性放大回 64x64，再做高斯平滑，用来代表“位置大致对、边界是糊的”的异常热图。几种得到掩码的办法都和真值算 IoU",
        **heat_rows,
        "面积不到 60 像素的缺陷个数": int((area < 60).sum()), "面积 60 像素以上的缺陷个数": int((area >= 60).sum()),
    }

    data, r0 = example[MAIN0]
    r1 = example[LOWC][1]
    cand = np.flatnonzero((r1["box"] > 0.8) & (r0["box"] < 0.45) & (r0["level"] == 0))
    ex = int(cand[0]) if len(cand) else int(np.argmax(r1["box"] - r0["box"]))
    i_img = int(r0["idx"][ex])
    tg = r0["target"][ex].numpy()
    bp, bl = ms.box_prompt(torch.from_numpy(mod.mask_to_box(tg[None])).float())
    preds = []
    with torch.no_grad():
        for name in (MAIN0, LOWC):
            lg, _ = models[name].decode(ms.repeat_feats(example[name][1]["f"], torch.tensor([i_img])), bp, bl)
            preds.append((lg[0, 0] > 0).numpy())
    NUM["低对比度"]["例图"] = {"说明": "图 7 左边三栏：灰度差 0.05 的一张图，框提示，两个模型的单输出和真值的 IoU",
                          MAIN0: r3(ms.mask_iou(torch.from_numpy(preds[0]), torch.from_numpy(tg))), LOWC: r3(ms.mask_iou(torch.from_numpy(preds[1]), torch.from_numpy(tg)))}
    fig, ax = plt.subplots(1, 5, figsize=(17.5, 3.7), gridspec_kw={"width_ratios": [1, 1, 1, 1.5, 1.5]}, layout="constrained")
    img = data.images[i_img, 0].numpy()
    x0, y0, x1, y1 = mod.mask_to_box(tg)
    ax[0].imshow(img, cmap="gray", vmin=0, vmax=1)
    ax[0].contour(tg, levels=[0.5], colors="cyan", linewidths=1.2)
    ax[0].add_patch(plt.Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0 + 1, y1 - y0 + 1, fill=False, ec=S.ORANGE, lw=1.4))
    ax[0].set_title("灰度差 0.05：真值和框提示")
    for a, p, title in ((ax[1], preds[0], "没见过低对比度的模型"), (ax[2], preds[1], "训练时见过的模型")):
        a.imshow(img, cmap="gray", vmin=0, vmax=1)
        a.contourf(p.astype(float), levels=[0.5, 1.5], colors=[S.ORANGE], alpha=0.4)
        a.contour(tg, levels=[0.5], colors="cyan", linewidths=1.0)
        a.set_title(title)
    for a in ax[:3]:
        S.image_axes(a)
    ax[3].axvspan(0.15, 0.4, color=S.GREEN, alpha=0.08)
    ax[3].text(0.155, 0.04, "主模型训练时\n见过的范围", fontsize=8.5, color=S.GREEN, va="bottom")
    for name, color, marker in ((MAIN0, S.RED, "o"), (LOWC, S.BLUE, "s")):
        ax[3].plot(contrasts, [v[1] for v in rows[name]], marker + "-", color=color, lw=1.9, ms=5, label=f"{'没见过' if name == MAIN0 else '见过'}低对比度：点提示")
        ax[3].plot(contrasts, [v[0] for v in rows[name]], marker + "--", color=color, lw=1.3, ms=4, alpha=0.75, label=f"{'没见过' if name == MAIN0 else '见过'}低对比度：框提示")
    ax[3].plot(contrasts, [v[2] for v in rows[MAIN0]], color=S.MUTED, lw=1.2, ls=":", label="基线：把框填满")
    ax[3].set_xscale("log")
    ax[3].set_xticks(contrasts, [f"{c:g}" for c in contrasts])
    ax[3].minorticks_off()
    ax[3].set_xlabel("目标和周围的灰度差（噪声合起来约 0.05）")
    ax[3].set_ylabel("和真值的 IoU")
    ax[3].set_ylim(0, 1.0)
    ax[3].set_title(TITLES["低对比度"])
    ax[3].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, columnspacing=1.0)
    S.clean(ax[3])
    labels = list(small)
    xs = np.arange(len(labels))
    ax[4].plot(xs, [small[k]["点提示"] for k in labels], "o-", color=S.BLUE, lw=1.9, ms=5, label="点提示（三个里最好的）")
    ax[4].plot(xs, [small[k]["框提示"] for k in labels], "s-", color=S.PURPLE, lw=1.9, ms=5, label="框提示（单输出）")
    ax[4].plot(xs, [small[k]["把框填满"] for k in labels], color=S.MUTED, lw=1.2, ls=":", label="基线：把框填满")
    ax[4].set_xticks(xs, [k.replace(" 像素", "").replace(" 到 ", "-").replace("以上", "+") for k in labels], fontsize=9)
    ax[4].set_xlabel("目标面积（像素；特征图一格是 16 像素）")
    ax[4].set_ylabel("和真值的 IoU")
    ax[4].set_ylim(0, 1.0)
    ax[4].set_title(TITLES["极小"])
    ax[4].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, columnspacing=1.0)
    S.clean(ax[4])
    S.save(fig, CH, "fig07_帮不上的地方")


TITLES.update({
    "预测IoU": "按预测 IoU 选比随机选好，按稳定性分数选不比它差",
    "交互": "第一下选错的，第二下拉回来一大截，之后提高很慢",
    "饱和": "三下之后再点，新增达标的很少",
    "位置": "贴着边缘只差一点；点在正中心最容易选错",
    "框和点": "框提示在三种目标上都比单点稳；小部件上不如把框填满",
    "自动": "三步把 185 个候选减到两三个",
    "自动召回": "部件主要丢在按预测 IoU 过滤这一步",
    "低对比度": "灰度差低于训练时见过的范围，IoU 很快掉到 0",
    "极小": "目标小于 50 像素时，模型出的掩码不如把框填满",
})


# ---------------------------------------------------------------- 演示页：把主模型的权重、三张图和引导步骤写进 demos/点提示的歧义.html
DEMO_SCENES = [("三层嵌套", 42, {}), ("另一张图", 20, {}), ("低对比度", 42, {"contrast": (0.05, 0.05)})]
DEMO_SUB, DEMO_PART, DEMO_WHOLE = (35, 23), (36, 29), (41, 15)      # 第一张图上：小部件里、部件里（小部件外）、整体里（部件外）各一个点
LEVEL_NAME = ["整体", "部件", "小部件"]


def demo_payload(model):
    """返回 (写进页面的数据, 每个预设的参考结果)。参考结果用“存成半精度再读回来”的权重算，和页面里 JavaScript 用的权重一致。"""
    state = model.state_dict()
    manifest, blobs = [], []
    for name, t in state.items():
        if name == "decoder.grid":
            continue
        manifest.append([name, list(t.shape)])
        blobs.append(t.numpy().astype("<f2").tobytes())
    ref = ms.MiniSAM(mod.point_position_encoding, seed=0)
    ref.load_state_dict({k: torch.from_numpy(v.numpy().astype(np.float16).astype(np.float32)) for k, v in state.items()})
    ref.eval()
    scenes, raw = [], []
    for name, seed, kw in DEMO_SCENES:
        img, masks, level, _ = ms.make_scene(np.random.default_rng(1000 + seed), n_obj=2, **kw)
        img8 = np.round(np.clip(img, 0, 1) * 255).astype(np.uint8)
        scenes.append({"name": name, "image": base64.b64encode(img8.tobytes()).decode("ascii"),
                       "masks": base64.b64encode(np.packbits(masks.astype(np.uint8).ravel()).tobytes()).decode("ascii"), "levels": [int(v) for v in level]})
        raw.append((img8.astype(np.float32) / 255, masks, level))

    def analyse(scene, points):
        img, masks, level = raw[scene]
        feats = ref.encode(torch.from_numpy(img)[None, None])
        pts, lab = ms.point_prompt(torch.tensor([[q[:2] for q in points]], dtype=torch.float32), torch.tensor([[q[2] for q in points]]))
        logits, ip = ref.decode(feats, pts, lab)
        logits, ip = logits[0].numpy(), ip[0].numpy()
        cands = []
        for m in range(4):
            ious = [float(ms.mask_iou(torch.from_numpy(logits[m] > 0), torch.from_numpy(g))) for g in masks]
            j = int(np.argmax(ious))
            cands.append({"质量": ious[j], "最像": LEVEL_NAME[level[j]], "预测 IoU": float(ip[m]), "稳定性": float(mod.stability_score(logits[m])), "面积": int((logits[m] > 0).sum())})
        first = next(q for q in points if q[2] == 1)
        return {"scene": scene, "points": [list(q) for q in points], "候选": cands, "模型选的": 1 + int(mod.select_by_predicted_iou((logits[None, 1:] > 0), ip[None, 1:])[1][0]),
                "合理答案个数": int(masks[:, first[1], first[0]].sum())}

    s, pt, w = (*DEMO_SUB, 1), (*DEMO_PART, 1), (*DEMO_WHOLE, 1)
    a = analyse(0, [s])
    b = analyse(0, [pt])
    c = analyse(0, [s, w])
    d = analyse(0, [s, (*DEMO_PART, 0)])
    e = analyse(2, [w])
    refs = {"第 1 步": a, "第 2 步": b, "第 3 步": c, "第 4 步": d, "第 5 步": e}

    def chosen(r):
        return r["候选"][r["模型选的"]]

    def f2(v):
        return f"{v:.2f}"

    # 引导语里的说法都由下面的断言保证；模型重新训练后行为变了，这里会直接报错，需要重写引导语
    assert a["合理答案个数"] == 3 and sorted(x["最像"] for x in a["候选"][1:]) == sorted(LEVEL_NAME) and min(x["质量"] for x in a["候选"][1:]) >= 0.75
    assert max(x["预测 IoU"] for x in a["候选"][1:]) < 0.55 and a["候选"][0]["稳定性"] < 0.5
    assert b["合理答案个数"] == 2 and {x["最像"] for x in b["候选"][1:]} == {"整体", "部件"} and b["候选"][0]["质量"] < 0.7
    assert chosen(c)["最像"] == "整体" and chosen(c)["质量"] > 0.9 and chosen(c)["预测 IoU"] > 0.8
    assert chosen(d)["最像"] == "小部件" and chosen(d)["质量"] > 0.7
    assert max(x["面积"] for x in e["候选"]) < 30
    order = "、".join(f"候选 {m} 最像{a['候选'][m]['最像']}（质量 {f2(a['候选'][m]['质量'])}）" for m in (1, 2, 3))
    b_whole = max((x for x in b["候选"][1:] if x["最像"] == "整体"), key=lambda x: x["质量"])
    b_part = max((x for x in b["候选"][1:] if x["最像"] == "部件"), key=lambda x: x["质量"])
    steps = [
        {"text": f"第一张图，在最里面的小部件上点一下。这个点有 3 个合理答案（小部件、部件、整体）。看右边：{order}，三个候选把三个答案都对上了。"
                 f"但三个预测 IoU 都不到 0.55（最高 {f2(max(x['预测 IoU'] for x in a['候选'][1:]))}），掩码明明不差。再看单输出：稳定性分数只有 {f2(a['候选'][0]['稳定性'])}，边界是糊的。",
         "preset": {"scene": 0, "points": [list(s)]}},
        {"text": f"换成点在部件上（小部件外面）。合理答案剩 2 个：候选里有整体（质量 {f2(b_whole['质量'])}）也有部件（质量 {f2(b_part['质量'])}）。"
                 f"单输出的质量只有 {f2(b['候选'][0]['质量'])}，它停在整体和部件之间，哪个都不像。",
         "preset": {"scene": 0, "points": [list(pt)]}},
        {"text": f"回到第 1 步的点，假设你要的是整体：在整体上（部件外面）再加一个正点。模型选的候选变成最像整体的那张，质量 {f2(chosen(c)['质量'])}，"
                 f"预测 IoU 升到 {f2(chosen(c)['预测 IoU'])}。歧义没了，预测 IoU 才敢给高分。",
         "preset": {"scene": 0, "points": [list(s), list(w)]}},
        {"text": f"还是第 1 步的点，假设你只要小部件：切到“负点”，在部件上（小部件外面）点一下。模型选的变成最像小部件的那张，质量 {f2(chosen(d)['质量'])}。"
                 "再自己试试：撤销这个负点，改成在部件上加正点，看选中的是不是变成部件。",
         "preset": {"scene": 0, "points": [list(s), [*DEMO_PART, 0]]}},
        {"text": f"切到“低对比度”这张图：形状和第一张完全一样，只是相邻两级的灰度差从 0.15 以上降到 0.05，训练时没见过。在整体上点一下，四个输出几乎都是空的"
                 f"（面积最大的只有 {max(x['面积'] for x in e['候选'])} 个像素）。打开真值轮廓能看到目标其实在那里。",
         "preset": {"scene": 2, "points": [list(w)]}},
    ]
    data = {"manifest": manifest, "weights": base64.b64encode(b"".join(blobs)).decode("ascii"), "scenes": scenes, "steps": steps}
    return data, refs


def demo_step(models):
    data, refs = demo_payload(models[MAIN0])
    NUM["演示页的预设"] = {
        "说明": f"demos/点提示的歧义.html 里装的是{MAIN0}的权重（存成半精度浮点）。下面是页面“按顺序试”五步的参考结果，用同一份半精度权重在 PyTorch 里算的；"
              "页面里的 JavaScript 实现用 node 核对过和它一致。候选的顺序是：单输出、候选 1、候选 2、候选 3。第一张图相邻两级的灰度差在 0.15 到 0.4 之间，“低对比度”那张固定为 0.05",
        **{k: {"合理答案个数": v["合理答案个数"], "模型选的是候选": v["模型选的"], "质量": [r3(c["质量"]) for c in v["候选"]], "最像": [c["最像"] for c in v["候选"]],
               "预测 IoU": [r3(c["预测 IoU"]) for c in v["候选"]], "稳定性分数": [r3(c["稳定性"]) for c in v["候选"]], "面积": [c["面积"] for c in v["候选"]]} for k, v in refs.items()},
        "页面里权重的个数": int(sum(np.prod(sh) for _, sh in data["manifest"])),
    }
    path = S.MAIN / "demos" / "点提示的歧义.html"
    block = "/*DATA_BEGIN*/\nconst DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n/*DATA_END*/"
    html = path.read_text(encoding="utf-8") if path.exists() else ""
    current = re.search(r"/\*DATA_BEGIN\*/.*?/\*DATA_END\*/", html, flags=re.S)
    if os.environ.get("CH27_WRITE_DEMO") == "1":
        if not current:
            raise RuntimeError("演示页里没有找到 /*DATA_BEGIN*/ 和 /*DATA_END*/ 标记")
        path.write_text(html[:current.start()] + block + html[current.end():], encoding="utf-8")
        ref_path = Path(os.environ.get("CH27_DEMO_REF", Path(__import__("tempfile").gettempdir()) / "ch27_demo_ref.json"))
        ref_path.write_text(json.dumps(refs, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"已写入演示页（{len(block) / 1024:.0f} KB 的数据），参考结果在 {ref_path}", flush=True)
    elif not current or current.group(0) != block:
        print("注意：演示页里的权重和这次训练的结果不一致。设置 CH27_WRITE_DEMO=1 重新写入，再用 node 核对。", flush=True)


def numbers_misc():
    freqs = torch.tensor([[1.0], [0.0]], dtype=torch.float64)
    pe = mod.point_position_encoding(torch.tensor([[0.0, 0.0], [31.5, 0.0], [63.0, 0.0]], dtype=torch.float64), (N, N), freqs)
    NUM["位置编码的例子"] = {
        "说明": "64x64 的图，只有一个频率（x 方向 1，y 方向 0）时，三个点的编码（sin, cos）。x=31.5 是图像正中，归一化后是 0",
        "x=0": [r3(v) for v in pe[0]], "x=31.5": [r3(v) + 0.0 for v in pe[1]], "x=63": [r3(v) for v in pe[2]],
        "x=0 归一化后的坐标（映射到 -1 到 1）": r3(2 * 0.5 / N - 1),
    }
    rng = np.random.default_rng(5)
    yy, xx = np.mgrid[0:N, 0:N]
    d = np.sqrt((yy - 32) ** 2 + (xx - 32) ** 2)
    sharp, soft = (12 - d) * 4.0, (12 - d) * 0.4
    NUM["稳定性分数的例子"] = {
        "说明": "半径 12 px 的圆，logit 取“半径减去到圆心的距离”乘一个斜率：斜率大就是边界清楚，斜率小就是边界糊。阈值 0，偏移 1",
        "斜率 4（边界清楚）": r3(mod.stability_score(sharp)), "斜率 0.4（边界糊）": r3(mod.stability_score(soft)),
        "斜率 0.4 时高阈值和低阈值掩码的像素数": [int((soft > 1).sum()), int((soft > -1).sum())],
    }
    a = np.zeros((N, N), bool)
    a[10:40, 10:40] = True
    b = np.zeros((N, N), bool)
    b[20:30, 20:30] = True
    ring = a & ~b
    NUM["嵌套掩码的 IoU"] = {
        "说明": "30x30 的方块里套一个 10x10 的方块。整体和部件的掩码 IoU，以及“整体去掉部件剩下的环”和整体的外接框 IoU（用来说明按外接框做 NMS 会把环和整体当成重复）",
        "整体和部件的掩码 IoU": r3((a & b).sum() / (a | b).sum()), "环和整体的掩码 IoU": r3((a & ring).sum() / (a | ring).sum()),
        "环和整体的外接框": [mod.mask_to_box(ring).tolist(), mod.mask_to_box(a).tolist()],
    }
    del rng
    disc = lambda r: d <= r  # noqa: E731
    whole, part, between = disc(12), disc(6), disc(9)

    def iou(p, q):
        return float((p & q).sum() / (p | q).sum())

    NUM["真模型的公开设计"] = {
        "说明": "SAM 论文和官方代码里公开的设计，不是本章实测的结果，写在这里只是为了讲义里的数字有出处",
        "输入图像的长边缩放到（像素）": 1024, "图像特征图": "64x64，每个位置 256 维", "一个提示输出的掩码": "3 个多输出加 1 个单输出，各带一个预测 IoU",
    }
    NUM["歧义下预测 IoU 的理论例子"] = {
        "说明": "同心的整体（半径 12 px）和部件（半径 6 px），点在部件里，两个都是合理答案、各有一半机会被抽成目标。预测 IoU 的分支拟合的是“和抽中的目标的 IoU”，"
              "它能学到的最好结果是这个量的期望。比较一张完美的整体掩码和一张介于两者之间的掩码（半径 9 px，哪个答案都不是）",
        "整体和部件的 IoU": r3(iou(whole, part)),
        "完美的整体掩码：对两个目标的 IoU 和它们的期望": [1.0, r3(iou(whole, part)), r3((1 + iou(whole, part)) / 2)],
        "介于两者之间的掩码：对两个目标的 IoU 和它们的期望": [r3(iou(between, whole)), r3(iou(between, part)), r3((iou(between, whole) + iou(between, part)) / 2)],
        "没有歧义时（目标只有整体）两张掩码的 IoU": [1.0, r3(iou(between, whole))],
        "有歧义时两张掩码的期望相差": r3((1 + iou(whole, part)) / 2 - (iou(between, whole) + iou(between, part)) / 2),
    }


def main() -> None:
    S.setup()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    t_all = time.perf_counter()
    models = train_all()
    print(f"训练共 {time.perf_counter() - t_all:.0f} 秒", flush=True)
    sec = S.pinned(CH, "训练耗时", lambda: {
        "说明": f"在一台 Mac 的 CPU 上测的一次结果：{len(JOBS)} 个模型各占一个进程同时训练（每个进程 2 线程），每个训练 {STEPS} 步。只保留两位有效数字；机器忙的时候会变，单独训练一个会更快",
        "每个模型的训练用时（秒，取平均）": sig2(np.mean(TRAIN_SEC))} if TRAIN_SEC else None, "CH27_RETIME")
    if sec:
        NUM["训练耗时"] = sec
    torch.set_grad_enabled(False)              # 训练都在子进程里做完了，主进程只做评估
    test = ms.SceneSet(1, N_TEST)
    feats = {name: m.encode(test.images) for name, m in models.items()}
    steps = [numbers_misc, lambda: fig_ambiguity(models, feats, test), lambda: fig_structure(models), fig_pred_iou, lambda: fig_interaction(models, feats, test),
             lambda: fig_position(models, feats, test), lambda: fig_auto(models, feats, test), lambda: fig_limits(models, feats, test), lambda: demo_step(models)]
    names = ["numbers_misc", "fig_ambiguity", "fig_structure", "fig_pred_iou", "fig_interaction", "fig_position", "fig_auto", "fig_limits", "demo_step"]
    for name, fn in zip(names, steps):
        t = time.perf_counter()
        fn()
        print(f"完成 {name}（{time.perf_counter() - t:.0f} 秒）", flush=True)
    out = S.MAIN / CH / "figures" / "实测数字.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(NUM, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(NUM, ensure_ascii=False, indent=2))
    print(f"总耗时 {time.perf_counter() - t_all:.0f} 秒")


if __name__ == "__main__":
    main()
