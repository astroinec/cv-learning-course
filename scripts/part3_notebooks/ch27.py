"""生成第 27 章的互动实验 Notebook。

用法（在仓库根目录运行）：python scripts/part3_notebooks/ch27.py
修改实验内容时改本文件再重新生成，不要直接手改 .ipynb。
Notebook 只调用学员自己在作业文件里的实现和本章目录里的 迷你SAM.py（那不是作业），不包含参考答案；
设置环境变量 COURSE_IMPL_DIR 可以让 Notebook 改用另一份作业实现（维护者验证用）。
HELPER 和 notebook() 抄自 scripts/build_track2_notebooks.py，本文件不依赖它。
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]

HELPER = '''%matplotlib inline
import importlib.util, os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "PingFang SC", "Heiti SC", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

CHAPTER_NAME = "{chapter}"
ASSIGNMENT = "{assignment}"

def find_chapter(name):
    here = Path.cwd().resolve()
    for p in [here, *here.parents]:
        if p.name == name:
            return p
        if (p / name).is_dir():
            return p / name
    raise FileNotFoundError("请在课程目录内启动 Jupyter")

CHAPTER = find_chapter(CHAPTER_NAME)
COURSE_ROOT = CHAPTER.parent

def load_module(chapter_dir, filename):
    override = os.environ.get("COURSE_IMPL_DIR")
    base = Path(override) / chapter_dir.name if override else chapter_dir
    spec = importlib.util.spec_from_file_location(Path(filename).stem, base / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

mod = load_module(CHAPTER, ASSIGNMENT)

def attempt(fn, *args, **kwargs):
    """调用作业函数；还没实现时打印提示并返回 None，而不是中断整个 Notebook。"""
    try:
        return fn(*args, **kwargs)
    except NotImplementedError as exc:
        print(f"先完成作业里的 {{fn.__name__}}：{{exc}}")
        return None

try:
    import cv2  # 可选，用于和 OpenCV 对照
except ImportError:
    cv2 = None
print("作业文件：", CHAPTER / ASSIGNMENT, "｜OpenCV：", "可用" if cv2 else "未安装（对照实验会跳过）")
'''


def notebook(title, intro, chapter, assignment, cells, outro):
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3 (cv-learning)", "language": "python", "name": "cv-learning"},
        "language_info": {"name": "python", "version": "3"},
    }
    nb.cells = [nbf.v4.new_markdown_cell(f"# {title}\n\n{intro}")]
    nb.cells.append(nbf.v4.new_code_cell(HELPER.format(chapter=chapter, assignment=assignment)))
    for kind, src in cells:
        nb.cells.append(nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src))
    nb.cells.append(nbf.v4.new_markdown_cell(outro))
    return nb


CHAPTER = "27_可提示分割SAM"
ASSIGNMENT = "27-1_提示与掩码后处理.py"
TARGET = f"{CHAPTER}/实验_可提示分割.ipynb"

CELLS = [
    ("code", '''import time
import torch

# 本章的小模型和合成数据在 迷你SAM.py 里（不是作业，可以直接读）。它总是从本章目录加载，不受 COURSE_IMPL_DIR 影响。
_spec = importlib.util.spec_from_file_location("mini_sam", CHAPTER / "迷你SAM.py")
ms = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ms)

torch.manual_seed(0)
torch.set_num_threads(2)
print("PyTorch", torch.__version__, "｜有 CUDA：", torch.cuda.is_available(), "｜小模型固定在 CPU 上跑（它太小，用不上显卡）")

rng = np.random.default_rng(3)
fig, ax = plt.subplots(1, 4, figsize=(13, 3.4))
for a in ax:
    img, masks, level, parent = ms.make_scene(rng)
    a.imshow(img, cmap="gray", vmin=0, vmax=1)
    for m, lv in zip(masks, level):
        a.contour(m, levels=[0.5], colors=[["tab:blue", "tab:orange", "tab:green"][lv]], linewidths=1.4)
    a.set_title(f"{len(masks)} 个掩码：整体 {int((level == 0).sum())}，部件 {int((level == 1).sum())}，小部件 {int((level == 2).sum())}")
    a.axis("off")
plt.suptitle("合成数据：蓝色是整体，橙色是部件，绿色是小部件。点在绿色圈里时，三个圈都是合理答案")
plt.show()'''),
    ("md", "## 实验 1：点提示的位置编码\n\n先预测：两个点离得越远，它们的编码越不像吗？频率取大一点或小一点，“像”的范围怎么变？\n\n下面把图中心那个点的编码和全图每个像素的编码做内积，画成一张图。"),
    ("code", '''gen = torch.Generator().manual_seed(0)
base = torch.randn(2, 16, generator=gen)
ys, xs = torch.meshgrid(torch.arange(64.0), torch.arange(64.0), indexing="ij")
grid = torch.stack([xs, ys], -1)                       # (64, 64, 2)，最后一维是 (x, y)
center = torch.tensor([31.0, 31.0])
pe_ok = attempt(mod.point_position_encoding, center, (64, 64), base) is not None
if pe_ok:
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
    for a, scale in zip(ax, (0.3, 1.0, 3.0)):
        f = base * scale
        sim = mod.point_position_encoding(grid, (64, 64), f) @ mod.point_position_encoding(center, (64, 64), f) / f.shape[1]
        a.imshow(sim, cmap="viridis", vmin=-0.3, vmax=1)
        a.plot(31, 31, "r+", ms=12, mew=2)
        a.set_title(f"频率乘 {scale}：和中心点编码的相似度")
        a.axis("off")
    plt.show()
    print("编码的形状：", tuple(mod.point_position_encoding(grid, (64, 64), base).shape), "（16 个频率，前 16 维是 sin，后 16 维是 cos）")'''),
    ("md", "## 实验 2：训练小模型\n\n`迷你SAM.py` 的模型会调用你写的 `point_position_encoding`，训练时造框提示要用你写的 `mask_to_box`，所以这两个函数要先写完。默认训练 1500 步（和讲义用的设置相同），在写课程的那台 Mac 的 CPU 上是一两分钟，你的机器上会不一样。\n\n训练时三个多输出只回传损失最小的那一个。打印出来的“三个候选里最好的 IoU”应该从 0.1 左右升到 0.8 以上。"),
    ("code", '''STEPS = int(os.environ.get("CH27_NB_STEPS", 1500))    # 想快点看到现象可以先改成 600，结论的方向不变，数字会低一些
model = None
box_ok = pe_ok and attempt(mod.mask_to_box, np.zeros((1, 4, 4), bool)) is not None
if pe_ok and box_ok:
    train_set = ms.SceneSet(0, 1500)
    test_set = ms.SceneSet(1, 100)
    t0 = time.perf_counter()
    model = ms.train_mini_sam(mod.point_position_encoding, mod.mask_to_box, train_set, steps=STEPS, seed=0, log=300)
    print(f"训练用时 {time.perf_counter() - t0:.0f} 秒；编码器参数 {sum(p.numel() for p in model.encoder.parameters())}，解码器参数 {sum(p.numel() for p in model.decoder.parameters())}")
    torch.set_grad_enabled(False)
    feats = model.encode(test_set.images)              # 100 张测试图各编码一次，后面所有实验都复用
else:
    print("point_position_encoding 和 mask_to_box 要先写完才能训练，后面的实验会跳过。")'''),
    ("md", "## 实验 3：点一下，看四个输出\n\n先预测：点在小部件里时，单输出会给出小部件、部件还是整体？三个多输出会不会都一样？"),
    ("code", '''def decode_points(i, points):
    """对第 i 张测试图给一组点 [(x, y, 标签), ...]，返回四张掩码的 logit 和四个预测 IoU。"""
    pts = torch.tensor([[p[:2] for p in points]], dtype=torch.float32)
    lab = torch.tensor([[p[2] for p in points]])
    logits, iou_pred = model.decode(ms.repeat_feats(feats, torch.tensor([i])), pts, lab)
    return logits[0].numpy(), iou_pred[0].numpy()

def show_outputs(i, points, title=""):
    logits, iou_pred = decode_points(i, points)
    gts = test_set.masks[i, :test_set.count[i]].numpy()
    fig, ax = plt.subplots(1, 5, figsize=(15, 3.3))
    ax[0].imshow(test_set.images[i, 0], cmap="gray", vmin=0, vmax=1)
    for x, y, l in points:
        ax[0].plot(x, y, "o", color="lime" if l == 1 else "red", ms=7, mec="white")
    ax[0].set_title(title or "提示")
    for m in range(4):
        q = max(float(ms.mask_iou(torch.from_numpy(logits[m] > 0), torch.from_numpy(g))) for g in gts)
        ax[1 + m].imshow(1 / (1 + np.exp(-np.clip(logits[m], -30, 30))), cmap="viridis", vmin=0, vmax=1)
        ax[1 + m].set_title(("单输出" if m == 0 else f"候选 {m}") + f"：预测 {iou_pred[m]:.2f}，质量 {q:.2f}")
    for a in ax:
        a.axis("off")
    plt.show()
    return logits, iou_pred

if model is not None:
    # 找一张有小部件的图，点在小部件里
    i = int(torch.nonzero((test_set.level == 2).any(1))[0])
    sub = test_set.masks[i][test_set.level[i] == 2][0].numpy()
    yy, xx = np.nonzero(sub)
    click = (int(xx[len(xx) // 2]), int(yy[len(yy) // 2]), 1)
    logits, iou_pred = show_outputs(i, [click], "点在小部件里（3 个合理答案）")
    r = attempt(mod.select_by_predicted_iou, logits[None, 1:], iou_pred[None, 1:])
    if r is not None:
        print("按预测 IoU 选中的是候选", int(r[1][0]) + 1, "，它的预测 IoU 是", round(float(r[2][0]), 3))'''),
    ("md", "## 实验 4：歧义下单输出和多输出差多少\n\n对 100 张测试图各抽 8 个点，每个点从“包含它的掩码”里随机抽一个当目标。按“这个点有几个合理答案”分组统计。\n\n先预测：有 3 个合理答案时，单输出和目标的 IoU 平均是多少？三个候选里最好的那个呢？"),
    ("code", '''if model is not None:
    gen = torch.Generator().manual_seed(100)
    idx = torch.arange(len(test_set)).repeat_interleave(8)
    p0, target, pick, K = ms.sample_targets(test_set, torch.arange(len(test_set)), 8, gen, depth_weights=(1.0, 1.0, 1.0))
    pts, lab = ms.point_prompt(p0[:, None], torch.full((len(p0), 1), ms.POS))
    logits, iou_pred = model.decode(ms.repeat_feats(feats, idx), pts, lab)
    iou = ms.mask_iou(logits > 0, target[:, None])                     # (提示数, 4)：和目标的 IoU
    print("合理答案个数  提示数  单输出   三个候选里最好的")
    for k in (1, 2, 3):
        s = K == k
        print(f"     {k}       {int(s.sum()):4d}   {iou[s, 0].mean():.3f}    {iou[s, 1:].max(1).values.mean():.3f}")'''),
    ("md", "## 实验 5：按什么选候选\n\n三个候选只能给用户看一个。比较四种选法：随机、按你写的稳定性分数、按模型预测的 IoU、事后知道哪个最好。\n\n这里只看没有歧义的提示（1 个合理答案），这样“选得好不好”和“用户想要哪个”不会混在一起。"),
    ("code", '''if model is not None:
    s = K == 1
    cand, true = logits[s][:, 1:], iou[s][:, 1:]
    print(f"随机选：            {true.mean():.3f}")
    stab = attempt(mod.stability_score, cand.numpy())
    if stab is not None:
        print(f"按稳定性分数选：    {true.gather(1, torch.from_numpy(np.asarray(stab)).argmax(1, keepdim=True)).mean():.3f}")
    r = attempt(mod.select_by_predicted_iou, (cand > 0).numpy(), iou_pred[s][:, 1:].numpy())
    if r is not None:
        print(f"按预测 IoU 选：     {true.gather(1, torch.from_numpy(r[1])[:, None]).mean():.3f}")
    print(f"三个里最好的：      {true.max(1).values.mean():.3f}")
    a, b = iou_pred[s][:, 1:].flatten().numpy(), true.flatten().numpy()
    print("预测 IoU 和真实 IoU 的相关系数：", round(float(np.corrcoef(a, b)[0, 1]), 3))
    plt.figure(figsize=(4.2, 4))
    plt.scatter(a, b, s=4, alpha=0.3)
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("模型预测的 IoU"); plt.ylabel("真实 IoU"); plt.title("没有歧义的提示")
    plt.show()'''),
    ("md", "## 实验 6：加正点和负点\n\n模拟用户：看到选中的掩码，在错的地方随机补一个点（漏掉的地方补正点，多出来的地方补负点），所有点一起重新解码，图像特征不重算。\n\n先预测：第 2 下能把平均 IoU 提高多少？第 5 下呢？"),
    ("code", '''if model is not None:
    n = len(idx)
    ar = torch.arange(n)
    pts, lab = ms.point_prompt(p0[:, None], torch.full((n, 1), ms.POS), capacity=6)
    for c in range(6):
        lg, ip = model.decode(ms.repeat_feats(feats, idx), pts, lab)
        shown = lg[ar, 1 + ip[:, 1:].argmax(1)] > 0                    # 给用户看的：预测 IoU 最高的候选
        cur = ms.mask_iou(shown, target)
        print(f"点 {c + 1} 下之后：平均 IoU {cur.mean():.3f}，IoU 达到 0.8 的比例 {(cur >= 0.8).float().mean():.0%}")
        if c == 5:
            break
        p, ok = ms.sample_pixel(shown ^ target, gen)                   # 在错的地方随机取一个像素
        pts[ar[ok], c + 1] = p[ok]
        lab[ar[ok], c + 1] = target[ar, p[:, 1].long(), p[:, 0].long()].long()[ok]
    # 一个例子：第一下点在小部件里，再用一个负点把部件排除掉
    part = test_set.masks[i][test_set.parent[i][test_set.level[i] == 2][0]].numpy() & ~sub
    yy, xx = np.nonzero(part)
    neg = (int(xx[len(xx) // 2]), int(yy[len(yy) // 2]), 0)
    show_outputs(i, [click, neg], "小部件里一个正点，部件上一个负点")'''),
    ("md", "## 实验 7：全图自动分割\n\n不给人工提示：在 8×8 的网格上撒 64 个正点，每个点取三个候选，共 192 张掩码；再按预测 IoU 和稳定性分数过滤，最后做掩码的非极大值抑制。\n\n先预测：192 张里大概有多少张是重复的？过滤掉大部分之后，会不会把真的目标也丢了？改一改三个阈值看看。"),
    ("code", '''T_IOU, T_STAB, T_NMS = 0.6, 0.85, 0.7
if model is not None:
    j = 3
    c = (np.arange(8) + 0.5) * 8 - 0.5
    gx, gy = np.meshgrid(c, c)
    grid_pts = torch.from_numpy(np.stack([gx.ravel(), gy.ravel()], -1).astype(np.float32))
    pts64, lab64 = ms.point_prompt(grid_pts[:, None], torch.full((64, 1), ms.POS))
    lg, ip = model.decode(ms.repeat_feats(feats, torch.full((64,), j)), pts64, lab64)
    cand_logits = lg[:, 1:].reshape(-1, 64, 64).numpy()
    cand_masks, cand_score = cand_logits > 0, ip[:, 1:].reshape(-1).numpy().astype(np.float64)
    keep = np.flatnonzero(cand_score >= T_IOU)
    print(f"全部候选 {len(cand_masks)} 张；预测 IoU 不低于 {T_IOU} 的 {len(keep)} 张")
    stab = attempt(mod.stability_score, cand_logits)
    final = None
    if stab is not None:
        keep = keep[np.asarray(stab)[keep] >= T_STAB]
        print(f"稳定性分数不低于 {T_STAB} 的 {len(keep)} 张")
        sub_keep = attempt(mod.mask_nms, cand_masks[keep], cand_score[keep], T_NMS)
        if sub_keep is not None:
            final = keep[sub_keep]
            print(f"掩码 NMS 之后 {len(final)} 张；这张图的真值有 {int(test_set.count[j])} 个掩码")
    if final is not None:
        fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.6))
        img = test_set.images[j, 0]
        ax[0].imshow(img, cmap="gray", vmin=0, vmax=1); ax[0].scatter(grid_pts[:, 0], grid_pts[:, 1], s=8, c="orange"); ax[0].set_title("网格撒点")
        ax[1].imshow(img, cmap="gray", vmin=0, vmax=1, alpha=0.6); ax[1].set_title(f"留下的 {len(final)} 张掩码")
        boxes = attempt(mod.mask_to_box, cand_masks[final])
        for n_, k in enumerate(final):
            ax[1].contour(cand_masks[k], levels=[0.5], colors=[f"C{n_ % 10}"], linewidths=1.5)
            if boxes is not None:
                x0, y0, x1, y1 = boxes[n_]
                ax[1].add_patch(plt.Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0 + 1, y1 - y0 + 1, fill=False, ec=f"C{n_ % 10}", lw=0.8, ls=":"))
        ax[2].imshow(img, cmap="gray", vmin=0, vmax=1, alpha=0.6); ax[2].set_title("真值")
        for n_, g in enumerate(test_set.masks[j, :test_set.count[j]].numpy()):
            ax[2].contour(g, levels=[0.5], colors=[f"C{n_ % 10}"], linewidths=1.5)
        for a in ax:
            a.axis("off")
        plt.show()'''),
    ("md", "## 实验 8：编码一次、解码多次，各花多少时间\n\n先预测：这个小模型里，编码器和解码器谁更慢？讲义图 2 说了真模型和这里不一样的地方。"),
    ("code", '''if model is not None:
    def median_ms(fn, n=100):
        for _ in range(10):
            fn()
        ts = []
        for _ in range(n):
            t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
        return float(np.median(ts) * 1e3)

    x = test_set.images[:1]
    f1 = model.encode(x)
    one_p, one_l = ms.point_prompt(torch.tensor([[[30.0, 30.0]]]), torch.tensor([[ms.POS]]))
    t_enc = median_ms(lambda: model.encode(x))
    t_dec = median_ms(lambda: model.decode(f1, one_p, one_l))
    t_64 = median_ms(lambda: model.decode(ms.repeat_feats(f1, torch.zeros(64, dtype=torch.long)), pts64, lab64), 30)
    print(f"编码一张图 {t_enc:.2f} ms，解码一个提示 {t_dec:.2f} ms，一次解码 64 个提示 {t_64:.1f} ms（逐个解码要 {64 * t_dec:.0f} ms）")
    print(f"同一张图点 10 下：只编码一次 {t_enc + 10 * t_dec:.1f} ms，每次都重新编码 {10 * (t_enc + t_dec):.1f} ms")'''),
    ("md", "## 选做：真模型\n\n需要自己安装 `segment-anything` 并下载官方权重，见讲义“在自己的机器上用真模型”一节。这个单元只检查库在不在，不下载任何东西。"),
    ("code", '''try:
    import segment_anything
    print("已安装 segment_anything。按讲义里的代码加载权重，把上面的实验 3、6、7 在一张真实图像上重做一遍。")
except Exception as exc:
    print("跳过：", type(exc).__name__, exc)'''),
]

BOOK = notebook(
    "第 27 章互动实验：小型可提示分割器",
    "先完成 `27-1_提示与掩码后处理.py` 里的 `point_position_encoding` 和 `mask_to_box`，就能训练本章的小模型；其余三个函数在实验 3、5、7 里用到，没写完的会打印提示并跳过。每个实验先预测，再运行。",
    CHAPTER, ASSIGNMENT, CELLS,
    "## 破坏实验与记录\n\n- 把实验 2 的训练改成 `ms.train_mini_sam(..., multimask=False)`（三个多输出都监督），重跑实验 3 和 4：三个候选还不一样吗？\n- 把你的 `point_position_encoding` 里“加 0.5”去掉再训练，实验 4 的数字变了多少？为什么这种错误不容易从结果里看出来？\n- 实验 7 里把 `T_IOU` 改成 0（不按预测 IoU 过滤），留下的掩码多了哪些？对照讲义第 7 节解释。\n- 把结果填进 `实验记录.md`。",
)


def main():
    path = ROOT / TARGET
    path.parent.mkdir(parents=True, exist_ok=True)
    for i, cell in enumerate(BOOK.cells):
        cell["id"] = f"c{i:03d}"  # 固定单元 id，重复生成不产生差异
    nbf.write(BOOK, path)
    print("写入", TARGET)


if __name__ == "__main__":
    main()
