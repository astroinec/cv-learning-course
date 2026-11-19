"""第 30 章配图与实测数字。

用法（在 solutions 分支的工作目录运行）：
    COURSE_MAIN_DIR=../cv-learning-course python solutions/figures/ch30.py

输出：<主分支>/30_神经渲染/figures/*.svg 和 实测数字.json。
讲义里引用的数字全部来自这个脚本，改讲义数字之前先重跑。

全部实验用代码生成的一个小场景（两个球、一个盒子、一块棋盘格底板）和 32x32 的图，在 CPU 上跑，不下载任何数据和权重。
体渲染权重、沿光线合成、位置编码、高斯投影和从前到后的合成调用本章作业的参考实现。
耗时默认沿用上次的记录，设置 CH30_RETIME=1 才重测。
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _style as S  # noqa: E402
from _style import plt  # noqa: E402

torch.set_num_threads(2)
torch.manual_seed(0)

CH = "30_神经渲染"
hw = S.load_solution(CH, "30-1_体渲染与高斯泼溅.py")
NUM: dict = {}
TIMES: dict = {}
SEEDS = (0, 1)

SIZE = 32
FX = FY = 44.0
CX = CY = (SIZE - 1) / 2            # 像素中心在整数坐标（OpenCV 的约定），32 px 宽的画面中心是 15.5
RADIUS = 3.2                         # 相机到原点的距离
BG = torch.tensor([1.0, 1.0, 1.0])
LIGHT = F.normalize(torch.tensor([0.4, 0.8, 0.6]), dim=0)
AMBIENT = 0.35
AABB = (torch.tensor([-1.05, -0.70, -1.05]), torch.tensor([1.05, 0.45, 1.05]))   # 场景包围盒，采样只在里面做
NERF_STEPS, NERF_BATCH, NERF_SAMPLES, NERF_EVAL_SAMPLES, NERF_LR = 1000, 512, 20, 48, 0.01
GS_STEPS, GS_LR, GS_BLUR = 800, 0.02, 0.3
L_DEFAULT, N_VIEWS, GS_N = 4, 12, 128

SCENE_NOTE = ("场景在世界坐标（右手系，y 向上）里，长度没有单位：条纹球（球心 (-0.38, -0.10, 0.15)，半径 0.45，条纹沿 y 方向、周期 0.3）、"
              "蓝球（球心 (0.48, -0.25, -0.25)，半径 0.30）、黄色盒子（0.55 x 0.60 x 0.50）和一块 2 x 2 的棋盘格底板（格子边长 0.5）。"
              "表面是漫反射：颜色 = 反射率 x (0.35 + 0.65 x max(0, n.l))，光的方向固定，所以同一个表面点从哪个方向看颜色都一样；背景是白色")
CAM_NOTE = (f"相机是针孔模型，OpenCV 约定（P_c = R P_w + t，看向 +Z，y 向下），{SIZE}x{SIZE} 像素，fx = fy = {FX:g}，主点 ({CX}, {CY})，"
            f"都在离原点 {RADIUS} 的球面上看向原点。训练视角按方位角均匀分布（第 i 个是 360 i / n + 10 度），仰角 25 度和 40 度交替；"
            "留出的 8 个新视角的方位角是 32、77、…、347 度（间隔 45 度），仰角在 32、18、48 度之间轮换。"
            "每张训练图每个像素用 4x4 条光线取平均（抗锯齿），深度真值是过像素中心那条光线到表面的距离")
NERF_NOTE = (f"迷你 NeRF：坐标除以 1.5 后做位置编码，3 层宽 64 的全连接网络（ReLU），输出密度（softplus）和颜色（sigmoid），不输入视角方向。"
             f"每条光线只在它穿过场景包围盒的那一段里分层采样 {NERF_SAMPLES} 个点（训练时加抖动），评价时取 {NERF_EVAL_SAMPLES} 个等间隔的中点。"
             f"Adam，学习率从 {NERF_LR} 按指数降到十分之一，{NERF_STEPS} 步，每步 {NERF_BATCH} 条光线，损失是均方误差")
GS_NOTE = (f"迷你高斯泼溅：每个高斯有中心、三个轴长（对数）、朝向（四元数）、不透明度和一个颜色（不随视角变），协方差 = R S S^T R^T。"
           f"中心在场景范围内随机初始化，轴长初值 0.12，个数固定，没有增删。渲染时在全图 1024 个像素上算每个高斯的贡献，二维协方差的对角线加 {GS_BLUR}（像素平方）。"
           f"Adam，{GS_STEPS} 步，每步 1 个训练视角，损失是 L1")
PSNR_NOTE = "峰值信噪比 PSNR = -10 log10(均方误差)，颜色范围 0 到 1，先对一张图的全部像素和三个通道取均方误差，再对各个视角的 PSNR 取平均"


def r1(v):
    return round(float(v), 1)


def r2(v):
    return round(float(v), 2)


def r3(v):
    return round(float(v), 3)


def gen(*seed):
    return torch.Generator().manual_seed(int(sum(s * 1000 ** i for i, s in enumerate(seed))))


def psnr(a, b):
    return float(-10 * torch.log10(((a - b) ** 2).mean()))


# ---------------------------------------------------------------- 场景：解析的几何体，光线求交得到真值
SPHERES = [(torch.tensor([-0.38, -0.10, 0.15]), 0.45), (torch.tensor([0.48, -0.25, -0.25]), 0.30)]
BOX = (torch.tensor([0.05, -0.55, 0.30]), torch.tensor([0.60, 0.05, 0.80]))
GROUND = (torch.tensor([-1.0, -0.65, -1.0]), torch.tensor([1.0, -0.55, 1.0]))
BASE_COLORS = torch.tensor([[0.85, 0.20, 0.15], [0.15, 0.35, 0.85], [0.95, 0.75, 0.10], [0.25, 0.55, 0.30]])


def albedo(pts, obj):
    out = torch.zeros(pts.shape[0], 3)
    m = obj == 0
    stripe = (torch.sin(2 * math.pi * pts[m, 1] / 0.3) > 0).float()[:, None]
    out[m] = stripe * BASE_COLORS[0] + (1 - stripe) * torch.tensor([0.95, 0.85, 0.75])
    out[obj == 1] = BASE_COLORS[1]
    out[obj == 2] = BASE_COLORS[2]
    m = obj == 3
    chk = ((torch.floor(pts[m, 0] / 0.5) + torch.floor(pts[m, 2] / 0.5)) % 2)[:, None]
    out[m] = chk * BASE_COLORS[3] + (1 - chk) * torch.tensor([0.80, 0.85, 0.80])
    return out


def ray_box(o, d, lo, hi):
    """光线和轴对齐盒子求交。返回进入、离开的距离和进入面的法线。"""
    inv = 1.0 / torch.where(d.abs() < 1e-9, torch.full_like(d, 1e-9), d)
    t0, t1 = (lo - o) * inv, (hi - o) * inv
    tn, axis = torch.minimum(t0, t1).max(dim=1)
    tf = torch.maximum(t0, t1).min(dim=1).values
    nrm = torch.zeros_like(o)
    rows = torch.arange(len(o))
    nrm[rows, axis] = -torch.sign(d[rows, axis])
    return tn, tf, nrm


def trace(o, d):
    """o, d: (P, 3)，d 是单位向量。返回颜色 (P, 3) 和沿光线到表面的距离 (P,)，没有打到物体时距离是 inf。"""
    P = o.shape[0]
    best = torch.full((P,), float("inf"))
    obj = torch.full((P,), -1, dtype=torch.long)
    nrm = torch.zeros(P, 3)
    for i, (c, r) in enumerate(SPHERES):
        oc = o - c
        b = (oc * d).sum(1)
        disc = b * b - (oc * oc).sum(1) + r * r
        t = -b - torch.sqrt(disc.clamp_min(0))
        ok = (disc > 0) & (t > 0) & (t < best)
        best = torch.where(ok, t, best)
        obj = torch.where(ok, torch.full_like(obj, i), obj)
        nrm = torch.where(ok[:, None], F.normalize(o + t[:, None] * d - c, dim=1), nrm)
    for i, (lo, hi) in ((2, BOX), (3, GROUND)):
        tn, tf, n = ray_box(o, d, lo, hi)
        ok = (tn < tf) & (tn > 0) & (tn < best)
        best = torch.where(ok, tn, best)
        obj = torch.where(ok, torch.full_like(obj, i), obj)
        nrm = torch.where(ok[:, None], n, nrm)
    pts = o + torch.where(torch.isfinite(best), best, torch.zeros_like(best))[:, None] * d
    shade = AMBIENT + (1 - AMBIENT) * (nrm @ LIGHT).clamp_min(0)
    col = albedo(pts, obj) * shade[:, None]
    return torch.where((obj >= 0)[:, None], col, BG.expand(P, 3)), best


# ---------------------------------------------------------------- 相机
def look_at_cv(eye):
    """OpenCV 约定的外参：相机在 eye，看向原点，世界的 +y 朝画面上方。返回 R (3, 3)、t (3,)，P_c = R P_w + t。"""
    eye = torch.as_tensor(eye, dtype=torch.float32)
    fwd = F.normalize(-eye, dim=0)                                   # 相机的 +Z
    right = F.normalize(torch.linalg.cross(fwd, torch.tensor([0.0, 1.0, 0.0])), dim=0)   # 相机的 +X
    down = torch.linalg.cross(fwd, right)                            # 相机的 +Y
    R = torch.stack([right, down, fwd])
    return R, -R @ eye


def orbit(az_deg, el_deg, radius=RADIUS):
    az, el = math.radians(az_deg), math.radians(el_deg)
    return look_at_cv([radius * math.cos(el) * math.sin(az), radius * math.sin(el), radius * math.cos(el) * math.cos(az)])


def rays(R, t, size=SIZE, sub=1, fx=FX):
    """每个像素 sub x sub 条光线，返回起点和单位方向。size 不是 32 时焦距和主点按比例缩放。"""
    offs = (torch.arange(sub, dtype=torch.float32) + 0.5) / sub - 0.5
    v, u = torch.meshgrid(torch.arange(size, dtype=torch.float32), torch.arange(size, dtype=torch.float32), indexing="ij")
    u = (u[..., None, None] + offs[None, None, None, :]).expand(size, size, sub, sub)
    v = (v[..., None, None] + offs[None, None, :, None]).expand(size, size, sub, sub)
    f, c = fx * size / SIZE, (size - 1) / 2
    dc = torch.stack([(u - c) / f, (v - c) / f, torch.ones_like(u)], -1).reshape(-1, 3)
    d = F.normalize(dc @ R, dim=1)                                   # 行向量右乘 R 就是 R^T d_c
    return (-R.T @ t).expand_as(d), d


def render_gt(R, t, size=SIZE, sub=4):
    o, d = rays(R, t, size, sub)
    img = trace(o, d)[0].reshape(size, size, sub * sub, 3).mean(2)
    o, d = rays(R, t, size, 1)
    return img, trace(o, d)[1].reshape(size, size)


_VIEWS: dict = {}


def make_views(key, azs, els):
    if key not in _VIEWS:
        cams = [orbit(a, e) for a, e in zip(azs, els)]
        imgs, deps = zip(*(render_gt(R, t) for R, t in cams))
        _VIEWS[key] = (cams, torch.stack(imgs), torch.stack(deps), list(azs), list(els))
    return _VIEWS[key]


def train_views(n):
    return make_views(("train", n), [360.0 * i / n + 10.0 for i in range(n)], [(25.0, 40.0)[i % 2] for i in range(n)])


def test_views():
    return make_views("test", [32.0 + 45.0 * i for i in range(8)], [(32.0, 18.0, 48.0)[i % 3] for i in range(8)])


def perturb(cams, angle_deg, seed):
    """每个相机的朝向绕一根随机的轴转 angle_deg 度，相机中心不动。模拟位姿估计的随机误差。"""
    g = gen(seed, 77)
    out = []
    for R, t in cams:
        ax = F.normalize(torch.randn(3, generator=g), dim=0)
        a = math.radians(angle_deg)
        K = torch.tensor([[0.0, -ax[2], ax[1]], [ax[2], 0.0, -ax[0]], [-ax[1], ax[0], 0.0]])
        R2 = (torch.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K) @ R
        out.append((R2, -R2 @ (-R.T @ t)))
    return out


def project(pts, R, t, fx=FX):
    cam = pts @ R.T + t
    return torch.stack([fx * cam[:, 0] / cam[:, 2] + CX, fx * cam[:, 1] / cam[:, 2] + CY], 1)


def reproj_rms(cams, cams_used, deps, fx_used=FX):
    """把每个训练视角里看得见的表面点，用“真位姿”和“拿去训练的位姿”各投影一次，像素距离的均方根。"""
    sq = []
    for (R, t), (R2, t2), dep in zip(cams, cams_used, deps):
        o, d = rays(R, t)
        m = torch.isfinite(dep.reshape(-1))
        pts = o[m] + dep.reshape(-1)[m, None] * d[m]
        sq.append(((project(pts, R, t) - project(pts, R2, t2, fx_used)) ** 2).sum(1))
    return float(torch.cat(sq).mean().sqrt())


# ---------------------------------------------------------------- 迷你 NeRF
class TinyNeRF(nn.Module):
    def __init__(self, num_freqs=L_DEFAULT, width=64, depth=3):
        super().__init__()
        self.num_freqs = num_freqs
        dims = [3 * (1 + 2 * num_freqs)] + [width] * depth
        self.body = nn.Sequential(*[m for a, b in zip(dims[:-1], dims[1:]) for m in (nn.Linear(a, b), nn.ReLU())])
        self.head = nn.Linear(width, 4)

    def forward(self, x):
        out = self.head(self.body(hw.positional_encoding(x / 1.5, self.num_freqs)))
        return 30.0 * F.softplus(out[:, 0] - 3.0), torch.sigmoid(out[:, 1:])


def scene_interval(o, d):
    tn, tf, _ = ray_box(o, d, *AABB)
    tn = tn.clamp_min(0)
    return tn, tf, tf > tn + 1e-6


def render_field(field, o, d, n_samples, jitter=None):
    """体渲染：field(pts) 返回密度和颜色。没有穿过场景包围盒的光线直接是背景。返回颜色、深度、累积不透明度。"""
    P = o.shape[0]
    tn, tf, hit = scene_interval(o, d)
    rgb, depth, acc = BG.expand(P, 3).clone(), torch.zeros(P), torch.zeros(P)
    if not bool(hit.any()):
        return rgb, depth, acc
    o, d, tn, tf = o[hit], d[hit], tn[hit], tf[hit]
    u = 0.5 if jitter is None else torch.rand(o.shape[0], n_samples, generator=jitter)
    tv = tn[:, None] + (tf - tn)[:, None] * (torch.arange(n_samples, dtype=torch.float32)[None] + u) / n_samples
    sigma, col = field((o[:, None] + tv[..., None] * d[:, None]).reshape(-1, 3))
    deltas = torch.cat([tv[:, 1:] - tv[:, :-1], tf[:, None] - tv[:, -1:]], 1)
    w = hw.volume_weights(sigma.reshape(-1, n_samples), deltas)
    rgb[hit], depth[hit], acc[hit] = hw.composite_rays(w, col.reshape(-1, n_samples, 3), tv, BG)
    return rgb, depth, acc


def train_nerf(cams, imgs, num_freqs=L_DEFAULT, seed=0, steps=NERF_STEPS, fx=FX, jitter=True):
    torch.manual_seed(seed)
    g = gen(seed, 11)
    net = TinyNeRF(num_freqs)
    o, d = zip(*(rays(R, t, fx=fx) for R, t in cams))
    o, d, c = torch.cat(o), torch.cat(d), imgs.reshape(-1, 3)
    opt = torch.optim.Adam(net.parameters(), lr=NERF_LR)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.1 ** (1 / steps))
    t0 = time.perf_counter()
    for _ in range(steps):
        idx = torch.randint(0, len(o), (NERF_BATCH,), generator=g)
        rgb, _, _ = render_field(net, o[idx], d[idx], NERF_SAMPLES, jitter=g if jitter else None)
        loss = ((rgb - c[idx]) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    TIMES.setdefault("nerf_train", []).append(time.perf_counter() - t0)
    return net.eval()


@torch.no_grad()
def render_views(field, cams, n_samples=NERF_EVAL_SAMPLES, fx=FX):
    """返回 (V, H, W, 3) 的颜色、(V, H, W) 的深度和累积不透明度。"""
    out = [render_field(field, *rays(R, t, fx=fx), n_samples) for R, t in cams]
    return (torch.stack([o[0].reshape(SIZE, SIZE, 3) for o in out]), torch.stack([o[1].reshape(SIZE, SIZE) for o in out]),
            torch.stack([o[2].reshape(SIZE, SIZE) for o in out]))


def mean_psnr(pred, imgs):
    return float(np.mean([psnr(p, im) for p, im in zip(pred, imgs)]))


def depth_error(depth, deps):
    """深度真值有限（打到物体）的像素上，估计深度和真值之差的绝对值的平均。"""
    m = torch.isfinite(deps)
    return float((depth[m] - deps[m]).abs().mean())


_CACHE: dict = {}


def cached(key, make):
    if key not in _CACHE:
        _CACHE[key] = make()
    return _CACHE[key]


def get_nerf(n=N_VIEWS, L=L_DEFAULT, seed=0, rot=0.0, fscale=1.0):
    def make():
        cams, imgs, deps, _, _ = train_views(n)
        used = perturb(cams, rot, seed) if rot else cams
        net = train_nerf(used, imgs, L, seed, fx=FX * fscale)
        tcams, timgs, tdeps, _, _ = test_views()
        tr = render_views(net, used, fx=FX * fscale)
        te = render_views(net, tcams, fx=FX * fscale)
        return {"net": net, "used": used, "train_psnr": mean_psnr(tr[0], imgs), "test_psnr": mean_psnr(te[0], timgs),
                "test_each": [psnr(p, im) for p, im in zip(te[0], timgs)], "depth_err": depth_error(te[1], tdeps), "test": te}
    return cached(("nerf", n, L, seed, rot, fscale), make)


# ---------------------------------------------------------------- 迷你高斯泼溅
PIX = torch.stack(torch.meshgrid(torch.arange(SIZE, dtype=torch.float32), torch.arange(SIZE, dtype=torch.float32), indexing="xy"), -1).reshape(-1, 2)


def quat_to_rot(q):
    w, x, y, z = F.normalize(q, dim=-1).unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


class Splats(nn.Module):
    def __init__(self, n, seed):
        super().__init__()
        g = gen(seed, 21)
        self.mean = nn.Parameter((torch.rand(n, 3, generator=g) * 2 - 1) * torch.tensor([1.0, 0.7, 1.0]))
        self.log_scale = nn.Parameter(torch.full((n, 3), math.log(0.12)))
        self.quat = nn.Parameter(torch.cat([torch.ones(n, 1), torch.zeros(n, 3)], 1))
        self.opacity = nn.Parameter(torch.full((n,), -1.0))
        self.color = nn.Parameter(torch.zeros(n, 3))

    def covs(self):
        m = quat_to_rot(self.quat) * torch.exp(self.log_scale)[:, None, :]
        return m @ m.transpose(1, 2)


def gaussian_alphas(uv, cov2d, opacity, pix=PIX, blur=GS_BLUR):
    """每个像素上每个高斯的不透明度：opacity x exp(-0.5 d^T Sigma^-1 d)，返回 (P, N)。"""
    a, b, c = cov2d[:, 0, 0] + blur, cov2d[:, 0, 1], cov2d[:, 1, 1] + blur
    det = (a * c - b * b).clamp_min(1e-8)
    dx, dy = pix[:, None, 0] - uv[None, :, 0], pix[:, None, 1] - uv[None, :, 1]
    return opacity[None] * torch.exp(-0.5 * (c * dx * dx - 2 * b * dx * dy + a * dy * dy) / det)


def render_splats(model, R, t, order=None):
    """order 为 None 时按真实深度排序；传一个函数时用它返回的值代替深度去排序（用来做“排序错误”的实验）。"""
    uv, cov2d, depth = hw.project_gaussians(model.mean, model.covs(), R, t, FX, FY, CX, CY)
    alphas = gaussian_alphas(uv, cov2d, torch.sigmoid(model.opacity))
    return hw.composite_gaussians(alphas, torch.sigmoid(model.color), depth if order is None else order(depth), BG)


def train_splats(cams, imgs, n, seed, steps=GS_STEPS, order=None):
    torch.manual_seed(seed)
    g = gen(seed, 31)
    model = Splats(n, seed)
    opt = torch.optim.Adam([{"params": [model.mean, model.log_scale, model.quat], "lr": GS_LR},
                            {"params": [model.opacity, model.color], "lr": 2.5 * GS_LR}])
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.1 ** (1 / steps))
    target = imgs.reshape(len(cams), -1, 3)
    t0 = time.perf_counter()
    for _ in range(steps):
        i = int(torch.randint(0, len(cams), (1,), generator=g))
        rgb, _ = render_splats(model, *cams[i], order)
        loss = (rgb - target[i]).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    TIMES.setdefault(f"gs_train_{n}", []).append(time.perf_counter() - t0)
    return model.eval()


@torch.no_grad()
def splat_views(model, cams, order=None):
    return torch.stack([render_splats(model, R, t, order)[0].reshape(SIZE, SIZE, 3) for R, t in cams])


def get_gs(n_views=N_VIEWS, n=GS_N, seed=0, rot=0.0):
    def make():
        cams, imgs, _, _, _ = train_views(n_views)
        used = perturb(cams, rot, seed) if rot else cams
        model = train_splats(used, imgs, n, seed)
        tcams, timgs, _, _, _ = test_views()
        te = splat_views(model, tcams)
        return {"model": model, "train_psnr": mean_psnr(splat_views(model, used), imgs), "test_psnr": mean_psnr(te, timgs),
                "test_each": [psnr(p, im) for p, im in zip(te, timgs)], "test": te}
    return cached(("gs", n_views, n, seed, rot), make)


def show(ax, img, title=None, **kw):
    ax.imshow(np.clip(np.asarray(img), 0, 1) if np.asarray(img).ndim == 3 else np.asarray(img), interpolation="nearest", **kw)
    S.image_axes(ax)
    if title:
        ax.set_title(title, fontsize=9.5, fontweight="normal")


# ---------------------------------------------------------------- 图 1：新视角合成的几种做法
def cam_center(R, t):
    return -R.T @ t


def fig_overview():
    cams, imgs, _, azs, els = train_views(N_VIEWS)
    tcams, timgs, _, tazs, tels = test_views()
    centers = torch.stack([cam_center(*c) for c in cams])
    near, avg2, gaps = [], [], []
    for R, t in tcams:
        cos = F.normalize(centers, dim=1) @ F.normalize(cam_center(R, t), dim=0)
        order = torch.argsort(-cos)
        near.append(imgs[order[0]])
        avg2.append(0.5 * (imgs[order[0]] + imgs[order[1]]))
        gaps.append(math.degrees(math.acos(float(cos[order[0]].clamp(-1, 1)))))
    near, avg2 = torch.stack(near), torch.stack(avg2)
    nerf = [get_nerf(seed=s) for s in SEEDS]
    gs = [get_gs(n=256, seed=s) for s in SEEDS]
    rows = {
        "最近的一张训练图": [psnr(p, im) for p, im in zip(near, timgs)],
        "最近两张训练图取平均": [psnr(p, im) for p, im in zip(avg2, timgs)],
        "迷你 NeRF": list(np.mean([m["test_each"] for m in nerf], axis=0)),
        "迷你高斯泼溅（256 个高斯）": list(np.mean([m["test_each"] for m in gs], axis=0)),
    }
    NUM["新视角合成的几种做法"] = {
        "说明": f"{N_VIEWS} 个训练视角，在 8 个留出的新视角上比较。前两种不建三维模型，直接拿相机位置最接近的训练图（或最近两张的平均）当作新视角的图；"
              f"后两种先从训练图重建场景再渲染，各 {len(SEEDS)} 个随机种子取平均。" + PSNR_NOTE,
        "场景": SCENE_NOTE, "相机": CAM_NOTE, "迷你 NeRF 的设置": NERF_NOTE, "迷你高斯泼溅的设置": GS_NOTE,
        "新视角 PSNR（8 个视角的平均，dB）": {k: r1(np.mean(v)) for k, v in rows.items()},
        "新视角 PSNR 最低的一个视角（dB）": {k: r1(np.min(v)) for k, v in rows.items()},
        "新视角 PSNR 最高的一个视角（dB）": {k: r1(np.max(v)) for k, v in rows.items()},
        "迷你 NeRF 比最近的一张训练图高（dB）": r1(np.mean(rows["迷你 NeRF"]) - np.mean(rows["最近的一张训练图"])),
        "迷你高斯泼溅比最近两张取平均高（dB）": r1(np.mean(rows["迷你高斯泼溅（256 个高斯）"]) - np.mean(rows["最近两张训练图取平均"])),
        "训练视角 PSNR（dB）": {"迷你 NeRF": r1(np.mean([m["train_psnr"] for m in nerf])), "迷你高斯泼溅（256 个高斯）": r1(np.mean([m["train_psnr"] for m in gs]))},
        "各种子的新视角 PSNR": {"迷你 NeRF": [r1(m["test_psnr"]) for m in nerf], "迷你高斯泼溅（256 个高斯）": [r1(m["test_psnr"]) for m in gs]},
        "新视角和最近训练视角的夹角（度，从原点看两个相机的夹角）：最小、平均、最大": [r1(min(gaps)), r1(np.mean(gaps)), r1(max(gaps))],
        "训练图里背景像素的比例": r3(float((~torch.isfinite(train_views(N_VIEWS)[2])).float().mean())),
        "图里画的新视角（方位角，仰角）": [tazs[0], tels[0]], "图里那个视角和最近训练视角的夹角（度）": r1(gaps[0]),
    }

    fig = plt.figure(figsize=(15.5, 3.3), layout="constrained")
    gs_ = fig.add_gridspec(1, 6, width_ratios=[1, 1, 1, 1, 1, 2.1])
    panels = [("新视角的真值", timgs[0]), ("最近的一张训练图", near[0]), ("最近两张取平均", avg2[0]),
              ("迷你 NeRF", nerf[0]["test"][0][0]), ("迷你高斯泼溅", gs[0]["test"][0])]
    for i, (title, img) in enumerate(panels):
        a = fig.add_subplot(gs_[0, i])
        show(a, img, title if i == 0 else f"{title}\n{psnr(img, timgs[0]):.1f} dB")
    a = fig.add_subplot(gs_[0, 5])
    names = list(rows)
    colors = [S.MUTED, "#9aa0a9", S.BLUE, S.ORANGE]
    for i, (name, color) in enumerate(zip(names, colors)):
        a.barh(i, np.mean(rows[name]), color=color, alpha=0.88, height=0.62)
        a.plot(rows[name], [i] * 8, ".", color=S.INK, ms=4.5)
        a.text(max(rows[name]) + 0.5, i, f"{np.mean(rows[name]):.1f}", va="center", fontsize=9.5)
    a.set_yticks(range(len(names)), ["最近的一张训练图", "最近两张取平均", "迷你 NeRF", "迷你高斯泼溅\n（256 个高斯）"])
    a.invert_yaxis()
    a.set_xlim(0, 36)
    a.set_xlabel("新视角 PSNR（dB），黑点是 8 个视角")
    a.set_title(f"直接搬训练图 {np.mean(rows[names[0]]):.1f} 和 {np.mean(rows[names[1]]):.1f} dB，先重建再渲染 {np.mean(rows[names[3]]):.1f} 和 {np.mean(rows[names[2]]):.1f} dB", fontsize=10)
    S.clean(a)
    a.grid(axis="y", visible=False)
    S.save(fig, CH, "fig01_新视角合成的几种做法")


# ---------------------------------------------------------------- 图 2：体渲染
SIG_SOLID = 40.0


def sdf_all(p):
    """四个物体各自的有符号距离，(P, 4)：物体内部为负。"""
    out = [(p - c).norm(dim=1) - r for c, r in SPHERES]
    for lo, hi in (BOX, GROUND):
        q = (p - 0.5 * (lo + hi)).abs() - 0.5 * (hi - lo)
        out.append(q.clamp_min(0).norm(dim=1) + q.max(dim=1).values.clamp_max(0))
    return torch.stack(out, 1)


def analytic_field(sig0, soft=0.0):
    """解析的密度场：物体内部密度 sig0，外部 0（soft=0）；soft 大于 0 时边界按 sigmoid(-距离 / soft) 平滑过渡。颜色是各物体的固定颜色按密度加权。"""
    def field(pts):
        sd = sdf_all(pts)
        dens = sig0 * ((sd < 0).float() if soft == 0 else torch.sigmoid(-sd / soft))
        sigma = dens.sum(1)
        return sigma, (dens[:, :, None] * BASE_COLORS[None]).sum(1) / sigma.clamp_min(1e-12)[:, None]
    return field


def fig_volume():
    tcams, timgs, tdeps, _, _ = test_views()
    R, t = tcams[0]
    o, d = rays(R, t)
    ns = (4, 8, 16, 32, 64, 128, 256)
    fields = {"硬边界": analytic_field(SIG_SOLID), "软边界": analytic_field(SIG_SOLID, 0.06)}
    conv = {}
    with torch.no_grad():
        for name, field in fields.items():
            ref = render_field(field, o, d, 4096)[0]
            mid = [float((render_field(field, o, d, n)[0] - ref).abs().mean()) for n in ns]
            g = gen(5, 2)
            rnd = [float(np.mean([float((render_field(field, o, d, n, jitter=g)[0] - ref).abs().mean()) for _ in range(4)])) for n in ns]
            ps = [psnr(render_field(field, o, d, n)[0], ref) for n in ns]
            conv[name] = {"mid": mid, "rnd": rnd, "psnr": ps}
        # 深度：硬边界的场和光线求交的真值比
        dep_true = tdeps[0].reshape(-1)
        m = torch.isfinite(dep_true)
        dep_rows = {}
        for n in ns:
            _, dep, acc = render_field(fields["硬边界"], o, d, n)
            part = m & (acc > 0.05) & (acc < 0.95)
            dep_rows[n] = {"norm": float((dep[m] - dep_true[m]).abs().mean()), "raw": float((dep[m] * acc[m] - dep_true[m]).abs().mean()),
                           "bias": float((dep[m & (acc > 0.99)] - dep_true[m & (acc > 0.99)]).mean()), "partial": float(part.float().sum() / m.float().sum()),
                           "part_norm": float((dep[part] - dep_true[part]).abs().mean()), "part_raw": float((dep[part] * acc[part] - dep_true[part]).abs().mean())}
        tn, tf, hit = scene_interval(o, d)
        mean_len = float((tf - tn)[hit].mean())
    # 训练好的迷你 NeRF 上，归一化和不归一化差多少
    nerf_te = get_nerf(seed=0)["test"]
    fin = torch.isfinite(tdeps)
    nerf_part = fin & (nerf_te[2] > 0.05) & (nerf_te[2] < 0.95)

    # 一条光线：半透明的密度，依次穿过盒子和条纹球
    row, col_ = ONE_RAY
    k = row * SIZE + col_
    field = analytic_field(SIG_DEMO)
    with torch.no_grad():
        t0, t1 = float(tn[k]), float(tf[k])
        fine = torch.linspace(t0, t1, 2001)
        sig_f, col_f = field(o[k][None] + fine[:, None] * d[k][None])
        tau = torch.cat([torch.zeros(1), torch.cumsum(0.5 * (sig_f[1:] + sig_f[:-1]) * (fine[1:] - fine[:-1]), 0)])
        trans_f = torch.exp(-tau)
        n_demo = 16
        tv = t0 + (t1 - t0) * (torch.arange(n_demo) + 0.5) / n_demo
        sig_s, col_s = field(o[k][None] + tv[:, None] * d[k][None])
        deltas = torch.full((n_demo,), (t1 - t0) / n_demo)
        w = hw.volume_weights(sig_s, deltas)
        rgb16, dep16, acc16 = hw.composite_rays(w, col_s, tv, BG)
        dens_ref = 4096
        tvr = t0 + (t1 - t0) * (torch.arange(dens_ref) + 0.5) / dens_ref
        sig_r, col_r = field(o[k][None] + tvr[:, None] * d[k][None])
        wr = hw.volume_weights(sig_r, torch.full((dens_ref,), (t1 - t0) / dens_ref))
        rgb_ref, dep_ref, acc_ref = hw.composite_rays(wr, col_r, tvr, BG)
        inside = sig_f > 0
        edges = fine[1:][inside[1:] != inside[:-1]]
        which = torch.where(sig_s > 0, sdf_all(o[k][None] + tv[:, None] * d[k][None]).argmin(1), torch.full((n_demo,), -1))
        per_obj = [float(w[which == j].sum()) for j in (2, 0, 3)]          # 盒子、条纹球、底板

    NUM["体渲染：一条光线"] = {
        "说明": f"新视角 0 里第 {row} 行第 {col_} 列像素的光线，在包围盒里的一段（距离 {t0:.2f} 到 {t1:.2f}）。物体内部密度取 {SIG_DEMO:g}（半透明），外部为 0。"
              f"取 {n_demo} 个等间隔的中点采样，和 {dens_ref} 个采样点的结果比",
        "物体边界的位置（沿光线的距离）": [r2(e) for e in edges],
        "每段的间隔": r3((t1 - t0) / n_demo), "一段里密度不为 0 时的不透明度": r3(1 - math.exp(-SIG_DEMO * (t1 - t0) / n_demo)),
        "16 个采样点的权重": [r3(v) for v in w], "权重之和（累积不透明度）": r3(acc16), "4096 个采样点的权重之和": r3(acc_ref),
        "盒子、条纹球、底板各自分到的权重（16 个采样点）": [r3(v) for v in per_obj], "落在盒子、条纹球、底板里的采样点个数": [int((which == j).sum()) for j in (2, 0, 3)],
        "留给背景的权重": r3(1 - float(acc16)),
        "期望深度比第一个表面远多少：16 个 / 4096 个采样点": [r2(float(dep16) - float(edges[0])), r2(float(dep_ref) - float(edges[0]))],
        "16 个采样点合成的颜色": [r3(v) for v in rgb16], "4096 个采样点合成的颜色": [r3(v) for v in rgb_ref],
        "16 个采样点的期望深度": r3(dep16), "4096 个采样点的期望深度": r3(dep_ref), "第一个表面的距离": r3(edges[0]),
        "穿出第一个物体时的透射率": r3(trans_f[(fine >= edges[1]).nonzero()[0, 0]]), "光线末端的透射率": r3(trans_f[-1]),
    }
    NUM["体渲染：采样点数与误差"] = {
        "说明": f"把场景换成解析的密度场（物体内部密度 {SIG_SOLID:g}，颜色是各物体的固定颜色），渲染新视角 0 的 32x32 图。误差是和 4096 个采样点的结果相比，"
              "全部像素、三个通道的平均绝对差。“硬边界”指密度在物体表面从 0 跳到 40；“软边界”指按 sigmoid(-到表面的距离 / 0.06) 平滑过渡。"
              "“等间隔中点”每条光线取各段的中点；“分层随机”每段里随机取一点，4 次的平均。采样只在光线穿过包围盒的那一段里做",
        "采样点数": list(ns), "光线在包围盒里的平均长度": r2(mean_len),
        **{f"{name}，{label}：平均绝对误差": [float(f"{v:.2g}") for v in conv[name][key]] for name in conv for key, label in (("mid", "等间隔中点"), ("rnd", "分层随机"))},
        **{f"{name}，等间隔中点：相对 4096 个采样点的 PSNR（dB）": [r1(v) for v in conv[name]["psnr"]] for name in conv},
        "同样的采样点数下，硬边界的误差是软边界的几倍（等间隔中点）": [r1(a / b) for a, b in zip(conv["硬边界"]["mid"], conv["软边界"]["mid"])],
        "同样是硬边界，分层随机的误差是等间隔中点的几倍": [r1(a / b) for a, b in zip(conv["硬边界"]["rnd"], conv["硬边界"]["mid"])],
        "采样点数翻倍时误差缩小的倍数（16 到 256 之间按对数坐标拟合的斜率换算）": {
            f"{name}，{label}": r2(2 ** -np.polyfit(np.log2(ns[2:]), np.log2(conv[name][key][2:]), 1)[0]) for name in conv for key, label in (("mid", "等间隔中点"), ("rnd", "分层随机"))},
    }
    NUM["体渲染：深度"] = {
        "说明": "硬边界的密度场（密度 40），期望深度和光线求交得到的真值比，统计打到物体的像素。“归一化”是作业 composite_rays 的做法：权重乘距离求和，再除以权重之和；"
              "“不归一化”是只求和不除。“半透明像素”指累积不透明度在 0.05 到 0.95 之间的像素（光线只擦过物体的边）。“系统偏差”只统计累积不透明度超过 0.99 的像素，正数表示估计得比真值远",
        "采样点数": list(ns),
        "归一化的期望深度：平均绝对误差": [r3(dep_rows[n]["norm"]) for n in ns],
        "不归一化：平均绝对误差": [r3(dep_rows[n]["raw"]) for n in ns],
        "归一化的期望深度：系统偏差": [r3(dep_rows[n]["bias"]) for n in ns],
        "半个采样间隔（平均长度 / 采样点数 / 2）": [r3(mean_len / n / 2) for n in ns],
        "半个采样间隔 + 1 / 密度": [r3(mean_len / n / 2 + 1 / SIG_SOLID) for n in ns],
        "半透明像素的比例": [r3(dep_rows[n]["partial"]) for n in ns],
        "64 个采样点时，半透明像素上的平均绝对误差：归一化 / 不归一化": [r2(dep_rows[64]["part_norm"]), r2(dep_rows[64]["part_raw"])],
        "训练好的迷你 NeRF（12 个视角，种子 0，8 个新视角）：归一化 / 不归一化的深度误差": [r3(depth_error(nerf_te[1], tdeps)), r3(depth_error(nerf_te[1] * nerf_te[2], tdeps))],
        "训练好的迷你 NeRF：半透明像素占物体像素的比例": r3(float(nerf_part.float().sum() / fin.float().sum())),
        "训练好的迷你 NeRF：半透明像素上的平均绝对误差，归一化 / 不归一化": [r2(float((nerf_te[1][nerf_part] - tdeps[nerf_part]).abs().mean())), r2(float((nerf_te[1][nerf_part] * nerf_te[2][nerf_part] - tdeps[nerf_part]).abs().mean()))],
    }

    fig, ax = plt.subplots(1, 4, figsize=(16.5, 3.9), gridspec_kw={"width_ratios": [1.25, 1.1, 1.05, 1.05]}, layout="constrained")
    a = ax[0]
    a.fill_between(fine, sig_f, step="mid", color="#c9cdd4", alpha=0.75, lw=0)
    a.set_ylabel(f"密度（物体内部 {SIG_DEMO:g}）", color=S.MUTED)
    a.set_ylim(0, SIG_DEMO * 1.6)
    a.set_yticks([0, SIG_DEMO])
    a.set_xlabel("沿光线的距离")
    b = a.twinx()
    b.plot(fine, trans_f, color=S.BLUE, lw=2)
    b.plot(tv - 0.5 * deltas, torch.cat([torch.ones(1), torch.cumprod(torch.exp(-sig_s * deltas), 0)[:-1]]), "o", color=S.BLUE, ms=4, mfc="white")
    b.set_ylim(0, 1.6)
    b.set_yticks([0, 0.5, 1.0])
    b.set_ylabel("透射率", color=S.BLUE)
    b.spines["top"].set_visible(False)
    a.spines["top"].set_visible(False)
    a.text(0.02, 0.97, "灰色：密度    蓝线：透射率\n圆圈：16 段各自起点处的透射率", transform=a.transAxes, va="top", fontsize=9, color=S.INK)
    a.set_title("透射率只在有密度的地方下降")
    a = ax[1]
    a.bar(tv, w, width=0.8 * (t1 - t0) / n_demo, color=[tuple(float(x) for x in c) for c in col_s], edgecolor="#8f959e", lw=0.6)
    a.set_xlim(t0, t1)
    a.set_ylim(0, max(float(w.max()) * 1.45, 0.3))
    a.set_xlabel("沿光线的距离")
    a.set_ylabel("权重 = 透射率 x 这一段的不透明度")
    for x0, rgb, label in ((0.60, rgb16, "16 点"), (0.80, rgb_ref, "4096 点")):
        a.add_patch(plt.Rectangle((x0, 0.76), 0.16, 0.16, transform=a.transAxes, color=tuple(float(x) for x in rgb.clamp(0, 1)), ec="#8f959e", lw=0.8))
        a.text(x0 + 0.08, 0.72, label, transform=a.transAxes, ha="center", va="top", fontsize=8.5)
    a.text(0.03, 0.96, f"权重之和 {float(acc16):.2f}\n剩下的 {1 - float(acc16):.2f} 给背景", transform=a.transAxes, va="top", fontsize=9)
    a.set_title("权重：前面挡住的越多，后面分到的越少")
    S.clean(a)
    a = ax[2]
    for name, color in (("硬边界", S.RED), ("软边界", S.GREEN)):
        a.plot(ns, conv[name]["mid"], "o-", color=color, lw=1.9, ms=5, label=f"{name}，等间隔中点")
        a.plot(ns, conv[name]["rnd"], "s--", color=color, lw=1.4, ms=4, alpha=0.75, label=f"{name}，分层随机")
    a.set_xscale("log", base=2)
    a.set_yscale("log")
    a.set_xticks(ns, [str(n) for n in ns])
    a.minorticks_off()
    a.set_yticks([1e-4, 1e-3, 1e-2, 1e-1], ["0.0001", "0.001", "0.01", "0.1"])
    a.set_xlabel("每条光线的采样点数")
    a.set_ylabel("颜色的平均绝对误差")
    a.set_title("点数翻倍误差大约减半，硬边界高出几倍", fontsize=10.5)
    a.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(a)
    a = ax[3]
    a.plot(ns, [dep_rows[n]["norm"] for n in ns], "o-", color=S.BLUE, lw=1.9, ms=5, label="归一化的期望深度")
    a.plot(ns, [dep_rows[n]["raw"] for n in ns], "s-", color=S.ORANGE, lw=1.9, ms=5, label="不归一化")
    a.plot(ns, [mean_len / n / 2 + 1 / SIG_SOLID for n in ns], ":", color=S.MUTED, lw=1.6, label="半个采样间隔 + 1 / 密度")
    a.set_xscale("log", base=2)
    a.set_yscale("log")
    a.set_xticks(ns, [str(n) for n in ns])
    a.set_yticks([0.03, 0.1, 0.3], ["0.03", "0.1", "0.3"])
    a.minorticks_off()
    a.set_xlabel("每条光线的采样点数")
    a.set_ylabel("深度的平均绝对误差")
    a.set_title("点数再多，深度误差也降不到 1 / 密度以下", fontsize=10.5)
    a.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(a)
    S.save(fig, CH, "fig02_体渲染")


# ---------------------------------------------------------------- 图 3：位置编码
PE_LS, PE_VIEWS = (0, 2, 4, 8), (3, 12)
FIT_LS, FIT_STEPS = (0, 2, 4, 6, 8, 10), 500


def fit_image(L, seed, xy, col, mask):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(2 * (1 + 2 * L), 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 3))
    opt = torch.optim.Adam(net.parameters(), 5e-3)
    for _ in range(FIT_STEPS):
        loss = ((torch.sigmoid(net(hw.positional_encoding(xy[mask], L))) - col[mask]) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return torch.sigmoid(net(hw.positional_encoding(xy, L)))


def fig_posenc():
    tcams, timgs, tdeps, _, _ = test_views()
    big = render_gt(*tcams[0], size=64, sub=3)[0]
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, 64), torch.linspace(-1, 1, 64), indexing="ij")
    xy, col = torch.stack([xs, ys], -1).reshape(-1, 2), big.reshape(-1, 3)
    mask = torch.rand(64 * 64, generator=gen(3, 4)) < 0.25
    fit, fit_img = {}, {}
    for L in FIT_LS:
        res = []
        for s in SEEDS:
            p = fit_image(L, s, xy, col, mask)
            res.append((psnr(p[mask], col[mask]), psnr(p[~mask], col[~mask])))
            if s == 0:
                fit_img[L] = p.reshape(64, 64, 3)
        fit[L] = np.array(res)
    NUM["位置编码：二维图像拟合"] = {
        "说明": f"把新视角 0 渲染成 64x64 的图，用一个 3 层宽 64 的全连接网络从像素坐标（归一化到 -1 到 1）回归颜色。只拿随机抽出的四分之一像素训练（{int(mask.sum())} 个），"
              f"其余像素留出。Adam，学习率 0.005，全部训练像素一起算，{FIT_STEPS} 步，{len(SEEDS)} 个种子取平均。L 是位置编码的频率个数，L = 0 就是直接输入坐标",
        "频率个数 L": list(FIT_LS), "训练像素个数": int(mask.sum()), "图的像素总数": 64 * 64,
        "训练像素之间的平均间距（2 个像素，归一化坐标）": r3(2 * 2 / 63),
        "训练像素上的 PSNR（dB）": [r1(fit[L][:, 0].mean()) for L in FIT_LS],
        "留出像素上的 PSNR（dB）": [r1(fit[L][:, 1].mean()) for L in FIT_LS],
        "各种子：留出像素上的 PSNR": [[r1(v) for v in fit[L][:, 1]] for L in FIT_LS],
        "训练像素的间距（相邻像素在归一化坐标里相距）": r3(2 / 63), "最高频率的周期（归一化坐标）：L = 2、4、6、8、10": [r3(2 / 2 ** (L - 1)) for L in (2, 4, 6, 8, 10)],
    }
    runs = {(n, L): [get_nerf(n=n, L=L, seed=s) for s in SEEDS] for n in PE_VIEWS for L in PE_LS}
    NUM["位置编码：迷你 NeRF"] = {
        "说明": f"迷你 NeRF 只改位置编码的频率个数 L，训练视角取 3 个和 12 个，各 {len(SEEDS)} 个种子取平均。新视角是留出的 8 个。深度误差是打到物体的像素上期望深度和真值之差的绝对值的平均。" + NERF_NOTE,
        "频率个数 L": list(PE_LS), "网络输入的维数": [3 * (1 + 2 * L) for L in PE_LS],
        **{f"{n} 个训练视角：{label}": [fmt(np.mean([m[key] for m in runs[(n, L)]])) for L in PE_LS]
           for n in PE_VIEWS for key, label, fmt in (("train_psnr", "训练视角 PSNR（dB）", r1), ("test_psnr", "新视角 PSNR（dB）", r1), ("depth_err", "新视角的深度误差", r3))},
        **{f"{n} 个训练视角：各种子的新视角 PSNR": [[r1(m["test_psnr"]) for m in runs[(n, L)]] for L in PE_LS] for n in PE_VIEWS},
        "最高频率的周期（世界坐标）：L = 2、4、8": [r3(3 / 2 ** (L - 1)) for L in (2, 4, 8)],
        "条纹的周期": 0.3, "一个像素在场景中心对应的宽度（相机距离 / 焦距）": r3(RADIUS / FX),
    }

    fig = plt.figure(figsize=(16, 5.7), layout="constrained")
    g = fig.add_gridspec(2, 6, width_ratios=[1, 1, 1, 1, 1, 1.85])
    seen = big.clone().reshape(-1, 3)
    seen[~mask] = 1.0
    top = [("目标（64x64）", big), ("训练用的四分之一像素", seen.reshape(64, 64, 3))] + [(f"L = {L}，留出 {fit[L][0, 1]:.1f} dB", fit_img[L]) for L in (0, 2, 8)]
    for i, (title, img) in enumerate(top):
        show(fig.add_subplot(g[0, i]), img, title)
    a = fig.add_subplot(g[0, 5])
    a.plot(FIT_LS, [fit[L][:, 0].mean() for L in FIT_LS], "o--", color=S.MUTED, lw=1.6, ms=5, label="训练像素")
    a.plot(FIT_LS, [fit[L][:, 1].mean() for L in FIT_LS], "o-", color=S.BLUE, lw=2, ms=5.5, label="留出像素")
    for j in range(len(SEEDS)):
        a.plot(FIT_LS, [fit[L][j, 1] for L in FIT_LS], ".", color=S.BLUE, alpha=0.4, ms=5)
    a.set_xticks(FIT_LS)
    a.set_xlabel("频率个数 L（二维图像拟合）")
    a.set_ylabel("PSNR（dB）")
    a.set_title(f"训练像素升到 {max(fit[L][:, 0].mean() for L in FIT_LS):.0f} dB，留出像素在 L = 2 之后变差", fontsize=10.5)
    a.legend(fontsize=9, loc="center right")
    S.clean(a)
    bottom = [("新视角的真值", timgs[0])] + [(f"{n} 个视角，L = {L}\n{psnr(runs[(n, L)][0]['test'][0][0], timgs[0]):.1f} dB", runs[(n, L)][0]["test"][0][0])
                                         for n, L in ((12, 0), (12, 8), (3, 2), (3, 8))]
    for i, (title, img) in enumerate(bottom):
        show(fig.add_subplot(g[1, i]), img, title)
    a = fig.add_subplot(g[1, 5])
    for n, color in ((12, S.BLUE), (3, S.ORANGE)):
        a.plot(range(len(PE_LS)), [np.mean([m["train_psnr"] for m in runs[(n, L)]]) for L in PE_LS], "o--", color=color, lw=1.4, ms=4.5, alpha=0.7, label=f"{n} 个视角：训练视角")
        a.plot(range(len(PE_LS)), [np.mean([m["test_psnr"] for m in runs[(n, L)]]) for L in PE_LS], "o-", color=color, lw=2, ms=5.5, label=f"{n} 个视角：新视角")
        for j in range(len(SEEDS)):
            a.plot(range(len(PE_LS)), [runs[(n, L)][j]["test_psnr"] for L in PE_LS], ".", color=color, alpha=0.4, ms=5)
    a.set_xticks(range(len(PE_LS)), [str(L) for L in PE_LS])
    a.set_xlabel("频率个数 L（迷你 NeRF）")
    a.set_ylabel("PSNR（dB）")
    a.set_title(POSENC_TITLE, fontsize=10.5)
    a.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    S.clean(a)
    S.save(fig, CH, "fig03_位置编码")


# ---------------------------------------------------------------- 图 4：视角数量
VIEW_NS, GS_VIEW_NS = (2, 3, 4, 6, 12), (3, 4, 6, 12)
VIEW_SHOW = 1               # 图 4 右上画的新视角（方位角 77 度）：离 2 个训练视角（10 度和 190 度）都远


def fig_views(fig, g):
    tcams, timgs, tdeps, _, _ = test_views()
    runs = {n: [get_nerf(n=n, seed=s) for s in SEEDS] for n in VIEW_NS}
    gs = {n: get_gs(n_views=n, seed=0) for n in GS_VIEW_NS}

    def cover(m):
        fin = torch.isfinite(tdeps)
        return float((m["test"][2][fin] > 0.5).float().mean())

    NUM["视角数量"] = {
        "说明": f"迷你 NeRF（L = {L_DEFAULT}）只改训练视角的个数，{len(SEEDS)} 个种子取平均；新视角是同一组留出的 8 个。深度误差是打到物体的像素上期望深度和真值之差的绝对值的平均；"
              "“物体像素里被重建出来的比例”指真值打到物体的像素中，累积不透明度超过 0.5 的比例。高斯泼溅是 128 个高斯、1 个种子。训练视角按方位角均匀分布：2 个视角时相隔 180 度，3 个时相隔 120 度",
        "训练视角个数": list(VIEW_NS),
        "训练视角 PSNR（dB）": [r1(np.mean([m["train_psnr"] for m in runs[n]])) for n in VIEW_NS],
        "新视角 PSNR（dB）": [r1(np.mean([m["test_psnr"] for m in runs[n]])) for n in VIEW_NS],
        "训练视角和新视角的 PSNR 之差（dB）": [r1(np.mean([m["train_psnr"] - m["test_psnr"] for m in runs[n]])) for n in VIEW_NS],
        "新视角的深度误差": [r3(np.mean([m["depth_err"] for m in runs[n]])) for n in VIEW_NS],
        "深度误差相对相机距离的比例": [f"{100 * np.mean([m['depth_err'] for m in runs[n]]) / RADIUS:.1f}%" for n in VIEW_NS],
        "物体像素里被重建出来的比例": [r3(np.mean([cover(m) for m in runs[n]])) for n in VIEW_NS],
        "各种子的新视角 PSNR": [[r1(m["test_psnr"]) for m in runs[n]] for n in VIEW_NS],
        "各种子的深度误差": [[r3(m["depth_err"]) for m in runs[n]] for n in VIEW_NS],
        "高斯泼溅（128 个高斯）的训练视角个数": list(GS_VIEW_NS),
        "高斯泼溅：训练视角 PSNR（dB）": [r1(gs[n]["train_psnr"]) for n in GS_VIEW_NS],
        "高斯泼溅：新视角 PSNR（dB）": [r1(gs[n]["test_psnr"]) for n in GS_VIEW_NS],
        "图里画的新视角的方位角（度）": test_views()[3][VIEW_SHOW], "2 个训练视角的方位角（度）": train_views(2)[3],
        "图里那个新视角：2、3、12 个训练视角时的 PSNR（dB，种子 0）": [r1(psnr(runs[n][0]["test"][0][VIEW_SHOW], timgs[VIEW_SHOW])) for n in (2, 3, 12)],
        "图里那个新视角：2、3、12 个训练视角时的深度误差（种子 0）": [r3(depth_error(runs[n][0]["test"][1][VIEW_SHOW], tdeps[VIEW_SHOW])) for n in (2, 3, 12)],
    }

    xs = range(len(VIEW_NS))
    a = fig.add_subplot(g[0:2, 0])
    a.plot(xs, [np.mean([m["train_psnr"] for m in runs[n]]) for n in VIEW_NS], "o--", color=S.BLUE, lw=1.5, ms=5, alpha=0.75, label="NeRF：训练视角")
    a.plot(xs, [np.mean([m["test_psnr"] for m in runs[n]]) for n in VIEW_NS], "o-", color=S.BLUE, lw=2.1, ms=6, label="NeRF：新视角")
    for j in range(len(SEEDS)):
        a.plot(xs, [runs[n][j]["test_psnr"] for n in VIEW_NS], ".", color=S.BLUE, alpha=0.4, ms=5)
    gx = [VIEW_NS.index(n) for n in GS_VIEW_NS]
    a.plot(gx, [gs[n]["train_psnr"] for n in GS_VIEW_NS], "s--", color=S.ORANGE, lw=1.5, ms=4.5, alpha=0.75, label="高斯泼溅：训练视角")
    a.plot(gx, [gs[n]["test_psnr"] for n in GS_VIEW_NS], "s-", color=S.ORANGE, lw=2.1, ms=5.5, label="高斯泼溅：新视角")
    a.set_xticks(xs, [str(n) for n in VIEW_NS])
    a.set_xlabel("训练视角个数")
    a.set_ylabel("PSNR（dB）")
    a.set_title("视角越少，训练视角和新视角差得越远")
    a.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    S.clean(a)
    a = fig.add_subplot(g[0:2, 1])
    a.plot(xs, [np.mean([m["depth_err"] for m in runs[n]]) for n in VIEW_NS], "o-", color=S.GREEN, lw=2.1, ms=6)
    for j in range(len(SEEDS)):
        a.plot(xs, [runs[n][j]["depth_err"] for n in VIEW_NS], ".", color=S.GREEN, alpha=0.45, ms=5)
    a.set_xticks(xs, [str(n) for n in VIEW_NS])
    a.set_ylim(0, None)
    a.set_xlabel("训练视角个数")
    a.set_ylabel("NeRF 新视角的深度误差")
    a.set_title("深度（形状）也跟着变差")
    S.clean(a)
    vmin, vmax = RADIUS - 1.3, RADIUS + 1.3
    cmap = plt.get_cmap("viridis_r").copy()
    cmap.set_bad("white")
    v = VIEW_SHOW
    cols = [("真值", timgs[v], torch.where(torch.isfinite(tdeps[v]), tdeps[v], torch.full_like(tdeps[v], float("nan"))))]
    for n in (2, 3, 12):
        te = runs[n][0]["test"]
        cols.append((f"{n} 个视角，{psnr(te[0][v], timgs[v]):.1f} dB", te[0][v], torch.where(te[2][v] > 0.5, te[1][v], torch.full_like(te[1][v], float("nan")))))
    for i, (title, img, dep) in enumerate(cols):
        show(fig.add_subplot(g[0, 2 + i]), img, title)
        show(fig.add_subplot(g[1, 2 + i]), dep, "深度（越亮越近）" if i == 0 else None, cmap=cmap, vmin=vmin, vmax=vmax)


# ---------------------------------------------------------------- 图 5：相机位姿有误差
POSE_ROTS = (0.5, 1.0, 2.0, 4.0)
F_SCALE = 1.1


def fitted_scale(m, fscale):
    """把新视角里估计的表面点和真值的表面点对上，求以原点为中心的最佳缩放（最小二乘）。"""
    tcams, _, tdeps, _, _ = test_views()
    num = den = 0.0
    for i, (R, t) in enumerate(tcams):
        o, d = rays(R, t)
        o2, d2 = rays(R, t, fx=FX * fscale)
        ok = (torch.isfinite(tdeps[i]) & (m["test"][2][i] > 0.9)).reshape(-1)
        gt = o[ok] + tdeps[i].reshape(-1)[ok, None] * d[ok]
        est = o2[ok] + m["test"][1][i].reshape(-1)[ok, None] * d2[ok]
        num += float((est * gt).sum())
        den += float((gt * gt).sum())
    return num / den


def fig_pose(fig, g):
    cams, imgs, deps, _, _ = train_views(N_VIEWS)
    tcams, timgs, tdeps, _, _ = test_views()
    base = [get_nerf(seed=s) for s in SEEDS]
    base_gs = get_gs(seed=0)
    rows, gs_rows = {}, {}
    for rot in POSE_ROTS:
        rows[rot] = [(reproj_rms(cams, m["used"], deps), m) for m in (get_nerf(seed=s, rot=rot) for s in SEEDS)]
        gs_rows[rot] = get_gs(seed=0, rot=rot)
    wrong_f = get_nerf(seed=0, fscale=F_SCALE)
    NUM["相机位姿有误差"] = {
        "说明": f"{N_VIEWS} 个训练视角，训练图不变，只把拿去训练的相机朝向各绕一根随机的轴转一个固定的角度（相机中心不动），模拟位姿估计的随机误差；每个种子是一组不同的随机轴。"
              "“重投影误差”是把各训练视角里看得见的表面点用真位姿和带误差的位姿各投影一次，像素距离的均方根（32 px 宽的图）。新视角用真位姿渲染，和真值比。"
              f"NeRF 是 {len(SEEDS)} 个种子的平均，高斯泼溅（128 个高斯）只跑了种子 0",
        "朝向误差（度）": [0.0] + list(POSE_ROTS),
        "重投影误差（px）": [0.0] + [r2(np.mean([e for e, _ in rows[r]])) for r in POSE_ROTS],
        "重投影误差占画面宽度的比例": ["0%"] + [f"{100 * np.mean([e for e, _ in rows[r]]) / SIZE:.1f}%" for r in POSE_ROTS],
        "NeRF：训练视角 PSNR（dB）": [r1(np.mean([m["train_psnr"] for m in base]))] + [r1(np.mean([m["train_psnr"] for _, m in rows[r]])) for r in POSE_ROTS],
        "NeRF：新视角 PSNR（dB）": [r1(np.mean([m["test_psnr"] for m in base]))] + [r1(np.mean([m["test_psnr"] for _, m in rows[r]])) for r in POSE_ROTS],
        "NeRF：新视角 PSNR 比没有误差时低（dB）": [0.0] + [r1(np.mean([b["test_psnr"] - m["test_psnr"] for b, (_, m) in zip(base, rows[r])])) for r in POSE_ROTS],
        "NeRF：新视角的深度误差": [r3(np.mean([m["depth_err"] for m in base]))] + [r3(np.mean([m["depth_err"] for _, m in rows[r]])) for r in POSE_ROTS],
        "NeRF：各种子的新视角 PSNR": [[r1(m["test_psnr"]) for m in base]] + [[r1(m["test_psnr"]) for _, m in rows[r]] for r in POSE_ROTS],
        "各种子的重投影误差（px）": [[0.0, 0.0]] + [[r2(e) for e, _ in rows[r]] for r in POSE_ROTS],
        "高斯泼溅：新视角 PSNR（dB）": [r1(base_gs["test_psnr"])] + [r1(gs_rows[r]["test_psnr"]) for r in POSE_ROTS],
        "高斯泼溅：新视角 PSNR 比没有误差时低（dB）": [0.0] + [r1(base_gs["test_psnr"] - gs_rows[r]["test_psnr"]) for r in POSE_ROTS],
    }
    f_rms = reproj_rms(cams, cams, deps, fx_used=FX * F_SCALE)
    NUM["所有相机的焦距都偏大"] = {
        "说明": f"位姿不动，训练和渲染新视角时都把焦距当成真值的 {F_SCALE} 倍（{FX * F_SCALE:g} 而不是 {FX:g}），1 个种子。这是所有相机共有的、方向一致的误差。"
              "“重建的大小”是把估计的表面点和真值的表面点对上之后，以原点为中心的最佳缩放比例；相机都在同一个球面上看向原点",
        "按错的焦距投影，表面点偏离的均方根（px）": r2(f_rms),
        "新视角 PSNR（dB）：焦距正确 / 焦距偏大": [r1(base[0]["test_psnr"]), r1(wrong_f["test_psnr"])],
        "训练视角 PSNR（dB）：焦距正确 / 焦距偏大": [r1(base[0]["train_psnr"]), r1(wrong_f["train_psnr"])],
        "新视角的深度误差：焦距正确 / 焦距偏大": [r3(base[0]["depth_err"]), r3(wrong_f["depth_err"])],
        "重建的大小相对真值：焦距正确 / 焦距偏大": [r3(fitted_scale(base[0], 1.0)), r3(fitted_scale(wrong_f, F_SCALE))],
        "焦距偏大时的大小 / 焦距正确时的大小": r3(fitted_scale(wrong_f, F_SCALE) / fitted_scale(base[0], 1.0)),
        "焦距偏大后新视角 PSNR 低了（dB）": r1(base[0]["test_psnr"] - wrong_f["test_psnr"]), "焦距偏大的比例": "10%",
        "尺寸偏小的比例": f"{100 * (1 - fitted_scale(wrong_f, F_SCALE) / fitted_scale(base[0], 1.0)):.1f}%",
        "焦距比例的倒数": r3(1 / F_SCALE),
    }

    a = fig.add_subplot(g[3:5, 0])
    xs = [0.0] + [np.mean([e for e, _ in rows[r]]) for r in POSE_ROTS]
    a.plot(xs, NUM["相机位姿有误差"]["NeRF：训练视角 PSNR（dB）"], "o--", color=S.BLUE, lw=1.5, ms=5, alpha=0.75, label="NeRF：训练视角（带误差的位姿）")
    a.plot(xs, NUM["相机位姿有误差"]["NeRF：新视角 PSNR（dB）"], "o-", color=S.BLUE, lw=2.1, ms=6, label="NeRF：新视角")
    for r in POSE_ROTS:
        a.plot([e for e, _ in rows[r]], [m["test_psnr"] for _, m in rows[r]], ".", color=S.BLUE, alpha=0.45, ms=6)
    a.plot([0.0] + [rows[r][0][0] for r in POSE_ROTS], NUM["相机位姿有误差"]["高斯泼溅：新视角 PSNR（dB）"], "s-", color=S.ORANGE, lw=2.1, ms=5.5, label="高斯泼溅：新视角")
    a.set_xlabel("训练位姿的重投影误差（px，画面宽 32 px）")
    a.set_ylabel("PSNR（dB）")
    a.set_title("位姿差 {:.1f} 和 {:.1f} px，NeRF 的新视角低 {:.1f} 和 {:.1f} dB".format(xs[1], xs[2], *NUM["相机位姿有误差"]["NeRF：新视角 PSNR 比没有误差时低（dB）"][1:3]), fontsize=10.5)
    a.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1)
    S.clean(a)
    a = fig.add_subplot(g[3:5, 1])
    a.plot(xs, NUM["相机位姿有误差"]["NeRF：新视角的深度误差"], "o-", color=S.GREEN, lw=2.1, ms=6)
    for r in POSE_ROTS:
        a.plot([e for e, _ in rows[r]], [m["depth_err"] for _, m in rows[r]], ".", color=S.GREEN, alpha=0.45, ms=6)
    a.set_ylim(0, None)
    a.set_xlabel("训练位姿的重投影误差（px）")
    a.set_ylabel("NeRF 新视角的深度误差")
    a.set_title("形状也一起变差")
    S.clean(a)
    shows = [("真值", timgs[0]), (f"没有误差\n{psnr(base[0]['test'][0][0], timgs[0]):.1f} dB", base[0]["test"][0][0])]
    for r in (1.0, 4.0):
        e, m = rows[r][0]
        shows.append((f"朝向差 {r:g} 度（{e:.1f} px）\n{psnr(m['test'][0][0], timgs[0]):.1f} dB", m["test"][0][0]))
    for i, (title, img) in enumerate(shows):
        show(fig.add_subplot(g[3:5, 2 + i]), img, title)


def fig_views_pose():
    fig = plt.figure(figsize=(16, 9.6), layout="constrained")
    g = fig.add_gridspec(5, 6, width_ratios=[2.0, 1.7, 1, 1, 1, 1], height_ratios=[1, 1, 0.42, 1, 1])
    fig_views(fig, g)
    fig_pose(fig, g)
    S.save(fig, CH, "fig04_视角数量与位姿误差")


# ---------------------------------------------------------------- 图 6：高斯的投影，个数与质量
GS_COUNTS = (16, 32, 64, 128, 256)
MC_N = 2_000_000


def projection_check(ratio, at_edge, seed):
    """相机坐标系里一个各向同性的高斯（标准差 = ratio x 深度）。用采样的办法得到投影点的真实协方差，和雅可比线性化的结果比。"""
    z = RADIUS
    x = z * (CX / FX) if at_edge else 0.0                           # 画面边缘：投影正好落在最右一列像素附近
    mean = torch.tensor([[x, 0.0, z]], dtype=torch.float64)
    s = ratio * z
    cov = (s * s * torch.eye(3, dtype=torch.float64))[None]
    eye, zero = torch.eye(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64)
    uv, cov2d, _ = hw.project_gaussians(mean, cov, eye, zero, FX, FY, CX, CY)
    pts = mean + s * torch.randn(MC_N, 3, generator=gen(seed, 41), dtype=torch.float64)
    front = pts[:, 2] > 0.05 * z
    p = pts[front]
    proj = torch.stack([FX * p[:, 0] / p[:, 2] + CX, FY * p[:, 1] / p[:, 2] + CY], 1)
    mc_mean, mc_cov = proj.mean(0), torch.cov(proj.T)
    return {"lin_uv": uv[0], "lin_cov": cov2d[0], "mc_mean": mc_mean, "mc_cov": mc_cov, "pts": proj[:1500],
            "rel": float((mc_cov - cov2d[0]).norm() / mc_cov.norm()), "shift": float((mc_mean - uv[0]).norm()), "behind": float(1 - front.double().mean()),
            "sx": (float(cov2d[0, 0, 0].sqrt()), float(mc_cov[0, 0].sqrt()))}


def ellipse(mean, cov, k=2.0, n=200):
    vals, vecs = torch.linalg.eigh(cov)
    th = torch.linspace(0, 2 * math.pi, n, dtype=cov.dtype)
    pts = torch.stack([th.cos(), th.sin()], 1) * (k * vals.clamp_min(0).sqrt())[None]
    return (pts @ vecs.T + mean).numpy()


def fig_splat():
    tcams, timgs, _, _, _ = test_views()
    ratios = (0.02, 0.05, 0.1, 0.15, 0.2)
    proj = {(r, e): projection_check(r, e, i) for i, r in enumerate(ratios) for e in (False, True)}
    runs = {n: [get_gs(n=n, seed=s) for s in SEEDS] for n in GS_COUNTS}
    model = runs[GS_N][0]["model"]
    with torch.no_grad():
        sizes = []
        for R, t in tcams:
            depth = (model.mean @ R.T + t)[:, 2]
            sizes.append(torch.exp(model.log_scale).max(1).values / depth)
        sizes = torch.cat(sizes)
        op = torch.sigmoid(model.opacity)
    NUM["高斯投影：雅可比近似的误差"] = {
        "说明": f"相机坐标系里一个各向同性的三维高斯，深度 {RADIUS}，标准差取深度的若干比例。“真实”的二维协方差用 {MC_N} 个采样点逐个做透视投影后统计得到，"
              "作业的 project_gaussians 用投影在高斯中心处的雅可比做线性近似。相对误差是两个协方差矩阵之差的 Frobenius 范数除以真实协方差的范数；"
              "“中心偏移”是采样点投影的平均位置和“把中心投影过去”之间的距离。“画面中心”指高斯在光轴上，“画面边缘”指它的中心投影到最右一列像素附近。"
              "比例 0.02 那一列的数已经接近采样本身的统计误差（百分之零点一的量级）。图里的散点和两个椭圆（2 倍标准差）画的是比例 0.2、画面边缘的情形",
        "标准差 / 深度": list(ratios),
        "画面中心：协方差的相对误差": [f"{100 * proj[(r, False)]['rel']:.1f}%" for r in ratios],
        "画面边缘：协方差的相对误差": [f"{100 * proj[(r, True)]['rel']:.1f}%" for r in ratios],
        "画面边缘：中心偏移（px）": [r2(proj[(r, True)]["shift"]) for r in ratios],
        "画面边缘：水平方向的标准差（px），线性近似 / 采样": [[r2(v) for v in proj[(r, True)]["sx"]] for r in ratios],
        "比例 0.2、画面边缘：线性近似的水平标准差比采样的小": f"{100 * (1 - proj[(0.2, True)]['sx'][0] / proj[(0.2, True)]['sx'][1]):.0f}%",
        "训练好的 128 个高斯在 8 个新视角里：最长轴 / 深度 的中位数和最大值": [r3(sizes.median()), r3(sizes.max())],
        "训练好的 128 个高斯：不透明度的中位数": r2(op.median()), "不透明度超过 0.5 的个数": int((op > 0.5).sum()),
    }
    NUM["高斯个数与质量"] = {
        "说明": f"{N_VIEWS} 个训练视角，只改高斯的个数，{len(SEEDS)} 个种子取平均。" + GS_NOTE,
        "高斯个数": list(GS_COUNTS), "参数个数（每个高斯 14 个数）": [14 * n for n in GS_COUNTS],
        "训练视角 PSNR（dB）": [r1(np.mean([m["train_psnr"] for m in runs[n]])) for n in GS_COUNTS],
        "新视角 PSNR（dB）": [r1(np.mean([m["test_psnr"] for m in runs[n]])) for n in GS_COUNTS],
        "各种子的新视角 PSNR": [[r1(m["test_psnr"]) for m in runs[n]] for n in GS_COUNTS],
        "个数每翻一倍新视角 PSNR 的增量（dB）": [r1(np.mean([m["test_psnr"] for m in runs[b]]) - np.mean([m["test_psnr"] for m in runs[a]])) for a, b in zip(GS_COUNTS[:-1], GS_COUNTS[1:])],
        "对照：迷你 NeRF（12 个视角，L = 4）的新视角 PSNR 和参数个数": [r1(np.mean([get_nerf(seed=s)["test_psnr"] for s in SEEDS])), sum(p.numel() for p in TinyNeRF().parameters())],
    }

    fig = plt.figure(figsize=(16.5, 4.6), layout="constrained")
    g = fig.add_gridspec(2, 6, width_ratios=[1.75, 1.6, 0.92, 0.92, 0.92, 1.9])
    a = fig.add_subplot(g[:, 0])
    big = proj[(0.2, True)]
    pts = big["pts"].numpy()
    a.plot(pts[:, 0], pts[:, 1], ".", color="#b9bec7", ms=2.2, alpha=0.7, zorder=1)
    e1, e2 = ellipse(big["mc_mean"], big["mc_cov"]), ellipse(big["lin_uv"], big["lin_cov"])
    a.plot(e1[:, 0], e1[:, 1], color=S.INK, lw=1.8, label="采样点的均值和协方差")
    a.plot(e2[:, 0], e2[:, 1], color=S.RED, lw=1.8, ls="--", label="雅可比线性近似")
    a.plot(*big["mc_mean"].numpy(), "+", color=S.INK, ms=9, mew=1.8)
    a.plot(*big["lin_uv"].numpy(), "x", color=S.RED, ms=7, mew=1.8)
    a.add_patch(plt.Rectangle((-0.5, -0.5), SIZE, SIZE, fill=False, ec=S.BLUE, lw=1.1, ls=":"))
    a.set_xlim(-8, 70)
    a.set_ylim(46, -15)
    a.set_aspect("equal")
    a.set_xlabel("u（px），蓝色点线框是 32x32 的画面")
    a.set_ylabel("v（px）")
    a.set_title("高斯大到深度的两成时，线性近似的椭圆偏小")
    a.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    S.clean(a)
    a = fig.add_subplot(g[:, 1])
    a.plot(ratios, [100 * proj[(r, False)]["rel"] for r in ratios], "o-", color=S.BLUE, lw=1.9, ms=5, label="画面中心")
    a.plot(ratios, [100 * proj[(r, True)]["rel"] for r in ratios], "s-", color=S.ORANGE, lw=1.9, ms=5, label="画面边缘")
    a.axvline(float(sizes.median()), color=S.MUTED, lw=1.1, ls=":")
    a.text(float(sizes.median()) * 1.08, 0.1, "训练好的高斯\n（中位数）", fontsize=8.5, color=S.MUTED, va="bottom")
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xticks(ratios, [str(r) for r in ratios])
    a.set_yticks([0.1, 1, 10, 100], ["0.1", "1", "10", "100"])
    a.minorticks_off()
    a.set_xlabel("高斯的标准差 / 深度")
    a.set_ylabel("二维协方差的相对误差（%）")
    a.set_title("近似误差随高斯的相对大小增长")
    a.legend(fontsize=8.5, loc="upper left")
    S.clean(a)
    shows = [("新视角的真值", timgs[0])] + [(f"{n} 个，{psnr(runs[n][0]['test'][0], timgs[0]):.1f} dB", runs[n][0]["test"][0]) for n in (16, 64, 256)]
    slots = [g[0, 2], g[0, 3], g[0, 4], g[1, 2]]
    for (title, img), slot in zip(shows, slots):
        show(fig.add_subplot(slot), img, title)
    # 把 16 个高斯的 2 倍标准差椭圆画出来
    a = fig.add_subplot(g[1, 3:5])
    m16 = runs[16][0]["model"]
    with torch.no_grad():
        uv, cov2d, depth = hw.project_gaussians(m16.mean, m16.covs(), *tcams[0], FX, FY, CX, CY)
        colr, opac = torch.sigmoid(m16.color), torch.sigmoid(m16.opacity)
    a.imshow(np.ones((SIZE, SIZE, 3)), interpolation="nearest")
    for i in torch.argsort(-depth).tolist():
        e = ellipse(uv[i], cov2d[i] + GS_BLUR * torch.eye(2), k=1.5, n=60)
        a.fill(e[:, 0], e[:, 1], color=tuple(float(v) for v in colr[i]), alpha=float(0.25 + 0.6 * opac[i]), ec="#646a73", lw=0.5)
    a.set_xlim(-0.5, SIZE - 0.5)
    a.set_ylim(SIZE - 0.5, -0.5)
    S.image_axes(a)
    a.set_title("16 个高斯各自的椭圆（1.5 倍标准差）", fontsize=9.5, fontweight="normal")
    a = fig.add_subplot(g[:, 5])
    xs = range(len(GS_COUNTS))
    a.plot(xs, [np.mean([m["train_psnr"] for m in runs[n]]) for n in GS_COUNTS], "s--", color=S.ORANGE, lw=1.5, ms=4.5, alpha=0.75, label="训练视角")
    a.plot(xs, [np.mean([m["test_psnr"] for m in runs[n]]) for n in GS_COUNTS], "s-", color=S.ORANGE, lw=2.1, ms=5.5, label="新视角")
    for j in range(len(SEEDS)):
        a.plot(xs, [runs[n][j]["test_psnr"] for n in GS_COUNTS], ".", color=S.INK, alpha=0.5, ms=4.5)
    a.axhline(np.mean([get_nerf(seed=s)["test_psnr"] for s in SEEDS]), color=S.BLUE, lw=1.2, ls=":")
    a.text(0.0, np.mean([get_nerf(seed=s)["test_psnr"] for s in SEEDS]) + 0.3, "迷你 NeRF 的新视角", fontsize=8.5, color=S.BLUE)
    a.set_xticks(xs, [str(n) for n in GS_COUNTS])
    a.set_xlabel("高斯个数")
    a.set_ylabel("PSNR（dB）")
    a.set_title("个数每翻一倍，新视角多 {:.1f} 到 {:.1f} dB".format(min(NUM["高斯个数与质量"]["个数每翻一倍新视角 PSNR 的增量（dB）"]), max(NUM["高斯个数与质量"]["个数每翻一倍新视角 PSNR 的增量（dB）"])))
    a.legend(fontsize=8.5, loc="lower right")
    S.clean(a)
    S.save(fig, CH, "fig05_高斯的投影与个数")


# ---------------------------------------------------------------- 图 7：排序，耗时
def time_frames():
    tcams = test_views()[0]
    net = get_nerf(seed=0)["net"]

    def best(fn, reps=5, inner=10):
        out = float("inf")
        for _ in range(reps):
            t = time.perf_counter()
            for _ in range(inner):
                fn()
            out = min(out, (time.perf_counter() - t) / inner)
        return out

    with torch.no_grad():
        o, d = rays(*tcams[0])
        t_nerf = best(lambda: render_field(net, o, d, NERF_EVAL_SAMPLES))
        t_gs = {n: best(lambda n=n: render_splats(get_gs(n=n, seed=0)["model"], *tcams[0])) for n in (GS_N, 256)}
    return t_nerf, t_gs


def fig_sort_time():
    tcams, timgs, _, tazs, _ = test_views()
    cams = train_views(N_VIEWS)[0]
    model = get_gs(n=256, seed=0)["model"]
    n = model.mean.shape[0]
    ref_cam = cams[0]
    with torch.no_grad():
        stale_depth = (model.mean @ ref_cam[0].T + ref_cam[1])[:, 2]
    orders = {"按深度从近到远（正确）": None, "不排序（按存储顺序）": lambda dep: torch.arange(n, dtype=dep.dtype),
              "排反了（从远到近）": lambda dep: -dep, "沿用另一个视角的顺序": lambda dep: stale_depth}
    out = {k: splat_views(model, tcams, v) for k, v in orders.items()}
    right = out["按深度从近到远（正确）"]
    changed = {k: float(((v - right).abs().amax(-1) > 0.05).float().mean()) for k, v in out.items()}
    each = {k: [psnr(p, im) for p, im in zip(v, timgs)] for k, v in out.items()}
    ang = [math.degrees(math.acos(float((F.normalize(cam_center(*c), dim=0) @ F.normalize(cam_center(*ref_cam), dim=0)).clamp(-1, 1)))) for c in tcams]

    # 不透明度只取 0 和 1 时，从前到后的合成退化成“最近的那个赢”（深度缓冲）
    g = gen(8, 8)
    centers, radii, depths = torch.rand(20, 2, generator=g) * SIZE, 3 + 6 * torch.rand(20, generator=g), torch.rand(20, generator=g)
    colors = torch.rand(20, 3, generator=g)
    hard = ((PIX[:, None] - centers[None]).norm(dim=-1) < radii[None]).float()
    rgb = hw.composite_gaussians(hard, colors, depths, BG, max_alpha=1.0)[0]
    nearest = torch.where(hard > 0, depths[None].expand_as(hard), torch.full_like(hard, float("inf"))).argmin(1)
    zbuf = torch.where((hard.sum(1) > 0)[:, None], colors[nearest], BG.expand(len(PIX), 3))
    rgb_rev = hw.composite_gaussians(hard, colors, -depths, BG, max_alpha=1.0)[0]
    NUM["排序错误的后果"] = {
        "说明": "取训练好的 256 个高斯（12 个训练视角，种子 0），渲染 8 个新视角时只改合成的顺序，其余不变。“沿用另一个视角的顺序”指一直按第 1 个训练视角算出的深度排序。"
              "“变了颜色的像素”指和正确顺序的结果相比，某个通道相差超过 0.05 的像素比例",
        "新视角 PSNR（dB）": {k: r1(np.mean(v)) for k, v in each.items()},
        "变了颜色的像素比例": {k: f"{100 * v:.1f}%" for k, v in changed.items() if k != "按深度从近到远（正确）"},
        "比正确顺序低（dB）": {k: r1(np.mean(each["按深度从近到远（正确）"]) - np.mean(v)) for k, v in each.items() if k != "按深度从近到远（正确）"},
        "沿用另一个视角的顺序：各新视角比正确顺序低（dB）": [r1(a - b) for a, b in zip(each["按深度从近到远（正确）"], each["沿用另一个视角的顺序"])],
        "沿用另一个视角的顺序：各新视角和那个视角的夹角（度）": [r1(a) for a in ang],
        "沿用另一个视角的顺序：各新视角的 PSNR（dB）": [r1(v) for v in each["沿用另一个视角的顺序"]],
        "正确顺序：各新视角的 PSNR（dB）": [r1(v) for v in each["按深度从近到远（正确）"]],
    }
    NUM["和三角形光栅化的关系"] = {
        "说明": "20 个随机的不透明圆片（不透明度只有 0 和 1，各有一个深度和颜色），画在 32x32 的图上。用作业的 composite_gaussians（max_alpha 取 1）从前到后合成，"
              "和“每个像素取盖住它的圆片里最近的那个”（第 21 章的深度缓冲）逐像素比较",
        "像素总数": len(PIX), "两种做法颜色相同的像素数": int(((rgb - zbuf).abs().amax(1) < 1e-6).sum()),
        "被至少一个圆片盖住的像素数": int((hard.sum(1) > 0).sum()), "被两个以上圆片盖住的像素数": int((hard.sum(1) > 1).sum()),
        "把顺序排反后颜色变了的像素数": int(((rgb_rev - zbuf).abs().amax(1) > 1e-6).sum()),
    }
    tn, tf, hit = scene_interval(*rays(*tcams[0]))
    mlp_mults = sum(m.in_features * m.out_features for m in TinyNeRF().modules() if isinstance(m, nn.Linear))
    NUM["渲染一帧的计算量"] = {
        "说明": f"渲染新视角 0 的一帧（{SIZE}x{SIZE}）各要做多少次基本运算，按代码数出来的，不是计时。NeRF 每条穿过包围盒的光线取 {NERF_EVAL_SAMPLES} 个点，每个点过一次网络；"
              "本章的高斯泼溅在每个像素上把每个高斯都算一遍（真正的实现只算盖到这个像素所在图块的高斯）",
        "穿过包围盒的光线数": int(hit.sum()), "NeRF：网络求值次数": int(hit.sum()) * NERF_EVAL_SAMPLES, "NeRF：一次网络求值的乘法次数（各线性层输入乘输出之和）": mlp_mults,
        "NeRF：一帧的乘法次数（百万）": r1(int(hit.sum()) * NERF_EVAL_SAMPLES * mlp_mults / 1e6),
        "高斯泼溅：像素数 x 高斯数（128 个 / 256 个）": [len(PIX) * GS_N, len(PIX) * 256],
        "NeRF 的网络求值次数 / 高斯泼溅（128 个）的像素高斯对数": r2(int(hit.sum()) * NERF_EVAL_SAMPLES / (len(PIX) * GS_N)),
    }

    def measure():
        t_nerf, t_gs = time_frames()
        return {
            "说明": "在一台 Apple 芯片 Mac 的 CPU 上（2 个线程，机器上同时有别的程序在跑）测的一次结果，只看量级和倍数关系，不是显卡上的数字。设置 CH30_RETIME=1 重测。"
                  f"渲染一帧是 {SIZE}x{SIZE}、不算梯度，每项连做 10 帧取平均、重复 5 组取最快的一组；NeRF 每条光线 {NERF_EVAL_SAMPLES} 个采样点。训练耗时是各次训练的平均",
            "渲染一帧（毫秒）：迷你 NeRF": float(f"{1000 * t_nerf:.2g}"),
            "渲染一帧（毫秒）：高斯泼溅 128 个": float(f"{1000 * t_gs[GS_N]:.2g}"), "渲染一帧（毫秒）：高斯泼溅 256 个": float(f"{1000 * t_gs[256]:.2g}"),
            "NeRF 是高斯泼溅（128 个）的几倍": float(f"{t_nerf / t_gs[GS_N]:.2g}"), "NeRF 是高斯泼溅（256 个）的几倍": float(f"{t_nerf / t_gs[256]:.2g}"),
            f"训练一个迷你 NeRF（{NERF_STEPS} 步，秒）": float(f"{np.mean(TIMES['nerf_train']):.2g}"),
            f"训练一组高斯（128 个，{GS_STEPS} 步，秒）": float(f"{np.mean(TIMES['gs_train_128']):.2g}"),
            f"训练一组高斯（256 个，{GS_STEPS} 步，秒）": float(f"{np.mean(TIMES['gs_train_256']):.2g}"),
        }

    timing = S.pinned(CH, "耗时", measure, "CH30_RETIME")
    NUM["耗时"] = timing

    fig = plt.figure(figsize=(16.5, 3.6), layout="constrained")
    g_ = fig.add_gridspec(1, 7, width_ratios=[1, 1, 1, 1, 1.9, 1.7, 1.6])
    for i, (k, title) in enumerate((("按深度从近到远（正确）", "正确顺序"), ("不排序（按存储顺序）", "不排序"), ("排反了（从远到近）", "排反了"), ("沿用另一个视角的顺序", "沿用另一个视角"))):
        show(fig.add_subplot(g_[0, i]), out[k][SORT_VIEW], f"{title}\n{each[k][SORT_VIEW]:.1f} dB")
    a = fig.add_subplot(g_[0, 4])
    names = list(each)
    for i, (k, color) in enumerate(zip(names, (S.GREEN, S.ORANGE, S.RED, S.PURPLE))):
        a.barh(i, np.mean(each[k]), color=color, alpha=0.88, height=0.62)
        a.plot(each[k], [i] * len(each[k]), ".", color=S.INK, ms=4.5)
        a.text(max(each[k]) + 0.5, i, f"{np.mean(each[k]):.1f}", va="center", fontsize=9.5)
    a.set_yticks(range(len(names)), ["正确顺序", "不排序", "排反了", "沿用另一个\n视角的顺序"])
    a.invert_yaxis()
    a.set_xlim(0, 34)
    a.set_xlabel("新视角 PSNR（dB），黑点是 8 个视角")
    a.set_title("同一组高斯，只改合成顺序")
    S.clean(a)
    a.grid(axis="y", visible=False)
    a = fig.add_subplot(g_[0, 5])
    a.plot(ang, each["按深度从近到远（正确）"], "o", color=S.GREEN, ms=6, label="正确顺序")
    a.plot(ang, each["沿用另一个视角的顺序"], "D", color=S.PURPLE, ms=6, label="沿用另一个视角的顺序")
    a.set_xlim(0, 180)
    a.set_xticks([0, 45, 90, 135, 180])
    a.set_xlabel("新视角和排序所用视角的夹角（度）")
    a.set_ylabel("PSNR（dB）")
    a.set_title("夹角 20 度几乎不掉，85 度以上掉 3 dB 以上", fontsize=10.5)
    a.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1)
    S.clean(a)
    a = fig.add_subplot(g_[0, 6])
    vals = [timing["渲染一帧（毫秒）：迷你 NeRF"], timing["渲染一帧（毫秒）：高斯泼溅 128 个"], timing["渲染一帧（毫秒）：高斯泼溅 256 个"]]
    bars = a.bar(range(3), vals, color=[S.BLUE, S.ORANGE, S.ORANGE], alpha=0.88, width=0.6)
    for b_, v in zip(bars, vals):
        a.text(b_.get_x() + b_.get_width() / 2, v, f"{v:g}", ha="center", va="bottom", fontsize=9.5)
    a.set_xticks(range(3), ["迷你 NeRF", "高斯 128 个", "高斯 256 个"], fontsize=9)
    a.set_ylim(0, max(vals) * 1.25)
    a.set_ylabel("渲染一帧 32x32（毫秒，CPU）")
    a.set_title("这台机器上的一次测量")
    S.clean(a)
    a.grid(axis="x", visible=False)
    S.save(fig, CH, "fig06_排序与耗时")


# ---------------------------------------------------------------- 训练步数：两种方法都没有训练到头
def numbers_steps():
    cams, imgs, _, _, _ = train_views(N_VIEWS)
    tcams, timgs, _, _, _ = test_views()
    net = train_nerf(cams, imgs, seed=0, steps=2 * NERF_STEPS)
    TIMES["nerf_train"].pop()
    gs = train_splats(cams, imgs, GS_N, 0, steps=2 * GS_STEPS)
    TIMES[f"gs_train_{GS_N}"].pop()
    NUM["训练步数加倍"] = {
        "说明": f"{N_VIEWS} 个训练视角、种子 0，把训练步数加倍（学习率同样按指数降到十分之一），看新视角 PSNR 还涨多少。用来说明本章各表里的数都是固定预算下的结果，两种方法都没有训练到头",
        f"迷你 NeRF：{NERF_STEPS} 步 / {2 * NERF_STEPS} 步的新视角 PSNR（dB）": [r1(get_nerf(seed=0)["test_psnr"]), r1(mean_psnr(render_views(net, tcams)[0], timgs))],
        f"高斯泼溅（128 个）：{GS_STEPS} 步 / {2 * GS_STEPS} 步的新视角 PSNR（dB）": [r1(get_gs(seed=0)["test_psnr"]), r1(mean_psnr(splat_views(gs, tcams), timgs))],
    }
    counts = (NERF_SAMPLES, 24, NERF_EVAL_SAMPLES, 96)
    rows = {}
    for name, jit in (("训练时每段里随机取点（默认）", True), ("训练时固定取各段的中点", False)):
        res = []
        for s in SEEDS:
            model = get_nerf(seed=s)["net"] if jit else train_nerf(cams, imgs, seed=s, jitter=False)
            if not jit:
                TIMES["nerf_train"].pop()
            res.append([mean_psnr(render_views(model, tcams, n_samples=n)[0], timgs) for n in counts])
        rows[name] = [r1(v) for v in np.mean(res, axis=0)]
    # 把作业函数换成数学上等价、只在浮点舍入上不同的写法，其余完全不变，看训练结果差多少
    orig = hw.volume_weights

    def equivalent(sigma, deltas):
        tau = sigma * deltas
        return torch.exp(-(torch.cumsum(tau, dim=-1) - tau)) * (1.0 - torch.exp(-tau))

    alt = []
    hw.volume_weights = equivalent
    try:
        for s in SEEDS:
            model = train_nerf(cams, imgs, seed=s)
            TIMES["nerf_train"].pop()
            alt.append(mean_psnr(render_views(model, tcams)[0], timgs))
    finally:
        hw.volume_weights = orig
    ref = [get_nerf(seed=s)["test_psnr"] for s in SEEDS]
    w_a, w_b = orig(torch.rand(4, 20, generator=gen(6, 6)) * 5, torch.full((20,), 0.06)), equivalent(torch.rand(4, 20, generator=gen(6, 6)) * 5, torch.full((20,), 0.06))
    NUM["实现上的浮点差异对训练结果的影响"] = {
        "说明": f"{N_VIEWS} 个训练视角、{NERF_STEPS} 步，种子、数据、超参数完全相同，只把 volume_weights 里“到达这一段之前的透射率”换成一种数学上等价的写法"
              "（先把各段的密度乘长度一直累加到自己，再减掉自己这一段）。两种写法算出的权重只在最后一两位有效数字上不同",
        "同一组输入下两种写法的权重最大相差": float(f"{float((w_a - w_b).abs().max()):.1g}"),
        "参考实现：两个种子的新视角 PSNR（dB）": [r1(v) for v in ref], "等价写法：两个种子的新视角 PSNR（dB）": [r1(v) for v in alt],
        "同一个种子两种写法相差（dB）": [r1(abs(a - b)) for a, b in zip(ref, alt)],
    }
    NUM["训练时采样点要不要抖动"] = {
        "说明": f"{N_VIEWS} 个训练视角，{len(SEEDS)} 个种子取平均。训练时每条光线 {NERF_SAMPLES} 个采样点：一种在每段里随机取一点（分层随机），一种固定取每段的中点。"
              "训练好之后都用等间隔的中点渲染 8 个新视角，只改渲染时的采样点数",
        "渲染时的采样点数": list(counts), **{f"{k}：新视角 PSNR（dB）": v for k, v in rows.items()},
    }


# ---------------------------------------------------------------- 三种实现错误训练出来是什么样
def numbers_bugs():
    """把作业函数换成写错的版本，其余代码不变，各训练一遍。训练和渲染用的是同一份（错的）实现。"""
    cams, imgs, _, _, _ = train_views(N_VIEWS)
    tcams, timgs, tdeps, _, _ = test_views()
    orig = hw.volume_weights
    counts = (NERF_SAMPLES, NERF_EVAL_SAMPLES, 96)

    def no_delta(sigma, deltas):                   # 不透明度里漏了每段的长度
        return orig(sigma, torch.ones_like(sigma))

    def inclusive(sigma, deltas):                  # 透射率把自己这一段也乘了进去
        tau = sigma * deltas
        return torch.exp(-torch.cumsum(tau, dim=-1)) * (1.0 - torch.exp(-tau))

    rows = {}
    for name, fn in (("正确的实现", orig), ("不透明度里漏了每段的长度", no_delta), ("透射率把自己这一段也乘了进去", inclusive)):
        res = []
        for s in SEEDS:
            hw.volume_weights = fn
            try:
                net = get_nerf(seed=s)["net"] if fn is orig else train_nerf(cams, imgs, seed=s)
                if fn is not orig:
                    TIMES["nerf_train"].pop()
                tr = mean_psnr(render_views(net, cams, n_samples=NERF_SAMPLES)[0], imgs)
                outs = [render_views(net, tcams, n_samples=n) for n in counts]
            finally:
                hw.volume_weights = orig
            res.append([tr] + [mean_psnr(o[0], timgs) for o in outs] + [depth_error(outs[1][1], tdeps), float(outs[1][2][torch.isfinite(tdeps)].mean())])
        m = np.mean(res, axis=0)
        rows[name] = {f"训练视角 PSNR（dB，渲染时 {NERF_SAMPLES} 个采样点，和训练时相同）": r1(m[0]),
                      **{f"新视角 PSNR（dB），渲染时 {n} 个采样点": r1(m[1 + i]) for i, n in enumerate(counts)},
                      f"新视角的深度误差（{NERF_EVAL_SAMPLES} 个采样点）": r3(m[4]), "物体像素上的累积不透明度（平均）": r2(m[5])}
    gs_ok = get_gs(seed=0)
    n = GS_N
    unsorted = lambda dep: torch.arange(n, dtype=dep.dtype)  # noqa: E731
    model = train_splats(cams, imgs, n, 0, order=unsorted)
    TIMES[f"gs_train_{n}"].pop()
    NUM["三种实现错误训练出来是什么样"] = {
        "说明": f"{N_VIEWS} 个训练视角。前两种把作业的 volume_weights 换成写错的版本训练迷你 NeRF（{len(SEEDS)} 个种子取平均），训练和渲染用的是同一份错的实现；"
              f"训练时每条光线 {NERF_SAMPLES} 个采样点，渲染时分别取 {NERF_SAMPLES}、{NERF_EVAL_SAMPLES}、96 个。第三种是高斯泼溅（128 个高斯，种子 0）训练和渲染时都不按深度排序、一直按存储顺序合成",
        **rows,
        "高斯泼溅：一直按深度排序 / 一直不排序（训练和渲染都不排）": {
            "训练视角 PSNR（dB）": [r1(gs_ok["train_psnr"]), r1(mean_psnr(splat_views(model, cams, unsorted), imgs))],
            "新视角 PSNR（dB）": [r1(gs_ok["test_psnr"]), r1(mean_psnr(splat_views(model, tcams, unsorted), timgs))],
            "不排序训练出来的高斯，渲染时改成按深度排序：新视角 PSNR（dB）": r1(mean_psnr(splat_views(model, tcams), timgs)),
        },
    }


# ---------------------------------------------------------------- 其余数字
ONE_RAY = (21, 12)          # 图 2 左边两栏用的那条光线：新视角 0 的第 21 行第 12 列，依次穿过盒子、条纹球和底板
SIG_DEMO = 4.0              # 那条光线上用的半透明密度
SORT_VIEW = 2               # 图 7 左边四张图用的新视角
POSENC_TITLE = "12 个视角：L 越大新视角越好；3 个视角：L = 2 之后变差"


def demo_numbers():
    """demos/体渲染.html 的几个预设，用和页面相同的公式在这里算一遍：光线从 0 到 4，前层从 1.0 开始，后层从 2.4 到 3.2，等间隔中点采样。"""
    length, back = 4.0, (2.4, 3.2)

    def sampled(s1, th1, s2, n, p1=1.0):
        d = length / n
        t = (np.arange(n) + 0.5) * d
        lay = np.where((t >= p1) & (t < p1 + th1), 1, np.where((t >= back[0]) & (t < back[1]), 2, 0))
        alpha = 1 - np.exp(-np.where(lay == 1, s1, np.where(lay == 2, s2, 0.0)) * d)
        w = np.concatenate([[1.0], np.cumprod(1 - alpha)[:-1]]) * alpha
        return {"front": w[lay == 1].sum(), "back": w[lay == 2].sum(), "acc": w.sum(), "raw": (w * t).sum(), "depth": (w * t).sum() / max(w.sum(), 1e-10)}

    def slab(t0, a, b, sig):
        e = math.exp(-sig * (b - a))
        return t0 * (1 - e), t0 * ((a + 1 / sig) - (b + 1 / sig) * e), t0 * e

    def exact(s1, th1, s2, p1=1.0):
        f, b = slab(1.0, p1, p1 + th1, s1), None
        b = slab(f[2], back[0], back[1], s2)
        return {"front": f[0], "back": b[0], "acc": f[0] + b[0], "depth": (f[1] + b[1]) / (f[0] + b[0])}

    a, ea = sampled(1.5, 0.6, 8, 32), exact(1.5, 0.6, 8)
    b = sampled(12, 0.6, 8, 32)
    thin = {n: sampled(40, 0.05, 8, n) for n in range(4, 129)}
    d, ed = sampled(1.5, 0.6, 40, 128), exact(1.5, 0.6, 40)
    e = sampled(0.5, 0.6, 0.5, 128)
    return {
        "说明": "demos/体渲染.html 的设置：一条光线上从 0 到 4 的一段，前层介质从 1.0 开始，后层从 2.4 到 3.2，最后是背景。页面里的数值在浏览器里实时算；讲义和页面引导语里引用的数用同样的公式在这里重算了一遍，"
              "并且和用 node 直接执行页面脚本的结果核对过",
        "光线的长度": 4, "默认的采样点数": 32, "采样点数的范围": [4, 128], "后层的范围": list(back),
        "第 1 步（前层密度 1.5、厚 0.6，后层密度 8）：前层 / 后层的精确权重": [r3(ea["front"]), r3(ea["back"])],
        "第 1 步：32 个采样点给出的前层 / 后层权重": [r3(a["front"]), r3(a["back"])],
        "第 2 步（前层密度 12）：前层 / 后层的权重": [r3(b["front"]), r3(b["back"])],
        "第 3 步（前层厚 0.05、密度 40）：精确的权重": r3(1 - math.exp(-40 * 0.05)), "32 个采样点的间隔": 0.125,
        "第 3 步：32 个 / 64 个采样点给出的前层权重": [r3(thin[32]["front"]), r3(thin[64]["front"])],
        "第 3 步：4 到 128 之间完全漏掉薄层的采样点数有几个": int(sum(1 for v in thin.values() if v["front"] == 0)),
        "第 3 步：分层随机时一次抽中薄层的机会（层厚 / 间隔）": r2(0.05 / 0.125),
        "第 4 步（前层密度 1.5，后层密度 40，128 个采样点）：期望深度，采样 / 精确": [r2(d["depth"]), r2(ed["depth"])], "前层的范围": [1.0, 1.6],
        "第 5 步（两层密度都是 0.5）：权重之和": r2(e["acc"]), "第 5 步：期望深度，除以权重之和 / 不除": [r2(e["depth"]), r2(e["raw"])],
    }


def numbers_misc():
    one = torch.tensor([[math.log(2.0)] * 3], dtype=torch.float64)
    w = hw.volume_weights(one, torch.ones(3, dtype=torch.float64))
    enc = hw.positional_encoding(torch.tensor([[0.5, -0.25]], dtype=torch.float64), 2)
    NUM["模型规模"] = {
        "说明": "迷你 NeRF 的参数个数随位置编码的频率个数 L 变（只有第一层的输入维数不同）；高斯泼溅每个高斯 14 个数（中心 3、轴长 3、四元数 4、不透明度 1、颜色 3）",
        "迷你 NeRF 的参数个数：L = 0、2、4、8": [sum(p.numel() for p in TinyNeRF(L).parameters()) for L in PE_LS],
        "高斯泼溅的参数个数：16、32、64、128、256 个高斯": [14 * n for n in GS_COUNTS],
        "每个高斯的参数个数": 14, "训练图的像素数（12 个视角）": N_VIEWS * SIZE * SIZE,
    }
    NUM["理论数值"] = {
        "说明": "由公式直接算出的数，不是实验结果",
        "相机朝向偏 1 度时画面中心的像移（px）= fx x tan(1 度)": r3(FX * math.tan(math.radians(1.0))),
        "相机位置横向偏 0.032（相机距离的 1%）时场景中心的像移（px）": r3(FX * 0.01),
        "一个像素在场景中心对应的宽度": r3(RADIUS / FX), "半视场角（度）= atan(16 / fx)": r1(math.degrees(math.atan(16 / FX))),
        "同样 1 度的朝向误差，画面宽 1000 px、视场角不变时的像移（px）": r1(FX * 1000 / SIZE * math.tan(math.radians(1.0))),
        "一段的光学厚度（密度 x 间隔）为 ln 2 时的不透明度": 0.5, "ln 2": r3(math.log(2.0)),
        "连续三段都是这样时的权重": [r3(v) for v in w[0]], "三段之后剩下的透射率": r3(1 - float(w.sum())),
        "密度 40、间隔 0.05 时一段的不透明度": r3(1 - math.exp(-SIG_SOLID * 0.05)), "密度 40 时光走多远透射率降到 1/e": r3(1 / SIG_SOLID),
        "位置编码输出的维数：3 维坐标、L = 4 / L = 10": [3 * (1 + 2 * 4), 3 * (1 + 2 * 10)],
        "x = 0.5、L = 2 的编码（保留原始坐标）": [r3(v) for v in enc[0, ::2]],
    }
    NUM["论文和官方实现里的设定（引用，不是实测）"] = {
        "说明": "讲义里引用的几个设定，出自各自的论文和官方仓库，以它们为准。放在这里只是为了让数字核对能过",
        "NeRF 论文：位置的频率个数 / 方向的频率个数": [10, 4], "NeRF 论文：粗采样 / 细采样的点数": [64, 128], "NeRF 论文：全连接层的层数 / 宽度": [8, 256],
        "3D Gaussian Splatting 官方实现：二维协方差对角线加的值（像素平方）": 0.3, "不透明度的上限": 0.99, "透射率低于这个值就停止合成": 0.0001,
        "球谐函数的最高阶数": 3, "损失里 D-SSIM 一项的权重": 0.2, "低于这个不透明度的贡献被跳过（1/255）": r3(1 / 255),
    }
    NUM["引用别的章（不是本章实测）"] = {
        "说明": "讲义里引用的别的章的数，出处是那一章的 实测数字.json 或讲义正文，抄在这里只是为了让数字核对能过",
        "第 22 章 2.2 节和 4.3 节：波长（nm）、NA、离焦 0.5 λ/NA² 对应的距离（µm）": [266, 0.9, 0.164],
        "第 17 章第 6 节：好的标定的重投影误差通常在（px）": [0.1, 0.5],
    }
    NUM["演示页"] = demo_numbers()
    NUM["作业自检"] = {
        "说明": "用参考实现核对的几个性质",
        "三段光学厚度都是 ln 2 时权重之和": r3(float(w.sum())),
        "把每一段再平分成两段（密度不变），权重之和的变化": float(f"{abs(float(hw.volume_weights(one.repeat_interleave(2, 1), torch.full((6,), 0.5, dtype=torch.float64)).sum() - w.sum())):.1g}"),
        "L = 2、保留原始坐标时，2 维输入的编码维数": int(enc.shape[-1]),
    }


def main() -> None:
    S.setup()
    plt.rcParams["svg.hashsalt"] = "ch30"
    t_all = time.perf_counter()
    for fn in (numbers_misc, fig_overview, fig_volume, fig_posenc, fig_views_pose, fig_splat, numbers_steps, numbers_bugs, fig_sort_time):
        t = time.perf_counter()
        fn()
        TIMES[fn.__name__] = time.perf_counter() - t
        print(f"完成 {fn.__name__}（{TIMES[fn.__name__]:.0f} 秒）", flush=True)
    total = time.perf_counter() - t_all
    NUM["训练规模"] = {
        "说明": "整个脚本一共训练了多少个模型",
        "迷你 NeRF": sum(1 for k in _CACHE if k[0] == "nerf"), "高斯泼溅": sum(1 for k in _CACHE if k[0] == "gs"), "二维图像拟合的小网络": len(FIT_LS) * len(SEEDS),
    }
    if "整个配图脚本（秒）" not in NUM["耗时"]:
        NUM["耗时"]["整个配图脚本（秒）"] = float(f"{total:.2g}")
    out = S.MAIN / CH / "figures" / "实测数字.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(NUM, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(NUM, ensure_ascii=False, indent=2))
    print(f"总耗时 {total:.0f} 秒")


if __name__ == "__main__":
    main()
