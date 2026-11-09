"""参考实现 20-1。学员完成本章前不要查看。"""

from __future__ import annotations

import numpy as np


def scratch_mask(shape, p0, p1, width, softness=1.0):
    if softness <= 0:
        raise ValueError("softness 必须大于 0")
    h, w = shape
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    ay, ax = map(float, p0)
    by, bx = map(float, p1)
    vy, vx = by - ay, bx - ax
    L2 = vy * vy + vx * vx
    if L2 < 1e-12:
        d = np.hypot(y - ay, x - ax)
    else:
        t = np.clip(((y - ay) * vy + (x - ax) * vx) / L2, 0.0, 1.0)
        d = np.hypot(y - (ay + t * vy), x - (ax + t * vx))
    return np.clip((width / 2.0 + softness / 2.0 - d) / softness, 0.0, 1.0)


def blend(background, foreground, mask):
    m = np.asarray(mask, dtype=np.float64)
    if m.min() < 0 or m.max() > 1:
        raise ValueError("mask 必须在 [0, 1]")
    bg = np.asarray(background, dtype=np.float64)
    return bg * (1.0 - m) + np.asarray(foreground, dtype=np.float64) * m


def linear_beta_schedule(steps, beta_start=1e-4, beta_end=0.02):
    return np.linspace(beta_start, beta_end, steps, dtype=np.float64)


def alpha_bars(betas):
    return np.cumprod(1.0 - np.asarray(betas, dtype=np.float64))


def q_sample(x0, t, abar, noise):
    if t < 0 or t >= len(abar):
        raise IndexError("t 越界")
    return np.sqrt(abar[t]) * np.asarray(x0, dtype=np.float64) + np.sqrt(1.0 - abar[t]) * noise


def reparameterize(mu, logvar, eps):
    return np.asarray(mu) + np.exp(0.5 * np.asarray(logvar)) * np.asarray(eps)
