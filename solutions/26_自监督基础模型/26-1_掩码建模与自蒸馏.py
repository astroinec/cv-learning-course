"""参考实现 26-1。学员完成本章前不要查看。"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def random_masking(x, mask_ratio, noise):
    if x.dim() != 3 or noise.shape != x.shape[:2]:
        raise ValueError("x 应为 (B, L, D)，noise 应为 (B, L)")
    if not 0.0 <= mask_ratio < 1.0:
        raise ValueError("mask_ratio 必须在 [0, 1) 内")
    b, n, d = x.shape
    n_keep = n - int(round(n * mask_ratio))
    if n_keep < 1:
        raise ValueError("遮挡比例太高，一个块都没留下")
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :n_keep]
    x_keep = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, d))
    mask = (ids_restore >= n_keep).to(x.dtype)
    return x_keep, mask, ids_restore


def masked_mse_loss(pred, target, mask, norm_pix=False, eps=1e-6):
    if pred.shape != target.shape or mask.shape != pred.shape[:2]:
        raise ValueError("pred 和 target 应同为 (B, L, P)，mask 应为 (B, L)")
    total = mask.sum()
    if float(total) == 0.0:
        raise ValueError("mask 全为 0，没有被遮住的块")
    if norm_pix:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + eps).sqrt()
    per_patch = ((pred - target) ** 2).mean(dim=-1)
    return (per_patch * mask).sum() / total


@torch.no_grad()
def ema_update(teacher, student, momentum):
    if not 0.0 <= momentum <= 1.0:
        raise ValueError("momentum 必须在 [0, 1] 内")
    t_params, s_params = list(teacher.parameters()), list(student.parameters())
    if len(t_params) != len(s_params):
        raise ValueError("教师和学生的参数个数不同")
    for t, s in zip(t_params, s_params):
        t.mul_(momentum).add_(s.detach(), alpha=1.0 - momentum)


def dino_loss(student_logits, teacher_logits, center, student_temp=0.1, teacher_temp=0.04):
    if student_temp <= 0 or teacher_temp <= 0:
        raise ValueError("温度必须大于 0")
    target = F.softmax((teacher_logits.detach() - center.detach()) / teacher_temp, dim=-1)
    log_p = F.log_softmax(student_logits / student_temp, dim=-1)
    return -(target * log_p).sum(dim=-1).mean()


def knn_classify(train_feats, train_labels, test_feats, k, num_classes, temperature=0.07):
    a = np.asarray(train_feats, dtype=np.float64)
    q = np.asarray(test_feats, dtype=np.float64)
    y = np.asarray(train_labels).astype(np.int64)
    if k < 1 or k > len(a):
        raise ValueError("k 必须在 1 到训练样本数之间")
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    sim = q @ a.T
    idx = np.argpartition(-sim, k - 1, axis=1)[:, :k]
    w = np.exp(np.take_along_axis(sim, idx, axis=1) / temperature)
    votes = np.zeros((len(q), num_classes), dtype=np.float64)
    np.add.at(votes, (np.arange(len(q))[:, None], y[idx]), w)
    return votes.argmax(axis=1).astype(np.int64)
