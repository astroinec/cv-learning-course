"""作业 25-1：ViT 的核心零件。

只允许使用 torch（和 Python 标准库的 math）。五个函数都是纯函数：不修改输入，不保存状态。
完成后在 Notebook 里用它们拼出一个很小的 ViT，不需要下载任何权重。

约定：
- 图像张量是 (B, C, H, W)，标记（token）序列是 (B, N, D)。
- 所有函数对任意浮点 dtype 都要能用，输出的 dtype 和输入相同（位置编码除外，见它的说明）。
- 输入可能不是内存连续的（例如刚从 HWC 用 permute 转过来的图像），也要能处理。
- 全程用 torch 运算，结果要能反向传播：不要转成 NumPy，不要 detach。
- 掩码是 bool 张量，True 表示“允许关注”，False 表示“屏蔽”。
  这和 torch.nn.functional.scaled_dot_product_attention 的 bool 掩码一致，
  和 torch.nn.MultiheadAttention 的 attn_mask 相反（那里 True 表示屏蔽）。
"""

from __future__ import annotations

import torch


def patchify(images: torch.Tensor, patch: int) -> torch.Tensor:
    """把图像切成不重叠的方块，每块展平成一个向量。

    images: (B, C, H, W)。patch: 块的边长 P，正整数。
    返回: (B, N, C*P*P)，N = (H/P) * (W/P)，dtype 和 images 相同。

    顺序约定（两条都要满足）：
    1. 块的顺序是行优先：先排第 0 行的块（从左到右），再排第 1 行。
       第 n 个块在第 n // (W/P) 行、第 n % (W/P) 列，覆盖像素
       images[:, :, 行*P:(行+1)*P, 列*P:(列+1)*P]。
    2. 块内部按 (C, P, P) 的顺序展平：先通道，再块内的行，再块内的列。
       也就是第 n 个标记等于上面那块子张量 (C, P, P) 直接 reshape(-1)。
       这个顺序和 nn.Conv2d(C, D, P, stride=P) 的权重展平顺序一致，
       所以“切块后接线性层”和“步长等于核大小的卷积”可以互相替换。

    H 或 W 不能被 P 整除时抛出 ValueError。不允许修改 images。
    """
    raise NotImplementedError("TODO: 实现 patchify")


def unpatchify(tokens: torch.Tensor, patch: int, channels: int, height: int, width: int) -> torch.Tensor:
    """patchify 的逆运算：把 (B, N, C*P*P) 的标记拼回 (B, C, H, W) 的图像。

    顺序约定和 patchify 完全相同，所以对任意图像都有
    unpatchify(patchify(x, P), P, C, H, W) == x（逐元素相等）。
    注意 H 和 W 可以不相等。

    下面任何一条不满足时抛出 ValueError：
    height、width 能被 patch 整除；N == (height/patch) * (width/patch)；
    tokens 最后一维 == channels * patch * patch。
    """
    raise NotImplementedError("TODO: 实现 unpatchify")


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """缩放点积注意力。

    q: (..., Lq, d)，k: (..., Lk, d)，v: (..., Lk, dv)。前面的 ... 是任意个批维度，三者相同。
    Lq 和 Lk 可以不相等，dv 和 d 可以不相等。
    mask: None，或者能广播到 (..., Lq, Lk) 的 bool 张量，True 表示这个查询可以关注这个键。

    计算：分数 = q 和 k 的点积除以 sqrt(d)（d 是 q 最后一维的长度）；
    被屏蔽的位置不参与 softmax；权重 = 对每个查询在 Lk 这一维上做 softmax；输出 = 权重乘 v。

    返回 (out, weights)：out 是 (..., Lq, dv)，weights 是 (..., Lq, Lk)。
    要求：
    - 每个查询的权重非负、和为 1，被屏蔽位置的权重严格等于 0。
    - 某个查询的所有键都被屏蔽时，这一行的权重规定为全 0（不能是 nan），对应的输出也是全 0。
    - mask 为 None 时所有位置都参与。
    """
    raise NotImplementedError("TODO: 实现缩放点积注意力")


def multi_head_attention(
    x: torch.Tensor,
    w_qkv: torch.Tensor,
    b_qkv: torch.Tensor,
    w_out: torch.Tensor,
    b_out: torch.Tensor,
    num_heads: int,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """多头自注意力的前向计算（权重由调用者给出，本函数不创建参数）。

    x: (B, L, D)。
    w_qkv: (3D, D)，b_qkv: (3D,)。线性层 y = x @ w_qkv.T + b_qkv 的输出 (B, L, 3D) 里，
        前 D 维是查询 q，中间 D 维是键 k，最后 D 维是值 v。
    w_out: (D, D)，b_out: (D,)。把各个头的结果拼回 D 维之后再过的输出线性层。
    num_heads: 头数 H，D 必须能被 H 整除，否则抛出 ValueError。
    mask: None，或者 (L, L)、(B, L, L) 的 bool 张量，True 表示允许关注；所有头共用同一个掩码。

    计算：
    1. 用 w_qkv、b_qkv 得到 q、k、v，各 (B, L, D)。
    2. 把最后一维按顺序切成 H 段，每段 D/H 维：第 h 个头用第 h*D/H 到 (h+1)*D/H 维。
    3. 每个头各自做缩放点积注意力，缩放用的是每个头的维数 D/H，不是 D。
    4. 把 H 个头的输出按头的顺序在最后一维拼回 (B, L, D)，再过输出线性层。

    返回 (out, attn)：out 是 (B, L, D)；attn 是 (B, H, L, L)，
    attn[b, h, i, j] 是第 h 个头里第 i 个标记对第 j 个标记的注意力权重，不对头取平均。

    权重的排列方式和 torch.nn.MultiheadAttention 的 in_proj_weight、in_proj_bias、
    out_proj.weight、out_proj.bias 相同，可以拿它来核对自己的结果（注意它的掩码含义相反）。
    """
    raise NotImplementedError("TODO: 实现多头注意力的前向")


def sincos_2d_position_embedding(dim: int, grid_h: int, grid_w: int, temperature: float = 10000.0) -> torch.Tensor:
    """固定的二维正余弦位置编码。

    返回 (grid_h * grid_w, dim) 的 float32 张量，第 n 行对应第 n // grid_w 行、第 n % grid_w 列的块
    （和 patchify 的块顺序相同）。

    记 m = dim / 4，频率 omega_t = 1 / temperature ** (t / m)，t = 0, 1, ..., m-1。
    位置（行 r，列 c）的编码是四段各 m 维按下面的顺序拼接：

        [ sin(r * omega_0..m-1), cos(r * omega_0..m-1), sin(c * omega_0..m-1), cos(c * omega_0..m-1) ]

    也就是前一半编码行号，后一半编码列号；每一半里先是全部 sin，再是全部 cos，不交错。
    r、c 从 0 开始，是整数的块下标，不做归一化。

    dim 不是 4 的正整数倍，或者 grid_h、grid_w 小于 1 时抛出 ValueError。
    建议中间用 float64 计算，最后再转成 float32。
    """
    raise NotImplementedError("TODO: 实现二维正余弦位置编码")
