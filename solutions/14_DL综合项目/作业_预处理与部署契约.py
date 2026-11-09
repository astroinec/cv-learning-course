"""参考实现（学员完成本章前不要查看）。"""

from __future__ import annotations

import numpy as np


def letterbox_metadata(src_hw, dst_hw):
    sh, sw = src_hw
    dh, dw = dst_hw
    scale = min(dh / sh, dw / sw)
    nh, nw = min(dh, int(round(sh * scale))), min(dw, int(round(sw * scale)))
    return {"scale": float(scale), "resized_hw": (nh, nw), "pad_xy": ((dw - nw) // 2, (dh - nh) // 2)}


def boxes_to_original(boxes_xyxy, metadata, src_hw):
    b = np.array(boxes_xyxy, dtype=np.float64, copy=True)
    px, py = metadata["pad_xy"]
    s = float(metadata["scale"])
    b[:, [0, 2]] = (b[:, [0, 2]] - px) / s
    b[:, [1, 3]] = (b[:, [1, 3]] - py) / s
    h, w = src_hw
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, w)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, h)
    return b
