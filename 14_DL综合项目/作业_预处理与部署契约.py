"""第 14 章作业：letterbox 几何与部署输出契约。"""

from __future__ import annotations

import numpy as np


def letterbox_metadata(src_hw: tuple[int, int], dst_hw: tuple[int, int]) -> dict[str, object]:
    """计算保持纵横比缩放到目标画布所需的元数据。

    src_hw/dst_hw: (height, width)，均为正整数。
    输出字典必须含：
      scale: 正 float；
      resized_hw: (new_h, new_w)，不超过 dst_hw；
      pad_xy: (pad_left, pad_top)，非负整数。
    奇数 padding 的多余 1 像素放到右侧或下侧。
    """
    # TODO: 使用 min(dst_h/src_h, dst_w/src_w)，四舍五入得到缩放尺寸。
    raise NotImplementedError("TODO: 实现 letterbox_metadata")


def boxes_to_original(boxes_xyxy: np.ndarray, metadata: dict[str, object], src_hw: tuple[int, int]) -> np.ndarray:
    """把 letterbox 画布上的框映射回原图并裁剪边界。

    boxes_xyxy: float array, shape (N, 4)，连续 xyxy 坐标。
    metadata: letterbox_metadata 的输出。
    src_hw: 原图 (height, width)。
    输出：float array，同 shape；不得原地修改输入。
    """
    # TODO: 减去左/上 padding，再除 scale，最后裁剪到原图边界。
    raise NotImplementedError("TODO: 实现 boxes_to_original")


if __name__ == "__main__":
    print("完成 letterbox 与坐标回映射 TODO 后运行检查器。")
