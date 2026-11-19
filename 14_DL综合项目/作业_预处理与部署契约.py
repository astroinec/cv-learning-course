"""第 14 章作业：letterbox 几何与部署输出契约。"""

from __future__ import annotations

import numpy as np


def letterbox_metadata(src_hw: tuple[int, int], dst_hw: tuple[int, int]) -> dict[str, object]:
    """计算保持纵横比缩放到目标画布所需的元数据。

    src_hw/dst_hw: (height, width)，均为正整数。
    输出字典必须含（可以有别的字段）：
      scale: 正 float，等于 min(dst_h / src_h, dst_w / src_w)，放大和缩小都允许；
      resized_hw: (new_h, new_w)，两个 int，不超过 dst_hw；
      pad_xy: (pad_left, pad_top)，非负 int。

    取整规则：new_h、new_w 是 src_h * scale、src_w * scale 取到最近的整数。
    正好落在 .5 上时取最近的偶数，也就是 Python 内置 round 的行为
    （np.round、np.rint 相同）：round(360.5) 是 360，round(361.5) 是 362。
    不是“逢 .5 进一”（C++ 的 std::round、math.floor(x + 0.5) 会得到 361 和 362），
    也不是截断。取整后如果超过画布，就取画布的尺寸。

    加边：两侧一共要加 dst - new 个像素，左（上）侧加其中的一半，向下取整。
    差是奇数时多出来的 1 个像素放到右侧或下侧，所以 pad_left、pad_top 是较小的那一半。
    """
    # TODO: 使用 min(dst_h/src_h, dst_w/src_w)，按上面的取整规则得到缩放尺寸。
    raise NotImplementedError("TODO: 实现 letterbox_metadata")


def boxes_to_original(boxes_xyxy: np.ndarray, metadata: dict[str, object], src_hw: tuple[int, int]) -> np.ndarray:
    """把 letterbox 画布上的框映射回原图并裁剪边界。

    boxes_xyxy: float array, shape (N, 4)，连续 xyxy 坐标；N 可以是 0。
    metadata: letterbox_metadata 的输出，只用其中的 scale 和 pad_xy。
    src_hw: 原图 (height, width)。
    输出：float array，同 shape；不得原地修改 boxes_xyxy 和 metadata。
    结果是连续坐标，不取整。裁剪范围：x 在 [0, 原图宽]，y 在 [0, 原图高]（不减 1）。
    """
    # TODO: 减去左/上 padding，再除 scale，最后裁剪到原图边界。
    raise NotImplementedError("TODO: 实现 boxes_to_original")


if __name__ == "__main__":
    print("完成 letterbox 与坐标回映射 TODO 后运行检查器。")
