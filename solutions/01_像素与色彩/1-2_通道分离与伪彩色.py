"""参考实现（学员完成本章前不要查看）。
题 1-2：通道分离与伪彩色

任务：
  1. 读入彩色图 → 分离 B、G、R 三通道
  2. 每个通道单独显示为灰度图（3 个子图）
  3. 将其中一个通道替换为全 0 → 合成回去 → 观察颜色偏移
  4. 延伸：BGR → HSV，拉伸 V 通道后再转回 BGR

学什么：
  - OpenCV 默认 BGR 不是 RGB！这是个千古大坑
  - HSV 的 V 通道独立于颜色，做亮度处理不会改变色调
  - 通道之间信息是冗余的——缺失一个通道后颜色偏移的方向揭示了各通道的角色
"""

from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


def read_image(path, flags):
    """读图片文件，文件不存在时返回 None。先读成字节再解码，路径里有中文时在 Windows 上也能读。"""
    if not path.exists():
        return None
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)


def create_test_color_image():
    """生成一张彩色测试图：红绿蓝三个色块 + 一个渐变"""
    img = np.zeros((300, 500, 3), dtype=np.uint8)
    img[0:300, 0:166, 2] = 255      # R：红色块
    img[0:300, 167:333, 1] = 255    # G：绿色块
    img[0:300, 334:500, 0] = 255    # B：蓝色块
    img[50:250, 50:450] = np.linspace(0, 255, 400)[None, :, None].astype(np.uint8)  # 叠加渐变
    return img


# ═══════════════════════════════════════════════════════════
# 任务 1：通道分离与显示
# ═══════════════════════════════════════════════════════════

def show_channels(img_bgr):
    """把 B、G、R 三通道分别显示为灰度图"""
    B = img_bgr[:, :, 0]
    G = img_bgr[:, :, 1]
    R = img_bgr[:, :, 2]
    if any(channel is Ellipsis for channel in (B, G, R)):
        raise NotImplementedError("请先实现 B、G、R 三通道分离")

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title('原图 (BGR→RGB显示)')
    axes[1].imshow(B, cmap='gray')
    axes[1].set_title('B 通道')
    axes[2].imshow(G, cmap='gray')
    axes[2].set_title('G 通道')
    axes[3].imshow(R, cmap='gray')
    axes[3].set_title('R 通道')
    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════
# 任务 2：去掉一个通道，看颜色偏移
# ═══════════════════════════════════════════════════════════

def remove_channel_effect(img_bgr, channel='R'):
    """把指定通道置零，然后合成回去。

    输入：img_bgr 为 (H, W, 3) 的 uint8 图，通道顺序是 OpenCV 的 B、G、R；channel 取 'B'、'G' 或 'R'。
    输出：同 shape 的 uint8 新数组，指定的通道全为 0，另外两个通道和输入相同。
    约定：不修改传进来的 img_bgr（下面先复制再改，就是为了这一条）。
    """
    # 复制原图
    modified = img_bgr.copy()

    # 提示：OpenCV BGR 顺序 —— modified[:,:,0]=B, modified[:,:,1]=G, modified[:,:,2]=R
    if channel == 'R':
        modified[:, :, 2] = 0    # R 通道置零 → 画面偏什么颜色？
    elif channel == 'G':
        modified[:, :, 1] = 0
    elif channel == 'B':
        modified[:, :, 0] = 0

    return modified


# ═══════════════════════════════════════════════════════════
# 任务 3（延伸）：HSV 空间单独拉伸亮度
# ═══════════════════════════════════════════════════════════

def enhance_brightness_hsv(img_bgr, v_scale=1.5):
    """在 HSV 空间拉伸 V 通道（亮度），不改变色调和饱和度。

    输入：img_bgr 为 (H, W, 3) 的 uint8 BGR 图；v_scale 是亮度倍数，大于等于 0。
    输出：同 shape 的 uint8 BGR 图；不修改输入。
    约定（验收按这几条检查）：
      1. 输出图的 V 等于输入图的 V 乘 v_scale；乘完超过 255 的截到 255，不能回绕成很小的数。
      2. H 和 S 不变（允许 HSV 量化带来的一两个单位的误差）。
         所以直接把 B、G、R 三个通道乘 v_scale 再截断不符合要求：发生截断的像素色调会变。
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    # 提示：hsv[:,:,2] 就是 V（Value）通道
    hsv[:, :, 2] *= v_scale

    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# ═══════════════════════════════════════════════════════════
# 运行
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    # 中文字体：Windows 的两种放在前面，找不到时依次往后试
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "PingFang SC", "Heiti SC", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    img_path = Path(__file__).resolve().parents[1] / "images" / "test_color.jpg"
    img = read_image(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print("没有找到 images/test_color.jpg，改用代码生成的测试图（要用文件版本，先运行 scripts/generate_sample_images.py）")
        img = create_test_color_image()

    print("=== 1. 三通道分离 ===")
    try:
        show_channels(img)
    except NotImplementedError as exc:
        print(f"TODO: {exc}。先完成任务 1，再运行本文件。")
        raise SystemExit(0)

    print("\n=== 2. 去掉 R 通道（变青蓝色调）===")
    modified = remove_channel_effect(img, 'R')
    plt.imshow(cv2.cvtColor(modified, cv2.COLOR_BGR2RGB))
    plt.title('去掉 R 通道')
    plt.show()

    # 试试去掉 G 和 B 分别是什么效果？自己改参数运行。

    print("\n=== 3. HSV 拉伸亮度 ===")
    try:
        bright = enhance_brightness_hsv(img)
    except NotImplementedError as exc:
        print(f"TODO: {exc}。先完成任务 3，再运行本文件。")
        raise SystemExit(0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_title('原图')
    ax2.imshow(cv2.cvtColor(bright, cv2.COLOR_BGR2RGB))
    ax2.set_title('HSV-V 拉伸 1.5×')
    plt.show()

    # 思考：如果直接在 BGR 上缩放所有通道会怎样？和 HSV 方式有什么区别？
