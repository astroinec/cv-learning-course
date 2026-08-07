"""
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
    B = ...
    G = ...
    R = ...
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
    """把指定通道置零，然后合成回去"""
    # 复制原图
    modified = img_bgr.copy()

    # 提示：OpenCV BGR 顺序 —— modified[:,:,0]=B, modified[:,:,1]=G, modified[:,:,2]=R
    if channel == 'R':
        modified[:, :, 2] = 0    # R 通道置零 → 画面偏什么颜色？
    elif channel == 'G':
        raise NotImplementedError("请实现 G 通道分支")
    elif channel == 'B':
        raise NotImplementedError("请实现 B 通道分支")

    return modified


# ═══════════════════════════════════════════════════════════
# 任务 3（延伸）：HSV 空间单独拉伸亮度
# ═══════════════════════════════════════════════════════════

def enhance_brightness_hsv(img_bgr, v_scale=1.5):
    """在 HSV 空间拉伸 V 通道（亮度），不改变色调和饱和度"""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    # 提示：hsv[:,:,2] 就是 V（Value）通道
    raise NotImplementedError("请实现 V 通道缩放")

    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# ═══════════════════════════════════════════════════════════
# 运行
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    img_path = Path(__file__).resolve().parents[1] / "images" / "test_color.jpg"
    img = cv2.imread(str(img_path))
    if img is None:
        print("没有找到测试图，自动生成一张...")
        img = create_test_color_image()

    print("=== 1. 三通道分离 ===")
    show_channels(img)

    print("\n=== 2. 去掉 R 通道（变青蓝色调）===")
    modified = remove_channel_effect(img, 'R')
    plt.imshow(cv2.cvtColor(modified, cv2.COLOR_BGR2RGB))
    plt.title('去掉 R 通道')
    plt.show()

    # 试试去掉 G 和 B 分别是什么效果？自己改参数运行。

    print("\n=== 3. HSV 拉伸亮度 ===")
    bright = enhance_brightness_hsv(img)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_title('原图')
    ax2.imshow(cv2.cvtColor(bright, cv2.COLOR_BGR2RGB))
    ax2.set_title('HSV-V 拉伸 1.5×')
    plt.show()

    # 思考：如果直接在 BGR 上缩放所有通道会怎样？和 HSV 方式有什么区别？
