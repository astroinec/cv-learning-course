"""
题 1-1：直方图均衡化（从零实现）

任务：不调 cv2.equalizeHist()，自己写。
  1. 读入灰度图
  2. 统计 0-255 每个灰度级的像素数
  3. 计算累积分布函数（CDF）
  4. 映射回 [0,255]
  5. 用 plt.subplot 对比原图和均衡化后的图 + 各自的直方图

学什么：图像的统计特性、CDF 为什么能拉伸对比度。
数学点：离散概率分布 → 累积分布 → 均匀化映射。
"""

from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════
# STEP 1：生成测试图（没有图可以直接用这个）
# ═══════════════════════════════════════════════════════════

def create_test_image():
    """生成一张欠曝 + 局部高亮的测试图，适合验证直方图均衡化"""
    img = np.zeros((400, 500), dtype=np.uint8)
    # 大面积暗区
    img[0:300, 0:300] = np.random.randint(20, 60, (300, 300))
    # 中间有一块亮区
    img[100:300, 200:500] = np.random.randint(160, 220, (200, 300))
    # 一个小高亮块
    img[50:100, 350:450] = 240
    return img


# ═══════════════════════════════════════════════════════════
# STEP 2：实现直方图均衡化（在这里动手写！）
# ═══════════════════════════════════════════════════════════

def histogram_equalization(img_gray):
    """
    输入：灰度图 (H, W)，dtype=uint8
    输出：均衡化后的灰度图 (H, W)，dtype=uint8

    提示：
      np.histogram(img.flatten(), 256, [0,256])  → (hist, bin_edges)
      hist.cumsum()                               → cdf
      cdf_normalized = cdf * 255 / cdf[-1]       → 映射到 [0,255]
      用 img 的值作为索引查表
    """
    # ---- 你的代码 ----
    pass
    # ---- 你的代码 ----


# ═══════════════════════════════════════════════════════════
# STEP 3：运行 & 对比
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    # 路径相对脚本文件，而不是相对终端当前目录
    img_path = Path(__file__).resolve().parents[1] / "images" / "test_underexposed.jpg"
    original = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if original is None:
        print("没有找到测试图，自动生成一张...")
        original = create_test_image()

    # 均衡化
    equalized = histogram_equalization(original)

    # 显示
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('原图')
    axes[0, 1].imshow(equalized, cmap='gray')
    axes[0, 1].set_title('均衡化后')

    axes[1, 0].hist(original.flatten(), 256, [0, 256], color='black')
    axes[1, 0].set_title('原图直方图')
    axes[1, 1].hist(equalized.flatten(), 256, [0, 256], color='black')
    axes[1, 1].set_title('均衡化后直方图')

    plt.tight_layout()
    plt.show()

    # 检验：均衡化后的直方图是不是比原图更"均匀"了？
