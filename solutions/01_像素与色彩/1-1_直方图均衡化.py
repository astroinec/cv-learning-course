"""参考实现（学员完成本章前不要查看）。
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


def read_image(path, flags):
    """读图片文件，文件不存在时返回 None。先读成字节再解码，路径里有中文时在 Windows 上也能读。"""
    if not path.exists():
        return None
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)


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
    输出：均衡化后的灰度图 (H, W)，dtype=uint8；不修改输入数组

    约定（验收按这几条检查）：
      1. 映射和 cv2.equalizeHist 相同：图里最暗的灰度映射到 0，最亮的映射到 255，
         中间的灰度按下面提示里的公式映射，结果四舍五入成整数。
      2. 全图只有一个灰度（纯色图）时，公式的分母是 0。规定这时输出和输入逐像素相同，
         不能除以零。cv2.equalizeHist 对纯色图也是原样输出。

    提示：
      np.histogram(img.flatten(), 256, [0,256])  → (hist, bin_edges)
      hist.cumsum()                               → cdf
      cdf_min = cdf 中第一个非零值（最暗像素对应的累计数）
      lut = round((cdf - cdf_min) * 255 / (cdf[-1] - cdf_min))   → 最暗映射到 0，最亮映射到 255
      用 img 的值作为索引查表；纯色图按约定 2 单独处理
    注意：直接用 cdf * 255 / cdf[-1] 时最暗像素不会映射到 0，和 cv2.equalizeHist 不一致，验收会 FAIL。
      查找表转成 uint8 之前要先四舍五入，直接 astype 是截断，会系统性偏低（讲义 1.1 节）。
    """
    hist = np.bincount(img_gray.ravel(), minlength=256)
    cdf = hist.cumsum()
    cdf_min = cdf[cdf > 0][0]
    total = cdf[-1]
    if total == cdf_min:
        return img_gray.copy()
    lut = np.round((cdf - cdf_min) * 255.0 / (total - cdf_min))
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return lut[img_gray]


# ═══════════════════════════════════════════════════════════
# STEP 3：运行 & 对比
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    # 中文字体：Windows 的两种放在前面，找不到时依次往后试
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "PingFang SC", "Heiti SC", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    # 路径相对脚本文件，而不是相对终端当前目录
    img_path = Path(__file__).resolve().parents[1] / "images" / "test_underexposed.jpg"
    original = read_image(img_path, cv2.IMREAD_GRAYSCALE)
    if original is None:
        print("没有找到 images/test_underexposed.jpg，改用代码生成的测试图（要用文件版本，先运行 scripts/generate_sample_images.py）")
        original = create_test_image()

    # 均衡化
    equalized = histogram_equalization(original)
    if equalized is None:
        print("histogram_equalization 还没有实现（仍返回 None）。先完成 STEP 2，再运行本文件。")
        raise SystemExit(0)

    # 显示
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('原图')
    axes[0, 1].imshow(equalized, cmap='gray')
    axes[0, 1].set_title('均衡化后')

    axes[1, 0].hist(original.flatten(), bins=256, range=(0, 256), color='black')
    axes[1, 0].set_title('原图直方图')
    axes[1, 1].hist(equalized.flatten(), bins=256, range=(0, 256), color='black')
    axes[1, 1].set_title('均衡化后直方图')

    plt.tight_layout()
    plt.show()

    # 检验：均衡化后的直方图是不是比原图更"均匀"了？
