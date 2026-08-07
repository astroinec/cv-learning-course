"""
题 1-3：手写椒盐噪声 + 中值滤波理解

任务：
  1. 给一张图随机加 5% 的椒盐噪声
  2. 用 3×3 和 5×5 的中值滤波（cv2.medianBlur）分别去噪 → 对比
  3. 自己实现中值滤波：对每个像素取邻域的中位数 → 和 OpenCV 的结果对比

学什么：
  - 中值滤波为什么能去椒盐而均值/Gaussian 不行？
  - 非线性滤波 vs 线性滤波的本质
  - 窗口大小的 trade-off：大窗口去噪强但模糊细节

数学点：中位数对 outliers 的鲁棒性——概率课上讲过，现在看到了。
"""

from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════
# STEP 1：椒盐噪声生成器
# ═══════════════════════════════════════════════════════════

def add_salt_pepper_noise(img, amount=0.05):
    """
    给灰度图添加椒盐噪声。
    amount: 噪声比例（比如 0.05 = 5%）
    椒 = 黑点 (0)，盐 = 白点 (255)

    提示：
      np.random.random(img.shape) < amount/2  → 椒的位置（mask）
      np.random.random(img.shape) < amount/2  → 盐的位置
      椒位置置 0，盐位置置 255
    """
    noisy = img.copy()
    # ---- 你的代码 ----
    pass
    # ---- 你的代码 ----
    return noisy


# ═══════════════════════════════════════════════════════════
# STEP 2：手写中值滤波
# ═══════════════════════════════════════════════════════════

def median_filter_manual(img, kernel_size=3):
    """
    不调 cv2.medianBlur，自己实现。

    提示：
      遍历每个像素（跳过边缘 pad）
      取 k*k 邻域 → np.median()
      写入输出图

    复杂度提示：np.median 比较慢但没关系，先写对。
    如果想加速，对于 3×3 可以用排序网络。
    """
    h, w = img.shape
    pad = kernel_size // 2
    output = np.zeros_like(img)

    # ---- 你的代码 ----
    pass
    # ---- 你的代码 ----
    return output


# ═══════════════════════════════════════════════════════════
# STEP 3：对比实验
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    # 路径相对脚本文件，而不是相对终端当前目录
    img_path = Path(__file__).resolve().parents[1] / "images" / "test_underexposed.jpg"
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        # 自动生成
        img = np.zeros((300, 400), dtype=np.uint8)
        img[0:300, 0:200] = np.random.randint(20, 80, (300, 200))
        img[0:300, 200:400] = np.random.randint(150, 220, (300, 200))
        # 画几条线模拟细节
        for i in range(5):
            cv2.line(img, (20+i*80, 20), (20+i*80, 280), 200, 2)

    # 1. 加噪
    noisy = add_salt_pepper_noise(img, amount=0.05)

    # 2. OpenCV 中值滤波（两个窗口大小对比）
    median_3x3 = cv2.medianBlur(noisy, 3)
    median_5x5 = cv2.medianBlur(noisy, 5)

    # 3. 你的中值滤波
    my_median = median_filter_manual(noisy, kernel_size=3)

    # 4. 来个对比——用均值滤波处理同样的噪声，看效果差异
    mean_3x3 = cv2.blur(noisy, (3, 3))

    # 显示
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))

    titles = ['原图', '加椒盐噪声 5%',
              '中值 3×3 (OpenCV)', '中值 5×5 (OpenCV)',
              '你的中值 3×3', '均值模糊 3×3（对比）',
              '原图局部', '噪声局部']
    images = [img, noisy,
              median_3x3, median_5x5,
              my_median, mean_3x3,
              img[100:200, 100:200], noisy[100:200, 100:200]]

    for ax, im, t in zip(axes.flatten(), images, titles):
        ax.imshow(im, cmap='gray')
        ax.set_title(t, fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    # 检验你的实现
    diff = np.abs(median_3x3.astype(float) - my_median.astype(float))
    print(f"\n你的中值滤波 vs OpenCV 的差异：max={diff.max():.1f}, mean={diff.mean():.3f}")
    print("差异应该几乎为 0——如果有明显差异，检查边缘处理。")

    # 思考题：
    # 1. 为什么均值模糊对椒盐噪声基本无效？
    #    （提示：一个 255 的噪点和 9 个正常像素求均值还是会把均值拉高很多）
    # 2. 中值滤波 5×5 比 3×3 更干净，但代价是什么？
    # 3. 如果噪声比例提高到 20%，3×3 中值滤波还能撑住吗？试试看。
