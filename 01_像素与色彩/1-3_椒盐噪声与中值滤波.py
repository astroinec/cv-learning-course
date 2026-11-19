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


def read_image(path, flags):
    """读图片文件，文件不存在时返回 None。先读成字节再解码，路径里有中文时在 Windows 上也能读。"""
    if not path.exists():
        return None
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)


# ═══════════════════════════════════════════════════════════
# STEP 1：椒盐噪声生成器
# ═══════════════════════════════════════════════════════════

def add_salt_pepper_noise(img, amount=0.05):
    """
    给灰度图添加椒盐噪声。
    amount: 噪声比例，取值 0 到 1（比如 0.05 = 5%）
    椒 = 黑点 (0)，盐 = 白点 (255)

    模型（和讲义 1.4 节一致，验收按它检查）：
      每个像素以 amount/2 的概率变成椒，以 amount/2 的概率变成盐，两者互斥，其余像素不变。
      所以被改动的像素期望占 amount，其中椒和盐各占一半；amount=0 时一个像素都不改。
    输出：同 shape 的 uint8 数组；不修改输入（下面已经先复制了一份）。

    提示：
      每个像素只抽一次 [0, 1) 上的均匀随机数：np.random.random(img.shape)
      在 [0, 1) 里划出两段互不重叠、长度都是 amount/2 的区间：随机数落在第一段的像素是椒，落在第二段的是盐
      椒位置置 0，盐位置置 255
    注意：给椒和盐各抽一次随机数是常见的错误写法。两张掩膜互相独立，会有一部分像素被两边同时选中，
      后写的覆盖先写的。结果是被改动的像素只占 1 - (1 - amount/2)**2（amount=0.10 时是 0.0975，不是 0.10），
      而且椒和盐不一样多。
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

    输入：img 为 (H, W) 的 uint8 灰度图；kernel_size 是奇数（3、5、7……）。
    输出：同 shape 的 uint8 数组，每个像素是以它为中心的 kernel_size×kernel_size 邻域的中位数；不修改输入。
    边界约定：最外一圈也要算，不能跳过。邻域伸出图像的部分，用离它最近的边界像素补上
      （复制边缘，np.pad 里对应 mode="edge"）。cv2.medianBlur 也是这样做的，验收会和它做全图比较。

    提示：
      先按边界约定在图像四周各补 pad 个像素
      遍历原图的每个像素，在补好的图上取 k*k 邻域 → np.median()
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
    # 中文字体：Windows 的两种放在前面，找不到时依次往后试
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "PingFang SC", "Heiti SC", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    # 路径相对脚本文件，而不是相对终端当前目录
    img_path = Path(__file__).resolve().parents[1] / "images" / "test_underexposed.jpg"
    img = read_image(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("没有找到 images/test_underexposed.jpg，改用代码生成的测试图（要用文件版本，先运行 scripts/generate_sample_images.py）")
        img = np.zeros((300, 400), dtype=np.uint8)
        img[0:300, 0:200] = np.random.randint(20, 80, (300, 200))
        img[0:300, 200:400] = np.random.randint(150, 220, (300, 200))
        # 画几条线模拟细节
        for i in range(5):
            cv2.line(img, (20+i*80, 20), (20+i*80, 280), 200, 2)

    # 1. 加噪
    noisy = add_salt_pepper_noise(img, amount=0.05)
    if np.array_equal(noisy, img):
        print("add_salt_pepper_noise 还没有实现（加噪后的图和原图相同）。下面的图只是占位，先完成 STEP 1。")

    # 2. OpenCV 中值滤波（两个窗口大小对比）
    median_3x3 = cv2.medianBlur(noisy, 3)
    median_5x5 = cv2.medianBlur(noisy, 5)

    # 3. 你的中值滤波
    my_median = median_filter_manual(noisy, kernel_size=3)
    if not np.any(my_median):
        print("median_filter_manual 还没有实现（输出全是 0）。下面“你的中值”一栏只是占位，先完成 STEP 2。")

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
    print("差异应该是 0（包括最外一圈）。只有最外一圈不同时，检查边界约定。")

    # 思考题：
    # 1. 为什么均值模糊对椒盐噪声基本无效？
    #    （提示：一个 255 的噪点和 9 个正常像素求均值还是会把均值拉高很多）
    # 2. 中值滤波 5×5 比 3×3 更干净，但代价是什么？
    # 3. 如果噪声比例提高到 20%，3×3 中值滤波还能撑住吗？试试看。
