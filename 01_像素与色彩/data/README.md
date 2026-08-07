# 第 1 章数据说明

本章不依赖大型数据集。课程根目录 `../images/` 已提供两张合成测试图：

- `test_underexposed.jpg`：用于直方图、亮度增强和噪声实验。
- `test_color.jpg`：用于 BGR/RGB/HSV 和通道实验。

合成图方便验证算法是否正确，但不能替代真实照片。建议额外准备 3-5 张自己拍摄的图片：

1. 一张明显欠曝的室内照片。
2. 一张同时含阴影和高光的照片。
3. 一张颜色鲜艳、包含红绿蓝物体的照片。
4. 一张带细线或文字的图片，用于观察滤波对细节的破坏。
5. 可选：一张高 ISO 夜景照片，用于观察真实传感器噪声。

## 推荐的公开测试图

如果需要标准图像，可从以下官方或学术来源按需下载；不要求一次性下载：

- OpenCV samples：`https://github.com/opencv/opencv/tree/4.x/samples/data`
- scikit-image 内置样例说明：`https://scikit-image.org/docs/stable/api/skimage.data.html`
- Kodak Lossless True Color Image Suite：适合图像质量与压缩实验

使用外部图片时，请记录来源和许可。课程目录只保存你实际使用的小图片；不要把整个大型数据集复制进来。

## 推荐目录

```text
data/
├── README.md
├── personal/       # 自己拍摄的图片
└── public/         # 公开样例，文件名旁记录来源
```

## 隐私注意

个人照片尽量避免包含身份证、住址、屏幕密码、他人正脸或其他敏感信息。需要分享实验时，优先使用公开样例或无隐私的静物照片。
