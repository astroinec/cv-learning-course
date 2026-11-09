# 第 21 章：3D 数学与渲染管线

本章用时建议 **12–20 小时**（OpenGL 选做部分另计）。目标：熟练掌握渲染需要的 3D 数学，从零写一个软件光栅化器理解渲染管线，看懂 OpenGL 和 GLSL 着色器代码，知道渲染在检测仿真里怎么用。

## 为什么重要

JD 里有“熟悉 3D 数学”“熟悉 OpenGL 渲染管线和 Shader 编程”“渲染引擎或 3D 仿真开发经验”。检测设备上的渲染主要用于两件事：把设计版图或 3D 结构渲染成“理想图像”做参考（die-to-database），以及批量生成带标签的仿真图像训练和测试算法。这一章打基础，第 22 章再加入光学和传感器。

## 学习顺序

1. 复习第 4 章（齐次坐标、变换矩阵）和第 7 章（相机模型），本章会把它们和图形学的约定对上。
2. 阅读 `教程_3D数学与渲染管线.md`。
3. 运行仓库根目录的 `demos/渲染管线.html`，拖动滑块观察 MVP 每一步对立方体的影响。
4. 实现 `21-1_软光栅.py`，运行验收。
5. 运行 `实验_软光栅.ipynb`：用自己的光栅器渲染一个立方体的深度图，再加上 Lambert 光照。
6. （选做）跟着 LearnOpenGL 的 Getting Started 部分写出第一个带着色器的三角形和立方体。
7. 在 `实验记录.md` 记录图形学和视觉两套相机约定的差异。

## 文件

| 文件 | 用途 |
|---|---|
| `教程_3D数学与渲染管线.md` | 向量矩阵、旋转表示、MVP、光栅化、着色、OpenGL 管线和 GLSL |
| `实验_软光栅.ipynb` | 软件渲染实验 |
| `21-1_软光栅.py` | 作业：look_at、透视矩阵、屏幕映射、z-buffer |
| `tests/check_progress.py` | 自动验收 |
| `实验记录.md` | 实验记录模板 |

## 通过标准

- 3 项测试 PASS。
- 能不看资料写出透视投影矩阵，并解释第三行为什么把深度非线性地映射到 [-1, 1]。
- 能说清 OpenGL 相机（看向 -Z、y 向上）和 OpenCV 相机（看向 +Z、y 向下）的差异，以及两者之间怎么转换。
- 能读懂一段顶点着色器加片元着色器，说出数据怎么从 CPU 流到屏幕像素。

## 推荐资源

- [LearnOpenGL](https://learnopengl.com/)：Getting Started（到 Camera 为止）和 Lighting 两部分，是学 OpenGL 管线和 GLSL 的标准入门。
- [tinyrenderer](https://github.com/ssloy/tinyrenderer)：用约 500 行 C++ 从零写一个软件渲染器的教程，和本章作业思路一致，可以作为 C++ 练习（顺便练第 15 章）。
- [Scratchapixel](https://www.scratchapixel.com/)：光栅化、透视矩阵、光线追踪的详细推导。
- 3Blue1Brown 的 Essence of Linear Algebra 系列视频：补线性代数直觉。
- [The Book of Shaders](https://thebookofshaders.com/)：片元着色器的交互式入门。
