# 第 15 章数据说明

本章不需要外部数据。测试图由 `tests/test_main.cpp` 用固定随机种子生成。

想用真实图片做实验时，先转成 8 位灰度 PGM（`cpp/image.hpp` 里的 `read_pgm` 只支持 P5 格式），可以用 Python：

```python
from PIL import Image
Image.open('in.jpg').convert('L').save('in.pgm')
```

个人图片放在 `data/personal/`，已被 `.gitignore` 排除的目录以外不要提交大图。
