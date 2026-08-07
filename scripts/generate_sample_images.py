"""生成第一章使用的两张无版权依赖合成测试图。"""

from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "images"
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    gray = np.zeros((400, 500), dtype=np.uint8)
    gray[:300, :300] = rng.integers(20, 60, (300, 300), dtype=np.uint8)
    gray[100:300, 200:500] = rng.integers(160, 220, (200, 300), dtype=np.uint8)
    gray[50:100, 350:450] = 240
    for x in range(30, 500, 100):
        cv2.line(gray, (x, 30), (x, 370), 200, 2)

    color = np.zeros((300, 500, 3), dtype=np.uint8)
    color[:, :166, 2] = 255
    color[:, 167:333, 1] = 255
    color[:, 334:, 0] = 255
    ramp = np.linspace(0, 255, 400, dtype=np.uint8)
    color[50:250, 50:450] = ramp[None, :, None]

    cv2.imwrite(str(output / "test_underexposed.jpg"), gray)
    cv2.imwrite(str(output / "test_color.jpg"), color)
    print(f"已生成测试图：{output}")


if __name__ == "__main__":
    main()
