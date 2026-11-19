"""第 16 章自动验收：TODO / FAIL / PASS。

环境变量 COURSE_IMPL_DIR 可指向另一份实现目录（例如参考实现），
用于维护者验证测试本身，学员正常使用时不需要设置。
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

CHAPTER = Path(__file__).resolve().parents[1]
ASSIGNMENT = "16-1_相位相关与亚像素.py"


def load_assignment():
    override = os.environ.get("COURSE_IMPL_DIR")
    base = Path(override) / CHAPTER.name if override else CHAPTER
    spec = importlib.util.spec_from_file_location("chapter16_assignment", base / ASSIGNMENT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def report(status: str, name: str, detail: str) -> None:
    print(f"[{status:4}] {name} — {detail}")


def smooth_texture(shape, seed):
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=shape)
    fy = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.fftfreq(shape[1])[None, :]
    lowpass = np.exp(-(fy ** 2 + fx ** 2) / (2 * 0.08 ** 2))
    return np.real(np.fft.ifft2(np.fft.fft2(noise) * lowpass))


def fourier_shift(image, dy, dx):
    fy = np.fft.fftfreq(image.shape[0])[:, None]
    fx = np.fft.fftfreq(image.shape[1])[None, :]
    phase = np.exp(-2j * np.pi * (fy * dy + fx * dx))
    return np.real(np.fft.ifft2(np.fft.fft2(image) * phase))


def close(got, want, tol) -> bool:
    """逐元素比较；出现 nan 或 inf 一律算不相等。"""
    diff = np.abs(np.asarray(got, dtype=np.float64) - np.asarray(want, dtype=np.float64))
    return bool(np.all(diff <= tol))


def fmt(pair) -> str:
    return f"({pair[0]:.3f}, {pair[1]:.3f})"


def run(name, func):
    try:
        ok, detail = func()
    except NotImplementedError as exc:
        report("TODO", name, str(exc))
        return False
    except Exception as exc:  # noqa: BLE001
        report("FAIL", name, f"运行异常：{type(exc).__name__}: {exc}")
        return False
    report("PASS" if ok else "FAIL", name, detail)
    return ok


def check_parabola(mod):
    f = mod.parabola_peak_offset
    # y = a (x - b)^2 + c 在 x = -1, 0, 1 处采样，顶点在 b
    for a, b, c in [(-1.0, 0.3, 2.0), (-3.0, -0.2, 0.5), (-0.5, 0.45, -1.0)]:
        got = f(*[a * (x - b) ** 2 + c for x in (-1, 0, 1)])
        if not isinstance(got, float):
            return False, "返回值必须是 Python float"
        if not close(got, b, 1e-9):
            return False, f"三个采样点取自顶点在 {b} 的抛物线，应精确恢复 {b}，得到 {got}。对照讲义 6.1 节检查分子的顺序和分母的系数"
    degenerate = [
        ((1.0, 1.0, 1.0), "三点相等"),
        ((0.0, 1.0, 2.0), "三点在一条斜线上（共线但不相等）"),
        ((0.0, 1.0, 2.0 + 1e-13), "三点几乎共线（left - 2*center + right 约为 1e-13）"),
    ]
    for vals, why in degenerate:
        got = f(*vals)
        if not isinstance(got, float) or got != 0.0:
            return False, f"{why}时没有曲率信息，应返回 Python float 的 0.0，得到 {got!r}。检查退化情况是怎么判断的"
    if not close(f(0.0, 1.0, 1.0), 0.5, 1e-12):
        return False, f"f(0, 1, 1) 的顶点正好在两个采样点正中间，应为 0.5，得到 {f(0.0, 1.0, 1.0)}"
    for vals, want in [((0.0, 2.0, 3.9), 0.5), ((3.9, 2.0, 0.0), -0.5)]:
        got = f(*vals)
        if not close(got, want, 1e-12):
            return False, f"f{vals} 的中间值不是最大值，抛物线顶点在半个像素之外，结果应限制为 {want}，得到 {got}"
    return True, "真抛物线精确，共线输入返回 0.0，越界结果被限制在半个像素内"


def corrcoef_map(image, template):
    """用 np.corrcoef 逐窗口算相关系数，作为独立对照（只用在没有平坦窗口的图上）。"""
    h, w = template.shape
    out = np.zeros((image.shape[0] - h + 1, image.shape[1] - w + 1))
    flat_tpl = template.ravel()
    for y in range(out.shape[0]):
        for x in range(out.shape[1]):
            out[y, x] = np.corrcoef(image[y:y + h, x:x + w].ravel(), flat_tpl)[0, 1]
    return out


def check_ncc(mod):
    rng = np.random.default_rng(1)
    image = rng.integers(0, 255, size=(20, 24)).astype(np.uint8)
    tpl = image[5:12, 7:15].copy()
    resp = mod.ncc_map(image, tpl)
    if not isinstance(resp, np.ndarray) or resp.shape != (14, 17) or resp.dtype != np.float64:
        return False, "输出必须是 (H-h+1, W-w+1) float64"
    if not np.all(np.isfinite(resp)):
        return False, "输出里有 nan 或 inf"
    if not close(resp[5, 7], 1.0, 1e-9):
        return False, "模板来源位置的 ZNCC 应为 1（输入是 uint8，检查计算前有没有转成浮点数）"
    if np.unravel_index(np.argmax(resp), resp.shape) != (5, 7):
        return False, "最大响应位置应是模板来源 (5, 7)"
    if resp.max() > 1 + 1e-9 or resp.min() < -1 - 1e-9:
        return False, "ZNCC 必须落在 [-1, 1]"
    img_f = image.astype(np.float64)
    tpl_f = tpl.astype(np.float64)
    img_keep, tpl_keep = img_f.copy(), tpl_f.copy()
    resp_f = mod.ncc_map(img_f, tpl_f)
    if not (np.array_equal(img_f, img_keep) and np.array_equal(tpl_f, tpl_keep)):
        return False, "函数修改了传入的数组，需要改动时先复制一份"
    if not close(resp_f, resp, 1e-9):
        return False, "同一组数据用 uint8 和 float64 两种类型传入，结果应该相同。检查计算前有没有转成浮点数"
    want = corrcoef_map(img_f, tpl_f)
    worst = np.unravel_index(np.argmax(np.abs(resp - want)), resp.shape)
    if not close(resp, want, 1e-9):
        return False, (f"和 np.corrcoef 逐窗口算出的相关系数不一致，例如位置 {tuple(int(v) for v in worst)} 应为 {want[worst]:.4f}，"
                       f"得到 {resp[worst]:.4f}。检查减的是不是各自窗口的均值、分母是不是两个范数的乘积、有没有丢掉符号")
    for gain, bias in [(2.0, 30.0), (0.5, 30.0)]:
        changed = mod.ncc_map(img_f * gain + bias, tpl)
        if not close(changed[5, 7], 1.0, 1e-9):
            return False, f"图像变成 {gain} 倍再加 {bias} 之后，模板来源位置的 ZNCC 仍应为 1，得到 {changed[5, 7]:.4f}"
    inverted = mod.ncc_map(255.0 - img_f, tpl)
    if not close(inverted[5, 7], -1.0, 1e-9):
        return False, f"图像反相（255 - 图像）后，模板来源位置的 ZNCC 应为 -1，得到 {inverted[5, 7]:.4f}"
    flat = np.full((10, 10), 7.0)
    if not close(mod.ncc_map(flat, flat[:3, :3]), 0.0, 1e-9):
        return False, "平坦窗口应输出 0"
    flat_tpl = mod.ncc_map(img_f, np.full((7, 8), 9.0))
    if flat_tpl.shape != (14, 17) or not close(flat_tpl, 0.0, 1e-9):
        return False, "模板完全平坦时整张响应图都应是 0.0，不能出现 nan"
    partly = img_f.copy()
    partly[0:9, 0:10] = 50.0
    mixed = mod.ncc_map(partly, tpl)
    if not np.all(np.isfinite(mixed)) or not close(mixed[0:3, 0:3], 0.0, 1e-9):
        return False, "图像里有一块平坦区域时，落在这块区域里的窗口应输出 0.0，其余位置应是有限的数"
    same = mod.ncc_map(tpl_f, tpl_f)
    if same.shape != (1, 1) or not close(same, 1.0, 1e-9):
        return False, "模板和图像一样大时，输出应是形状 (1, 1)、值为 1 的数组"
    return True, "和 np.corrcoef 一致，对增益和偏置不变，反相得 -1，平坦窗口和平坦模板输出 0"


def check_phase(mod):
    f = mod.phase_correlation
    ref = smooth_texture((96, 96), 2)
    mov = np.roll(ref, (7, -11), axis=(0, 1))
    dy, dx = f(ref, mov, use_window=False)
    if not all(isinstance(v, float) for v in (dy, dx)):
        return False, "返回值必须是两个 Python float"
    if not close((dy, dx), (7, -11), 0.5):
        return False, f"整数循环平移 (7, -11) 估计为 {fmt((dy, dx))}。检查方向（哪一幅图取共轭）、返回值的行列顺序和循环卷绕"
    if not close((dy, dx), (7, -11), 0.05):
        return False, f"整数循环平移 (7, -11) 在 use_window=False 时估计为 {fmt((dy, dx))}，接近但不精确。循环平移在不加窗时应当精确，检查 use_window=False 时是不是仍然加了窗"

    # 非方形图：两个方向的尺寸不同，各有一个方向的位移超过半幅
    rect = smooth_texture((72, 96), 5)
    for true in [(-30, 40), (25, -40)]:
        est = f(rect, np.roll(rect, true, axis=(0, 1)), use_window=False)
        if not close(est, true, 0.05):
            return False, f"72x96 的图上整数循环平移 {true} 估计为 {fmt(est)}。检查 y、x 两个方向各自用的是哪个尺寸"

    # 亚像素：前三组在方形图上，后两组的整数峰落在最后一行一列或第 0 行第 0 列，邻点要循环取
    errors = []
    cases = [(ref, 2.3, -4.6), (ref, -5.75, 3.2), (ref, 0.5, 0.0), (rect, -0.7, -0.6), (rect, 0.3, 0.4)]
    for base, true_dy, true_dx in cases:
        est = f(base, fourier_shift(base, true_dy, true_dx), use_window=False)
        if not np.all(np.isfinite(est)):
            return False, (f"小数平移 ({true_dy}, {true_dx}) 返回了 {est}。这组数据里有的频率分量恰好为 0，"
                           "检查互功率谱归一化的分母在这些频率上会不会是 0")
        err = np.max(np.abs(np.asarray(est, dtype=np.float64) - (true_dy, true_dx)))
        if not err <= 0.25:
            return False, (f"小数平移 ({true_dy}, {true_dx}) 估计为 {fmt(est)}，误差超过 0.25 px。"
                           "检查峰值两侧的邻点取的是哪个方向、下标到头时怎么取、小数部分加的方向对不对")
        errors.append(float(err))
    worst = max(errors)

    # 两幅图叠加同一个不随内容移动的低频明暗起伏（例如照明不均）
    yy, xx = np.mgrid[0:72, 0:96]
    shading = 5.0 * rect.std() * (np.cos(2 * np.pi * xx / 96) + np.sin(2 * np.pi * yy / 72))
    est = f(rect + shading, np.roll(rect, (-9, 14), axis=(0, 1)) + shading, use_window=False)
    if not close(est, (-9, 14), 0.05):
        return False, (f"两幅图叠加同一个不动的低频明暗起伏后，循环平移 (-9, 14) 估计为 {fmt(est)}。"
                       "对照文档字符串第 3 步，检查互功率谱的归一化有没有做")

    # 从大图裁出两个视场：边界不连续，必须靠加窗
    big = smooth_texture((220, 240), 20)
    for sy, sx in [(7, -9), (-6, -15)]:
        a = big[60:132, 70:166]
        b = big[60 - sy:132 - sy, 70 - sx:166 - sx]
        a_keep, b_keep = a.copy(), b.copy()
        est = f(a, b, use_window=True)
        if not (np.array_equal(a, a_keep) and np.array_equal(b, b_keep)):
            return False, "函数修改了传入的数组（加窗时要先复制，不能原地相乘）"
        if not close(est, (sy, sx), 0.5):
            return False, (f"从大图裁出的两个视场（边界不连续）在 use_window=True 时，平移 ({sy}, {sx}) 估计为 {fmt(est)}。"
                           "检查 use_window=True 时是不是两幅图都乘了二维的窗")

    for value in (0.0, 3.0):
        const = np.full((16, 16), value)
        est = f(const, const, use_window=False)
        if not np.all(np.isfinite(est)):
            return False, f"两幅图都是常数 {value} 时返回了 {est}，应是有限的数。检查互功率谱归一化的分母"

    rng = np.random.default_rng(3)
    noisy = np.roll(ref, (4, 6), axis=(0, 1)) + rng.normal(scale=0.05 * ref.std(), size=ref.shape)
    dy, dx = f(ref, noisy, use_window=True)
    if not close((dy, dx), (4, 6), 0.3):
        return False, f"加噪声和窗函数后 (4, 6) 估计为 {fmt((dy, dx))}"
    return True, f"方向、卷绕、非方形图、归一化和加窗开关都正确，亚像素最大误差 {worst:.3f} px"


def main() -> int:
    print("第 16 章进度检查\n" + "=" * 52)
    try:
        mod = load_assignment()
    except Exception as exc:  # noqa: BLE001
        report("FAIL", "加载作业", f"{type(exc).__name__}: {exc}")
        return 1
    results = [
        run("抛物线亚像素", lambda: check_parabola(mod)),
        run("ZNCC 模板匹配", lambda: check_ncc(mod)),
        run("相位相关", lambda: check_phase(mod)),
    ]
    print("=" * 52)
    print(f"通过 {sum(results)}/{len(results)}")
    if not all(results):
        print("提示：TODO 是正常学习状态；实现后重新运行。")
        return 1
    print("基本不变量通过；还要在实验记录里写清周期图案下的失败案例。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
