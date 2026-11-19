"""第 28 章自动验收：TODO / FAIL / PASS。"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path

import numpy as np

CHAPTER = Path(__file__).resolve().parents[1]
ASSIGNMENT = "28-1_零样本分类与图文检索.py"
R2 = math.sqrt(0.5)


def load_assignment():
    override = os.environ.get("COURSE_IMPL_DIR")
    base = Path(override) / CHAPTER.name if override else CHAPTER
    spec = importlib.util.spec_from_file_location("chapter28_assignment", base / ASSIGNMENT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def report(status, name, detail):
    print(f"[{status:4}] {name} — {detail}")


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


def raises_value_error(fn, *args, **kwargs):
    """只有抛出 ValueError 才算对；不抛异常或者抛出别的异常都返回 False。"""
    try:
        fn(*args, **kwargs)
    except ValueError:
        return True
    except NotImplementedError:
        raise
    except Exception:  # noqa: BLE001
        return False
    return False


def is_array(x, shape, dtype):
    return isinstance(x, np.ndarray) and x.shape == shape and x.dtype == dtype


def close(a, b, tol=1e-6):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return a.shape == b.shape and bool(np.all(np.isfinite(a))) and bool(np.max(np.abs(a - b), initial=0.0) <= tol)


# ---------------------------------------------------------------- 零样本分类器
def check_classifier(mod):
    # 2 个类别、3 个模板、3 维。类别 0 的三个模板分别指向三个坐标轴，长度 2、3、0.5；类别 1 是 e2、e2、e3，长度 4、1、7
    t = np.array([[[2.0, 0, 0], [0, 3.0, 0], [0, 0, 0.5]], [[0, 4.0, 0], [0, 1.0, 0], [0, 0, 7.0]]])
    keep = t.copy()
    out = mod.build_zeroshot_classifier(t)
    if not np.array_equal(t, keep):
        return False, "不要修改传入的 text_embeds（归一化时不要原地除）"
    if not is_array(out, (2, 3), np.float64):
        return False, f"2 个类别、3 个模板、3 维的输入应返回 (2, 3) 的 float64 数组；得到 {getattr(out, 'shape', type(out))}、{getattr(out, 'dtype', '')}。检查是不是对模板那一维取的平均"
    s3, s5 = 1 / math.sqrt(3.0), 1 / math.sqrt(5.0)
    want = np.array([[s3, s3, s3], [0.0, 2 * s5, s5]])
    if not close(out, want):
        return False, (f"小例子：类别 0 的三个模板归一化后是三个坐标轴，平均再归一化应为 [0.577, 0.577, 0.577]；类别 1 应为 [0, 0.894, 0.447]。得到 {np.round(out, 3).tolist()}。"
                       "检查：每个模板是不是各除各的范数（不是整个类别一起除）、是不是先归一化再平均、平均之后有没有再归一化一次")

    rng = np.random.default_rng(0)
    t = rng.normal(size=(5, 4, 6))
    out = mod.build_zeroshot_classifier(t.astype(np.float32))
    if not is_array(out, (5, 6), np.float64):
        return False, "输入是 float32 时也应返回 float64 的 (C, D) 数组"
    if not close(np.linalg.norm(out, axis=1), np.ones(5), 1e-6):
        return False, f"每个类别的分类器向量长度都应是 1，得到 {np.round(np.linalg.norm(out, axis=1), 3).tolist()}。平均之后还要再归一化一次"
    scaled = t * rng.uniform(0.01, 50.0, size=(5, 4, 1))
    if not close(mod.build_zeroshot_classifier(scaled), mod.build_zeroshot_classifier(t), 1e-9):
        return False, "把每个模板的向量各自乘上不同的正数，结果不应该变：每个模板要先各自归一化，再参与平均"

    flat = mod.build_zeroshot_classifier(np.array([[3.0, 4.0], [0.0, 2.0]]))
    if not is_array(flat, (2, 2), np.float64) or not close(flat, [[0.6, 0.8], [0.0, 1.0]]):
        return False, "输入是 (C, D) 时当作只有 1 个模板：[[3, 4], [0, 2]] 应得到 [[0.6, 0.8], [0, 1]]"

    # 全 0 的模板、互相抵消的模板
    z = np.array([[[0.0, 0.0], [0.0, 2.0]], [[1.0, 0.0], [-3.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]])
    out = mod.build_zeroshot_classifier(z)
    if not isinstance(out, np.ndarray) or not np.all(np.isfinite(out)):
        return False, "有全 0 的模板、或者两个模板正好抵消时，结果里不能出现 nan 或 inf（范数要和 eps 取较大值再除）"
    if not close(out, [[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]):
        return False, f"全 0 的模板归一化后仍是 0：[0, 0] 和 [0, 2] 平均再归一化应为 [0, 1]；[1, 0] 和 [-3, 0] 归一化后正好抵消，应为 [0, 0]。得到 {np.round(out, 3).tolist()}"

    for bad in (np.ones(4), np.ones((2, 2, 2, 2)), np.ones((0, 3, 4)), np.ones((2, 0, 4)), np.ones((2, 3, 0))):
        if not raises_value_error(mod.build_zeroshot_classifier, bad):
            return False, f"形状为 {bad.shape} 的输入应抛出 ValueError（只接受二维或三维，且每一维都不为 0）"
    return True, "小例子、模板各自归一化、类别向量长度为 1、全 0 的情形和异常都正确"


# ---------------------------------------------------------------- 零样本预测
def check_predict(mod):
    ln3 = math.log(3.0)
    x = np.array([[3.0, 0.0], [0.0, 0.5], [2.0, 2.0]])
    w = np.eye(2)
    keep = (x.copy(), w.copy())
    out = mod.zeroshot_predict(x, w, ln3)
    if not (isinstance(out, (tuple, list)) and len(out) == 2):
        return False, "应返回 (probs, pred) 两个数组"
    probs, pred = out
    if not np.array_equal(x, keep[0]) or not np.array_equal(w, keep[1]):
        return False, "不要修改传入的数组"
    if not is_array(probs, (3, 2), np.float64) or not is_array(pred, (3,), np.int64):
        return False, f"3 张图、2 个类别：probs 应为 (3, 2) 的 float64，pred 应为 (3,) 的 int64；得到 {getattr(probs, 'shape', None)}/{getattr(probs, 'dtype', None)} 和 {getattr(pred, 'shape', None)}/{getattr(pred, 'dtype', None)}"
    want = [[0.75, 0.25], [0.25, 0.75], [0.5, 0.5]]
    if not close(probs, want):
        return False, (f"小例子（放大倍数 ln 3）：三张图归一化后是 [1, 0]、[0, 1]、[0.707, 0.707]，概率应为 {want}，得到 {np.round(probs, 4).tolist()}。"
                       "检查：图像向量有没有归一化、相似度是不是乘了 logit_scale、softmax 是不是对每一行的各个类别做的")
    if pred.tolist() != [0, 1, 0]:
        return False, f"小例子的预测应为 [0, 1, 0]（第三张图两类并列，取编号小的），得到 {pred.tolist()}"

    probs, pred = mod.zeroshot_predict(np.array([[1.0, 1.0]]), np.array([[1.0, 0.0], [0.0, 3.0]]), 1.0)
    p1 = 1.0 / (1.0 + math.exp(-2.0 * R2))
    if pred.tolist() != [1] or not close(probs, [[1 - p1, p1]]):
        return False, (f"classifier 要原样使用，不要再归一化：图像 [1, 1]、分类器 [[1, 0], [0, 3]]、放大倍数 1 时，两类的打分是 0.707 和 2.121，"
                       f"应预测类别 1，概率 [{1 - p1:.4f}, {p1:.4f}]；得到 {pred.tolist()}、{np.round(probs, 4).tolist()}")

    c99 = np.array([[1.0, 0.0], [0.99, math.sqrt(1 - 0.99 ** 2)]])
    probs, _ = mod.zeroshot_predict(np.array([[5.0, 0.0]]), c99)
    pe = 1.0 / (1.0 + math.exp(-1.0))
    if not close(probs, [[pe, 1 - pe]]):
        return False, f"不传 logit_scale 时应使用默认值 100：两类的余弦是 1 和 0.99，概率应为 [{pe:.4f}, {1 - pe:.4f}]，得到 {np.round(probs, 4).tolist()}"

    probs, pred = mod.zeroshot_predict(np.eye(2), np.eye(2), 2000.0)
    if not isinstance(probs, np.ndarray) or not np.all(np.isfinite(probs)) or not close(probs, np.eye(2)) or pred.tolist() != [0, 1]:
        return False, "logit_scale=2000 时概率里不能出现 inf 或 nan：exp(2000) 会溢出，按文档字符串先减去每行的最大值再取指数"

    rng = np.random.default_rng(1)
    x = rng.normal(size=(40, 6)).astype(np.float32)
    w = rng.normal(size=(5, 6)).astype(np.float32)
    p1_, pred1 = mod.zeroshot_predict(x, w, 1.0)
    p50, pred50 = mod.zeroshot_predict(x, w, 50.0)
    if not is_array(p1_, (40, 5), np.float64) or not is_array(pred1, (40,), np.int64):
        return False, "输入是 float32 时 probs 也应是 float64，pred 是 int64"
    if not close(p1_.sum(axis=1), np.ones(40)) or not close(p50.sum(axis=1), np.ones(40)):
        return False, "每一行的概率加起来应该是 1"
    if not np.array_equal(pred1, pred50):
        return False, "logit_scale 只改变概率有多尖，不应改变预测的类别"
    if not (p50.max(axis=1).mean() > p1_.max(axis=1).mean() + 0.2):
        return False, "logit_scale 越大，概率应该越尖（最大概率更接近 1）。检查是乘以 logit_scale 还是除以它"
    xn = x.astype(np.float64)
    xn = xn / np.linalg.norm(xn, axis=1, keepdims=True)
    if not np.array_equal(pred1, np.argmax(xn @ w.astype(np.float64).T, axis=1)):
        return False, "pred 应是归一化后的图像向量和各类分类器向量内积最大的那一类"
    if not np.array_equal(mod.zeroshot_predict(x * rng.uniform(0.1, 30.0, size=(40, 1)).astype(np.float32), w, 50.0)[1], pred50):
        return False, "把每个图像向量各自乘上一个正数，预测不应该变（图像向量要先归一化）"
    if not close(mod.zeroshot_predict(3.0 * x, w, 50.0)[0], p50, 1e-5):
        return False, "把图像向量整体放大 3 倍，概率不应该变：图像向量要先归一化，否则向量越长概率越尖"

    probs, pred = mod.zeroshot_predict(np.zeros((1, 3)), np.eye(3), 100.0)
    if not isinstance(probs, np.ndarray) or not close(probs, [[1 / 3, 1 / 3, 1 / 3]]) or pred.tolist() != [0]:
        return False, "图像向量全 0 时归一化后仍是 0，三个类别打分相同：概率应各为 1/3，预测取编号最小的 0，不能出现 nan"

    if not raises_value_error(mod.zeroshot_predict, x, w[:, :5], 1.0) or not raises_value_error(mod.zeroshot_predict, x[0], w, 1.0):
        return False, "image_embeds 不是二维，或者它和 classifier 的维数 D 不同时应抛出 ValueError"
    if not raises_value_error(mod.zeroshot_predict, x, w, 0.0) or not raises_value_error(mod.zeroshot_predict, x, w, -5.0):
        return False, "logit_scale 小于等于 0 时应抛出 ValueError"
    return True, "小例子、分类器原样使用、默认放大倍数、数值稳定、并列和异常都正确"


# ---------------------------------------------------------------- Recall@K
def brute_recall(sim, pos, k):
    """独立的参照：用 Python 的排序逐个查询地数，相似度相同按下标从小到大。"""
    hits = 0
    for row, prow in zip(sim.tolist(), pos.tolist()):
        order = sorted(range(len(row)), key=lambda j: (-row[j], j))[:k]
        hits += any(bool(prow[j]) for j in order)
    return hits / len(sim)


def check_recall(mod):
    # 3 个查询、4 个候选。查询 0 的正确答案排第 3，查询 1 有两个正确答案（最好的排第 2），查询 2 的排第 1
    sim = np.array([[0.9, 0.1, 0.5, 0.3], [0.2, 0.8, 0.1, 0.7], [0.3, 0.2, 0.1, 0.6]])
    pos = np.array([[False, False, False, True], [True, False, False, True], [False, False, False, True]])
    keep = (sim.copy(), pos.copy())
    got = []
    for k in (1, 2, 3, 4):
        got.append(mod.recall_at_k(sim, pos, k))
        if not np.array_equal(sim, keep[0]) or not np.array_equal(pos, keep[1]):  # 每调用一次就查，连续两次原地取负会互相抵消
            return False, "不要修改传入的数组（排序时不要原地改 sim）"
    if not all(isinstance(v, float) for v in got):
        return False, "应返回 Python 的 float"
    want = [1 / 3, 2 / 3, 1.0, 1.0]
    if not close(got, want, 1e-9):
        return False, (f"小例子：三个查询的正确答案分别排在第 3、第 2、第 1 位，Recall@1 到 @4 应为 [0.333, 0.667, 1, 1]，得到 {[round(v, 3) for v in got]}。"
                       "检查：是不是按相似度从大到小排、取的是不是正好 k 个、是不是“前 k 个里有一个对就算命中”、正确答案是不是从 positives 里读的（不在对角线上）、返回的是不是比例")

    one = np.array([[0.5, 0.5, 0.5]])
    a = mod.recall_at_k(one, np.array([[True, False, False]]), 1)
    b = mod.recall_at_k(one, np.array([[False, True, False]]), 1)
    c = mod.recall_at_k(one, np.array([[False, True, False]]), 2)
    d = mod.recall_at_k(one, np.array([[False, False, True]]), 2)
    if [a, b, c, d] != [1.0, 0.0, 1.0, 0.0]:
        return False, (f"三个候选的相似度完全相同时，下标小的排前面：正确答案在第 0、1 个候选时 Recall@1 应为 1 和 0，在第 1、2 个候选时 Recall@2 应为 1 和 0；得到 {[a, b, c, d]}。"
                       "先从小到大排再整个倒过来，会把并列的顺序也倒过来")

    rng = np.random.default_rng(2)
    sim = rng.normal(size=(30, 11))
    pos = rng.random((30, 11)) < 0.15
    pos[np.arange(30), rng.integers(0, 11, size=30)] = True
    for k in (1, 3, 5, 11):
        want = brute_recall(sim, pos, k)
        got = mod.recall_at_k(sim.astype(np.float32), pos.astype(np.int64), k)
        if abs(got - want) > 1e-9:
            return False, f"30 个查询、11 个候选、每个查询 1 到几个正确答案的随机例子，Recall@{k} 应为 {want:.4f}，得到 {got:.4f}（sim 是 float32、positives 是 0/1 整数时也要能用）"
    if mod.recall_at_k(sim, np.ones_like(pos), 1) != 1.0:
        return False, "所有候选都是正确答案时 Recall@1 应为 1"

    if not raises_value_error(mod.recall_at_k, sim, pos, 0) or not raises_value_error(mod.recall_at_k, sim, pos, 12):
        return False, "k 小于 1 或大于候选个数时应抛出 ValueError"
    if not raises_value_error(mod.recall_at_k, sim, pos[:, :10], 1) or not raises_value_error(mod.recall_at_k, sim[0], pos[0], 1):
        return False, "sim 不是二维，或者 positives 的形状和 sim 不同时应抛出 ValueError"
    none = pos.copy()
    none[4] = False
    if not raises_value_error(mod.recall_at_k, sim, none, 3):
        return False, "有的查询一个正确答案都没有时应抛出 ValueError，而不是悄悄算成没命中"
    return True, "小例子、多个正确答案、并列时的顺序、和逐个查询的排序对照、异常都正确"


# ---------------------------------------------------------------- 模态间隙
def check_gap(mod):
    # 图像：e1、e1、e2（长度 2、7、3），中心 [2/3, 1/3]；文本：两个都是 -e2，中心 [0, -1]
    u = np.array([[2.0, 0.0], [7.0, 0.0], [0.0, 3.0]])
    v = np.array([[0.0, -1.0], [0.0, -3.0]])
    keep = (u.copy(), v.copy())
    out = mod.modality_gap(u, v)
    if not (isinstance(out, (tuple, list)) and len(out) == 3 and all(isinstance(x, float) for x in out)):
        return False, "应返回 (gap, spread_image, spread_text) 三个 Python 的 float"
    if not np.array_equal(u, keep[0]) or not np.array_equal(v, keep[1]):
        return False, "不要修改传入的数组"
    want = (math.sqrt(20.0) / 3.0, 2.0 / 3.0, 0.0)
    if not close(out, want):
        return False, (f"小例子：图像向量归一化后是 e1、e1、e2，中心 [0.667, 0.333]；文本向量都是 -e2，中心 [0, -1]。"
                       f"应为间隙 {want[0]:.4f}、图像散布 {want[1]:.4f}、文本散布 0；得到 {[round(x, 4) for x in out]}。"
                       "检查：有没有先归一化、中心是不是没有再归一化、间隙是不是两个中心之差的长度（不是平方，也不是 1 减余弦）、"
                       "散布是不是“到本组中心的距离的平方取平均再开方”（除以个数，不是个数减 1）、三个返回值的顺序")

    rng = np.random.default_rng(3)
    u = rng.normal(size=(12, 5)) + np.array([1.5, 0, 0, 0, 0])
    v = rng.normal(size=(7, 5)) - np.array([0, 2.0, 0, 0, 0])
    gap, su, sv = mod.modality_gap(u.astype(np.float32), v.astype(np.float32))
    un = u / np.linalg.norm(u, axis=1, keepdims=True)
    vn = v / np.linalg.norm(v, axis=1, keepdims=True)
    pair = lambda a, b: float(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1).mean())  # noqa: E731  全部两两距离的平方的平均
    if abs(2 * su * su - pair(un, un)) > 1e-5 or abs(2 * sv * sv - pair(vn, vn)) > 1e-5:
        return False, "一组单位向量里，全部两两距离的平方的平均应等于散布平方的 2 倍。检查散布是不是对整个向量算距离（各维的平方先加起来），而不是每一维各算一个标准差再平均"
    if abs(gap * gap + su * su + sv * sv - pair(un, vn)) > 1e-5:
        return False, "图像向量和文本向量之间全部两两距离的平方的平均，应等于 间隙平方 + 两个散布的平方。检查间隙和中心的算法，以及两组个数不同时（12 和 7）是否仍然正确"
    scaled = mod.modality_gap(u * rng.uniform(0.05, 40.0, size=(12, 1)), v * rng.uniform(0.05, 40.0, size=(7, 1)))
    if not close(scaled, (gap, su, sv), 1e-5):
        return False, "把每个向量各自乘上一个正数，三个返回值都不应该变（两组都要先归一化）"

    same = mod.modality_gap(u, 3.0 * u)
    if abs(same[0]) > 1e-9 or abs(same[1] - same[2]) > 1e-9:
        return False, "同一组向量和它自己的 3 倍之间，间隙应为 0，两个散布应相等"
    far = mod.modality_gap(np.array([[2.0, 0.0], [5.0, 0.0]]), np.array([[-1.0, 0.0]]))
    if not close(far, (2.0, 0.0, 0.0)):
        return False, f"图像向量都指向 e1、文本向量指向 -e1 时，间隙应为 2，两个散布都是 0；得到 {[round(x, 4) for x in far]}"

    if not raises_value_error(mod.modality_gap, u, v[:, :4]) or not raises_value_error(mod.modality_gap, u[0], v):
        return False, "任何一组不是二维，或者两组的维数 D 不同时应抛出 ValueError"
    if not raises_value_error(mod.modality_gap, np.zeros((0, 5)), v) or not raises_value_error(mod.modality_gap, u, np.zeros((0, 5))):
        return False, "任何一组是空的时应抛出 ValueError"
    return True, "小例子、两两距离的恒等式、对向量长度不敏感、两组个数不同和异常都正确"


# ---------------------------------------------------------------- 区域打分
def check_regions(mod):
    r = np.array([[4.0, 0.0], [0.0, 0.5], [3.0, 3.0]])
    q = np.array([[0.0, 2.0], [1.0, 1.0]])
    keep = (r.copy(), q.copy())
    out = mod.score_regions(r, q)
    if not (isinstance(out, (tuple, list)) and len(out) == 3):
        return False, "应返回 (scores, best_region, best_score) 三个数组"
    scores, best, top = out
    if not np.array_equal(r, keep[0]) or not np.array_equal(q, keep[1]):
        return False, "不要修改传入的数组"
    if not is_array(scores, (3, 2), np.float64) or not is_array(best, (2,), np.int64) or not is_array(top, (2,), np.float64):
        return False, (f"3 个区域、2 个查询：scores 应为 (3, 2) 的 float64（行是区域、列是查询），best_region 应为 (2,) 的 int64，best_score 应为 (2,) 的 float64；"
                       f"得到 {getattr(scores, 'shape', None)}、{getattr(best, 'shape', None)}/{getattr(best, 'dtype', None)}、{getattr(top, 'shape', None)}。检查取最大是不是沿区域那一维")
    want = np.array([[0.0, R2], [1.0, R2], [R2, 1.0]])
    if not close(scores, want):
        return False, (f"小例子：区域归一化后是 e1、e2、[0.707, 0.707]，查询是 e2、[0.707, 0.707]，scores 应为 {np.round(want, 3).tolist()}，得到 {np.round(scores, 3).tolist()}。"
                       "检查区域特征和查询向量是不是都归一化了")
    if best.tolist() != [1, 2] or not close(top, [1.0, 1.0]):
        return False, f"小例子里两个查询得分最高的区域应是 1 和 2，分数都是 1；得到 {best.tolist()}、{np.round(top, 3).tolist()}"

    _, best, top = mod.score_regions(np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0]]), np.array([[1.0, 0.0]]))
    if best.tolist() != [0] or not close(top, [1.0]):
        return False, "两个区域方向相同、只是长度不同（[1, 0] 和 [2, 0]）时分数并列，应取下标小的 0；长的那个不应该赢"

    scores, best, top = mod.score_regions(np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([[1.0, 0.0], [-1.0, -1.0]]))
    if not all(isinstance(a, np.ndarray) and np.all(np.isfinite(a)) for a in (scores, top)):
        return False, "特征全 0 的区域和任何查询的分数都应是 0，不能出现 nan"
    if not close(scores, [[0.0, 0.0], [R2, -1.0]]) or best.tolist() != [1, 0] or not close(top, [R2, 0.0]):
        return False, f"全 0 的区域分数是 0：第二个查询和另一个区域的余弦是 -1，所以它得分最高的是全 0 的那个区域（下标 0，分数 0）。得到 {best.tolist()}、{np.round(top, 3).tolist()}"

    rng = np.random.default_rng(4)
    r = rng.normal(size=(9, 6))
    q = rng.normal(size=(4, 6))
    scores, best, top = mod.score_regions(r.astype(np.float32), q.astype(np.float32))
    if not is_array(scores, (9, 4), np.float64) or scores.min() < -1 - 1e-6 or scores.max() > 1 + 1e-6:
        return False, "输入是 float32 时 scores 也应是 float64；余弦相似度应在 -1 到 1 之间"
    cols = np.arange(4)
    if not np.array_equal(best, np.argmax(scores, axis=0)) or not close(top, scores[best, cols], 1e-12):
        return False, "best_region 应是 scores 每一列最大值所在的行，best_score 应等于 scores[best_region[q], q]"
    for qi in range(4):
        for ri in range(9):
            cos = float(r[ri] @ q[qi] / math.sqrt(float(r[ri] @ r[ri]) * float(q[qi] @ q[qi])))
            if abs(scores[ri, qi] - cos) > 1e-5:
                return False, f"scores[{ri}, {qi}] 应是第 {ri} 个区域和第 {qi} 个查询的余弦相似度 {cos:.4f}，得到 {scores[ri, qi]:.4f}"
    s2, b2, _ = mod.score_regions(r * rng.uniform(0.05, 40.0, size=(9, 1)), q * rng.uniform(0.05, 40.0, size=(4, 1)))
    if not close(s2, scores, 1e-5) or not np.array_equal(b2, best):
        return False, "把每个区域特征、每个查询向量各自乘上一个正数，结果不应该变"
    neg_scores, neg_best, neg_top = mod.score_regions(np.array([[-1.0, 0.0], [-1.0, -1.0], [0.0, -1.0]]), np.array([[1.0, 0.2]]))
    if neg_best.tolist() != [2] or neg_top[0] >= 0:
        return False, "所有区域的分数都是负数时，得分最高的是最不负的那个（不要取绝对值）"

    if not raises_value_error(mod.score_regions, r, q[:, :5]) or not raises_value_error(mod.score_regions, r, q[0]) or not raises_value_error(mod.score_regions, r[0], q):
        return False, "任何一个输入不是二维，或者两者的维数 D 不同时应抛出 ValueError"
    if not raises_value_error(mod.score_regions, np.zeros((0, 6)), q) or not raises_value_error(mod.score_regions, r, np.zeros((0, 6))):
        return False, "没有区域（R = 0）或者没有查询（Q = 0）时应抛出 ValueError"
    return True, "小例子、两边都归一化、并列、全 0 的区域和异常都正确"


def main() -> int:
    print("第 28 章进度检查\n" + "=" * 52)
    try:
        mod = load_assignment()
    except Exception as exc:  # noqa: BLE001
        report("FAIL", "加载作业", f"{type(exc).__name__}: {exc}")
        return 1
    results = [
        run("多模板平均的零样本分类器", lambda: check_classifier(mod)),
        run("零样本预测", lambda: check_predict(mod)),
        run("检索指标 Recall@K", lambda: check_recall(mod)),
        run("模态间隙的度量", lambda: check_gap(mod)),
        run("区域打分并取最大", lambda: check_regions(mod)),
    ]
    print("=" * 52)
    print(f"通过 {sum(results)}/{len(results)}")
    if not all(results):
        print("提示：TODO 是正常学习状态；实现后重新运行。")
        return 1
    print("零件通过；下一步在 Notebook 里把它们接到从零训练的双塔小模型上。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
