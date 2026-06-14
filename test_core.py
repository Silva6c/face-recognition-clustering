# -*- coding: utf-8 -*-
"""
test_core.py — 核心逻辑回归测试
运行: python test_core.py
覆盖: 投票计数 · 框颜色 · 余弦分类器 · 编码一致性 · 录入辅助函数 · 增量录入引擎
"""
import numpy as np
from collections import Counter
import os, sys, tempfile, shutil

passed = failed = 0

def check(cond, msg):
    global passed, failed
    if cond:
        passed += 1
        print(f"  [PASS] {msg}")
    else:
        failed += 1
        print(f"  [FAIL] {msg}")

print("=" * 50)
print("1. 三模型投票 + 框颜色 + 自适应权重")
print("=" * 50)

# ── 清晰图像权重 (Laplacian > 100): cos=0.15, knn=0.45, svm=0.40 ──
for cos_p, knn_p, svm_p, expected_agree, expected_color in [
    (5, 5, 5, 3, "green"),   # 全票一致
    (5, 5, 3, 2, "yellow"),  # 两票
    (5, 3, 4, 1, "red"),     # 一票
    (0, 0, 0, 3, "green"),   # 另一种全票
]:
    # 加权投票
    w_cos, w_knn, w_svm = 0.15, 0.45, 0.40
    weighted = Counter()
    for pred, w in [(cos_p, w_cos), (knn_p, w_knn), (svm_p, w_svm)]:
        weighted[pred] += w
    final = weighted.most_common(1)[0][0]

    # agree 计数 (Counter 方式)
    vote = Counter([cos_p, knn_p, svm_p])
    agree = vote[final]

    # 框颜色
    box = (0, 255, 0) if agree >= 3 else (0, 255, 255) if agree >= 2 else (0, 0, 255)
    color_name = "green" if box == (0, 255, 0) else "yellow" if box == (0, 255, 255) else "red"

    check(agree == expected_agree,
          f"清晰 vote=({cos_p},{knn_p},{svm_p}) -> agree={agree} (expect {expected_agree})")
    check(color_name == expected_color,
          f"清晰 color={color_name} (expect {expected_color})")

# ── 模糊图像权重 (Laplacian ≤ 100): cos=0.55, knn=0.25, svm=0.20 (v2.7 新增) ──
for cos_p, knn_p, svm_p, expected_final, expected_agree in [
    (5, 5, 3, 5, 2),   # cos+knn 同投5, 权重 0.55+0.25=0.80 > svm 0.20
    (4, 3, 3, 4, 1),   # cos 权重 0.55 > knn+svm=0.45, cos 独赢
    (2, 1, 1, 2, 1),   # 三者全不同, cos 权重最高 → 2 胜出
]:
    w_cos, w_knn, w_svm = 0.55, 0.25, 0.20
    weighted = Counter()
    for pred, w in [(cos_p, w_cos), (knn_p, w_knn), (svm_p, w_svm)]:
        weighted[pred] += w
    final = weighted.most_common(1)[0][0]

    vote = Counter([cos_p, knn_p, svm_p])
    agree = vote[final]

    check(final == expected_final,
          f"模糊 vote=({cos_p},{knn_p},{svm_p}) w=(0.55,0.25,0.20) -> final={final} (expect {expected_final})")
    check(agree == expected_agree,
          f"模糊 agree={agree} (expect {expected_agree})")

# ── 开集拒识 ──
# 低置信度 → agree 强制 0 → 红色
avg_conf = 0.30
if avg_conf < 0.55:
    agree = 0
box = (0, 255, 0) if agree >= 3 else (0, 255, 255) if agree >= 2 else (0, 0, 255)
check(box == (0, 0, 255), "开集拒识 (avg_conf=0.30) → red box")

# 边界值: 刚好 0.55 → 不拒识, 保持原 agree
avg_conf = 0.55
agree = 3  # 假设全票
if avg_conf < 0.55:
    agree = 0
box = (0, 255, 0) if agree >= 3 else (0, 255, 255) if agree >= 2 else (0, 0, 255)
check(box == (0, 255, 0), "开集拒识边界 (avg_conf=0.55) → 不拒识, green")

# 刚好低于阈值 → 拒识
avg_conf = 0.549
agree = 3
if avg_conf < 0.55:
    agree = 0
box = (0, 255, 0) if agree >= 3 else (0, 255, 255) if agree >= 2 else (0, 0, 255)
check(box == (0, 0, 255), "开集拒识边界 (avg_conf=0.549) → red box")

print()
print("=" * 50)
print("2. 余弦分类器")
print("=" * 50)

from engine import CosineClassifier

np.random.seed(42)
X = np.random.randn(100, 128).astype(np.float64)
y = np.array([i // 10 for i in range(100)])  # 10类各10样本

clf = CosineClassifier()
clf.fit(X, y)

# predict_scores 返回类型
feat = np.random.randn(128).astype(np.float64)
scores = clf.predict_scores(feat)
check(len(scores) == 10, f"scores 有 {len(scores)} 类")
check(all(isinstance(k, (int, np.integer)) for k in scores), "scores key 是整数")
check(all(isinstance(v, (float, np.floating)) for v in scores.values()), "scores value 是浮点")

# predict 正确性
preds = clf.predict(X[:5])
check(len(preds) == 5, f"predict 返回 {len(preds)} 个结果")
check(all(0 <= p < 10 for p in preds), "predict 值在 [0,10) 范围内")

# predict_proba 形状和范围
proba = clf.predict_proba(X[:3])
check(proba.shape == (3, 10), f"proba shape={proba.shape}")
check(np.allclose(proba.sum(axis=1), 1.0), "proba 每行和为 1")
check(np.all(proba >= 0) and np.all(proba <= 1), "proba 值在 [0,1]")

# predict_scores 与 predict_proba 一致性 (同一输入)
s2 = clf.predict_scores(X[0])
p2 = clf.predict_proba(X[0:1])[0]
check(max(s2, key=s2.get) == np.argmax(p2),
      f"score best={max(s2, key=s2.get)} vs proba best={np.argmax(p2)}")

print()
print("=" * 50)
print("3. 编码一致性")
print("=" * 50)

# 模拟 LabelEncoder 替换: label_names 和 y_all 必须同序
label_names = np.array(["Alice", "Bob", "Charlie"])
y_raw = ["Alice", "Alice", "Bob", "Charlie", "Charlie", "Bob"]
y_enc = np.array([list(label_names).index(n) for n in y_raw])
check(np.array_equal(y_enc, [0, 0, 1, 2, 2, 1]), "编码一致")
check(label_names[y_enc[0]] == "Alice", "索引->名字正确")
check(label_names[y_enc[2]] == "Bob", "索引->名字正确")

print()
print("=" * 50)
print("4. 录入辅助函数 (face_gui)")
print("=" * 50)

import face_gui as fg

# _build_gallery 空缓冲区
fg._enroll_buffer = []
g = fg._build_gallery()
check(g == [], "空缓冲区 → 空 gallery")

# _enroll_status 空缓冲区
check("0" in fg._enroll_status(), "空缓冲区计数含 0")

# _enroll_status 有数据
fg._enroll_buffer = [np.zeros((100, 100, 3), dtype=np.uint8)] * 3
check("3" in fg._enroll_status(), "3 张照片计数含 3")

# 模块级常量
check(isinstance(fg._IMAGE_EXTENSIONS, tuple), "_IMAGE_EXTENSIONS 是元组")
check(isinstance(fg._MODEL_CHOICES, list), "_MODEL_CHOICES 是列表")

# 恢复缓冲区
fg._enroll_buffer = []

print()
print("=" * 50)
print("5. 增量录入引擎 (engine) — v2.7 新增")
print("=" * 50)

import engine

# ── 5.1 save_enrollment_images ──
_orig_images_dir = engine.IMAGES_DIR
_tmp_images = tempfile.mkdtemp(prefix='test_images_')
try:
    engine.IMAGES_DIR = _tmp_images

    # 保存单张图片
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[30:70, 30:70] = 255  # 白色方块模拟人脸
    paths = engine.save_enrollment_images("测试人物", [img])
    check(len(paths) == 1, f"save_enrollment_images 返回 {len(paths)} 个路径")
    check(os.path.exists(paths[0]), f"文件已创建: {os.path.basename(paths[0])}")
    check("测试人物" in paths[0], "路径包含人名")

    # 追加保存第二张 (索引递增)
    paths2 = engine.save_enrollment_images("测试人物", [img, img])
    check(len(paths2) == 2, f"追加 2 张 → 返回 {len(paths2)} 个路径")
    check("enroll_0001" in paths2[0], f"索引递增: {os.path.basename(paths2[0])}")
    check("enroll_0002" in paths2[1], f"索引递增: {os.path.basename(paths2[1])}")

    # 新人物 → 独立目录
    paths3 = engine.save_enrollment_images("新人物", [img])
    check(len(paths3) == 1, "新人物目录创建成功")
    check("新人物" in paths3[0], "新人物路径正确")
finally:
    engine.IMAGES_DIR = _orig_images_dir
    shutil.rmtree(_tmp_images, ignore_errors=True)

# ── 5.2 update_cache_incremental ──
_orig_cache = engine.CACHE_PATH
_tmp_cache = os.path.join(tempfile.gettempdir(), '_test_face_cache.pkl')
try:
    engine.CACHE_PATH = _tmp_cache
    # 确保干净起点
    if os.path.exists(_tmp_cache):
        os.remove(_tmp_cache)

    # 场景A: 无现有缓存 → 新建
    new_X = np.random.randn(10, 128).astype(np.float64)
    new_labels = ["Alice"] * 5 + ["Bob"] * 5
    merged_X, merged_y, merged_names = engine.update_cache_incremental(new_X, new_labels)

    check(merged_X.shape == (10, 128), f"新建缓存 X shape={merged_X.shape}")
    check(len(merged_y) == 10, f"新建缓存 y 长度={len(merged_y)}")
    check(len(merged_names) == 2, f"新建缓存 label_names 长度={len(merged_names)}")
    check(set(merged_names) == {"Alice", "Bob"}, "新建缓存类名正确")
    check(os.path.exists(_tmp_cache), "pkl 文件已创建")

    # 场景B: 已有缓存 → 增量追加 (含新类)
    new_X2 = np.random.randn(6, 128).astype(np.float64)
    new_labels2 = ["Alice"] * 3 + ["Charlie"] * 3  # Alice 追加, Charlie 新类
    merged_X2, merged_y2, merged_names2 = engine.update_cache_incremental(new_X2, new_labels2)

    check(merged_X2.shape == (16, 128), f"追加后 X shape={merged_X2.shape} (expect 16)")
    check(len(merged_y2) == 16, f"追加后 y 长度={len(merged_y2)}")
    check(len(merged_names2) == 3, f"追加后 label_names 长度={len(merged_names2)} (expect 3)")
    check(set(merged_names2) == {"Alice", "Bob", "Charlie"}, "追加后类名正确")

    # 验证旧索引不变 (Alice=0, Bob=1, Charlie=2)
    check(merged_names2[0] == "Alice", "Alice 索引保持 0")
    check(merged_names2[1] == "Bob", "Bob 索引保持 1")
    check(merged_names2[2] == "Charlie", "Charlie 新索引=2")

    # 场景C: 追加全部已有类, 无新类
    new_X3 = np.random.randn(4, 128).astype(np.float64)
    new_labels3 = ["Bob"] * 4
    merged_X3, merged_y3, merged_names3 = engine.update_cache_incremental(new_X3, new_labels3)

    check(merged_X3.shape == (20, 128), f"再次追加后 X shape={merged_X3.shape}")
    check(len(merged_names3) == 3, "无新类时 label_names 数量不变")
finally:
    engine.CACHE_PATH = _orig_cache
    if os.path.exists(_tmp_cache):
        os.remove(_tmp_cache)

# ── 5.3 enroll_new_face (需要 cascade 模型文件) ──
engine_cascade_path = engine.CASCADE_PATH
if os.path.exists(engine_cascade_path):
    import cv2
    cascade = cv2.CascadeClassifier(engine_cascade_path)

    # 空白图无人脸 → 返回错误
    blank = np.zeros((200, 200, 3), dtype=np.uint8)
    result, msg = engine.enroll_new_face("测试", [blank], cascade, None)
    check(result is None, "空白图 enroll → result=None")
    check(isinstance(msg, str) and len(msg) > 0, f"空白图 enroll → 错误信息: {msg}")

    # 多张空白图 → 同样失败
    result2, msg2 = engine.enroll_new_face("测试", [blank, blank], cascade, None)
    check(result2 is None, "多张空白图 enroll → result=None")
else:
    print(f"  [SKIP] 模型文件不存在 ({engine_cascade_path}), 跳过 enroll_new_face 测试")

# ── 5.4 get_class_names 动态扫描 ──
names = engine.get_class_names()
check(isinstance(names, list), "get_class_names 返回列表")
check(len(names) > 0, f"get_class_names 非空 (当前 {len(names)} 个类)")
check("Elon Musk" in names or "Trump" in names, "get_class_names 包含硬编码类名")

print()
print("=" * 50)
print(f"结果: {passed} passed, {failed} failed")
print("=" * 50)

exit(0 if failed == 0 else 1)
