# -*- coding: utf-8 -*-
"""
test_core.py — 核心逻辑回归测试
运行: python test_core.py
覆盖: 投票计数 · 框颜色 · 余弦分类器 · 编码一致性 · 录入辅助函数
"""
import numpy as np
from collections import Counter

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
print("1. 三模型投票 + 框颜色")
print("=" * 50)

# 模拟: 三个预测, 加权投票, agree 计数, 框颜色
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
          f"vote=({cos_p},{knn_p},{svm_p}) -> agree={agree} (expect {expected_agree})")
    check(color_name == expected_color,
          f"color={color_name} (expect {expected_color})")

# 开集拒识: 低置信度 → agree 强制 0 → 红色
avg_conf = 0.30
if avg_conf < 0.55:
    agree = 0
box = (0, 255, 0) if agree >= 3 else (0, 255, 255) if agree >= 2 else (0, 0, 255)
check(box == (0, 0, 255), "开集拒识 (avg_conf=0.30) → red box")

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
print("4. 录入辅助函数")
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

print()
print("=" * 50)
print(f"结果: {passed} passed, {failed} failed")
print("=" * 50)

exit(0 if failed == 0 else 1)
