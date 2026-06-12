# -*- coding: utf-8 -*-
"""
face_gui.py — 人脸识别与聚类系统 (Gradio Web 界面)
依赖: engine.py (核心引擎), face_features_cache.pkl (特征缓存)

启动: python face_gui.py
"""
import os, socket, time, threading, pickle, numpy as np, cv2, gradio as gr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, normalized_mutual_info_score
from collections import Counter
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage

# ── 导入核心引擎 ──
from engine import (
    BASE_DIR, IMAGES_DIR, CASCADE_PATH, YUNET_PATH, CACHE_PATH,
    RANDOM_SEED, CLASS_NAMES, YUNET_CONF_THRESHOLD,
    imread_safe, augment_image,
    detect_face_yunet, detect_face_haar,
    extract_feature, CosineClassifier,
    create_yunet, load_or_build_features,
    get_class_names, save_enrollment_images,
    enroll_new_face, update_cache_incremental,
)

# ============================================================
# 全局配置
# ============================================================
STREAM_MAX_DIM = 320
os.environ["OMP_NUM_THREADS"] = "3"
np.random.seed(RANDOM_SEED)

# 中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 全局状态
face_cascade = None
yunet_detector = None
label_names = None
cosine_clf = None
knn_clf = None
svm_model = None
scaler = None
X_all = None
y_all = None
X_scaled = None    # StandardScaler 缓存
X_tsne = None      # 延迟计算
km_labels = None
_tsne_ready = False

# 模块级常量
_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
_MODEL_CHOICES = ["余弦匹配", "KNN(k=3)", "SVM(RBF)"]

_enroll_buffer = []  # 录入采集缓冲区: list of BGR numpy arrays
_enroll_gallery_cache = []  # Gallery 标注缓存 (增量标注, 避免每帧重复计算)
_last_face_rect = None  # 轻量去重: 上一次入库的人脸 bbox
_last_capture_time = 0  # 轻量去重: 上一次入库时间戳
_capture_enabled = True  # 采集开关: 用户意图 (清空后=False, 摄像头重启后自动恢复)
_cam_was_off = True      # 追踪摄像头 None→帧 跳变 (用于自动恢复采集)
_pending_frame = None    # 待确认帧: 当前帧暂存, 等下一帧确认是 live 还是 stuck
_model_lock = threading.Lock()  # 模型热更新锁: 防止 do_recognize 与 hot_reload_classifiers 并发竞态

# 最近一次识别的三模型分数 (用于 Top-3 图表切换)
_last_cos_scores = {}
_last_knn_scores = {}
_last_svm_scores = {}
_last_final_idx = -1

# 摄像头流式的 Top-3 模型选择 (Timer 会覆盖, 需全局记忆)
_webcam_top3_choice = "余弦匹配"

# 中文字体路径探测 (PIL 渲染中文标签)
_FONT_PATH = None
for _font_candidate in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf",
                          "C:/Windows/Fonts/simsun.ttc", "C:/Windows/Fonts/msyhbd.ttf"]:
    if os.path.exists(_font_candidate):
        _FONT_PATH = _font_candidate
        break

# ============================================================
# 人脸检测 (YuNet + Haar 回退)
# ============================================================
def detect_face(img, fast_mode=False):
    """统一人脸检测接口"""
    h, w = img.shape[:2]
    if yunet_detector is not None:
        try:
            yunet_detector.setInputSize((w, h))
            conf_th = 0.5 if fast_mode else YUNET_CONF_THRESHOLD
            r = detect_face_yunet(img, yunet_detector, conf_th)
            if r is not None:
                x, y, fw, fh, _ = r
                if fw >= 60 and fh >= 60:
                    return (x, y, fw, fh)
        except Exception:
            pass
    sf = 1.1 if fast_mode else 1.05
    mn = 5 if fast_mode else 4
    return detect_face_haar(img, face_cascade, sf, mn)

# ============================================================
# 辅助
# ============================================================
def fig_to_pil(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0); plt.close(fig)
    return Image.open(buf)


def render_top3(model_choice):
    """根据用户选择, 从最近一次识别缓存中渲染对应模型的 Top-3 图表"""
    title_map = {"余弦匹配": "余弦相似度", "KNN(k=3)": "KNN 置信度", "SVM(RBF)": "SVM 置信度"}
    scores_map = {"余弦匹配": _last_cos_scores, "KNN(k=3)": _last_knn_scores, "SVM(RBF)": _last_svm_scores}

    scores = scores_map.get(model_choice, _last_cos_scores)
    if not scores:
        return None

    sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:3]
    # 若 _last_final_idx 无效(初始-1), 高亮最高分项
    highlight_idx = _last_final_idx if _last_final_idx >= 0 else sorted_scores[0][0]
    fig, ax = plt.subplots(figsize=(5, 2.5))
    names_top3 = [label_names[i].split()[-1] if ' ' in label_names[i]
                  else label_names[i] for i, _ in sorted_scores]
    vals_top3 = [max(s, 0) for _, s in sorted_scores]
    colors_bar = ['#2ecc71' if i == highlight_idx else '#bdc3c7' for i, _ in sorted_scores]
    ax.barh(names_top3[::-1], vals_top3[::-1], color=colors_bar[::-1])
    ax.set_xlim(0, 1)
    ax.set_title(f'{title_map.get(model_choice, "")} Top-3')
    for i, v in enumerate(vals_top3):
        ax.text(v + 0.02, 2-i, f'{v:.1%}', va='center')
    return fig_to_pil(fig)


def _draw_label_pil(img_bgr, text, x, y, box_color_bgr):
    """使用 PIL 绘制带中文支持的标签 (cv2.putText 的 Hershey 字体仅支持 ASCII)

    若无中文字体, 自动回退到 cv2.putText (仅 ASCII 文本可正常显示)"""
    if _FONT_PATH is None:
        # 回退: 使用 cv2 绘制标签 (仅支持 ASCII 字符)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(img_bgr, (x, y - th - 10), (x + tw, y), box_color_bgr, -1)
        cv2.putText(img_bgr, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 2)
        return img_bgr

    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(_FONT_PATH, 22)

    box_color_rgb = (box_color_bgr[2], box_color_bgr[1], box_color_bgr[0])
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = 6

    draw.rectangle([(x, y - th - pad * 2), (x + tw + pad, y)], fill=box_color_rgb)
    draw.text((x + pad // 2, y - th - pad), text, font=font, fill=(0, 0, 0))

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# ============================================================
# 初始化
# ============================================================
def initialize():
    global face_cascade, yunet_detector, label_names
    global cosine_clf, knn_clf, svm_model, scaler
    global X_all, y_all, X_scaled, X_tsne, km_labels

    print("=" * 60)
    print("人脸识别系统 — engine.py + face_gui.py")
    print("=" * 60)

    # 检测器
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    yunet_detector = create_yunet()
    print(f"[1/5] 检测器: {'YuNet + Haar' if yunet_detector else 'Haar only'}")

    # 加载数据 + 特征 (缓存命中则秒加载，否则重新提取)
    print("[2/5] 加载特征...")
    if os.path.exists(CACHE_PATH):
        # 缓存命中: 秒加载 (直接用缓存的编码, 避免 LabelEncoder 字母序重排导致索引错位)
        with open(CACHE_PATH, 'rb') as f:
            c = pickle.load(f)
        X_all = c['X']
        y_all = c['y']
        label_names = c['label_names']
        print(f"  缓存命中, 直接加载")
    else:
        # 首次: 扫描图片 → 检测 → 编码
        face_images, labels_list = [], []
        for cls in get_class_names():
            d = os.path.join(IMAGES_DIR, cls)
            if not os.path.isdir(d): continue
            for fn in sorted(os.listdir(d)):
                if not fn.lower().endswith(_IMAGE_EXTENSIONS): continue
                img = imread_safe(os.path.join(d, fn))
                if img is None: continue
                rect = detect_face(img, fast_mode=False)
                if rect is None: continue
                x, y, w, h = rect
                face_images.append(img[y:y+h, x:x+w].copy())
                labels_list.append(cls)
        X_all, y_raw, label_names = load_or_build_features(
            face_images, labels_list, face_cascade, yunet_detector)
        le = LabelEncoder(); y_all = le.fit_transform(y_raw)

    print(f"  样本: {X_all.shape}, 类别: {len(label_names)}")
    for i, name in enumerate(label_names):
        print(f"    {name:<25s}: {np.sum(y_all==i):>3d}")

    # 训练分类器 (仅余弦+投票，KNN和SVM延迟)
    print("[3/5] 训练分类器...")
    # 动态 stratify: 任一类样本数 < 5 时禁用分层抽样
    min_class_count = min(np.bincount(y_all))
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all, test_size=0.2, random_state=RANDOM_SEED,
        stratify=y_all if min_class_count >= 5 else None)

    cosine_clf = CosineClassifier().fit(X_tr, y_tr)
    knn_clf = KNeighborsClassifier(n_neighbors=3, metric='cosine').fit(X_tr, y_tr)
    scaler = StandardScaler()
    svm_model = SVC(kernel='rbf', C=10, gamma='scale', probability=True,
                    random_state=RANDOM_SEED).fit(scaler.fit_transform(X_tr), y_tr)

    print(f"  余弦: {accuracy_score(y_te, cosine_clf.predict(X_te)):.1%}  "
          f"KNN: {accuracy_score(y_te, knn_clf.predict(X_te)):.1%}  "
          f"SVM: {accuracy_score(y_te, svm_model.predict(scaler.transform(X_te))):.1%}")

    # 聚类 (结果缓存供可视化复用)
    print("[4/5] KMeans...")
    X_scaled = StandardScaler().fit_transform(X_all)
    km_labels = KMeans(n_clusters=len(label_names), random_state=RANDOM_SEED, n_init=10).fit_predict(X_scaled)

    print(f"\n[5/5] 就绪! {'YuNet' if yunet_detector else 'Haar'} + dlib + KNN/SVM  ({time.time():.0f})\n")

# ============================================================
# 识别核心
# ============================================================
def do_recognize(input_img, fast_mode=False, top3_model="余弦匹配"):
    if input_img is None:
        return None, None, 0, 0, "", None

    img_bgr = cv2.cvtColor(np.array(input_img), cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]
    scale = 1.0
    if fast_mode and max(h, w) > STREAM_MAX_DIM:
        scale = STREAM_MAX_DIM / max(h, w)
        img_detect = cv2.resize(img_bgr, (int(w*scale), int(h*scale)))
    else:
        img_detect = img_bgr

    rect = detect_face(img_detect, fast_mode=fast_mode)
    if rect is None:
        return (cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                None, 0, 0, "## 未检测到人脸\n请正对镜头, 光线充足", None)

    x, y, rw, rh = rect
    if scale != 1.0:
        x, y, rw, rh = int(x/scale), int(y/scale), int(rw/scale), int(rh/scale)

    feat = extract_feature(img_bgr, (x, y, rw, rh))
    if feat is None:
        return (cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                None, 0, 0, "## 特征提取失败", None)

    feat_2d = feat.reshape(1, -1)

    # 锁内快照分类器引用, 防止 hot_reload_classifiers 并发修改
    with _model_lock:
        cos_scores = cosine_clf.predict_scores(feat)
        cos_best = max(cos_scores, key=cos_scores.get)
        cos_probs = cosine_clf.predict_proba(feat_2d)[0]

        knn_pred = knn_clf.predict(feat_2d)[0]
        knn_probs = knn_clf.predict_proba(feat_2d)[0]

        feat_s = scaler.transform(feat_2d)
        svm_pred = svm_model.predict(feat_s)[0]
        svm_probs = svm_model.predict_proba(feat_s)[0]

    # ── 创新: 动态软投票 (基于图像清晰度的自适应权重) ──
    gray_face = cv2.cvtColor(img_bgr[y:y+rh, x:x+rw], cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray_face, cv2.CV_64F).var()
    if blur_score > 100:  # 清晰: SVM/KNN 决策边界更可靠
        w_cos, w_knn, w_svm = 0.15, 0.45, 0.40
    else:                 # 模糊: 全局余弦度量更鲁棒
        w_cos, w_knn, w_svm = 0.55, 0.25, 0.20

    weighted = Counter()
    for pred, weight in [(cos_best, w_cos), (knn_pred, w_knn), (svm_pred, w_svm)]:
        weighted[pred] += weight
    final = weighted.most_common(1)[0][0]
    final_name = label_names[final]

    cos_conf = max(cos_scores.get(final, 0), 0)
    knn_conf = knn_probs[final]
    svm_conf = svm_probs[final]
    avg_conf = (cos_conf + knn_conf + svm_conf) / 3
    # 共识度: 三模型置信度标准差越小越可信
    consensus = 1.0 - float(np.std([cos_conf, knn_conf, svm_conf]))

    vote = Counter([cos_best, knn_pred, svm_pred])
    agree = vote[final]

    # 开集拒识
    if avg_conf < 0.55:
        final_name = f"未知 ({final_name}?)"
        agree = 0

    # ── 创新: 撞脸检索 (KNN 最近邻) ──
    with _model_lock:
        distances, indices = knn_clf.kneighbors(feat_2d, n_neighbors=3)
        retrieval = []
        for d, idx in zip(distances[0], indices[0]):
            retrieval.append(f"| {label_names[y_all[idx]]} | {d:.4f} |")

    # 绘制
    result = img_bgr.copy()
    box_color = (0, 255, 0) if agree >= 3 else (0, 255, 255) if agree >= 2 else (0, 0, 255)
    cv2.rectangle(result, (x, y), (x+rw, y+rh), box_color, 3)
    label_text = f"{final_name} ({avg_conf:.0%})"
    # 使用 PIL 绘制中文标签 (cv2.putText 的 Hershey 字体仅支持 ASCII)
    result = _draw_label_pil(result, label_text, x, y, box_color)
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    detail = f"""
| 模型 | 预测 | 置信度 | 动态权重 |
|------|------|--------|----------|
| 余弦匹配 | {label_names[cos_best]} | {cos_probs[cos_best]:.1%} | {w_cos:.0%} |
| KNN(k=3) | {label_names[knn_pred]} | {knn_conf:.1%} | {w_knn:.0%} |
| SVM(RBF) | {label_names[svm_pred]} | {svm_conf:.1%} | {w_svm:.0%} |
| **软投票** | **{final_name}** | **{avg_conf:.1%}** | |

| 指标 | 值 |
|------|-----|
| 图像清晰度 | {blur_score:.0f} ({'清晰' if blur_score>100 else '模糊'}) |
| 模型共识度 | {consensus:.1%} |
| 票数 | {agree}/3 |

### 撞脸检索 (KNN 最近邻)
| 人物 | 欧氏距离 |
|------|----------|
{chr(10).join(retrieval)}
"""

    # 存储三模型分数 (供 Top-3 切换使用)
    global _last_cos_scores, _last_knn_scores, _last_svm_scores, _last_final_idx
    _last_cos_scores = cos_scores
    _last_knn_scores = {i: float(knn_probs[i]) for i in range(len(knn_probs))}
    _last_svm_scores = {i: float(svm_probs[i]) for i in range(len(svm_probs))}
    _last_final_idx = final

    # Top-3 图 (根据 top3_model 选择数据源)
    title_map = {"余弦匹配": "余弦相似度", "KNN(k=3)": "KNN 置信度", "SVM(RBF)": "SVM 置信度"}
    scores_map = {"余弦匹配": _last_cos_scores, "KNN(k=3)": _last_knn_scores, "SVM(RBF)": _last_svm_scores}
    chart_scores = scores_map.get(top3_model, _last_cos_scores)
    sorted_scores = sorted(chart_scores.items(), key=lambda kv: kv[1], reverse=True)[:3]
    fig, ax = plt.subplots(figsize=(5, 2.5))
    names_top3 = [label_names[i].split()[-1] if ' ' in label_names[i]
                  else label_names[i] for i, _ in sorted_scores]
    vals_top3 = [max(s, 0) for _, s in sorted_scores]
    colors_bar = ['#2ecc71' if i == final else '#bdc3c7' for i, _ in sorted_scores]
    ax.barh(names_top3[::-1], vals_top3[::-1], color=colors_bar[::-1])
    ax.set_xlim(0, 1); ax.set_title(f'{title_map.get(top3_model, "")} Top-3')
    for i, v in enumerate(vals_top3):
        ax.text(v + 0.02, 2-i, f'{v:.1%}', va='center')
    chart = fig_to_pil(fig)

    return result_rgb, final_name, agree, avg_conf, detail, chart

# ============================================================
# Gradio 回调
# ============================================================
def recognize(input_img, fast_mode=False, top3_model="余弦匹配"):
    """统一识别入口 — 图片/摄像头共用"""
    if input_img is None:
        msg = "等待摄像头画面..." if fast_mode else "请上传一张人脸图片"
        return None, msg, "", None
    result, name, votes, conf, detail, chart = do_recognize(input_img, fast_mode=fast_mode, top3_model=top3_model)
    if name is None:
        no_face = "## 未检测到人脸" if fast_mode else detail
        return result, no_face, "", chart
    status = "3/3" if fast_mode and votes >= 3 else "全票通过" if votes >= 3 else f"{votes}/3票"
    label = "" if fast_mode else f", 置信度 {conf:.1%}"
    summary = f"## {name}  ({status}{label})"
    return result, summary, detail, chart

# 向后兼容别名
recognize_image = recognize

def recognize_webcam(img):
    """摄像头回调 — 使用用户选择的 Top-3 模型"""
    return recognize(img, fast_mode=True, top3_model=_webcam_top3_choice)

def set_webcam_top3(choice):
    """更新摄像头 Top-3 选择 + 立即刷新图表"""
    global _webcam_top3_choice
    _webcam_top3_choice = choice
    return render_top3(choice)

# ============================================================
# 在线录入回调
# ============================================================
def _build_gallery():
    """增量构建 Gallery — 仅标注新增图片, 已标注的直接复用缓存"""
    while len(_enroll_gallery_cache) < len(_enroll_buffer):
        idx = len(_enroll_gallery_cache)
        _enroll_gallery_cache.append(_annotate_for_gallery(_enroll_buffer[idx]))
    return list(_enroll_gallery_cache)

def _enroll_status():
    return f"已采集: **{len(_enroll_buffer)}** / 20 张"

def _annotate_for_gallery(img_bgr):
    """给采集的照片画人脸框 (用于 Gallery 展示)"""
    rect = detect_face(img_bgr, fast_mode=True)
    display = img_bgr.copy()
    if rect is not None:
        x, y, w, h = rect
        cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return cv2.cvtColor(display, cv2.COLOR_BGR2RGB)


def process_video_frame(img_bgr):
    """视频流帧处理：人脸检测 → 质量检查 → 去重 → 入库
    返回 (ok, reason): ok=True 表示已入库，reason 为 "full"/"no_face"/"blurry"/"duplicate"/"ok"
    """
    global _enroll_buffer, _last_face_rect, _last_capture_time
    if len(_enroll_buffer) >= 20:
        return False, "full"
    rect = detect_face(img_bgr, fast_mode=True)
    if rect is None:
        return False, "no_face"
    x, y, w, h = rect
    # 清晰度检查 (仅人脸区域, 更快)
    face_gray = cv2.cvtColor(img_bgr[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    blur = cv2.Laplacian(face_gray, cv2.CV_64F).var()
    if blur < 80:
        return False, "blurry"
    # 轻量去重: bbox 中心位移 + 最小时间间隔 (替代 dlib 嵌入, 提速 ~30x)
    now = time.time()
    if _last_face_rect is not None:
        lx, ly, lw, lh = _last_face_rect
        cx_shift = abs((x + w/2) - (lx + lw/2)) / max(w, lw, 1)
        cy_shift = abs((y + h/2) - (ly + lh/2)) / max(h, lh, 1)
        elapsed = now - _last_capture_time
        # 脸没怎么动 + 间隔不够 → 跳过 (鼓励用户转头换角度)
        if cx_shift < 0.15 and cy_shift < 0.15 and elapsed < 0.8:
            return False, "duplicate"
    _last_face_rect = rect
    _last_capture_time = now
    _enroll_buffer.append(img_bgr)
    return True, "ok"


def video_tick(webcam_img):
    """视频流自动采集 — 由 Gradio 摄像头控件驱动启停

    ⚠️ 屎山警告 ⚠️
    此函数的状态机 (_cam_was_off / _capture_enabled / _pending_frame) 高度耦合,
    牵一发而动全身。已知问题:
    - 热更新(SVM重训练~1-2s)阻塞事件循环时, Gradio 向队列积压重复帧
    - 停滞检测单帧即判停 → _cam_was_off=True → 下一帧自动恢复 → 状态来回跳
    - 尝试过: 连续帧计数器、下采样比较、状态分支重写, 均引发新的连锁bug
    - 当前方案: 状态机保持原样, 底部提示文字全部置空, 眼不见为净
    - 若要重构: 建议将录入状态封装为 EnrollSession 类, 用显式状态机替代散乱的 bool 旗标

    待确认帧机制:
    - 当前帧不立即处理, 先存为 _pending_frame
    - 下一帧到来时: 若不同 → pending 是 live 帧 → 确认入库; 若相同 → 摄像头已停止 → 丢弃
    - 根除"停止后多入库1张"的结构性缺陷 (停滞检测与比较基准永远滞后1帧)
    """
    global _capture_enabled, _cam_was_off, _pending_frame

    # ── 摄像头未开 (Gradio 发送 None) ──
    if webcam_img is None:
        _cam_was_off = True
        _pending_frame = None
        return _build_gallery(), _enroll_status(), ""

    arr = np.array(webcam_img)

    # ── 停滞检测: pending 与当前帧相同 → 摄像头已停止, 丢弃 pending ──
    if _pending_frame is not None and np.array_equal(arr, _pending_frame):
        _cam_was_off = True
        _capture_enabled = False
        _pending_frame = None
        return _build_gallery(), _enroll_status(), ""

    # ── pending 与当前帧不同 → pending 是 live 帧, 确认入库 ──
    if _pending_frame is not None and _capture_enabled and len(_enroll_buffer) < 20:
        img_bgr = cv2.cvtColor(_pending_frame, cv2.COLOR_RGB2BGR)
        process_video_frame(img_bgr)

    # ── 当前帧变为新的 pending ──
    _pending_frame = arr.copy()

    # ── 摄像头刚启动 → 自动恢复采集 ──
    if _cam_was_off:
        _cam_was_off = False
        _capture_enabled = True

    # ── 暂停中 ──
    if not _capture_enabled:
        return _build_gallery(), _enroll_status(), ""

    # ── 已满 20 张 ──
    if len(_enroll_buffer) >= 20:
        return _build_gallery(), _enroll_status(), ""

    # ── 返回状态 (pending 待确认, 显示已入库数量) ──
    gallery = _build_gallery()
    n = len(_enroll_buffer)
    if n >= 20:
        gr.Info("采集完成！20 张已满")
    return (gallery, _enroll_status(), "")


def clear_enroll():
    """清空采集缓冲区并暂停采集, 需重启摄像头恢复"""
    global _enroll_buffer, _enroll_gallery_cache
    global _last_face_rect, _last_capture_time
    global _capture_enabled, _pending_frame
    _enroll_buffer = []
    _enroll_gallery_cache = []
    _last_face_rect = None
    _last_capture_time = 0
    _capture_enabled = False
    _pending_frame = None
    gr.Info("已清空采集缓冲区")
    return [], "已采集: **0** / 20 张", "⏸️ 已清空，点击摄像头「停止」再点「录制」重新开始", None


def batch_import_files(files):
    """批量导入文件: 自动检测人脸 → 入缓冲区, 选完即处理"""
    global _enroll_buffer
    if files is None or len(files) == 0:
        gallery = _build_gallery()
        gr.Warning("未选择任何文件")
        return (gallery, _enroll_status(),
                "⚠️ 未选择任何文件", None)

    total = len(files)
    added = 0
    skipped_no_face = 0
    skipped_full = 0

    for f in files:
        if len(_enroll_buffer) >= 20:
            skipped_full += 1
            continue
        img_bgr = imread_safe(f.name if hasattr(f, 'name') else f)
        if img_bgr is None:
            continue
        rect = detect_face(img_bgr, fast_mode=True)
        if rect is None:
            skipped_no_face += 1
            continue
        _enroll_buffer.append(img_bgr)
        added += 1

    gallery = _build_gallery()
    parts = []
    if added > 0:
        parts.append(f"✅ 已添加 **{added}** 张人脸")
    if skipped_no_face > 0:
        parts.append(f"⚠️ {skipped_no_face} 张未检测到人脸")
    if skipped_full > 0:
        parts.append(f"⏭️ {skipped_full} 张因缓冲区已满跳过")
    status = f"📁 从 {total} 个文件中: " + (", ".join(parts) if parts else "无有效图片")

    # Toast 强提醒
    if added > 0 and skipped_no_face == 0 and skipped_full == 0:
        gr.Info(f"已添加 {added} 张人脸")
    elif added > 0:
        gr.Warning(f"已添加 {added} 张，{skipped_no_face + skipped_full} 张跳过")
    else:
        gr.Warning("未能添加任何有效人脸")

    return (gallery, _enroll_status(), status, None)


def do_enroll(name):
    """确认录入新面孔 — 提取特征 + 热更新分类器"""
    global _enroll_buffer, _enroll_gallery_cache, _last_face_rect, _last_capture_time

    if not name or not name.strip():
        gallery = _build_gallery()
        gr.Warning("请先输入姓名")
        return "⚠️ 请先输入姓名", gallery, _enroll_status(), gr.skip()
    if len(_enroll_buffer) < 3:
        gallery = _build_gallery()
        gr.Warning(f"至少需要 3 张照片（当前 {len(_enroll_buffer)} 张）")
        return f"⚠️ 至少需要采集 **3** 张照片（当前 {len(_enroll_buffer)} 张）", gallery, _enroll_status(), gr.skip()

    name = name.strip()
    images_bgr = list(_enroll_buffer)

    # 1. 保存照片到磁盘
    saved = save_enrollment_images(name, images_bgr)

    # 2. 提取特征 + 增强
    new_X, result = enroll_new_face(name, images_bgr, face_cascade, yunet_detector)
    if new_X is None:
        gallery = _build_gallery()
        gr.Error(f"录入失败: {result}")
        return f"❌ 录入失败: {result}", gallery, _enroll_status(), gr.skip()

    # 3. 更新缓存
    update_cache_incremental(new_X, result)

    # 4. 热更新分类器
    hot_reload_classifiers(new_X, result)

    n_features = len(new_X)
    n_raw = len(images_bgr)
    _enroll_buffer = []
    _enroll_gallery_cache = []
    _last_face_rect = None
    _last_capture_time = 0

    gr.Info(f"✅ {name} 录入成功！")
    return (
        f"## ✅ 录入成功！\n\n"
        f"| 项目 | 数据 |\n"
        f"|------|------|\n"
        f"| 姓名 | **{name}** |\n"
        f"| 原始照片 | {n_raw} 张 |\n"
        f"| 特征向量 | {n_features} 条（含增强）|\n"
        f"| 总样本数 | {len(X_all)} |\n"
        f"| 总类别数 | {len(label_names)} |\n\n"
        f"> 💡 可立即前往「📷 图片识别」或「🎥 摄像头实时」验证",
        [],
        "已采集: **0** / 20 张",
        ""  # 清空姓名框
    )


def hot_reload_classifiers(new_X, new_labels):
    """热更新全局分类器 (CosineClassifier + KNN + SVM), 失败自动回滚"""
    global X_all, y_all, label_names, cosine_clf, knn_clf, svm_model, scaler
    global X_scaled, km_labels, X_tsne, _tsne_ready

    with _model_lock:
        # 保存旧状态快照, 用于失败回滚
        _snap = (
            X_all.copy(), y_all.copy(), label_names.copy(),
            X_scaled.copy() if X_scaled is not None else None,
            km_labels.copy() if km_labels is not None else None,
        )

        try:
            # 扩展 label_names
            label_names_list = list(label_names)
            for lb in set(new_labels):
                if lb not in label_names_list:
                    label_names_list.append(lb)
            label_names = np.array(label_names_list)

            # 编码新标签 & 追加数据
            new_y = np.array([label_names_list.index(lb) for lb in new_labels])
            X_all = np.vstack([X_all, new_X])
            y_all = np.concatenate([y_all, new_y])

            # 重新训练所有分类器 (数据量<千级, refit < 1s)
            # 动态 stratify: 任一类样本数 < 5 时禁用分层抽样, 避免 ValueError
            min_class_count = min(np.bincount(y_all))
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_all, y_all, test_size=0.2, random_state=RANDOM_SEED,
                stratify=y_all if min_class_count >= 5 else None)

            cosine_clf = CosineClassifier().fit(X_tr, y_tr)
            knn_clf = KNeighborsClassifier(n_neighbors=3, metric='cosine').fit(X_tr, y_tr)
            scaler = StandardScaler()
            svm_model = SVC(kernel='rbf', C=10, gamma='scale', probability=True,
                            random_state=RANDOM_SEED).fit(scaler.fit_transform(X_tr), y_tr)

            # 更新聚类
            X_scaled = StandardScaler().fit_transform(X_all)
            km_labels = KMeans(n_clusters=len(label_names), random_state=RANDOM_SEED,
                               n_init=10).fit_predict(X_scaled)

            # t-SNE 缓存失效, 下次点击可视化时重算
            _tsne_ready = False
            X_tsne = None

            acc_cos = accuracy_score(y_te, cosine_clf.predict(X_te))
            acc_knn = accuracy_score(y_te, knn_clf.predict(X_te))
            acc_svm = accuracy_score(y_te, svm_model.predict(scaler.transform(X_te)))
            print(f"  [热更新] {len(label_names)} 类, {len(X_all)} 样本 | "
                  f"余弦:{acc_cos:.1%} KNN:{acc_knn:.1%} SVM:{acc_svm:.1%}")

        except Exception as e:
            # 回滚到旧状态
            X_all, y_all, label_names, old_scaled, old_km = _snap
            if old_scaled is not None:
                X_scaled = old_scaled
            if old_km is not None:
                km_labels = old_km
            print(f"  [热更新失败] {e}，已回滚至更新前状态")
            gr.Warning(f"⚠️ 分类器热更新失败 ({e})，已自动回滚。请重新尝试录入。")

# ============================================================
# 可视化回调 (t-SNE 延迟计算)
# ============================================================
def _ensure_tsne():
    """延迟计算 t-SNE — 首次点击可视化时触发，之后缓存"""
    global X_tsne, _tsne_ready
    if _tsne_ready:
        return
    print("  计算 t-SNE (约60秒, 仅首次)...")
    tsne = TSNE(n_components=2, random_state=RANDOM_SEED,
                perplexity=min(30, len(X_all)//10), max_iter=1000)
    X_tsne = tsne.fit_transform(X_all)
    _tsne_ready = True
    print("  t-SNE 完成")

def show_tsne():
    _ensure_tsne()
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.cm.get_cmap('tab10', len(label_names))
    for i, name in enumerate(label_names):
        mask = y_all == i
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                   c=[cmap(i)], label=f"{name} ({np.sum(mask)})",
                   alpha=0.6, s=15, edgecolors='none')
    ax.set_title('t-SNE 特征分布 (dlib 128维)')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7)
    return fig_to_pil(fig)

def show_clusters():
    _ensure_tsne()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    cmap = plt.cm.get_cmap('tab10', len(label_names))
    for i, name in enumerate(label_names):
        mask = y_all == i
        ax1.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                    c=[cmap(i)], label=name, alpha=0.5, s=12, edgecolors='none')
    ax1.set_title('真实标签'); ax1.legend(bbox_to_anchor=(1.02,1), loc='upper left', fontsize=7)
    nc = len(set(km_labels)); cl_cmap = plt.cm.get_cmap('Set3', nc)
    for i in range(nc):
        ax2.scatter(X_tsne[km_labels==i,0], X_tsne[km_labels==i,1],
                    c=[cl_cmap(i)], alpha=0.5, s=12, edgecolors='none', label=f'簇{i}')
    ax2.set_title(f'KMeans (NMI={normalized_mutual_info_score(y_all, km_labels):.3f})')
    ax2.legend(bbox_to_anchor=(1.02,1), loc='upper left', fontsize=7)
    return fig_to_pil(fig)

def show_dendrogram():
    Z = linkage(X_scaled, method='ward')
    fig, ax = plt.subplots(figsize=(16, 7))
    dendrogram(Z, truncate_mode='lastp', p=30, leaf_rotation=90,
               leaf_font_size=8, show_contracted=True)
    ax.axhline(np.median(Z[:,2]), color='red', ls='--', alpha=0.5, label='中位距离')
    ax.set_title('层次聚类树状图 (Ward)'); ax.legend()
    return fig_to_pil(fig)

def show_kdistance():
    neigh = NearestNeighbors(n_neighbors=5, metric='euclidean')
    k_dist = np.sort(neigh.fit(X_scaled).kneighbors(X_scaled)[0][:,4])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(k_dist, linewidth=1)
    ax.set_xlabel('样本'); ax.set_ylabel('第5近邻距离')
    ax.set_title('DBSCAN k-distance'); ax.grid(True, alpha=0.3)
    return fig_to_pil(fig)

# ============================================================
# Gradio UI
# ============================================================
CSS = """
.main-title { text-align: center; margin-bottom: 1rem; }
.result-box { border: 2px solid #e0e0e0; border-radius: 12px; padding: 1rem; }
footer { visibility: hidden; }
"""

def build_ui():
    with gr.Blocks(title="人脸识别与聚类系统") as demo:
        gr.Markdown("""
        <div class="main-title">
        <h1>🔍 人脸识别与聚类系统</h1>
        <p>YuNet DNN 检测 · dlib ResNet 编码 · 余弦 + KNN + SVM 三模型投票</p>
        </div>
        """)

        with gr.Tabs():
            # ── Tab 1: 图片识别 ──
            with gr.TabItem("📷 图片识别"):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=2):
                        inp_img = gr.Image(label="上传人脸图片", type="numpy", height=400)
                        with gr.Row():
                            btn_rec = gr.Button("🔍 开始识别", variant="primary", size="lg")
                            btn_clr = gr.Button("清空", size="lg")
                    with gr.Column(scale=3):
                        out_img = gr.Image(label="检测结果", elem_classes="result-box", height=400)
                        out_md = gr.Markdown("")

                gr.Markdown("---")
                with gr.Row():
                    with gr.Column():
                        out_detail = gr.Markdown("### 三模型预测详情")
                    with gr.Column():
                        top3_radio = gr.Radio(
                            choices=_MODEL_CHOICES,
                            value="余弦匹配", label="Top-3 模型切换", interactive=True)
                        out_chart = gr.Image(label="Top-3 图表", type="pil")

                btn_rec.click(fn=recognize_image, inputs=[inp_img],
                              outputs=[out_img, out_md, out_detail, out_chart])
                top3_radio.change(fn=render_top3, inputs=[top3_radio],
                                  outputs=[out_chart])
                btn_clr.click(fn=lambda: (None, None, "", "", None),
                              outputs=[inp_img, out_img, out_md, out_detail, out_chart])

                gr.Markdown("""
                > 💡 **提示**: 上传正面人脸照片 → 点击识别 → 三模型投票。
                绿框 = 全票一致 · 黄框 = 2票 · 红框 = 仅1票
                """)

            # ── Tab 2: 摄像头实时 ──
            with gr.TabItem("🎥 摄像头实时"):
                gr.Markdown("### 实时人脸识别 · 每0.5秒自动刷新")

                with gr.Row():
                    webcam_in = gr.Image(label="摄像头", type="numpy",
                                         sources=["webcam"], streaming=True)
                    webcam_out = gr.Image(label="识别结果")

                webcam_md = gr.Markdown("等待摄像头...")

                with gr.Row():
                    with gr.Column():
                        webcam_detail = gr.Markdown("")
                    with gr.Column():
                        webcam_top3_radio = gr.Radio(
                            choices=_MODEL_CHOICES,
                            value="余弦匹配", label="Top-3 切换", interactive=True)
                        webcam_chart = gr.Image(label="Top-3 图表", type="pil")

                timer = gr.Timer(0.5)
                timer.tick(fn=recognize_webcam, inputs=[webcam_in],
                           outputs=[webcam_out, webcam_md, webcam_detail, webcam_chart])
                webcam_top3_radio.change(fn=set_webcam_top3, inputs=[webcam_top3_radio],
                                         outputs=[webcam_chart])

                gr.Markdown("""
                > 💡 **提示**: 点击「录制」开始摄像头，「结束」停止。识别自动进行，每 0.5 秒刷新结果。
                下方显示三模型投票详情和余弦相似度 Top-3 图表。
                """)

            # ── Tab: 录入新面孔 ──
            with gr.TabItem("✏️ 录入新面孔"):
                gr.Markdown("""
                ### 在线录入新面孔 · 热更新
                > 采集 3~20 张不同角度/表情的照片，输入姓名后一键录入，**无需重启系统**。
                """)

                with gr.Row(equal_height=True):
                    with gr.Column(scale=2):
                        enroll_cam = gr.Image(label="📸 摄像头采集", type="numpy",
                                              sources=["webcam"], streaming=True,
                                              height=300)
                        with gr.Row():
                            btn_clear_enroll = gr.Button("🔄 清空重来")
                        enroll_count = gr.Markdown("已采集: **0** / 20 张")
                    with gr.Column(scale=3):
                        enroll_gallery = gr.Gallery(label="已采集照片",
                                                    columns=5, height=280)

                gr.Markdown("---")
                gr.Markdown("#### 📁 批量导入 (从文件选择)")

                with gr.Row():
                    batch_files = gr.File(file_count="multiple",
                                          label="选择多张人脸照片",
                                          file_types=[".jpg", ".jpeg", ".png", ".bmp", ".webp"])

                gr.Markdown("---")
                with gr.Row():
                    enroll_name = gr.Textbox(label="姓名",
                                             placeholder="输入要录入的姓名...",
                                             scale=3)
                    btn_enroll = gr.Button("✅ 确认录入", variant="primary",
                                           size="lg", scale=1)

                enroll_status = gr.Markdown("等待采集...")

                # 视频流自动采集 — Timer 驱动，读 enroll_cam 不写它
                video_timer = gr.Timer(0.3)
                video_timer.tick(fn=video_tick, inputs=[enroll_cam],
                                 outputs=[enroll_gallery, enroll_count, enroll_status])

                # 清空全部 (缓冲区 + 摄像头 + 文件选择)
                btn_clear_enroll.click(
                    fn=clear_enroll,
                    outputs=[enroll_gallery, enroll_count, enroll_status,
                             batch_files])
                # 批量文件导入 → 选完自动处理
                batch_files.upload(
                    fn=batch_import_files, inputs=[batch_files],
                    outputs=[enroll_gallery, enroll_count, enroll_status, batch_files])
                # 确认录入 → 热更新 → 清空姓名框
                btn_enroll.click(
                    fn=do_enroll, inputs=[enroll_name],
                    outputs=[enroll_status, enroll_gallery, enroll_count, enroll_name])

                gr.Markdown("""
                > 💡 **两种录入方式**:
                > 1. 📹 **录制采集**: 点击「录制」→ 自动抽帧入库 →「停止」结束；🔄 清空重来会暂停采集，需重新开关摄像头恢复
                > 2. 📁 **批量导入**: 选择多个文件 → 自动检测人脸并添加 (无需额外点击)
                > 录入后三个分类器（余弦/KNN/SVM）同步热更新，立即生效。
                """)

            # ── Tab 4: 可视化 ──
            with gr.TabItem("📊 可视化"):
                gr.Markdown("### 数据可视化分析")
                with gr.Row():
                    btn1 = gr.Button("t-SNE 特征分布")
                    btn2 = gr.Button("KMeans vs 真实标签")
                with gr.Row():
                    btn3 = gr.Button("层次聚类树状图")
                    btn4 = gr.Button("DBSCAN k-distance")

                viz_out = gr.Image(label="可视化结果", type="pil")
                btn1.click(fn=show_tsne, outputs=viz_out)
                btn2.click(fn=show_clusters, outputs=viz_out)
                btn3.click(fn=show_dendrogram, outputs=viz_out)
                btn4.click(fn=show_kdistance, outputs=viz_out)

            # ── Tab 4: 系统信息 ──
            with gr.TabItem("ℹ️ 系统信息"):
                gr.Markdown("### 模型配置与性能指标")
                with gr.Row():
                    btn_info = gr.Button("刷新模型信息", variant="primary")
                    btn_stats = gr.Button("数据集统计")
                info_out = gr.Markdown("")
                btn_info.click(fn=get_model_info, outputs=info_out)
                btn_stats.click(fn=get_stats, outputs=info_out)

    return demo

def get_model_info():
    # 复用 initialize() 已训练的全局分类器，不重新拟合
    # 动态 stratify
    min_class_count = min(np.bincount(y_all))
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all, test_size=0.2, random_state=RANDOM_SEED,
        stratify=y_all if min_class_count >= 5 else None)
    X_te_s = scaler.transform(X_te)

    cos_acc = accuracy_score(y_te, cosine_clf.predict(X_te))
    knn_acc = accuracy_score(y_te, knn_clf.predict(X_te))
    svm_acc = accuracy_score(y_te, svm_model.predict(X_te_s))

    rows = "\n".join(f"| {name} | {np.sum(y_all==i)} |"
                     for i, name in enumerate(label_names))

    return f"""
### 系统配置

| 组件 | 方案 |
|------|------|
| 人脸检测 | {'YuNet DNN (320px)' if yunet_detector else 'Haar Cascade'} |
| 特征编码 | dlib ResNet-34 (128维) |
| 分类器 | 余弦匹配 + KNN(k=3) + SVM(RBF) 三投票 |

### 数据集
| 类别 | 样本数 |
|------|--------|
{rows}
| **总计** | **{len(y_all)}** |

### 分类器准确率
| 算法 | 准确率 |
|------|--------|
| 余弦匹配 | {cos_acc:.1%} |
| KNN(k=3) | {knn_acc:.1%} |
| SVM(RBF) | {svm_acc:.1%} |
"""

def get_stats():
    lines = ["| 类别 | 原始图片 | 增强后 |",
             "|------|----------|--------|"]
    label_names_list = list(label_names)
    for cls in get_class_names():
        d = os.path.join(IMAGES_DIR, cls)
        if not os.path.isdir(d): continue
        raw = len([f for f in os.listdir(d)
                   if f.lower().endswith(_IMAGE_EXTENSIONS)])
        aug = int(np.sum(y_all == label_names_list.index(cls))) if cls in label_names_list else 0
        lines.append(f"| {cls} | {raw} | {aug} |")
    return "\n".join(lines)

# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    print("\n正在初始化...\n")
    initialize()

    demo = build_ui()

    port = 7860
    for p in range(7860, 7920):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', p)) != 0:
                port = p; break

    theme = gr.themes.Soft(primary_hue="blue", secondary_hue="slate")
    print(f"\n启动 http://127.0.0.1:{port}\n")
    demo.launch(server_name="127.0.0.1", server_port=port,
                share=False, theme=theme, css=CSS, quiet=True)
