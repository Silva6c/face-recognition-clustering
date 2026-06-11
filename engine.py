# -*- coding: utf-8 -*-
"""
engine.py — 人脸识别核心引擎
共享模块: face_gui.py 和 report.py 均导入此模块

├── 配置常量
├── 图片读取 (imread_safe)
├── 人脸检测 (YuNet + Haar Cascade)
├── 特征提取 (dlib ResNet)
├── 数据增强
├── 余弦相似度分类器
└── 特征缓存 (load / build)
"""
import os, pickle, time, numpy as np, cv2, face_recognition
from sklearn.preprocessing import LabelEncoder

# ============================================================
# 配置
# ============================================================
BASE_DIR = r"G:\cvcv\No1"
IMAGES_DIR = os.path.join(BASE_DIR, "images")
CASCADE_PATH = os.path.join(BASE_DIR, "models", "haarcascade_frontalface_alt2.xml")
YUNET_PATH = os.path.join(BASE_DIR, "models", "face_detection_yunet_2023mar.onnx")
CACHE_PATH = os.path.join(BASE_DIR, "cache", "face_features_cache.pkl")

RANDOM_SEED = 42

CLASS_NAMES = [
    "Angelina Jolie", "Brad Pitt", "Denzel Washington",
    "Elon Musk", "Hugh Jackman", "Jen-hsun Huang",
    "Kobe", "Trump", "Xuefeng Zhang", "myself"
]
SMALL_THRESHOLD = 22
AUGMENT_FACTOR = 6
YUNET_CONF_THRESHOLD = 0.6

# ============================================================
# 图片读取
# ============================================================
def imread_safe(path):
    """兼容中文路径"""
    img = cv2.imread(path)
    if img is not None:
        return img
    with open(path, 'rb') as f:
        return cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)

# ============================================================
# 人脸检测
# ============================================================
def create_yunet():
    """创建 YuNet 检测器"""
    if not os.path.exists(YUNET_PATH):
        return None
    det = cv2.FaceDetectorYN.create(YUNET_PATH, '', (320, 320))
    det.setScoreThreshold(0.5)
    det.setNMSThreshold(0.3)
    return det

def detect_face_yunet(img, detector, conf_threshold=0.5):
    """YuNet 检测 → (x, y, w, h, conf) 或 None"""
    _, faces = detector.detect(img)
    if faces is None or len(faces) == 0:
        return None
    valid = faces[faces[:, -1] >= conf_threshold]
    if len(valid) == 0:
        return None
    best = valid[valid[:, -1].argmax()]
    x, y, w, h = best[:4].astype(int)
    return (max(0, x), max(0, y), w, h, float(best[-1]))

def detect_face_haar(img, cascade, scale_factor=1.05, min_neighbors=4):
    """Haar 回退检测 → (x, y, w, h) 或 None"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray, scaleFactor=scale_factor, minNeighbors=min_neighbors,
        minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE)
    if len(faces) == 0:
        return None
    return max(faces, key=lambda r: r[2] * r[3])

# ============================================================
# 特征提取
# ============================================================
def extract_feature(img_bgr, face_rect):
    """
    dlib ResNet 编码 → 128维 L2归一化向量

    创新: CLAHE 光照自适应 — 在编码前对全图做自适应直方图均衡，
    归一化全局亮度分布，提升低光/侧光条件下特征一致性。
    face_rect: (x, y, w, h)
    """
    x, y, w, h = face_rect
    try:
        # CLAHE 光照归一化
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab_eq = cv2.merge([clahe.apply(l), a, b])
        img_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

        img_rgb = cv2.cvtColor(img_eq, cv2.COLOR_BGR2RGB)
        encs = face_recognition.face_encodings(
            img_rgb, known_face_locations=[(y, x + w, y + h, x)])
        if encs:
            feat = encs[0].astype(np.float64)
            return feat / (np.linalg.norm(feat) + 1e-8)
    except Exception:
        pass
    return None

# ============================================================
# 数据增强
# ============================================================
def augment_image(img):
    """翻转 + 亮度±20% + 旋转±5°/±8° → 6个变体"""
    h, w = img.shape[:2]
    results = [cv2.flip(img, 1)]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    for factor in [1.2, 0.8]:
        adj = hsv.copy(); adj[:, :, 2] = np.clip(adj[:, :, 2] * factor, 0, 255)
        results.append(cv2.cvtColor(adj.astype(np.uint8), cv2.COLOR_HSV2BGR))

    for angle in [5, 8, -5]:
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        results.append(cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE))
    return results

# ============================================================
# 余弦相似度分类器
# ============================================================
class CosineClassifier:
    """对每类计算特征均值(原型), 新样本与各原型比余弦相似度"""
    def __init__(self):
        self.prototypes = {}
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for lb in self.classes_:
            proto = X[y == lb].mean(axis=0)
            proto /= (np.linalg.norm(proto) + 1e-8)
            self.prototypes[lb] = proto
        return self

    def _sims(self, X):
        """归一化 + 计算与所有原型的余弦相似度"""
        x_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        return np.column_stack([np.dot(x_norm, self.prototypes[lb])
                                for lb in self.classes_])

    def predict_scores(self, x):
        x_norm = x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
        return {lb: float(np.dot(x_norm, self.prototypes[lb]))
                for lb in self.classes_}

    def predict(self, X):
        return self.classes_[np.argmax(self._sims(X), axis=1)]

    def predict_proba(self, X):
        sims = np.clip(self._sims(X), 0, None)
        exp = np.exp((sims - sims.max(axis=1, keepdims=True)) * 10)
        return exp / exp.sum(axis=1, keepdims=True)

# ============================================================
# 特征缓存
# ============================================================
def load_or_build_features(face_images, labels_list, cascade, yunet):
    """加载特征缓存，不存在则重新提取并缓存"""
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, 'rb') as f:
            c = pickle.load(f)
        X = c['X']
        y_raw = [c['label_names'][i] for i in c['y']]
        return X, y_raw, c['label_names']

    print("  [首次] 提取 dlib 特征 (约 6-7 分钟)...")
    t0 = time.time()
    features, valid_labels = [], []
    for img, lb in zip(face_images, labels_list):
        rect = _detect_with_fallback(img, cascade, yunet)
        if rect is None: continue
        feat = extract_feature(img, rect)
        if feat is not None:
            features.append(feat); valid_labels.append(lb)
        if len(features) % 100 == 0 and features:
            print(f"    {len(features)} faces...")

    # 数据增强
    label_counts = {}
    for lb in valid_labels: label_counts[lb] = label_counts.get(lb, 0) + 1
    small_set = {lb for lb, c in label_counts.items() if c <= SMALL_THRESHOLD}

    aug_f, aug_l = [], []
    for feat, lb, img in zip(features, valid_labels, face_images):
        aug_f.append(feat); aug_l.append(lb)
        if lb in small_set:
            for aimg in augment_image(img)[:AUGMENT_FACTOR]:
                rect2 = _detect_with_fallback(aimg, cascade, yunet)
                if rect2 is None: continue
                af = extract_feature(aimg, rect2)
                if af is not None: aug_f.append(af); aug_l.append(lb)

    X = np.array(aug_f)
    le = LabelEncoder(); y_enc = le.fit_transform(aug_l)

    with open(CACHE_PATH, 'wb') as f:
        pickle.dump({'X': X, 'y': y_enc, 'label_names': le.classes_,
                     'label_counts': label_counts}, f)
    print(f"  特征已缓存 ({time.time()-t0:.0f}s)")
    return X, aug_l, le.classes_

def _detect_with_fallback(img, cascade, yunet):
    """内部检测器 — YuNet 主力 + Haar 回退"""
    h, w = img.shape[:2]
    if yunet is not None:
        try:
            yunet.setInputSize((w, h))
            r = detect_face_yunet(img, yunet, YUNET_CONF_THRESHOLD)
            if r is not None:
                x, y, fw, fh, _ = r
                if fw >= 60 and fh >= 60:
                    return (x, y, fw, fh)
        except Exception:
            pass
    return detect_face_haar(img, cascade)

# ============================================================
# 在线增量录入 (热更新)
# ============================================================
def get_class_names():
    """动态获取类名: 硬编码列表 + images/ 目录自动扫描新增文件夹"""
    names = list(CLASS_NAMES)
    if os.path.isdir(IMAGES_DIR):
        for d in sorted(os.listdir(IMAGES_DIR)):
            if os.path.isdir(os.path.join(IMAGES_DIR, d)) and d not in names:
                names.append(d)
    return names


def save_enrollment_images(name, images_bgr):
    """将采集的人脸照片保存到 images/{name}/ 目录, 供下次冷启动使用"""
    save_dir = os.path.join(IMAGES_DIR, name)
    os.makedirs(save_dir, exist_ok=True)
    existing = [f for f in os.listdir(save_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
    start_idx = len(existing)
    saved_paths = []
    for i, img in enumerate(images_bgr):
        fname = f"enroll_{start_idx + i:04d}.jpg"
        fpath = os.path.join(save_dir, fname)
        ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if ok:
            buf.tofile(fpath)
            saved_paths.append(fpath)
    return saved_paths


def enroll_new_face(name, images_bgr, cascade, yunet):
    """
    在线增量录入核心:
    1. 对每张照片检测 + 提取 dlib 128维特征
    2. 样本不足时自动数据增强
    返回 (new_X, new_labels) 或 (None, error_msg)
    """
    features, valid_imgs = [], []
    for img in images_bgr:
        rect = _detect_with_fallback(img, cascade, yunet)
        if rect is None:
            continue
        feat = extract_feature(img, rect)
        if feat is not None:
            features.append(feat)
            valid_imgs.append(img)

    if len(features) == 0:
        return None, "未能从任何照片中提取到人脸特征"

    # 数据增强 (少量样本自动扩充)
    aug_f, aug_l = [], []
    for feat, img in zip(features, valid_imgs):
        aug_f.append(feat)
        aug_l.append(name)
        if len(features) <= SMALL_THRESHOLD:
            for aimg in augment_image(img)[:AUGMENT_FACTOR]:
                rect2 = _detect_with_fallback(aimg, cascade, yunet)
                if rect2 is None:
                    continue
                af = extract_feature(aimg, rect2)
                if af is not None:
                    aug_f.append(af)
                    aug_l.append(name)

    new_X = np.array(aug_f)
    return new_X, aug_l


def update_cache_incremental(new_X, new_labels):
    """
    增量追加特征到 pkl 缓存 (不重建全量)
    - 加载现有缓存 → 合并新特征 & 标签 → 保存
    - 新类名自动扩展 label_names
    """
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, 'rb') as f:
            c = pickle.load(f)
        old_X = c['X']
        old_label_names = list(c['label_names'])
        old_y = list(c['y'])

        # 扩展 label_names (新类追加到末尾, 保持旧索引不变)
        for lb in set(new_labels):
            if lb not in old_label_names:
                old_label_names.append(lb)

        new_y = [old_label_names.index(lb) for lb in new_labels]
        merged_X = np.vstack([old_X, new_X])
        merged_y = np.array(old_y + new_y)
        merged_names = np.array(old_label_names)
    else:
        merged_names = np.array(sorted(set(new_labels)))
        merged_y = np.array([list(merged_names).index(lb) for lb in new_labels])
        merged_X = new_X

    with open(CACHE_PATH, 'wb') as f:
        pickle.dump({
            'X': merged_X,
            'y': merged_y,
            'label_names': merged_names,
        }, f)

    return merged_X, merged_y, merged_names
