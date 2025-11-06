from __future__ import annotations
import cv2
import numpy as np
from skimage import feature, color
from dataclasses import dataclass, asdict
import json
# ========= 修复 float32 无法保存到 JSON 的问题 =========
import numpy as np
def np_convert(obj):
    """将 numpy 类型（float32 等）转换为普通 Python 类型"""
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError
import os
from tqdm import tqdm
import mediapipe as mp
import exifread
import matplotlib.pyplot as plt

# ========== LOONOOL Skin Vision Engine v1.0 ==========
# 功能：对 1~N 张自拍照片进行标准化、分析、对比与趋势输出

@dataclass
class Features:
    brightness_mean: float
    brightness_cv: float
    redness_proxy: float
    yellowness_proxy: float
    texture_entropy: float
    sharpness_lap_var: float
    highfreq_energy: float
    gloss_ratio: float

def read_exif_datetime(path: str) -> str | None:
    try:
        with open(path, 'rb') as f:
            tags = exifread.process_file(f, details=False)
        dt = tags.get('EXIF DateTimeOriginal') or tags.get('Image DateTime')
        return str(dt) if dt else None
    except Exception:
        return None
# ---------- 图像标准化 ----------
mp_face = mp.solutions.face_mesh

def align_and_normalize(rgb: np.ndarray) -> np.ndarray:
    """使用 Mediapipe 对齐人脸并标准化光照"""
    h, w, _ = rgb.shape
    with mp_face.FaceMesh(static_image_mode=True, max_num_faces=1) as fm:
        res = fm.process(rgb)
        if not res.multi_face_landmarks:
            # 没检测到人脸，直接居中裁剪
            side = min(h, w)
            cx, cy = w // 2, h // 2
            crop = rgb[cy - side//2:cy + side//2, cx - side//2:cx + side//2]
            return cv2.resize(crop, (512, 512))
        lm = res.multi_face_landmarks[0]
        pts = np.array([[p.x * w, p.y * h] for p in lm.landmark], dtype=np.float32)
    # 取两眼作为对齐基准
    left_eye = pts[[33, 133]].mean(axis=0)
    right_eye = pts[[362, 263]].mean(axis=0)
    dx, dy = right_eye - left_eye
    angle = np.degrees(np.arctan2(dy, dx))
    M = cv2.getRotationMatrix2D(tuple(((left_eye + right_eye) / 2)), angle, 1.0)
    rotated = cv2.warpAffine(rgb, M, (w, h), flags=cv2.INTER_LINEAR)
    # 裁剪中心区域
    side = min(rotated.shape[0], rotated.shape[1])
    cx, cy = rotated.shape[1] // 2, rotated.shape[0] // 2
    crop = rotated[cy - side//2:cy + side//2, cx - side//2:cx + side//2]
    resized = cv2.resize(crop, (512, 512))
    # 光照均衡（CLAHE）
    ycrcb = cv2.cvtColor(resized, cv2.COLOR_RGB2YCrCb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

# ---------- 特征提取 ----------
def lbp_entropy(gray: np.ndarray) -> float:
    lbp = feature.local_binary_pattern(gray, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10), density=True)
    hist += 1e-8
    return float(-(hist * np.log(hist)).sum())

def extract_features(rgb: np.ndarray) -> Features:
    lab = color.rgb2lab(rgb / 255.0)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    y = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)[:, :, 0]
    bright_mean = np.mean(y)
    bright_cv = np.std(y) / (np.mean(y) + 1e-6)
    red = np.mean(lab[:, :, 1])
    yellow = np.mean(lab[:, :, 2])
    texture = lbp_entropy(gray)
    sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
    highfreq = np.mean(np.abs(gray.astype(np.float32) - cv2.GaussianBlur(gray, (0, 0), 1)))
    gloss = ((y > 240).sum() / (y.size + 1e-6))
    return Features(bright_mean, bright_cv, red, yellow, texture, sharp, highfreq, gloss)
# ---------- 单张分析、对比与趋势 ----------

def load_rgb(path: str) -> np.ndarray:
    arr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise ValueError(f"无法读取图片：{path}")
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

def analyze_one(path: str) -> dict:
    rgb = load_rgb(path)
    exif_dt = read_exif_datetime(path)
    rgb_std = align_and_normalize(rgb)
    feats = extract_features(rgb_std)
    return {
        "path": path,
        "exif_datetime": exif_dt,
        "features": asdict(feats)
    }

def pairwise_diff(a: dict, b: dict) -> dict:
    fa, fb = a["features"], b["features"]
    keys = list(fa.keys())
    deltas = {k: float(fb[k] - fa[k]) for k in keys}
    mag = float(np.linalg.norm([deltas[k] for k in keys]))
  # ----- 置信度计算 -----
    try:
        illum = 1 / (1 + abs(fb["brightness_mean"] - fa["brightness_mean"]) / max(1, fa["brightness_mean"]))
        uniform = 1 / (1 + abs(fb["brightness_cv"] - fa["brightness_cv"]) * 100)
        sharp = 1 / (1 + abs(fb["sharpness_lap_var"] - fa["sharpness_lap_var"]) / 500)
        color = 1 / (1 + (abs(fb["redness_proxy"] - fa["redness_proxy"]) +
                          abs(fb["yellowness_proxy"] - fa["yellowness_proxy"])) / 20)
        confidence = round(0.4*illum + 0.2*uniform + 0.3*sharp + 0.1*color, 3)
    except Exception:
        confidence = None

    deltas = {k: float(fb[k] - fa[k]) for k in keys}
    magnitude = float(np.linalg.norm([deltas[k] for k in keys]))
    return {"a": a["path"], "b": b["path"], "deltas": deltas, "magnitude": magnitude, "confidence": confidence}

def trend_summary(all_items: list[dict]) -> dict:
    keys = list(all_items[0]["features"].keys())
    Y = {k: [it["features"][k] for it in all_items] for k in keys}
    idx = np.arange(len(all_items)).astype(np.float32)
    slopes = {}
    for k in keys:
        y = np.array(Y[k], dtype=np.float32)
        x = idx - idx.mean()
        y2 = y - y.mean()
        sxx = (x * x).sum() + 1e-6
        sxy = (x * y2).sum()
        slopes[k] = float(sxy / sxx)
    return {"slopes": slopes, "count": len(all_items)}

# ---------- 可视化输出（可选 HTML + 折线图） ----------

def render_html(report: dict, out_html: str):
    os.makedirs(os.path.dirname(out_html) or ".", exist_ok=True)
    per = report["per_photo"]
    feats = list(per[0]["features"].keys())
    # 逐特征画趋势图
    for feat in feats:
        ys = [p["features"][feat] for p in per]
        plt.figure()
        plt.plot(range(1, len(ys) + 1), ys, marker="o")
        plt.title(f"Trend: {feat}")
        plt.xlabel("Photo index (time)")
        plt.ylabel(feat)
        png_path = out_html.replace(".html", f"_{feat}.png")
        plt.savefig(png_path, bbox_inches="tight")
        plt.close()

    # 简单 HTML 汇总
    with open(out_html, "w", encoding="utf-8") as f:
        f.write("<html><body>")
        f.write("<h2>LOONOOL · Skin Vision Engine · MVP</h2>")
        f.write("<p>本页同目录下包含每个特征的趋势图（PNG）。</p >")
        f.write("<ul>")
        for p in per:
            base = os.path.basename(p["path"])
            dt = p["exif_datetime"] or "-"
            f.write(f"<li>{base} | 时间：{dt}</li>")
        f.write("</ul>")
        f.write("</body></html>")

# ---------- 命令行入口 ----------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="LOONOOL Skin Vision Engine · MVP")
    ap.add_argument("images", nargs="+", help="1~N 张图片路径（JPG/PNG）")
    ap.add_argument("--out", default="report.json", help="输出 JSON 路径")
    ap.add_argument("--html", default=None, help="可选：输出 HTML 汇总（会生成同名 PNG 图）")
    args = ap.parse_args()

    # 分析
    items = []
    for p in tqdm(args.images, desc="Analyzing"):
        items.append(analyze_one(p))

    # 两两相邻对比
    diffs = []
    for i in range(len(items) - 1):
        diffs.append(pairwise_diff(items[i], items[i + 1]))

    # 趋势
    trend = trend_summary(items) if len(items) >= 3 else None

    report = {
        "per_photo": items,
        "pairwise_diffs": diffs,
        "trend": trend,
        "version": "SVE-MVP v1.0"
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=np_convert)

    if args.html:
        render_html(report, args.html)

    print(f"✅ 已保存 JSON 报告：{args.out}")
    if args.html:
        print(f"✅ 已生成 HTML：{args.html}（同目录包含趋势 PNG 图）")
    # -------- 置信度自然语言解释 --------
    try:
        conf = report["pairwise_diffs"][0].get("confidence", None)
        if conf is not None:
            if conf >= 0.85:
                conf_text = f"置信度 {conf:.2f} → 光线与角度稳定，结果高度可信。"
            elif conf >= 0.70:
                conf_text = f"置信度 {conf:.2f} → 拍摄条件较好，结果中等可信。"
            else:
                conf_text = f"置信度 {conf:.2f} → 光照或角度差异较大，建议重新拍摄以提高精度。"
            print(f"🌤️ {conf_text}")
        else:
            print("未检测到置信度数据。")
    except Exception as e:
        print(f"⚠️ 无法生成置信度说明：{e}")
# ========= 修复 float32 无法保存到 JSON 的问题 =========
import numpy as np
def np_convert(obj):
    """将 numpy 类型（float32 等）转换为普通 Python 类型"""
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError

# 修改主保存逻辑
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="LOONOOL Skin Vision Engine · MVP")
    ap.add_argument("images", nargs="+", help="1~N 张图片路径（JPG/PNG）")
    ap.add_argument("--out", default="report.json", help="输出 JSON 路径")
    ap.add_argument("--html", default=None, help="输出 HTML 报告")
    args = ap.parse_args()

    items = []
    for p in args.images:
        items.append(analyze_one(p))

    diffs = []
    for i in range(len(items) - 1):
        diffs.append(pairwise_diff(items[i], items[i + 1]))

    trend = trend_summary(items) if len(items) >= 3 else None

    report = {
        "per_photo": items,
        "pairwise_diffs": diffs,
        "trend": trend,
        "version": "SVE-MVP v1.0"
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=np_convert)

    if args.html:
        render_html(report, args.html)

    print(f"✅ 已保存 JSON 报告：{args.out}")
    if args.html:
        print(f"✅ 已生成 HTML：{args.html}（同目录包含趋势 PNG 图）")

import numpy as np
def np_convert(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError

