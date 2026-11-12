# -*- coding: utf-8 -*-
"""
DeepFace 免安装版 · 主程序

用法：双击同目录下的「双击运行.bat」即可。
照片放进「照片」文件夹，结果图会输出到「结果图」文件夹。
"""
import os
import subprocess
import sys

# ---- 关键：让 DeepFace 从本文件夹的 models 目录读取模型权重（不联网）----
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("DEEPFACE_HOME", os.path.join(HERE, "models"))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BUNDLED_PY = os.path.join(HERE, "python", "python.exe")

try:
    from deepface import DeepFace  # noqa: E402
    from PIL import Image, ImageDraw, ImageFont  # noqa: E402
except ImportError:
    # 被 VS Code / 终端用错了解释器（例如系统 Python）：自动切换到包内自带的 Python 重跑
    if os.path.exists(BUNDLED_PY) and os.path.abspath(sys.executable) != os.path.abspath(BUNDLED_PY):
        print("[提示] 当前解释器没装 deepface，自动切换到包内自带的 Python 重新运行……")
        print("       ", BUNDLED_PY)
        print()
        sys.stdout.flush()
        sys.exit(subprocess.call([BUNDLED_PY, os.path.abspath(__file__)]))
    raise

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
PHOTO_DIR = os.path.join(HERE, "照片")
OUT_DIR = os.path.join(HERE, "结果图")


def find_font(size=34):
    for name in ("msyh.ttc", "simhei.ttf", "simsun.ttc", "arial.ttf"):
        p = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", name)
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def list_photos():
    if not os.path.isdir(PHOTO_DIR):
        return []
    names = sorted(n for n in os.listdir(PHOTO_DIR) if n.lower().endswith(IMAGE_EXTS))
    return [os.path.join(PHOTO_DIR, n) for n in names]


def annotate(img_path, faces, info, out_path):
    """在人脸上画框，标注年龄/性别/情绪，保存结果图"""
    im = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(im)
    font = find_font(34)
    for f in faces:
        a = f["facial_area"]
        draw.rectangle([a["x"], a["y"], a["x"] + a["w"], a["y"] + a["h"]], outline=(0, 255, 0), width=6)
        label = f"年龄 {info['age']}   性别 {info['dominant_gender']}   情绪 {info['dominant_emotion']}"
        x0, y0 = a["x"], max(a["y"] - 52, 2)
        box = draw.textbbox((0, 0), label, font=font)
        draw.rectangle([x0, y0, x0 + (box[2] - box[0]) + 24, y0 + (box[3] - box[1]) + 22], fill=(0, 0, 0))
        draw.text((x0 + 12, y0 + 8), label, fill=(0, 255, 0), font=font)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    im.save(out_path)


def process(img_path):
    print("=" * 62)
    print("照片:", os.path.basename(img_path))

    faces = DeepFace.extract_faces(img_path=img_path, detector_backend="opencv", enforce_detection=False)
    if not faces:
        print("  [跳过] 这张照片里没检测到人脸")
        return None

    res = DeepFace.analyze(
        img_path, actions=["age", "gender", "emotion"], detector_backend="opencv", enforce_detection=False
    )
    info = res[0] if isinstance(res, list) else res
    print(f"  [检测] 人脸 {len(faces)} 张")
    print(f"  [属性] 年龄 {info['age']} | 性别 {info['dominant_gender']} | 情绪 {info['dominant_emotion']}")
    print(f"  [情绪分布] " + "  ".join(f"{k}:{v:.1f}%" for k, v in info["emotion"].items()))

    emb = DeepFace.represent(img_path=img_path, model_name="Facenet", detector_backend="opencv")[0]["embedding"]
    print(f"  [特征] Facenet 特征向量 {len(emb)} 维")

    out_png = os.path.join(OUT_DIR, "识别结果_" + os.path.splitext(os.path.basename(img_path))[0] + ".png")
    annotate(img_path, faces, info, out_png)
    print(f"  [结果图] 已保存 -> 结果图\\{os.path.basename(out_png)}")
    return emb


def main():
    photos = list_photos()
    print()
    print("  DeepFace 人脸识别演示（免安装版）")
    print("  照片目录:", PHOTO_DIR)
    print()

    if not photos:
        print("  [!] 照片文件夹里还没有图片。")
        print("      请把 jpg / png 照片放进「照片」文件夹，再重新双击本程序。")
        input("\n  按回车键退出...")
        return

    print(f"  共找到 {len(photos)} 张照片，开始处理……\n")

    embeddings = []
    for p in photos:
        try:
            emb = process(p)
            if emb is not None:
                embeddings.append((os.path.basename(p), emb))
        except Exception as err:
            print(f"  [失败] {type(err).__name__}: {err}")

    # 两张以上时，做一次两两比对
    if len(embeddings) >= 2:
        print("=" * 62)
        print("  两两比对（余弦距离，越小越像；阈值 0.4）")
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                name1, e1 = embeddings[i]
                name2, e2 = embeddings[j]
                a = os.path.join(PHOTO_DIR, name1)
                b = os.path.join(PHOTO_DIR, name2)
                v = DeepFace.verify(
                    img1_path=a, img2_path=b, model_name="Facenet",
                    detector_backend="opencv", enforce_detection=False,
                )
                flag = "同一人" if v["verified"] else "不同人"
                print(f"  {name1}  vs  {name2}  ->  {flag}   距离 {v['distance']:.4f}")

    print("=" * 62)
    print(f"  处理完成，共 {len(embeddings)} 张成功。结果图在「结果图」文件夹。")


if __name__ == "__main__":
    main()
