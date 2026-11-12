# DeepFace-Portable

DeepFace 人脸识别课程实验 · **免安装绿色版**（Windows / 纯 CPU / 无需联网）。

> 完整实验报告见 [说明.md](说明.md)。

## 本仓库包含什么

| 文件 / 目录 | 说明 |
|---|---|
| `main.py` | 主程序：人脸检测 → 年龄/性别/情绪分析 → Facenet 特征提取 → 两两比对 |
| `双击运行.bat` | 启动入口，双击即运行 |
| `说明.md` | 完整实验报告（环境、结果、原理、踩坑记录） |
| `使用说明.txt` | 面向使用者的简明操作说明 |
| `运行结果.txt` | 一次完整运行的实际输出 |
| `示例照片/` | 3 张测试照片 |
| `结果图/` | 程序生成的标注结果图 |

## 本仓库**不**包含什么

绿包体积主要来自两部分，均超出 Git 托管限制，因此排除在仓库外：

- `python/` —— 内置 Python 3.14 embeddable 运行时与依赖库，约 **307 MB**
- `models/` —— Facenet / age / gender / emotion 四个 ONNX 权重，约 **1.1 GB**

完整绿色包（约 1.4 GB）请从网盘下载后解压，目录结构如下：

```
DeepFace-Portable/
├── 双击运行.bat
├── main.py
├── python/        ← 从网盘包补齐
├── models/        ← 从网盘包补齐
├── 照片/          ← 待识别的照片放这里
└── 结果图/        ← 标注结果自动输出到这里
```

## 自己从零搭建（不想用绿色包）

```powershell
pip install deepface onnxruntime opencv-python pillow pandas numpy
python main.py
```

模型权重首次运行会自动下载到 `~/.deepface/weights`；程序里用
`DEEPFACE_HOME` 环境变量指向本地 `models` 目录，因此离线也能跑。

## 运行效果

- 检测：OpenCV Haar Cascade，0.5 ~ 1.2 s/张
- 属性分析：age / gender / emotion 三个 ONNX 模型，首次约 3 s（含加载），之后约 0.6 s/张
- 特征提取：Facenet，128 维，约 0.5 s/张
- 比对：128 维余弦距离，阈值 0.4

⚠️ 注意：整包必须放在**纯英文路径**下，OpenCV 的 `FileStorage` 读不了非 ASCII 路径。

## 许可

实验代码基于 [DeepFace](https://github.com/serengil/deepface)（MIT License）。
