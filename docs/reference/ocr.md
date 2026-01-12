# 本地 OCR 方案研究报告

> 调研日期：2026-01-12
> MarkIt 版本：0.1.2

## 摘要

本报告针对 MarkIt 项目的扫描 PDF OCR 需求，调研了 7 种本地 OCR 解决方案。MarkIt 是一个文档转 Markdown 工具，已使用 pymupdf4llm 作为主要 PDF 转换引擎。本报告重点评估各方案的离线可用性、中英文支持、Python 集成复杂度和性能特征。

---

## 1. PyMuPDF4LLM 内置 OCR 支持

### 现有集成状态

**已在代码库中使用**：`src/markit/converters/pdf/pymupdf4llm.py`

### 功能特性

pymupdf4llm 已内置 Tesseract OCR 支持：
- **自动 OCR 触发**：当页面检测到图片内容或不可识别字符时自动启用 OCR
- **启发式判断**：
  - 页面几乎无文本但充满图片或向量图形 → 触发 OCR
  - 页面有文本但太多字符不可读（如乱码 ""）→ 对受影响区域执行 OCR

### 依赖要求

```bash
# 已在 pyproject.toml 中定义为可选依赖
[project.optional-dependencies]
ocr = [
    "opencv-python>=4.10.0",  # pymupdf4llm OCR 必需
]

# 系统级依赖
apt-get install tesseract-ocr tesseract-ocr-chi-sim tesseract-ocr-chi-tra
```

### 中文支持

需要安装 Tesseract 中文语言包：
- `chi_sim`：简体中文
- `chi_tra`：繁体中文

设置 `TESSDATA_PREFIX` 环境变量指向 tessdata 目录：
```python
import os
os.environ["TESSDATA_PREFIX"] = "/usr/share/tesseract-ocr/4.00/tessdata"
```

### 评估

| 维度 | 评分 | 说明 |
|------|------|------|
| 集成复杂度 | ★★★★★ | 零额外代码，已内置 |
| 依赖增量 | ★★★★ | 仅需 opencv-python（已在 optional） |
| 中文准确度 | ★★★ | 依赖 Tesseract，中文表现一般 |
| 性能 | ★★★★ | CPU 执行，中等速度 |
| 维护成本 | ★★★★★ | 由 Artifex 维护，活跃更新 |

---

## 2. Tesseract OCR

### 概述

Google 维护的经典开源 OCR 引擎，使用 LSTM 模型，支持 100+ 语言。

### 安装

```bash
# Ubuntu/Debian
apt-get install tesseract-ocr tesseract-ocr-chi-sim tesseract-ocr-chi-tra

# macOS
brew install tesseract tesseract-lang

# Python 包装器
pip install pytesseract
```

### Python 集成

```python
import pytesseract
from PIL import Image

# 基础用法
text = pytesseract.image_to_string(Image.open('image.png'), lang='chi_sim+eng')

# 带配置
custom_config = r'--oem 3 --psm 6'
text = pytesseract.image_to_string(image, lang='chi_sim', config=custom_config)
```

### 中文支持

- 官方支持简体/繁体中文
- 存在社区重训练模型 ([tessdata_chi](https://github.com/gumblex/tessdata_chi))，准确率更高
- 需要图像预处理（二值化、去噪）以提升准确率

### 评估

| 维度 | 评分 | 说明 |
|------|------|------|
| 集成复杂度 | ★★★★ | 成熟稳定，文档丰富 |
| 依赖增量 | ★★★ | 系统级安装 + pytesseract |
| 中文准确度 | ★★★ | 中等，复杂布局表现差 |
| 性能 | ★★★★ | CPU 执行，轻量快速 |
| 维护成本 | ★★★★★ | 20+ 年历史，极其稳定 |

---

## 3. PaddleOCR

### 概述

百度开源的深度学习 OCR 工具包，2025 年发布 PP-OCRv5，中文识别领先。

### 最新版本

- **PaddleOCR 3.0.3** (2025-06-26)：PP-OCRv5 准确率比 v4 提升 13%
- **PaddleOCR-VL** (2025-10-16)：0.9B 视觉语言模型，支持 109 种语言

### 安装

```bash
# 基础安装
pip install paddlepaddle paddleocr

# GPU 版本
pip install paddlepaddle-gpu paddleocr
```

### Python 集成

```python
from paddleocr import PaddleOCR

# 初始化（自动下载模型）
ocr = PaddleOCR(use_angle_cls=True, lang='ch')  # 中英文混合

# 执行识别
result = ocr.ocr('image.png', cls=True)
for line in result[0]:
    print(line[1][0])  # 识别文本
```

### 中文支持

- **原生优化**：专门针对中文优化训练
- **混合识别**：单模型支持简体/繁体中文、拼音、英文、日文
- **手写支持**：PP-OCRv5 显著提升手写识别能力

### 性能特点

- 模型小于 100M 参数，推理快
- 支持 CPU/GPU，GPU 推理速度显著提升
- 比 MinerU 2.5 快 14.2%，比 dots.ocr 快 253%

### 评估

| 维度 | 评分 | 说明 |
|------|------|------|
| 集成复杂度 | ★★★ | 需要 PaddlePaddle 框架 |
| 依赖增量 | ★★ | paddlepaddle + paddleocr（较重） |
| 中文准确度 | ★★★★★ | 中文 OCR 领先水平 |
| 性能 | ★★★★ | 轻量模型，内存占用低 |
| 维护成本 | ★★★★ | 百度持续维护，更新频繁 |

---

## 4. EasyOCR

### 概述

Jaided AI 开发的即用型 OCR，基于 PyTorch，支持 80+ 语言。

### 安装

```bash
pip install easyocr
```

### Python 集成

```python
import easyocr

# 初始化（首次运行自动下载模型）
reader = easyocr.Reader(['ch_sim', 'en'])  # 简体中文 + 英语

# 执行识别
results = reader.readtext('image.png')
for (bbox, text, confidence) in results:
    print(f"{text} (置信度: {confidence:.2f})")
```

### 中文支持

- 支持简体中文 (`ch_sim`) 和繁体中文 (`ch_tra`)
- 中英文混合识别效果好
- 相比 Tesseract，对中文/日文/阿拉伯文等语言表现更好

### 性能特点

- 基于深度学习，对噪声图像容忍度高
- 识别速度中等偏慢
- 第二代模型更小更快，准确率持平

### 评估

| 维度 | 评分 | 说明 |
|------|------|------|
| 集成复杂度 | ★★★★★ | API 简洁，开箱即用 |
| 依赖增量 | ★★★ | PyTorch + 模型文件 |
| 中文准确度 | ★★★★ | 优于 Tesseract，略逊于 PaddleOCR |
| 性能 | ★★★ | 识别较慢，需 GPU 加速 |
| 维护成本 | ★★★ | 更新频率一般 |

---

## 5. Surya OCR

### 概述

Datalab 团队 2025 年发布的轻量级文档 OCR 和分析工具包。

### 安装

```bash
pip install surya-ocr
# 需要 Python 3.10+, PyTorch
```

### Python 集成

```python
from surya.ocr import run_ocr
from surya.model.detection import load_model, load_processor

# 加载模型
det_model, det_processor = load_model(), load_processor()

# 执行 OCR
results = run_ocr([image], det_model, det_processor, languages=['zh', 'en'])
```

### 特性

- 支持 90+ 语言的 OCR、布局分析、阅读顺序、表格识别
- 布局分析准确率 88%，A10 GPU 上 0.4 秒/页
- 在真实世界 PDF 基准测试中优于 Tesseract

### 局限

- **需要 GPU**：推荐 16GB VRAM 用于批处理
- 对复杂背景和手写文本支持有限
- GPL-3.0 许可证

### 评估

| 维度 | 评分 | 说明 |
|------|------|------|
| 集成复杂度 | ★★★ | API 设计合理，需熟悉 |
| 依赖增量 | ★★★ | PyTorch + 模型（GPU 推荐） |
| 中文准确度 | ★★★★ | 多语言支持好 |
| 性能 | ★★★★★ | GPU 下极快 |
| 维护成本 | ★★★ | 新项目，活跃开发 |

---

## 6. RapidOCR

### 概述

基于 PaddleOCR 模型的多运行时 OCR 工具，支持 ONNX Runtime、OpenVINO、PyTorch。

### 最新版本

**RapidOCR 3.5.0** (2026-01-06)

### 安装

```bash
# ONNX Runtime 版本（推荐，无需 PaddlePaddle）
pip install rapidocr-onnxruntime

# 或完整版
pip install rapidocr
```

### Python 集成

```python
from rapidocr_onnxruntime import RapidOCR

engine = RapidOCR()
result, elapse = engine('image.png')
for line in result:
    print(line[1])  # 识别文本
```

### 特性

- **无需 PaddlePaddle**：使用 ONNX Runtime 推理
- 内存占用最低，适合资源受限环境
- 支持 Mobile（更快）和 Server（更准）两种模型

### 中文支持

- 继承 PaddleOCR 的中英文原生支持
- 其他语言需自助转换模型

### 评估

| 维度 | 评分 | 说明 |
|------|------|------|
| 集成复杂度 | ★★★★★ | API 极简，无框架依赖 |
| 依赖增量 | ★★★★★ | 仅需 onnxruntime，轻量 |
| 中文准确度 | ★★★★★ | 与 PaddleOCR 同源 |
| 性能 | ★★★★ | 最佳性价比，内存效率高 |
| 维护成本 | ★★★★ | RapidAI 社区活跃维护 |

---

## 7. MarkItDown OCR 能力

### 现有能力

根据调研（参考 `docs/reference/markitdown_pdf_research.md`）：

- **基础 PDF 转换**：使用 pdfminer.six，**无内置 OCR**
- **Azure Document Intelligence**：云服务，支持高分辨率 OCR
- **LLM 图像描述**：仅支持图片和 PPTX 中的图片，**不支持直接分析 PDF**

### 结论

MarkItDown **不提供本地 OCR 能力**，扫描 PDF 需要：
1. Azure Document Intelligence（云服务）
2. 预处理转图片后用 LLM 描述

---

## 综合对比

| 方案 | 中文准确度 | 性能 | 依赖复杂度 | 集成难度 | 推荐场景 |
|------|------------|------|------------|----------|----------|
| **PyMuPDF4LLM 内置** | ★★★ | ★★★★ | ★★★★★ | ★★★★★ | **零成本启用** |
| Tesseract | ★★★ | ★★★★ | ★★★★ | ★★★★ | 轻量需求 |
| PaddleOCR | ★★★★★ | ★★★★ | ★★ | ★★★ | 中文最优 |
| EasyOCR | ★★★★ | ★★★ | ★★★ | ★★★★★ | 快速原型 |
| Surya | ★★★★ | ★★★★★ | ★★★ | ★★★ | GPU 环境 |
| **RapidOCR** | ★★★★★ | ★★★★ | ★★★★★ | ★★★★★ | **最佳平衡** |
| MarkItDown | 不支持 | - | - | - | 需云服务 |

---

## 推荐方案

### 对 MarkIt 项目的建议

考虑到项目已使用 pymupdf4llm 作为主要 PDF 转换引擎，推荐**分阶段实施**：

### 阶段 1：启用 PyMuPDF4LLM 内置 OCR（立即可用）

**优势**：
- 零代码修改，仅需安装依赖
- 已在 `pyproject.toml` 定义为可选依赖
- 自动触发，用户无感知

**实施步骤**：
1. 安装系统依赖：`apt-get install tesseract-ocr tesseract-ocr-chi-sim`
2. 安装 Python 依赖：`uv sync --extra ocr`
3. 在配置中启用 OCR 选项

**配置示例**（`markit.yaml`）：
```yaml
pdf:
  engine: pymupdf4llm
  ocr_enabled: true  # 已有此选项
  ocr_language: chi_sim+eng  # 建议新增：OCR 语言设置
```

### 阶段 2：集成 RapidOCR（可选增强）

**适用场景**：需要更高的中文准确率

**优势**：
- 与 PaddleOCR 同源，中文准确率领先
- 基于 ONNX Runtime，无需 PaddlePaddle 框架
- 依赖轻量，仅 `rapidocr-onnxruntime`

**实施方案**：
```python
# markit/ocr/rapidocr_engine.py
from rapidocr_onnxruntime import RapidOCR

class RapidOCREngine:
    def __init__(self):
        self.engine = RapidOCR()

    def extract_text(self, image_path: str) -> str:
        result, _ = self.engine(image_path)
        return '\n'.join([line[1] for line in result]) if result else ''
```

**配置扩展**：
```yaml
pdf:
  ocr_engine: pymupdf  # 或 rapidocr
  ocr_language: chi_sim+eng
```

### 不推荐方案

1. **PaddleOCR 直接集成**：依赖过重（需要 PaddlePaddle 框架）
2. **Surya**：GPL-3.0 许可证限制，需要 GPU
3. **EasyOCR**：性能较慢，无明显优势

---

## 参考资料

- [PyMuPDF4LLM PyPI](https://pypi.org/project/pymupdf4llm/)
- [PyMuPDF OCR 文档](https://pymupdf.readthedocs.io/en/latest/recipes-ocr.html)
- [PyMuPDF4LLM 文档](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/)
- [PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR)
- [PaddleOCR 3.0 技术报告](https://arxiv.org/html/2507.05595v1)
- [EasyOCR GitHub](https://github.com/JaidedAI/EasyOCR)
- [Surya OCR GitHub](https://github.com/datalab-to/surya)
- [RapidOCR GitHub](https://github.com/RapidAI/RapidOCR)
- [RapidOCR PyPI](https://pypi.org/project/rapidocr-onnxruntime/)
- [MarkItDown GitHub](https://github.com/microsoft/markitdown)
- [MarkItDown 扫描 PDF 讨论](https://github.com/microsoft/markitdown/discussions/1361)
- [Tesseract 中文重训练模型](https://github.com/gumblex/tessdata_chi)
- [OCR 工具对比指南](https://modal.com/blog/8-top-open-source-ocr-models-compared)
