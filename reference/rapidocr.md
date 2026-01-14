# RapidOCR 深度调研报告

*2025年1月*

---

## 1. 项目概述

RapidOCR 是由 RapidAI 团队开发的开源 OCR 工具包，基于 PaddleOCR 的模型，支持多种推理引擎（ONNXRuntime、OpenVINO、PaddlePaddle、PyTorch）。该项目主打轻量、快速、低成本和智能化的特点，在 GitHub 上拥有超过 5,400 颗星。

### 1.1 基本信息

| 属性 | 详情 |
|------|------|
| 项目名称 | RapidOCR |
| 开发团队 | RapidAI Team |
| 开源协议 | Apache 2.0 |
| 最新版本 | v3.4.2 (2025年10月) |
| GitHub Stars | 5,400+ |
| 支持语言 | Python, C++, Java, C#, Android, iOS |
| 文档地址 | https://rapidai.github.io/RapidOCRDocs/ |

### 1.2 核心特性

- **多平台支持**：Linux、Windows、macOS、Android、iOS
- **多推理引擎**：ONNXRuntime、OpenVINO、PaddlePaddle、PyTorch
- **多语言支持**：支持简体中文、繁体中文、英文、日文和拼音五种书写系统，可识别超过 40 种语言
- **轻量部署**：无需 GPU，CPU 即可运行
- **完全开源**：免费使用，支持离线部署
- **PP-OCRv5 支持**：集成最新的 PaddleOCR v5 模型

---

## 2. 安装指南

### 2.1 Python 环境要求

| 要求 | 说明 |
|------|------|
| Python 版本 | >=3.6, <3.13 |
| 操作系统 | Linux / Windows / macOS |
| 内存要求 | 最小 512MB，建议 1GB+ |

### 2.2 安装方式

**方式一：统一包（推荐）**
```bash
pip install rapidocr onnxruntime
```

**方式二：ONNXRuntime 后端**
```bash
pip install rapidocr-onnxruntime
```

**方式三：OpenVINO 后端（Intel 优化）**
```bash
pip install rapidocr-openvino
```

**方式四：PaddlePaddle 后端**
```bash
pip install rapidocr-paddle
```

**方式五：Docker 部署**
```bash
docker pull rapidai/rapidocr:latest
```

---

## 3. API 使用指南

### 3.1 基本用法

```python
from rapidocr import RapidOCR

engine = RapidOCR()
result = engine("image.jpg")
print(result)
result.vis("vis_result.jpg")  # 可视化结果
```

### 3.2 核心参数配置

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| use_det | bool | True | 是否启用文本检测 |
| use_cls | bool | True | 是否启用方向分类 |
| use_rec | bool | True | 是否启用文本识别 |
| return_word_box | bool | False | 是否返回单词级别边界框 |
| return_single_char_box | bool | False | 是否返回单字级别边界框 |

### 3.3 高级配置

通过 params 字典进行详细配置：

```python
from rapidocr import RapidOCR, LangRec, ModelType, OCRVersion

engine = RapidOCR(params={
    "Rec.lang_type": LangRec.CH,
    "Rec.model_type": ModelType.SERVER,
    "Rec.ocr_version": OCRVersion.PPOCRV5,
})
```

### 3.4 输出格式

- `result.txts` - 识别的文本列表
- `result.boxes` - 文本框坐标列表
- `result.scores` - 置信度分数列表
- `result.vis()` - 可视化识别结果
- `result.to_markdown()` - 输出 Markdown 格式
- `result.to_json()` - 输出 JSON 格式

---

## 4. 推理引擎对比

| 引擎 | 适用场景 | GPU 支持 | 模型格式 |
|------|----------|----------|----------|
| ONNXRuntime | 通用部署、跨平台 | CUDA / DirectML | .onnx |
| OpenVINO | Intel 硬件优化 | Intel GPU | .onnx |
| PaddlePaddle | 原生支持、Ascend NPU | CUDA / NPU | .pdmodel |
| PyTorch | 研发实验 | CUDA | .pth |

### 4.1 GPU 加速配置

**ONNXRuntime CUDA 配置：**
```python
engine = RapidOCR(params={
    "Global.use_cuda": True,
})
```

**Windows DirectML 配置：**
```python
engine = RapidOCR(params={
    "Global.use_dml": True,
})
```

ONNX Runtime 后端支持：CPU 多线程（通过 intra_op_num_threads 和 inter_op_num_threads 控制）、NVIDIA GPU 的 CUDA 加速（use_cuda: true）、Windows 的 DirectML 加速（use_dml: true）

---

## 5. 语言支持

### 5.1 检测语言

- ch - 中文
- en - 英文
- multi - 多语言

### 5.2 识别语言

| 语言代码 | 语言名称 | PPOCRv5 支持 |
|----------|----------|--------------|
| ch | 中文（简体） | ✓ |
| en | 英文 | ✓ |
| chinese_cht | 中文（繁体） | ✓ |
| japan | 日文 | v4 支持 |
| korean | 韩文 | ✓ |
| latin | 拉丁文 | ✓ |
| arabic | 阿拉伯文 | v4 支持 |
| cyrillic | 西里尔文 | ✓ |
| devanagari | 梵文 | v4 支持 |

---

## 6. 性能评估

### 6.1 基准测试结果

根据 Nanonets 2025 年 OCR 基准测试报告，在开源模型中，PaddleOCR 和 RapidOCR 是最轻量的选项，非常适合低内存场景。

| 指标 | RapidOCR | EasyOCR | Tesseract |
|------|----------|---------|-----------|
| 内存占用 | 低（最优） | 中 | 低 |
| 推理速度 | 快 | 中 | 快 |
| 中文识别精度 | 高 | 中 | 低 |
| GPU 支持 | CUDA/DirectML | CUDA | 有限 |

### 6.2 优化建议

1. 使用 Server 模型提高精度，使用 Mobile 模型提高速度
2. 对于纯文本图片，可禁用 use_cls 提升速度
3. 调整 limit_side_len 控制图像缩放
4. Intel CPU 优先使用 OpenVINO 后端
5. NVIDIA GPU 优先使用 CUDA 加速

---

## 7. 生态集成

### 7.1 主要集成项目

使用 RapidOCR 的知名项目包括：Docling、CnOCR、api-for-open-llm、arknights-mower、pensieve、ChatLLM、langchain、Langchain-Chatchat、JamAIBase、PAI-RAG、OpenAdapt、Umi-OCR 等。

- **Docling** - IBM 文档解析框架，内置 RapidOCR 支持
- **Langchain** - LLM 应用框架
- **Langchain-Chatchat** - 本地知识库问答
- **CnOCR** - 中文 OCR 工具
- **Umi-OCR** - 桌面端 OCR 应用
- **PAI-RAG** - 阿里云 RAG 应用

### 7.2 Web 服务

RapidOCR 提供独立的 Web 服务包 rapidocr-web，支持：
- REST API 接口
- Web UI 界面
- 支持 WebP 格式
- Nuitka 打包为 exe

---

## 8. 竞品对比

| 特性 | RapidOCR | PaddleOCR | EasyOCR | Tesseract |
|------|----------|-----------|---------|-----------|
| 开源免费 | ✓ | ✓ | ✓ | ✓ |
| 中文支持 | 优秀 | 优秀 | 良好 | 一般 |
| GPU 加速 | 多种后端 | CUDA | CUDA | 有限 |
| 部署复杂度 | 低 | 中 | 低 | 低 |
| 内存占用 | 低 | 中 | 高 | 低 |
| 离线部署 | ✓ | ✓ | ✓ | ✓ |
| 模型格式 | ONNX等 | Paddle | PyTorch | 原生 |

---

## 9. 总结与建议

### 9.1 优势

- **轻量级**：内存占用低，是开源 OCR 中最轻量的选项之一，非常适合资源受限环境
- **多引擎**：支持 ONNXRuntime、OpenVINO、Paddle、PyTorch
- **跨平台**：支持主流操作系统和移动端
- **易于集成**：简单的 API，丰富的生态集成
- **持续更新**：已支持 PP-OCRv5 最新模型

### 9.2 局限性

- 文档主要为中文，英文文档较少
- 部分小语种在 PPOCRv5 中尚未支持（如日文）
- 依赖百度的模型版权

### 9.3 推荐场景

1. **中文 OCR 应用** - PP-OCR 模型对中文支持极佳
2. **边缘部署** - 轻量级，无需 GPU
3. **文档处理流水线** - 与 Docling、Langchain 等集成
4. **Intel 平台** - OpenVINO 后端提供优化
5. **快速原型** - 简单 API 快速验证 OCR 方案
