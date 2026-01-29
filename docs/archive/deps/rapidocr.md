---
source: 
- https://raw.githubusercontent.com/RapidAI/RapidOCR/refs/heads/main/README-CN.md
- https://raw.githubusercontent.com/RapidAI/RapidOCRDocs/main/docs/install_usage/rapidocr/how_to_convert_to_markdown.md
- https://raw.githubusercontent.com/RapidAI/RapidOCRDocs/main/docs/install_usage/rapidocr/parameters.md
---

# RapidOCR README

## 📝 简介

目前，我们自豪地推出了运行速度最为迅猛、兼容性最为广泛的多平台多语言OCR工具，它完全开源免费，并支持离线环境下的快速部署。

**支持语言概览：** 默认支持中文与英文识别，对于其他语言的识别需求，我们提供了便捷的自助转换方案。具体转换指南，请参见[这里](https://rapidai.github.io/RapidOCRDocs/main/blog/2022/09/28/%E6%94%AF%E6%8C%81%E8%AF%86%E5%88%AB%E8%AF%AD%E8%A8%80/)。

**项目缘起：** 鉴于[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)在工程化方面仍有进一步优化的空间，为了简化并加速在各种终端设备上进行OCR推理的过程，我们创新地将PaddleOCR中的模型转换为了高度兼容的ONNX格式，并利用Python、C++、Java、C#等多种编程语言，实现了跨平台的无缝移植，让广大开发者能够轻松上手，高效应用。

**名称寓意：** RapidOCR，这一名称蕴含着我们对产品的深刻期待——轻快（操作简便，响应迅速）、好省（资源占用低，成本效益高）并智能（基于深度学习的强大技术，精准高效）。我们专注于发挥人工智能的优势，打造小巧而强大的模型，将速度视为不懈追求，同时确保识别效果的卓越。

**使用指南：**

- 直接部署：若本仓库中已提供的模型能满足您的需求，那么您只需参考[官方文档](https://rapidai.github.io/RapidOCRDocs/main/quickstart/)进行RapidOCR的部署与使用即可。
- 定制化微调：若现有模型无法满足您的特定需求，您可以在PaddleOCR的基础上，利用自己的数据进行微调，随后再将其应用于RapidOCR的部署中，实现个性化定制。

## 🛠️ 安装

```bash
pip install rapidocr onnxruntime
```

## 📋 使用

```python
from rapidocr import RapidOCR

engine = RapidOCR()

img_url = "https://github.com/RapidAI/RapidOCR/blob/main/python/tests/test_files/ch_en_num.jpg?raw=true"
result = engine(img_url)
print(result)

result.vis("vis_result.jpg")
```

## 📚 文档

完整文档请移步：[docs](https://rapidai.github.io/RapidOCRDocs)

## 👥 谁在使用？([更多](https://github.com/RapidAI/RapidOCR/network/dependents))

- [Docling](https://github.com/DS4SD/docling)
- [CnOCR](https://github.com/breezedeus/CnOCR)
- [api-for-open-llm](https://github.com/xusenlinzy/api-for-open-llm)
- [arknights-mower](https://github.com/ArkMowers/arknights-mower)
- [pensieve](https://github.com/arkohut/pensieve)
- [genshin_artifact_auxiliary](https://github.com/SkeathyTomas/genshin_artifact_auxiliary)
- [ChatLLM](https://github.com/yuanjie-ai/ChatLLM)
- [langchain](https://github.com/langchain-ai/langchain)
- [Langchain-Chatchat](https://github.com/chatchat-space/Langchain-Chatchat)
- [JamAIBase](https://github.com/EmbeddedLLM/JamAIBase)
- [PAI-RAG](https://github.com/aigc-apps/PAI-RAG)
- [ChatAgent_RAG](https://github.com/junyuyang7/ChatAgent_RAG)
- [OpenAdapt](https://github.com/OpenAdaptAI/OpenAdapt)
- [Umi-OCR](https://github.com/hiroi-sora/Umi-OCR)

> 更多使用RapidOCR的项目，欢迎在[登记地址](https://github.com/RapidAI/RapidOCR/discussions/286)登记，登记仅仅为了产品推广。

## ⚖️ 开源许可证

OCR模型版权归百度所有，其他工程代码版权归本仓库所有者所有。

该项目采用 [Apache 2.0 license](../LICENSE) 开源许可证。

---

# 如何将识别结果导出为markdown格式？

在`rapidocr>=3.2.0`中粗略支持了导出markdown格式排版，后续会逐步优化。使用方法：

```python linenums="1" hl_lines="10"
from rapidocr import RapidOCR

engine = RapidOCR()

img_url = "https://img1.baidu.com/it/u=3619974146,1266987475&fm=253&fmt=auto&app=138&f=JPEG?w=500&h=516"
result = engine(img_url, return_word_box=True, return_single_char_box=True)
print(result)

result.vis("vis_result.jpg")
print(result.to_markdown())
```

---

# 参数介绍

### `config.yaml`的生成

```bash linenums="1"
rapidocr config
```

### `default_rapidocr.yaml`常用参数介绍

#### Global

该部分为全局配置。

```yaml linenums="1"
Global:
    text_score: 0.5

    use_det: true
    use_cls: true
    use_rec: true

    min_height: 30
    width_height_ratio: 8
    max_side_len: 2000
    min_side_len: 30

    return_word_box: false
    return_single_char_box: false

    font_path: null
    log_level: "info" # debug / info / warning / error / critical
```

`text_score (float)`: 文本识别结果置信度，值越大，把握越大。取值范围：`[0, 1]`, 默认值是0.5。

`use_det (bool)`: 是否使用文本检测。默认为`True`。

`use_cls (bool)`: 是否使用文本行方向分类。默认为`True`。

`use_rec (bool)`: 是否使用文本行识别。默认为`True`。

`min_height (int)` : 图像最小高度（单位是像素），低于这个值，会跳过文本检测阶段，直接进行后续识别。默认值为30。`min_height`是用来过滤只有一行文本的图像（如下图），这类图像不会进入文本检测模块，直接进入后续过程。

![](https://github.com/RapidAI/RapidOCR/releases/download/v1.1.0/single_line_text.jpg)

`width_height_ratio (float)`: 如果输入图像的宽高比大于`width_height_ratio`，则会跳过文本检测，直接进行后续识别，取值为-1时：不用这个参数. 默认值为8。

`max_side_len (int)`: 如果输入图像的最大边大于`max_side_len`，则会按宽高比，将最大边缩放到`max_side_len`。默认为2000px。

`min_side_len (int)`: 如果输入图像的最小边小于`min_side_len`，则会按宽高比，将最小边缩放到`min_side_len`。默认为30px。

`return_word_box (bool)`: 是否返回文字的单字坐标。默认为`False`。

> 在`rapidocr>=2.1.0`中，纯中文、中英文混合返回单字坐标，纯英文返回单词坐标。

> 在`rapidocr<=2.0.7`中，纯中文、中英文混合和纯英文均返回单字坐标。

> 在`rapidocr_onnxruntime>=1.4.1`中，汉字返回单字坐标，英语返回单字母坐标。

> 在`rapidocr_onnxruntime==1.4.0`中，汉字会返回单字坐标，英语返回单词坐标。

`return_single_char_box (bool)`: 文本内容只有英文和数字情况下，是否返回单字坐标。默认为`False`。

> 在`rapidocr>=3.1.0`中添加该参数，该参数只有在`return_word_box=True`时，才能生效。

```python
result = engine(img_url, return_word_box=True, return_single_char_box=True)
```

`font_path (str)`: 字体文件路径。如不提供，程序会自动下载预置的字体文件模型到本地。默认为`null`。

`log_level (str)`: 日志级别设置。可选择的有`debug / info / warning / error / critical`，默认为`info`，会打印加载模型等日志。如果设置`critical`，则不会打印任何日志。

> 在`rapidocr>=3.4.0`中，才添加此参数。

#### EngineConfig

!!! note

    下面显示的为最新版本配置。如果遇到某些字段未找到等问题。请切换为对应版本的当前文档查看。

该部分为相关推理引擎的配置文件，大家可按需配置。该部分后面可能会增删部分关键字，如果有需求，可以在文档下面评论区指出。

```yaml linenums="1"
EngineConfig:
    onnxruntime:
        intra_op_num_threads: -1
        inter_op_num_threads: -1
        enable_cpu_mem_arena: false

        cpu_ep_cfg:
            arena_extend_strategy: "kSameAsRequested"

        use_cuda: false
        cuda_ep_cfg:
            device_id: 0
            arena_extend_strategy: "kNextPowerOfTwo"
            cudnn_conv_algo_search: "EXHAUSTIVE"
            do_copy_in_default_stream: true

        use_dml: false
        dm_ep_cfg: null

        use_cann: false
        cann_ep_cfg:
            device_id: 0
            arena_extend_strategy: "kNextPowerOfTwo"
            npu_mem_limit:  21474836480 # 20 * 1024 * 1024 * 1024
            op_select_impl_mode: "high_performance"
            optypelist_for_implmode: "Gelu"
            enable_cann_graph: true

    openvino:
        inference_num_threads: -1
        performance_hint: null
        performance_num_requests: -1
        enable_cpu_pinning: null
        num_streams: -1
        enable_hyper_threading: null
        scheduling_core_type: null

    paddle:
        cpu_math_library_num_threads: -1

        use_npu: false
        npu_ep_cfg:
            device_id: 0
            envs:
                FLAGS_npu_jit_compile: 0
                FLAGS_use_stride_kernel: 0
                FLAGS_allocator_strategy: "auto_growth"
                CUSTOM_DEVICE_BLACK_LIST: "pad3d,pad3d_grad,set_value,set_value_with_tensor"
                FLAGS_npu_scale_aclnn: "True"
                FLAGS_npu_split_aclnn: "True"

        use_cuda: false
        cuda_ep_cfg:
            device_id: 0
            gpu_mem: 500

    torch:
        use_cuda: false
        cuda_ep_cfg:
            device_id: 0

        use_npu: false
        npu_ep_cfg:
            device_id: 0
```

该部分的详细使用，请参见：[如何使用不同推理引擎？](./how_to_use_infer_engine.md)

各个推理引擎的API：

- ONNXRuntime Python API 参见：[Python API](https://onnxruntime.ai/docs/api/python/api_summary.html)
- OpenVINO Python API 参见：[OpenVINO Python API](https://docs.openvino.ai/2025/api/ie_python_api/api.html)
- PaddlePaddle API 参见：[API 文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/index_cn.html)
- PyTorch API 参见：[PyTorch documentation](https://docs.pytorch.org/docs/stable/index.html)

以下三部分前4个参数基本类似，对应关系如下表，具体请参见[模型列表](../../model_list.md)文档：

| YAML 参数       | 对应枚举类       | 可用枚举值（示例）                 |导入方式 | 备注                                |
|-----------------|------------------|------------------|-------------------|-------------------------------------|
| `engine_type`   | `EngineType`     | `ONNXRUNTIME`（onnxruntime）<br>`OPENVINO`（openvino）<br>`PADDLE`（paddle）<br>`TORCH`（torch） | `from rapidocr import EngineType`|推理引擎类型         |
| `lang_type`     |  `LangDet`<br> `LangCls`<br> `LangRec` | **检测（Det）**：`CH`/`EN`/`MULTI`<br>**分类（Cls）**：`CH`<br>**识别（Rec）**：`CH`/`CH_DOC`/`EN`/`ARABIC`/... |`from rapidocr import LangDet`<br/> `from rapidocr import LangCls` <br/>`from rapidocr import LangRec`| 根据OCR处理阶段选择不同枚举值 |
| `model_type`    | `ModelType`      | `MOBILE`（mobile）<br>`SERVER`（server） |`from rapidocr import ModelType`| 模型大小与性能级别      |
| `ocr_version`   | `OCRVersion`     | `PPOCRV4`（PP-OCRv4）<br>`PPOCRV5`（PP-OCRv5） |`from rapidocr import OCRVersion`| 模型版本    |

#### Det

```yaml linenums="1"
Det:
    engine_type: "onnxruntime"
    lang_type: "ch"
    model_type: "mobile"
    ocr_version: "PP-OCRv4"

    task_type: "det"

    model_path: null
    model_dir: null

    limit_side_len: 736
    limit_type: min
    std: [ 0.5, 0.5, 0.5 ]
    mean: [ 0.5, 0.5, 0.5 ]

    thresh: 0.3
    box_thresh: 0.5
    max_candidates: 1000
    unclip_ratio: 1.6
    use_dilation: true
    score_mode: fast
```

`engine_type (str)`: 选定推理引擎。支持`onnxruntime`、`openvino`、`paddle`和`torch`四个值。默认为`onnxruntime`。

`lang_type (str)`: 支持检测的语种类型。这里指的是`LangDet`，具体支持`ch`、`en`和`multi`3个值。`ch`可以识别中文和中英文混合文本检测。`en`支持英文文字检测。`multi`支持多语言文本检测。默认为`ch`。详细参见：[docs](https://rapidai.github.io/RapidOCRDocs/main/model_list/#_1)

`model_type (str)`: 模型量级选择，支持`mobile`（轻量型）和`server`（服务型）。默认为`mobile`。

`ocr_version (str)`: ocr版本的选择，支持`PP-OCRv4`和`PP-OCRv5`，默认为`PP-OCRv4`。

`model_path (str)`: 文本检测模型路径，仅限于基于PaddleOCR训练所得DBNet文本检测模型。默认值为`null`。

`model_dir (str)`: 模型存放路径或目录。如果是PaddlePaddle，该参数则对应模型存在目录。其余推理引擎请使用`model_path`参数。

`limit_side_len (float)`: 限制图像边的长度的像素值。默认值为736。

`limit_type (str)`: 限制图像的最小边长度还是最大边为`limit_side_len`。 示例解释：当`limit_type=min`和`limit_side_len=736`时，图像最小边小于736时，会将图像最小边拉伸到736，另一边则按图像原始比例等比缩放。 取值范围为：`[min, max]`，默认值为`min`。

`thresh (float)`: 图像中文字部分和背景部分分割阈值。值越大，文字部分会越小。取值范围：`[0, 1]`，默认值为0.3。

`box_thresh (float)`: 文本检测所得框是否保留的阈值，值越大，召回率越低。取值范围：`[0, 1]`，默认值为0.5。

`max_candidates (int)`: 最大候选框数目。默认是1000。

`unclip_ratio (float)`: 控制文本检测框的大小，值越大，检测框整体越大。取值范围：`[1.6, 2.0]`，默认值为1.6。

`use_dilation (bool)`: 是否使用膨胀。默认为`true`。该参数用于将检测到的文本区域做形态学的膨胀处理。

`score_mode (str)`: 计算文本框得分的方式。取值范围为：`[slow, fast]`，默认值为`fast`。

#### Cls

```yaml linenums="1"
Cls:
    engine_type: "onnxruntime"
    lang_type: "ch"
    model_type: "mobile"
    ocr_version: "PP-OCRv4"

    task_type: "cls"

    model_path: null
    model_dir: null

    cls_image_shape: [3, 48, 192]
    cls_batch_num: 6
    cls_thresh: 0.9
    label_list: ["0", "180"]
```

`engine_type (str)`: 同Det部分介绍。

`lang_type (str)`: 支持检测的语种类型。这里指的是`LangCls`，目前只有一种选项：`ch`。默认为`ch`。

`model_type (str)`: 同Det部分介绍。

`ocr_version (str)`: 同Det部分介绍。

`model_path (str)`: 文本行方向分类模型路径，仅限于PaddleOCR训练所得二分类分类模型。默认值为`None`。

`model_dir (str)`: 占位参数，暂时无效。

`cls_image_shape (List[int])`: 输入方向分类模型的图像Shape(CHW)。默认值为`[3, 48, 192]`。

`cls_batch_num (int)`: 批次推理的batch大小，一般采用默认值即可，太大并没有明显提速，效果还可能会差。默认值为6。

`cls_thresh (float)`: 方向分类结果的置信度。取值范围：`[0, 1]`，默认值为0.9。

`label_list (List[str])`: 方向分类的标签，0°或者180°，**该参数不能动** 。默认值为`["0", "180"]`。

#### Rec

```yaml linenums="1"
Rec:
    engine_type: "onnxruntime"
    lang_type: "ch"
    model_type: "mobile"
    ocr_version: "PP-OCRv4"

    task_type: "rec"

    model_path: null
    model_dir: null

    rec_keys_path: null
    rec_img_shape: [3, 48, 320]
    rec_batch_num: 6
```

`engine_type (str)`: 同Det部分介绍。

`lang_type (str)`: 支持检测的语种类型。这里指的是`LangRec`，具体支持的语种参见：[model_list](../../model_list.md).

`model_type (str)`: 同Det部分介绍。

`ocr_version (str)`: 同Det部分介绍。

`model_path (str)`: 文本识别模型路径，仅限于PaddleOCR训练文本识别模型。默认值为`None`。

`model_dir (str)`: 模型存放路径或目录。如果是PaddlePaddle，该参数则对应模型存在目录。其余推理引擎请使用`model_path`参数。

`rec_keys_path (str)`: 文本识别模型对应的字典文件，默认为`None`。

`rec_img_shape (List[int])`: 输入文本识别模型的图像Shape(CHW)。默认值为`[3, 48, 320]`。

`rec_batch_num (int)`: 批次推理的batch大小，一般采用默认值即可，太大并没有明显提速，效果还可能会差。默认值为6。

---

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
