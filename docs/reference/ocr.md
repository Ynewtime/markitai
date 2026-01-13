# OCR 方案研究报告

> 调研日期：2026-01-13 (更新)
> 目标 MarkIt 版本：0.2.0
> 状态：已确定方案

## 摘要

本报告针对 MarkIt 项目的扫描 PDF OCR 需求，深度调研了 Python 社区主流 OCR 解决方案。经过技术验证和社区调研，最终确定采用 **RapidOCR + PaddleOCR 双引擎方案**：

- **默认模式**：RapidOCR + PaddleOCR 双引擎，确保最佳输出效果
- **快速模式** (`--fast`)：仅启用 RapidOCR，平衡速度与质量

---

## 1. 技术验证

### 1.1 pymupdf4llm 内置 OCR 的局限性

**验证结论**：pymupdf4llm 存在"图片避让机制"，**无法直接用于扫描件 OCR**。

```python
# 测试代码
import pymupdf4llm

# 扫描件 PDF（图片 PDF，无文本层）
result = pymupdf4llm.to_markdown('scanned.pdf')
print(result)  # 输出为空！

# 原因：pymupdf4llm 的版面分析算法会忽略图片区域内的文本
```

**PyMuPDF 底层 OCR API 可用**，但 pymupdf4llm 不会使用其结果：

```python
import fitz

doc = fitz.open('scanned.pdf')
page = doc[0]

# 直接调用 OCR API - 可以工作
tp = page.get_textpage_ocr(language='eng', full=True)
text = page.get_text(textpage=tp)
print(text)  # 有输出
```

### 1.2 OCR 引擎实测对比

使用模拟中文发票 PDF 进行测试：

| 引擎 | 耗时 | 中文识别 | 英文识别 | 置信度 |
|------|------|---------|---------|--------|
| **RapidOCR** | 0.89s | 完美 | 完美 | 96-100% |
| PyMuPDF + Tesseract (eng) | 0.12s | 完全失败 | 部分错误 | - |

**RapidOCR 测试结果**：

```
发票号码：INV-2026-001234 (置信度: 0.98)
客户名称：北京科技有限公司 (置信度: 1.00)
金额：￥12,580.00 (置信度: 0.96)
Date: 2026-01-13 (置信度: 1.00)
备注：此发票仅供测试使用 (置信度: 1.00)
```

**PyMuPDF + Tesseract 测试结果**（仅英文语言包）：

```
ASH: INV-2026-001234
PARR: TOREARAB]
4:3: ¥ 12,580.00
Date: 2026-01-13
SIE 2 WEAR UE FA
```

---

## 2. Python 社区 OCR 方案深度调研

### 2.1 主流方案对比

| 方案 | 中文准确度 | 速度 | 依赖复杂度 | 语言支持 | 许可证 |
|------|-----------|------|-----------|---------|--------|
| **PaddleOCR** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ (需 PaddlePaddle) | 80+ 语言 | Apache 2.0 |
| **RapidOCR** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ (仅 ONNX) | 中英文 | Apache 2.0 |
| EasyOCR | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ (需 PyTorch) | 80+ 语言 | Apache 2.0 |
| Tesseract | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ (系统依赖) | 100+ 语言 | Apache 2.0 |
| Surya | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ (GPU) | ⭐⭐⭐ (需 PyTorch) | 90+ 语言 | **GPL-3.0** |
| docTR | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | 有限 | Apache 2.0 |

### 2.2 RapidOCR 与 PaddleOCR 的关系

> RapidOCR 是 PaddleOCR 模型的 **ONNX 转换版本**，两者使用相同的底层模型，准确率基本一致。

- **PaddleOCR**：功能更丰富（版面分析、表格识别、PP-StructureV3），但依赖 PaddlePaddle 框架
- **RapidOCR**：轻量级部署，仅需 `onnxruntime`，跨平台更友好

### 2.3 性能基准（2025 年实测）

| 方案 | 内存占用 | 吞吐量 | 备注 |
|------|---------|--------|------|
| PaddleOCR + RapidOCR | **最低** | 高 | 适合低资源环境 |
| EasyOCR (GPU) | 中 | **最快** (4x CPU) | 需 GPU |
| Tesseract | 低 | 快 | 预处理耗时 |

### 2.4 中文识别专项评估

广州软件应用技术研究院 2024 年对 12 款 OCR 工具的评测显示：

- **PaddleOCR 排名第二**，生态系统最完善
- 覆盖场景：印刷中文、印刷英文、手写中文、复杂自然场景、变形字体
- GitHub Star 数超过 50,000（截至 2025 年 6 月）

### 2.5 各方案详细分析

#### PaddleOCR

**优势**：
- 中文识别准确率领先（PP-OCRv5 比 v4 提升 13%）
- 功能最全面：PP-StructureV3（版面分析）、PP-ChatOCRv4（信息提取）
- 支持 80+ 语言，中英文混合识别优秀
- 被 MinerU、RAGFlow、UmiOCR 等项目采用为核心 OCR 引擎

**劣势**：
- 依赖 PaddlePaddle 框架（约 500MB+）
- 首次运行需下载模型文件

#### RapidOCR

**优势**：
- 与 PaddleOCR 同源模型，准确率一致
- 仅依赖 `onnxruntime`（约 14MB）
- 纯 Python，跨平台无系统依赖
- 支持多种推理后端：ONNXRuntime、OpenVINO、PyTorch
- 被 langchain、Docling、PAI-RAG 等 121+ 项目采用

**劣势**：
- 默认仅支持中英文，其他语言需自行转换模型
- 功能相比 PaddleOCR 较少（无版面分析）

#### Surya

**优势**：
- 版面分析 + 阅读顺序检测 + 表格识别，功能最全面
- 支持 90+ 语言
- GPU 下性能极佳（A10 GPU 上 0.4 秒/页）

**劣势**：
- **GPL-3.0 许可证**（代码）+ 商业限制（模型权重）
- 推荐 16GB VRAM GPU
- 不适合 Apache 2.0 项目集成

#### EasyOCR

**优势**：
- API 简洁，开箱即用
- GPU 下速度最快（4x CPU）
- 支持 80+ 语言

**劣势**：
- 需要 PyTorch 依赖
- 准确率略逊于 PaddleOCR
- 更新频率一般

#### Tesseract

**优势**：
- 最成熟稳定（20+ 年历史）
- 支持 100+ 语言
- CPU 执行速度快

**劣势**：
- 中文识别准确率一般
- 需要系统级安装 + 语言包
- 复杂布局表现差

---

## 3. 最终方案

### 3.1 选型决策

基于 MarkIt 的定位（面向中文文档转换、开箱即用、Apache 2.0 许可），采用 **RapidOCR + PaddleOCR 双引擎方案**：

| 模式 | 引擎组合 | 适用场景 |
|------|---------|---------|
| **默认模式** | RapidOCR + PaddleOCR | 最佳输出质量，适合重要文档 |
| **快速模式** (`--fast`) | 仅 RapidOCR | 平衡速度与质量，适合批量处理 |

### 3.2 双引擎策略

```
输入图片
    │
    ├─→ RapidOCR ──→ 结果 A
    │
    └─→ PaddleOCR ─→ 结果 B
                         │
                         ▼
                    结果融合/选优
                         │
                         ▼
                    最终输出
```

**融合策略**：
1. 对比两个引擎的识别结果
2. 按置信度加权选择
3. 低置信度区域采用另一引擎结果补充

### 3.3 配置设计

```yaml
pdf:
  engine: pymupdf4llm
  extract_images: true

  # OCR 配置
  ocr_enabled: true
  ocr_auto_detect: true      # 自动检测扫描件
  ocr_dpi: 200               # OCR 渲染分辨率

  # OCR 引擎策略
  # dual: RapidOCR + PaddleOCR 双引擎（默认，最佳质量）
  # rapid: 仅 RapidOCR（快速模式）
  # paddle: 仅 PaddleOCR
  ocr_engine: dual

# --fast 模式会自动将 ocr_engine 切换为 rapid
```

### 3.4 依赖配置

```toml
# pyproject.toml
[project]
dependencies = [
    "rapidocr-onnxruntime>=1.4.0",  # 默认 OCR 引擎
]

[project.optional-dependencies]
ocr-full = [
    "paddlepaddle>=2.6.0",
    "paddleocr>=2.9.0",
]
```

---

## 4. 实施要点

### 4.1 OCR 引擎抽象层

```python
# src/markit/ocr/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class OCRTextBlock:
    text: str
    confidence: float
    bbox: tuple[float, float, float, float]

class BaseOCREngine(ABC):
    name: str = "base"

    @abstractmethod
    async def recognize(self, image_data: bytes) -> list[OCRTextBlock]:
        ...
```

### 4.2 双引擎实现

```python
# src/markit/ocr/dual.py
class DualOCREngine(BaseOCREngine):
    """RapidOCR + PaddleOCR 双引擎"""

    name = "dual"

    def __init__(self):
        self.rapid = RapidOCREngine()
        self.paddle = PaddleOCREngine() if PADDLE_AVAILABLE else None

    async def recognize(self, image_data: bytes) -> list[OCRTextBlock]:
        # 并行执行两个引擎
        rapid_result = await self.rapid.recognize(image_data)

        if self.paddle is None:
            return rapid_result

        paddle_result = await self.paddle.recognize(image_data)

        # 结果融合
        return self._merge_results(rapid_result, paddle_result)

    def _merge_results(
        self,
        rapid: list[OCRTextBlock],
        paddle: list[OCRTextBlock],
    ) -> list[OCRTextBlock]:
        """按置信度融合两个引擎的结果"""
        # 实现细节：按区域匹配，选择置信度高的结果
        ...
```

### 4.3 性能优化

- **并行执行**：RapidOCR 和 PaddleOCR 在不同线程并行运行
- **惰性加载**：PaddleOCR 仅在首次使用时加载（模型约 100MB）
- **结果缓存**：相同图片不重复 OCR
- **并发控制**：`ocr_workers` 限制并发数，避免 CPU 过载

---

## 5. 排除方案

| 方案 | 排除原因 |
|------|---------|
| PyMuPDF4LLM 内置 OCR | 存在"图片避让"问题，无法用于扫描件 |
| Surya | GPL-3.0 许可证与项目不兼容 |
| EasyOCR | 速度较慢，准确率无明显优势 |
| 纯 Tesseract | 中文识别准确率不足 |

---

## 6. 参考资料

### 官方文档

- [RapidOCR GitHub](https://github.com/RapidAI/RapidOCR)
- [RapidOCR PyPI](https://pypi.org/project/rapidocr-onnxruntime/)
- [PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR)
- [PaddleOCR 3.0 技术报告](https://arxiv.org/html/2507.05595v1)
- [PyMuPDF OCR 文档](https://pymupdf.readthedocs.io/en/latest/recipes-ocr.html)

### 社区调研

- [Modal: 8 Top Open-Source OCR Models Compared](https://modal.com/blog/8-top-open-source-ocr-models-compared)
- [E2E Networks: 7 Best Open-Source OCR Models 2025](https://www.e2enetworks.com/blog/complete-guide-open-source-ocr-models-2025)
- [KDnuggets: 10 Awesome OCR Models for 2025](https://www.kdnuggets.com/10-awesome-ocr-models-for-2025)
- [Plugger: OCR Comparison](https://www.plugger.ai/blog/comparison-of-paddle-ocr-easyocr-kerasocr-and-tesseract-ocr)
- [OCR Ranking 2025 - Pragmile](https://pragmile.com/ocr-ranking-2025-comparison-of-the-best-text-recognition-and-document-structure-software/)

### 其他参考

- [Surya OCR GitHub](https://github.com/datalab-to/surya)
- [EasyOCR GitHub](https://github.com/JaidedAI/EasyOCR)
- [Tesseract 中文重训练模型](https://github.com/gumblex/tessdata_chi)
- [MarkItDown 扫描 PDF 讨论](https://github.com/microsoft/markitdown/discussions/1361)

---

## 7. 2025-2026 OCR 技术演进：VLM 时代

### 7.1 范式转变

2025 年 OCR 领域发生了根本性的范式转变：从 **管道式 OCR** 转向 **端到端 VLM-based OCR**。

**传统管道式 OCR**：
```
图像 → 文本检测 → 文本识别 → 版面分析 → 结构化输出
      (模型 A)    (模型 B)    (模型 C)
```

**VLM-based OCR**：
```
图像 → Vision Language Model → 结构化 Markdown/HTML/JSON
              (单一模型)
```

关键突破点：
- **单次推理**：VLM 在一次前向传播中完成检测、识别、版面理解
- **语义理解**：不仅识别字符，还理解文档结构和上下文
- **多模态融合**：同时处理文本、表格、图表、公式、手写内容
- **多语言原生支持**：大型 VLM 训练数据覆盖 100+ 语言

### 7.2 2025 年 OCR 模型爆发

2025 年 10 月被称为"OCR 模型大爆发月"，六款重量级开源 OCR 模型集中发布：

| 模型 | 参数量 | olmOCR-Bench 得分 | 特点 |
|------|--------|------------------|------|
| **Chandra-OCR** | 9B | 83.1 | 当前开源最高分 |
| **olmOCR-2** | 7B | 82.4 | Allen AI，基于 Qwen2.5-VL |
| **Nanonets OCR2** | 3B | ~80 | 专为文档优化 |
| **DeepSeek-OCR** | 3B | - | 10x 压缩率，97% 解码精度 |
| **LightOn OCR** | 1B | 76.1 | 轻量级，低成本 |
| **PaddleOCR-VL** | 0.9B | - | 百度，超轻量 |

### 7.3 主流 VLM 的 OCR 能力对比

| 模型 | DocVQA | OCRBench | 手写识别 | 表格提取 | 中文支持 |
|------|--------|----------|---------|---------|---------|
| **Gemini 2.5 Pro** | 95%+ | - | 85% | 优秀 | ⭐⭐⭐⭐⭐ |
| **GPT-4o** | 90%+ | - | 优秀 | 优秀 | ⭐⭐⭐⭐ |
| **Claude 4.5 Opus** | - | - | 优秀 | 优秀 | ⭐⭐⭐⭐ |
| **Qwen3-VL-235B** | 96.4% | 88.8% | 优秀 | 优秀 | ⭐⭐⭐⭐⭐ |
| **Qwen2.5-VL-72B** | 96.4% | 88.8% | 优秀 | 优秀 | ⭐⭐⭐⭐⭐ |
| **MiniCPM-V 4.5** | - | 最高 | 优秀 | 优秀 | ⭐⭐⭐⭐⭐ |
| **Mistral OCR 3** | - | - | 88.9% | 96.6% | ⭐⭐⭐⭐ |

> **注**：MiniCPM-o 2.6 在 OCRBench 排行榜上超越 GPT-4o、Gemini 1.5 Pro

---

## 8. 云端 OCR 服务：OpenRouter 与主流提供商

### 8.1 OpenRouter 可用的 OCR 模型

[OpenRouter](https://openrouter.ai) 聚合了多家提供商的 VLM 模型，支持统一 API 调用：

#### 推荐模型（按性价比排序）

| 模型 | 输入价格 | 输出价格 | OCR 能力 | 推荐场景 |
|------|---------|---------|---------|---------|
| **Qwen3-VL-235B-A22B** | $0.35/M | $1.50/M | ⭐⭐⭐⭐⭐ | 高精度中文文档 |
| **Qwen2.5-VL-72B** | $0.40/M | $0.40/M | ⭐⭐⭐⭐⭐ | 性价比首选 |
| **Gemini 2.5 Flash** | $0.075/M | $0.30/M | ⭐⭐⭐⭐ | 大批量处理 |
| **DeepSeek-VL** | $0.14/M | $0.14/M | ⭐⭐⭐⭐ | 极致成本优化 |
| **Claude 3.5 Sonnet** | $3/M | $15/M | ⭐⭐⭐⭐⭐ | 复杂版面理解 |
| **GPT-4o** | $2.50/M | $10/M | ⭐⭐⭐⭐ | 通用场景 |

> **注**：价格单位为 USD/百万 token，实际图片消耗 token 数因分辨率而异（1024×1024 ≈ 1290 tokens）

#### OpenRouter 特性

- **统一 API**：一个 API Key 访问所有模型
- **自动回退**：主模型超时可自动切换备用模型
- **PDF 原生支持**：智能 PDF 解析，自动处理扫描件
- **免费模型**：部分模型有免费配额（有速率限制）

### 8.2 主流提供商专用 OCR API

#### Mistral OCR

Mistral 提供专门的 OCR API（`mistral-ocr-latest` / `mistral-ocr-2512`）：

| 特性 | 说明 |
|------|------|
| **价格** | $1/1,000 页（$0.001/页） |
| **速度** | 2,000 页/分钟（单 GPU 节点） |
| **准确率** | 94.9%（内部基准） |
| **手写识别** | 88.9%（优于 Azure 78.2%） |
| **表格提取** | 96.6%（优于 Textract 84.8%） |
| **特色** | 支持提取嵌入图片、LaTeX 公式、复杂版面 |
| **限制** | 单文件 ≤50MB，≤1,000 页 |

```python
# Mistral OCR 示例
from mistralai import Mistral

client = Mistral(api_key="your-api-key")
result = client.ocr.process(
    model="mistral-ocr-latest",
    document={"type": "pdf", "source": {"type": "url", "url": "..."}}
)
```

#### Google Gemini API

| 特性 | Gemini 2.5 Pro | Gemini 2.5 Flash |
|------|---------------|-----------------|
| **输入价格** | $1.25/M tokens | $0.075/M tokens |
| **输出价格** | $10/M tokens | $0.30/M tokens |
| **批处理折扣** | 50% | 50% |
| **缓存折扣** | 90% | 90% |
| **图片处理** | 1 页 PDF = 1 图片 ≈ 1290 tokens | 同左 |
| **免费额度** | 有（速率限制） | 有（速率限制） |

**成本估算**：
- 单页 PDF（1024×1024）：~1290 tokens
- 使用 Gemini 2.5 Flash：$0.075 × 1.29 = **$0.0001/页**
- 批处理模式：**$0.00005/页**

#### DeepSeek OCR

DeepSeek-OCR（2025 年 10 月发布）采用 **10x 压缩** 技术：

| 特性 | 说明 |
|------|------|
| **核心技术** | 将文本压缩为视觉 token，10x 压缩率 |
| **解码精度** | 97%@10x 压缩，60%@20x 压缩 |
| **速度** | ~2500 tokens/s（A100-40G） |
| **开源** | 模型权重已在 HuggingFace 发布 |
| **API 价格** | $0.28/M tokens（缓存命中 $0.028/M） |

**成本优势**：10 页文档从 10,000 tokens 压缩到 ~1,000 tokens，输入成本降低 90%。

### 8.3 传统云端 OCR 服务对比

| 服务商 | 基础 OCR 价格 | 结构化提取价格 | 特点 |
|-------|-------------|--------------|------|
| **Google Document AI** | $1.50/1K 页 | $10-50/1K 页 | 表单解析、实体提取 |
| **Azure Document Intelligence** | $1.50/1K 页 | $10-50/1K 页 | 预构建模型丰富 |
| **AWS Textract** | $1.50/1K 页 | $50/1K 页 | 表格、表单提取 |
| **ABBYY Cloud** | $2-5/1K 页 | 按需报价 | 企业级准确率 |

> **对比**：VLM-based OCR（如 Gemini Flash）成本仅为传统云端 OCR 的 **1/10 到 1/100**

---

## 9. 本地方案 vs 云端方案对比

### 9.1 综合对比表

| 维度 | 本地方案 | 云端 VLM API | 传统云端 OCR |
|------|---------|-------------|-------------|
| **准确率** | ⭐⭐⭐⭐ (PaddleOCR 领先) | ⭐⭐⭐⭐⭐ (Gemini/Claude 最佳) | ⭐⭐⭐⭐ |
| **中文支持** | ⭐⭐⭐⭐⭐ (PaddleOCR 专精) | ⭐⭐⭐⭐⭐ (Qwen 系列最佳) | ⭐⭐⭐⭐ |
| **手写识别** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **版面理解** | ⭐⭐⭐ (需额外模型) | ⭐⭐⭐⭐⭐ (原生支持) | ⭐⭐⭐⭐ |
| **表格提取** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **数据隐私** | ⭐⭐⭐⭐⭐ (完全本地) | ⭐⭐ (需信任提供商) | ⭐⭐ |
| **延迟** | ⭐⭐⭐⭐⭐ (无网络开销) | ⭐⭐⭐ (100-500ms 网络延迟) | ⭐⭐⭐ |
| **批量成本** | ⭐⭐⭐⭐⭐ (~$0.09/1K 页) | ⭐⭐⭐⭐ (~$0.10-1/1K 页) | ⭐⭐ ($1.50+/1K 页) |
| **少量成本** | ⭐⭐ (需 GPU 硬件) | ⭐⭐⭐⭐⭐ (按需付费) | ⭐⭐⭐⭐ |
| **部署复杂度** | ⭐⭐ (需配置环境) | ⭐⭐⭐⭐⭐ (API 调用) | ⭐⭐⭐⭐⭐ |
| **离线使用** | ⭐⭐⭐⭐⭐ | ❌ | ❌ |

### 9.2 成本深度分析

#### 少量文档（<1,000 页/月）

| 方案 | 成本 | 推荐度 |
|------|------|-------|
| 本地 RapidOCR | 免费（CPU 即可） | ⭐⭐⭐⭐⭐ |
| Gemini 2.5 Flash 免费额度 | 免费 | ⭐⭐⭐⭐⭐ |
| OpenRouter 免费模型 | 免费（有限额） | ⭐⭐⭐⭐ |

#### 中等规模（1,000-100,000 页/月）

| 方案 | 成本估算 | 推荐度 |
|------|---------|-------|
| 本地 RapidOCR (CPU) | ~$0（电费） | ⭐⭐⭐⭐ |
| Gemini 2.5 Flash | $10-100/月 | ⭐⭐⭐⭐⭐ |
| Mistral OCR | $1-100/月 | ⭐⭐⭐⭐⭐ |
| DeepSeek-VL API | $5-50/月 | ⭐⭐⭐⭐⭐ |

#### 大规模（>100,000 页/月）

| 方案 | 成本估算 | 推荐度 |
|------|---------|-------|
| 自建 GPU 集群 + PaddleOCR | ~$0.09/1K 页 | ⭐⭐⭐⭐⭐ |
| Gemini Flash 批处理 | ~$0.05/1K 页 | ⭐⭐⭐⭐ |
| 传统云端 OCR | $1.50+/1K 页 | ⭐⭐ |

**结论**：1000 万页/月规模下：
- 自建 GPU 集群：~$900/月
- Gemini Flash 批处理：~$500/月
- 传统云端 OCR：$15,000+/月

### 9.3 场景化推荐

#### 场景 1：企业敏感文档（合同、财务、医疗）

**推荐**：本地方案（RapidOCR + PaddleOCR）

理由：
- 数据完全不出内网
- 满足合规要求（HIPAA、GDPR 等）
- 一次部署，无持续成本

#### 场景 2：高精度复杂文档（学术论文、技术手册）

**推荐**：云端 VLM（Gemini 2.5 Pro / Claude）+ 本地后处理

理由：
- VLM 对复杂版面、公式、图表理解最佳
- 成本仍然可控
- 可结合本地模型做后处理校验

#### 场景 3：大批量简单文档（发票、收据、表单）

**推荐**：Mistral OCR API 或 Gemini Flash

理由：
- 极低成本（$0.001/页以下）
- 高吞吐量
- 结构化输出原生支持

#### 场景 4：离线/边缘场景（移动设备、嵌入式）

**推荐**：RapidOCR（ONNX）或 MiniCPM-V

理由：
- 完全离线运行
- 轻量级，支持 CPU 推理
- 无网络延迟

#### 场景 5：中文文档专精

**推荐**：本地 PaddleOCR + 云端 Qwen-VL 增强

理由：
- PaddleOCR 中文准确率最高
- Qwen 系列对中文理解最深
- 双引擎互补提升效果

### 9.4 MarkIt 集成建议

对于 MarkIt 项目，建议采用 **混合策略**：

```yaml
# markit.yaml 配置示例
ocr:
  # 本地 OCR 引擎（默认）
  local_engine: dual  # rapid | paddle | dual

  # 云端增强（可选）
  cloud_enhance: false
  cloud_provider: openrouter  # openrouter | gemini | mistral
  cloud_model: qwen/qwen-vl-max

  # 回退策略
  fallback_to_cloud: true  # 本地失败时使用云端
  confidence_threshold: 0.8  # 低于此置信度启用云端验证
```

**工作流程**：
1. 默认使用本地 RapidOCR + PaddleOCR 双引擎
2. 识别置信度 < 0.8 的区域，可选调用云端 VLM 增强
3. 用户可通过 `--llm` 参数启用云端增强
4. 敏感文档可强制 `--local-only` 禁用云端

---

## 10. 更新日志

### 2026-01-13 更新

- 新增 2025-2026 OCR 技术演进章节（VLM 范式转变）
- 新增云端 OCR 服务对比（OpenRouter、Mistral、Gemini、DeepSeek）
- 新增本地 vs 云端方案深度对比分析
- 新增场景化推荐指南
- 新增 MarkIt 混合策略集成建议
- 更新参考资料

---

## 11. 参考资料（补充）

### VLM OCR 相关

- [State of OCR in 2026 - AIMultiple](https://research.aimultiple.com/ocr-technology/)
- [Top 10 Vision Language Models 2026 - DextraLabs](https://dextralabs.com/blog/top-10-vision-language-models/)
- [Comparing Top 6 OCR Models 2025 - MarkTechPost](https://www.marktechpost.com/2025/11/02/comparing-the-top-6-ocr-optical-character-recognition-models-systems-in-2025/)
- [OCRBench v2 - arXiv](https://arxiv.org/html/2501.00321v2)
- [OmniDocBench - CVPR 2025](https://github.com/opendatalab/OmniDocBench)

### 模型与 API

- [OpenRouter Models](https://openrouter.ai/models)
- [Mistral OCR 官方文档](https://mistral.ai/news/mistral-ocr)
- [Mistral OCR 3 发布](https://mistral.ai/news/mistral-ocr-3)
- [Gemini API 定价](https://ai.google.dev/gemini-api/docs/pricing)
- [DeepSeek OCR - HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
- [olmOCR 2 - Allen AI](https://allenai.org/blog/olmocr-2)
- [Qwen3-VL GitHub](https://github.com/QwenLM/Qwen3-VL)

### 基准测试与对比

- [OmniAI OCR Benchmark](https://getomni.ai/blog/ocr-benchmark)
- [olmOCR-Bench Dataset](https://huggingface.co/datasets/allenai/olmOCR-bench)
- [Top 5 Vision LLMs for OCR - DocsRouter](https://docs.docsrouter.com/blog/top-5-vision-llms-for-ocr-in-2025-ranked-by-elo-score)
- [DeepSeek-OCR vs GPT-4 Vision - Skywork](https://skywork.ai/blog/ai-agent/deepseek-ocr-vs-gpt-4-vision-2025-comparison/)

### 成本与部署

- [7 Best Open-Source OCR Models 2025 - E2E Networks](https://www.e2enetworks.com/blog/complete-guide-open-source-ocr-models-2025)
- [Cloud vs On-Prem OCR 2025 - Veryfi](https://www.veryfi.com/technology/cloud-vs-on-premise-bank-check-ocr/)
- [OpenRouter Review 2025 - Skywork](https://skywork.ai/blog/openrouter-review-2025/)
