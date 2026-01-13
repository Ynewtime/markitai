# v0.2.0 实施规格书

本版本进行**全量架构重构**，聚焦两个核心目标：

1. **本地 OCR 能力**：RapidOCR + PaddleOCR 双引擎方案，支持离线中文 OCR
2. **LLM 层重构**：全面切换至 LiteLLM，废弃现有 Provider 层代码

---

## 1. 本地 OCR 能力 (Local OCR)

### 1.1 技术背景

`pymupdf4llm` 存在**图片避让机制 (Image Avoidance)**：即便通过 OCR 注入了隐藏文本层，只要这些文本位于图片区域内，`pymupdf4llm` 的版面分析算法会将其忽略。因此传统的 "Sandwich PDF" 方案无法使用。

**测试验证**（详见 `docs/reference/ocr.md`）：

| 引擎 | 中文识别 | 英文识别 | 置信度 |
|------|---------|---------|--------|
| **RapidOCR** | 完美 | 完美 | 96-100% |
| PyMuPDF + Tesseract | 完全失败 | 部分错误 | - |

### 1.2 方案概述

采用 **RapidOCR + PaddleOCR 双引擎方案**：

| 模式 | 引擎组合 | 适用场景 |
|------|---------|---------|
| **默认模式** | RapidOCR + PaddleOCR | 最佳输出质量，适合重要文档 |
| **快速模式** (`--fast`) | 仅 RapidOCR | 平衡速度与质量，适合批量处理 |

### 1.3 核心流程

```
输入 PDF
    │
    ▼
┌─────────────────────────────┐
│  扫描件自动检测              │
│  (综合判断：文字密度 +        │
│   图片覆盖率 + 图片数量)      │
└─────────────────────────────┘
    │
    ├─ 非扫描件 ──→ 常规 pymupdf4llm 转换
    │
    └─ 扫描件 ──→ OCR 处理流程
                    │
                    ▼
            ┌───────────────────┐
            │  逐页渲染为图片     │
            │  (DPI 可配置)      │
            └───────────────────┘
                    │
                    ▼
            ┌───────────────────┐
            │  双引擎 OCR 识别   │
            │  (置信度选优融合)   │
            └───────────────────┘
                    │
                    ▼
            ┌───────────────────┐
            │  智能图片保留判断   │
            │  - 纯文字页：丢弃图片│
            │  - 含插图页：保留原图│
            └───────────────────┘
                    │
                    ▼
            ┌───────────────────┐
            │  组装 Markdown     │
            │  + 可选图片附件     │
            └───────────────────┘
```

### 1.4 扫描件自动检测

采用**综合判断策略**，结合多个指标评分：

```python
def _detect_scanned_pdf(page: fitz.Page) -> bool:
    """
    综合判断页面是否为扫描件

    评分标准：
    1. 文字密度：可提取文字 < 50 字符 → +40 分
    2. 图片覆盖率：图片区域 > 80% → +40 分
    3. 图片数量：页面仅包含 1 张大图 → +20 分

    总分 >= 60 → 判定为扫描件
    """
    score = 0

    # 1. 文字密度
    text = page.get_text().strip()
    if len(text) < 50:
        score += 40

    # 2. 图片覆盖率
    images = page.get_images()
    if images:
        image_area = _calculate_image_coverage(page, images)
        page_area = page.rect.width * page.rect.height
        if image_area / page_area > 0.8:
            score += 40

    # 3. 图片数量
    if len(images) == 1:
        score += 20

    return score >= 60
```

### 1.5 双引擎融合策略

采用**置信度选优**方案：

```python
class DualOCREngine(BaseOCREngine):
    """RapidOCR + PaddleOCR 双引擎，置信度选优"""

    async def recognize(self, image_data: bytes) -> list[OCRTextBlock]:
        # 1. 两个引擎并行执行
        rapid_result, paddle_result = await asyncio.gather(
            self.rapid.recognize(image_data),
            self.paddle.recognize(image_data) if self.paddle else [],
        )

        if not paddle_result:
            return rapid_result

        # 2. 按区域匹配，选择置信度更高的结果
        return self._merge_by_confidence(rapid_result, paddle_result)

    def _merge_by_confidence(
        self,
        rapid: list[OCRTextBlock],
        paddle: list[OCRTextBlock],
    ) -> list[OCRTextBlock]:
        """
        按区域匹配融合:
        - 计算两个引擎结果的 IoU (Intersection over Union)
        - IoU > 0.5 的区域，选择置信度更高的结果
        - 无匹配的区域，保留原结果
        """
        merged = []
        used_paddle = set()

        for r_block in rapid:
            best_match = None
            best_iou = 0.5  # IoU 阈值

            for i, p_block in enumerate(paddle):
                if i in used_paddle:
                    continue
                iou = _calculate_iou(r_block.bbox, p_block.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_match = (i, p_block)

            if best_match:
                i, p_block = best_match
                used_paddle.add(i)
                # 选择置信度更高的
                merged.append(
                    r_block if r_block.confidence >= p_block.confidence else p_block
                )
            else:
                merged.append(r_block)

        # 添加未匹配的 paddle 结果
        for i, p_block in enumerate(paddle):
            if i not in used_paddle:
                merged.append(p_block)

        return merged
```

### 1.6 智能图片保留

根据页面内容智能判断是否保留原始图片：

```python
def _should_preserve_image(
    page: fitz.Page,
    ocr_result: list[OCRTextBlock],
) -> bool:
    """
    智能判断是否保留原始页面图片

    保留条件（满足任一）：
    1. 检测到图表/流程图/示意图等非文字内容
    2. OCR 结果中存在低置信度区域 (< 0.7)
    3. 页面包含多个独立图片区域

    丢弃条件：
    - 纯文字扫描件，OCR 高置信度覆盖全页
    """
    # 1. 检查是否有低置信度区域
    low_conf_blocks = [b for b in ocr_result if b.confidence < 0.7]
    if low_conf_blocks:
        return True

    # 2. 检查图片数量和分布
    images = page.get_images()
    if len(images) > 1:
        return True

    # 3. 检查 OCR 覆盖率
    ocr_coverage = _calculate_ocr_coverage(page, ocr_result)
    if ocr_coverage < 0.9:
        return True

    return False
```

**输出格式**：

```markdown
<!-- 纯文字扫描件 -->
# 合同标题

第一条 甲方责任...

<!-- 含插图的扫描件 -->
# 产品说明书

![page_3](assets/page_3.png)

产品参数表：
| 型号 | 规格 | 价格 |
|------|------|------|
| A001 | 100x50 | ¥999 |
```

### 1.7 模块设计

#### 1.7.1 OCR 引擎抽象层 (`src/markit/ocr/`)

```
src/markit/ocr/
├── __init__.py          # 导出 create_ocr_engine()
├── base.py              # BaseOCREngine 抽象类，OCRTextBlock 数据类
├── rapid.py             # RapidOCREngine 实现
├── paddle.py            # PaddleOCREngine 实现
└── dual.py              # DualOCREngine 实现
```

**抽象接口**：

```python
# src/markit/ocr/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class OCRTextBlock:
    """OCR 识别结果块"""
    text: str
    confidence: float
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1)
    line_num: int = 0  # 行号，用于排序

class BaseOCREngine(ABC):
    """OCR 引擎抽象基类"""
    name: str = "base"

    @abstractmethod
    async def recognize(self, image_data: bytes) -> list[OCRTextBlock]:
        """识别图片中的文字"""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """检查引擎是否可用"""
        ...
```

**工厂函数**：

```python
# src/markit/ocr/__init__.py
def create_ocr_engine(engine: str = "dual") -> BaseOCREngine:
    """
    创建 OCR 引擎

    Args:
        engine: 引擎类型
            - "dual": RapidOCR + PaddleOCR 双引擎（默认）
            - "rapid": 仅 RapidOCR
            - "paddle": 仅 PaddleOCR

    Returns:
        OCR 引擎实例

    Raises:
        OCREngineNotAvailableError: 引擎不可用时抛出
    """
    if engine == "dual":
        return DualOCREngine()
    elif engine == "rapid":
        return RapidOCREngine()
    elif engine == "paddle":
        return PaddleOCREngine()
    else:
        raise ValueError(f"Unknown OCR engine: {engine}")
```

#### 1.7.2 PDF 转换器集成

修改 `src/markit/converters/pdf/pymupdf4llm.py`：

```python
class PyMuPDF4LLMConverter(BaseConverter):
    async def convert(self, file_path: Path, config: ConversionConfig) -> str:
        doc = fitz.open(file_path)

        # OCR 处理
        if config.pdf.ocr_enabled:
            if config.pdf.ocr_auto_detect:
                # 自动检测是否需要 OCR
                needs_ocr = self._detect_scanned_pdf(doc)
            else:
                needs_ocr = True

            if needs_ocr:
                return await self._convert_with_ocr(doc, config)

        # 常规转换
        return self._convert_normal(doc, config)

    async def _convert_with_ocr(
        self,
        doc: fitz.Document,
        config: ConversionConfig,
    ) -> str:
        """OCR 转换流程"""
        ocr_engine = create_ocr_engine(config.pdf.ocr_engine)
        markdown_parts = []
        images_to_save = []

        for page_num, page in enumerate(doc):
            # 1. 渲染页面为图片
            pix = page.get_pixmap(dpi=config.pdf.ocr_dpi)
            image_data = pix.tobytes("png")

            # 2. OCR 识别
            ocr_result = await ocr_engine.recognize(image_data)

            # 3. 智能图片保留判断
            if self._should_preserve_image(page, ocr_result):
                image_name = f"page_{page_num + 1}.png"
                images_to_save.append((image_name, image_data))
                markdown_parts.append(f"![{image_name}](assets/{image_name})\n\n")

            # 4. 组装文字
            text = self._ocr_blocks_to_text(ocr_result)
            markdown_parts.append(text)

        return "\n".join(markdown_parts), images_to_save
```

### 1.8 配置设计

#### 1.8.1 配置模型变更 (`src/markit/config/settings.py`)

**PDFConfig 扩展**（新增 3 个字段）：

```python
class PDFConfig(BaseModel):
    """PDF 转换配置"""
    # 现有字段
    engine: Literal["pymupdf4llm", "pymupdf", "pdfplumber", "markitdown"] = "pymupdf4llm"
    extract_images: bool = True
    ocr_enabled: bool = False  # 已存在

    # 新增字段
    ocr_auto_detect: bool = True  # 自动检测扫描件
    ocr_dpi: int = 200            # OCR 渲染分辨率
    ocr_engine: Literal["dual", "rapid", "paddle"] = "dual"
```

**ConcurrencyConfig 扩展**（新增 1 个字段）：

```python
class ConcurrencyConfig(BaseModel):
    """并发配置"""
    # 现有字段
    file_workers: int = 4     # 并行文档转换数
    image_workers: int = 8    # 并行图片处理数
    llm_workers: int = 5      # 并行 LLM 请求数

    # 新增字段
    ocr_workers: int = 2      # OCR 并发数（CPU 密集型，默认较小）
```

#### 1.8.2 配置文件示例 (`markit.example.yaml`)

```yaml
pdf:
  engine: pymupdf4llm
  extract_images: true

  # OCR 配置（v0.2.0 新增）
  ocr_enabled: true
  ocr_auto_detect: true    # 自动检测扫描件
  ocr_dpi: 200             # OCR 渲染分辨率（越高越清晰，但越慢）
  ocr_engine: dual         # dual / rapid / paddle

concurrency:
  file_workers: 4          # 并行文档转换数
  image_workers: 8         # 并行图片处理数
  llm_workers: 5           # 并行 LLM 请求数
  ocr_workers: 2           # OCR 并发数（v0.2.0 新增）

# 注意：--fast 模式会自动将 ocr_engine 切换为 rapid
```

### 1.9 CLI 集成

#### 1.9.1 现有 convert 命令参数

```bash
# 现有参数（保持不变）
markit convert document.docx                    # 基础转换
markit convert document.docx -o ./output        # 指定输出目录
markit convert document.docx --llm              # LLM Markdown 优化
markit convert document.docx --analyze-image    # LLM 图片分析（alt text）
markit convert document.docx --analyze-image-with-md  # 图片分析 + .md 描述文件
markit convert document.docx --no-compress      # 禁用图片压缩
markit convert document.pdf --pdf-engine pdfplumber   # 指定 PDF 引擎
markit convert document.docx --llm-provider openai    # 指定 LLM 提供商
markit convert document.docx --llm-model gpt-4        # 指定 LLM 模型
markit convert document.docx --verbose          # 详细输出
markit convert document.docx --fast             # 快速模式
markit convert document.docx --dry-run          # 预览模式
```

#### 1.9.2 新增 `--ocr` 参数

修改 `src/markit/cli/commands/convert.py` 和 `src/markit/cli/commands/batch.py`：

```python
# 新增参数
ocr: Annotated[
    bool,
    typer.Option(
        "--ocr",
        help="Enable OCR for scanned PDF documents.",
    ),
] = False,
```

**使用示例**：

```bash
# 启用 OCR
markit convert scanned.pdf --ocr

# OCR + 快速模式（自动切换为仅 RapidOCR）
markit convert scanned.pdf --ocr --fast

# OCR + LLM 增强
markit convert scanned.pdf --ocr --llm

# OCR + 图片分析（OCR 后对保留的图片进行 LLM 分析）
markit convert scanned.pdf --ocr --analyze-image

# 批量 OCR
markit batch ./scans -o ./output --ocr

# 批量 OCR + LLM 增强
markit batch ./scans -o ./output --ocr --llm
```

**参数交互逻辑**：

| 参数组合 | 行为 |
|---------|------|
| `--ocr` | 启用双引擎 OCR（RapidOCR + PaddleOCR） |
| `--ocr --fast` | 仅使用 RapidOCR（跳过 PaddleOCR） |
| `--ocr --llm` | OCR 提取文字 → LLM 优化 Markdown |
| `--ocr --analyze-image` | OCR + 智能图片保留 → LLM 分析保留的图片 |

#### 1.9.3 新增 `markit check` 命令

新增 `src/markit/cli/commands/check.py`：

```python
# src/markit/cli/commands/check.py
import typer
from rich.console import Console
from rich.table import Table

console = Console()


def check() -> None:
    """Check MarkIt environment and dependencies."""
    console.print("\n[bold blue]MarkIt Environment Check[/bold blue]\n")

    # Python version
    import sys
    console.print(f"✓ Python {sys.version.split()[0]}")

    # PyMuPDF
    try:
        import fitz
        console.print(f"✓ PyMuPDF {fitz.version[0]}")
    except ImportError:
        console.print("[red]✗ PyMuPDF not installed[/red]")

    console.print("\n[bold]OCR Engines:[/bold]")

    # RapidOCR
    try:
        from rapidocr_onnxruntime import __version__ as rapid_version
        console.print(f"✓ RapidOCR {rapid_version} (rapidocr-onnxruntime)")
    except ImportError:
        console.print("[red]✗ RapidOCR not installed[/red]")
        console.print("  Install: pip install rapidocr-onnxruntime")

    # PaddleOCR
    try:
        import paddleocr
        console.print(f"✓ PaddleOCR {paddleocr.__version__} (paddleocr + paddlepaddle)")
    except ImportError:
        console.print("[yellow]○ PaddleOCR not installed (optional)[/yellow]")
        console.print("  Install: pip install paddlepaddle paddleocr")

    console.print("\n[bold]LLM Support:[/bold]")

    # LiteLLM
    try:
        import litellm
        console.print(f"✓ LiteLLM {litellm.__version__}")
    except ImportError:
        console.print("[red]✗ LiteLLM not installed[/red]")

    console.print()
```

**输出示例**：

```bash
$ markit check

MarkIt Environment Check

✓ Python 3.12.0
✓ PyMuPDF 1.25.0

OCR Engines:
✓ RapidOCR 1.4.0 (rapidocr-onnxruntime)
✓ PaddleOCR 2.9.0 (paddleocr + paddlepaddle)

LLM Support:
✓ LiteLLM 1.50.0
```

#### 1.9.4 CLI 命令注册

修改 `src/markit/cli/main.py`：

```python
from markit.cli.commands.check import check

app.command()(check)
```

### 1.10 依赖配置

```toml
# pyproject.toml
[project]
dependencies = [
    # ... 现有依赖
    "rapidocr-onnxruntime>=1.4.0",  # OCR 核心引擎
]

[project.optional-dependencies]
ocr-full = [
    "paddlepaddle>=2.6.0",
    "paddleocr>=2.9.0",
]
ocr-gpu = [
    "paddlepaddle-gpu>=2.6.0",
    "paddleocr>=2.9.0",
]
```

### 1.11 性能与并发控制

OCR 是 **CPU 密集型操作**，需严格控制并发：

```python
# src/markit/services/ocr_service.py
class OCRService:
    def __init__(self, config: ConcurrencyConfig):
        self.semaphore = asyncio.Semaphore(config.ocr_workers)
        self.executor = ProcessPoolExecutor(max_workers=config.ocr_workers)

    async def process_page(self, image_data: bytes) -> list[OCRTextBlock]:
        async with self.semaphore:
            # 使用进程池执行 CPU 密集型 OCR
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.executor,
                self._sync_ocr,
                image_data,
            )
```

---

## 2. LLM 架构重构 (LiteLLM Migration)

### 2.1 技术背景

当前系统手写了多个 Provider 实现（OpenAI、Anthropic、Gemini、Ollama、OpenRouter），存在以下问题：

1. **维护成本高**：每个 Provider 需要独立处理认证、重试、错误处理
2. **功能不一致**：不同 Provider 的 streaming、tool calling 实现差异大
3. **扩展困难**：新增 Provider 需要大量代码

LiteLLM 提供统一接口，支持 100+ 提供商，可大幅简化代码。

### 2.2 现有接口分析

当前 `BaseLLMProvider` 定义了以下核心接口（需保持兼容）：

```python
# 现有接口 (src/markit/llm/base.py)
class BaseLLMProvider(ABC):
    name: str = "base"

    @abstractmethod
    async def complete(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs,
    ) -> LLMResponse: ...

    @abstractmethod
    async def stream(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs,
    ) -> AsyncIterator[str]: ...

    @abstractmethod
    async def analyze_image(
        self,
        image_data: bytes,
        prompt: str,
        image_format: str = "png",
        **kwargs,
    ) -> LLMResponse: ...

    async def validate(self) -> bool: ...
```

**关键数据类**（需保留）：
- `LLMMessage`: 消息结构（role, content，支持多模态）
- `LLMResponse`: 响应结构（content, usage, model, finish_reason, estimated_cost）
- `TokenUsage`: Token 用量统计
- `ResponseFormat`: 结构化输出配置

### 2.3 架构映射

| 维度 | 现状 | LiteLLM 方案 |
|------|------|--------------|
| **Provider** | 手写 5 个 Provider 类 | `litellm.acompletion()` 统一接口 |
| **接口** | `BaseLLMProvider` 抽象类 | **保留**，新实现继承此接口 |
| **Streaming** | 自定义 AsyncIterator | 统一 `ModelResponse` chunk |
| **Tokenizer** | 依赖 `tiktoken` | `litellm.encode()` 多 tokenizer |
| **Cost** | `ModelCostConfig` 配置表 | 内置价格表 + 自定义覆盖 |

### 2.4 代码变更

#### 2.4.1 删除旧代码

```
删除:
- src/markit/llm/openai.py
- src/markit/llm/anthropic.py
- src/markit/llm/gemini.py
- src/markit/llm/ollama.py
- src/markit/llm/openrouter.py
```

#### 2.4.2 保留/修改的文件

```
保留（类型定义）:
- src/markit/llm/base.py          # LLMMessage, LLMResponse, TokenUsage 等

修改:
- src/markit/llm/manager.py       # 适配 LiteLLM
- src/markit/llm/enhancer.py      # 适配新 Provider
- src/markit/llm/queue.py         # 适配新 Provider

新增:
- src/markit/llm/provider.py      # LiteLLM 统一 Provider
```

#### 2.4.3 统一 Provider 实现

```python
# src/markit/llm/provider.py
from collections.abc import AsyncIterator

import litellm
from litellm import acompletion

from markit.llm.base import (
    BaseLLMProvider,
    LLMMessage,
    LLMResponse,
    TokenUsage,
)
from markit.utils.logging import get_logger

log = get_logger(__name__)


class LiteLLMProvider(BaseLLMProvider):
    """基于 LiteLLM 的统一 Provider，兼容现有 BaseLLMProvider 接口"""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
        name: str | None = None,
        **kwargs,
    ):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.name = name or model
        self.extra_params = kwargs

    async def complete(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs,
    ) -> LLMResponse:
        """调用 LLM 完成，兼容现有接口"""
        try:
            # 转换消息格式
            converted_messages = self._convert_messages(messages)

            response = await acompletion(
                model=self.model,
                messages=converted_messages,
                api_key=self.api_key,
                api_base=self.api_base,
                temperature=temperature,
                max_tokens=max_tokens,
                **self.extra_params,
                **kwargs,
            )

            # 转换响应格式
            return self._convert_response(response)

        except Exception as e:
            self._handle_api_error(e, "completion", log)

    async def stream(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """流式调用，兼容现有接口"""
        converted_messages = self._convert_messages(messages)

        response = await acompletion(
            model=self.model,
            messages=converted_messages,
            api_key=self.api_key,
            api_base=self.api_base,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **self.extra_params,
            **kwargs,
        )

        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def analyze_image(
        self,
        image_data: bytes,
        prompt: str,
        image_format: str = "png",
        **kwargs,
    ) -> LLMResponse:
        """图片分析，兼容现有接口"""
        # 构造多模态消息
        message = LLMMessage.user_with_image(prompt, image_data, image_format)
        return await self.complete([message], **kwargs)

    async def validate(self) -> bool:
        """验证 Provider 配置"""
        try:
            # 使用 LiteLLM 的模型列表 API 验证
            await acompletion(
                model=self.model,
                messages=[{"role": "user", "content": "hi"}],
                api_key=self.api_key,
                api_base=self.api_base,
                max_tokens=1,
            )
            return True
        except Exception:
            return False

    def _convert_response(self, response) -> LLMResponse:
        """转换 LiteLLM 响应为 LLMResponse"""
        usage = None
        if response.usage:
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
            )

        return LLMResponse(
            content=response.choices[0].message.content or "",
            usage=usage,
            model=response.model,
            finish_reason=response.choices[0].finish_reason or "stop",
        )
```

#### 2.4.4 ProviderManager 适配

`ProviderManager` 保持现有的核心逻辑（懒加载、凭证级验证、能力路由、fallback），仅替换 Provider 实例化：

```python
# src/markit/llm/manager.py 关键变更

class ProviderManager:
    """保持现有接口，内部使用 LiteLLMProvider"""

    async def _create_provider(
        self,
        model_config: LLMModelConfig,
        credential: LLMCredentialConfig,
    ) -> BaseLLMProvider:
        """创建 Provider 实例（变更点）"""
        model_name = self._build_litellm_model_name(model_config, credential)

        return LiteLLMProvider(
            model=model_name,
            api_key=credential.api_key,
            api_base=credential.base_url,
            name=model_config.name,
        )

    def _build_litellm_model_name(
        self,
        model_config: LLMModelConfig,
        credential: LLMCredentialConfig,
    ) -> str:
        """
        构建 LiteLLM 模型名称

        LiteLLM 格式: provider/model
        例如: openai/gpt-4, anthropic/claude-3, ollama/llama3
        """
        provider = credential.provider
        model = model_config.model

        # 特殊处理
        if provider == "openrouter":
            return f"openrouter/{model}"
        elif provider == "ollama":
            return f"ollama/{model}"
        else:
            return f"{provider}/{model}"

    # 以下方法保持不变：
    # - complete_with_fallback()
    # - analyze_image_with_fallback()
    # - complete_with_concurrent_fallback()
    # - _ensure_provider_initialized()
    # - _validate_provider()
    # - has_capability()
    # - calculate_cost()
```

### 2.4 配置兼容

保持配置文件格式不变，仅内部实现切换：

```yaml
# markit.yaml - 配置格式不变
credentials:
  - id: openai-main
    provider: openai
    api_key: ${OPENAI_API_KEY}

  - id: ollama-local
    provider: ollama
    base_url: http://localhost:11434

models:
  - name: gpt-4-turbo
    credential: openai-main
    model: gpt-4-turbo
    capabilities: [text, vision]

  - name: llama3-local
    credential: ollama-local
    model: llama3:8b
    capabilities: [text]
```

### 2.5 依赖变更

```toml
# pyproject.toml
[project]
dependencies = [
    # 移除
    # "openai>=1.0.0",
    # "anthropic>=0.20.0",
    # "google-generativeai>=0.5.0",

    # 新增
    "litellm>=1.50.0",
]
```

### 2.6 验证场景

迁移后需验证：

1. **多 Provider 调用**：OpenAI / Anthropic / Gemini / Ollama / OpenRouter
2. **Ollama 本地调用**：`model="ollama/llama3"` + `api_base`
3. **流式响应**：确保 streaming chunk 正确解析
4. **异常处理**：`ContextWindowExceededError`、`RateLimitError` 等
5. **Fallback 机制**：多模型回退正常工作
6. **Vision 任务**：图片分析功能正常

---

## 3. 测试计划

### 3.1 OCR 测试

#### 单元测试

```python
# tests/unit/ocr/test_rapid_engine.py
class TestRapidOCREngine:
    async def test_recognize_chinese(self, chinese_image):
        engine = RapidOCREngine()
        result = await engine.recognize(chinese_image)
        assert any("中文" in b.text for b in result)

    async def test_confidence_threshold(self, noisy_image):
        engine = RapidOCREngine()
        result = await engine.recognize(noisy_image)
        assert all(b.confidence >= 0.5 for b in result)

# tests/unit/ocr/test_dual_engine.py
class TestDualOCREngine:
    async def test_merge_by_confidence(self):
        engine = DualOCREngine()
        # 验证置信度选优逻辑
        ...

    async def test_fallback_when_paddle_unavailable(self):
        # PaddleOCR 不可用时降级为 RapidOCR
        ...
```

#### 集成测试

```python
# tests/integration/test_pdf_ocr.py
class TestPDFOCR:
    async def test_scanned_pdf_detection(self, scanned_pdf, normal_pdf):
        """测试扫描件自动检测"""
        ...

    async def test_ocr_with_image_preservation(self, mixed_pdf):
        """测试智能图片保留"""
        ...
```

### 3.2 LiteLLM 测试

```python
# tests/unit/llm/test_litellm_provider.py
class TestLiteLLMProvider:
    async def test_openai_completion(self, mock_litellm):
        ...

    async def test_ollama_local(self, mock_litellm):
        ...

    async def test_streaming(self, mock_litellm):
        ...
```

### 3.3 性能基准

| 测试项 | 目标 |
|--------|------|
| 10 页扫描 PDF 双引擎 OCR | < 60 秒 |
| 10 页扫描 PDF 单引擎 OCR | < 30 秒 |
| 100 页扫描 PDF 内存峰值 | < 2GB |

---

## 4. 验收标准

### 4.1 OCR 功能验收

- [ ] `markit convert scanned.pdf --ocr` 成功提取扫描件文字
- [ ] 自动检测扫描件并启用 OCR
- [ ] 默认模式使用双引擎，结果融合正确
- [ ] `--fast` 模式自动切换为仅 RapidOCR
- [ ] 智能图片保留：纯文字页不输出图片，含插图页保留
- [ ] `markit check` 正确显示 OCR 引擎状态
- [ ] 仅安装 RapidOCR 时，双引擎模式降级并给出警告

### 4.2 LLM 功能验收

- [ ] LiteLLM 统一接口正常调用各提供商
- [ ] 多模型 fallback 正常工作
- [ ] 流式响应正常
- [ ] 配置文件格式完全兼容

### 4.3 性能验收

- [ ] 10 页扫描 PDF 双引擎 OCR 转换 < 60 秒
- [ ] 10 页扫描 PDF 单引擎 OCR 转换 < 30 秒
- [ ] 100 页扫描 PDF 内存峰值 < 2GB

### 4.4 测试验收

- [ ] 单元测试覆盖率 >= 80%
- [ ] OCR 单元测试：RapidOCR、PaddleOCR、DualOCREngine
- [ ] 集成测试：PDF OCR 转换、双引擎融合、智能图片保留

---

## 附录 A：代码变更清单

### A.1 新增文件

| 文件路径 | 描述 |
|---------|------|
| `src/markit/ocr/__init__.py` | OCR 模块入口，导出 `create_ocr_engine()` |
| `src/markit/ocr/base.py` | `BaseOCREngine` 抽象类，`OCRTextBlock` 数据类 |
| `src/markit/ocr/rapid.py` | `RapidOCREngine` 实现 |
| `src/markit/ocr/paddle.py` | `PaddleOCREngine` 实现 |
| `src/markit/ocr/dual.py` | `DualOCREngine` 双引擎实现 |
| `src/markit/llm/provider.py` | `LiteLLMProvider` 统一实现 |
| `src/markit/cli/commands/check.py` | `markit check` 命令 |
| `tests/unit/ocr/test_rapid_engine.py` | RapidOCR 单元测试 |
| `tests/unit/ocr/test_paddle_engine.py` | PaddleOCR 单元测试 |
| `tests/unit/ocr/test_dual_engine.py` | 双引擎单元测试 |
| `tests/unit/llm/test_litellm_provider.py` | LiteLLM Provider 单元测试 |
| `tests/integration/test_pdf_ocr.py` | PDF OCR 集成测试 |

### A.2 修改文件

| 文件路径 | 变更内容 |
|---------|---------|
| `src/markit/config/settings.py` | `PDFConfig` 新增 3 字段，`ConcurrencyConfig` 新增 1 字段 |
| `src/markit/converters/pdf/pymupdf4llm.py` | 新增 OCR 相关方法 |
| `src/markit/llm/manager.py` | 适配 `LiteLLMProvider` |
| `src/markit/llm/enhancer.py` | 适配新 Provider 接口 |
| `src/markit/llm/queue.py` | 适配新 Provider 接口 |
| `src/markit/services/llm_orchestrator.py` | 适配新 Provider |
| `src/markit/image/analyzer.py` | 适配新 Provider |
| `src/markit/cli/commands/convert.py` | 新增 `--ocr` 参数 |
| `src/markit/cli/commands/batch.py` | 新增 `--ocr` 参数 |
| `src/markit/cli/main.py` | 注册 `check` 命令 |
| `pyproject.toml` | 新增依赖 |

### A.3 删除文件

| 文件路径 | 原因 |
|---------|------|
| `src/markit/llm/openai.py` | 由 `LiteLLMProvider` 替代 |
| `src/markit/llm/anthropic.py` | 由 `LiteLLMProvider` 替代 |
| `src/markit/llm/gemini.py` | 由 `LiteLLMProvider` 替代 |
| `src/markit/llm/ollama.py` | 由 `LiteLLMProvider` 替代 |
| `src/markit/llm/openrouter.py` | 由 `LiteLLMProvider` 替代 |

### A.4 保留文件（类型定义）

| 文件路径 | 说明 |
|---------|------|
| `src/markit/llm/base.py` | 保留 `LLMMessage`, `LLMResponse`, `TokenUsage` 等类型定义 |

### A.5 依赖变更 (`pyproject.toml`)

**新增**：
```toml
dependencies = [
    "rapidocr-onnxruntime>=1.4.0",  # OCR 核心引擎
    "litellm>=1.50.0",               # LLM 统一接口
]

[project.optional-dependencies]
ocr-full = [
    "paddlepaddle>=2.6.0",
    "paddleocr>=2.9.0",
]
ocr-gpu = [
    "paddlepaddle-gpu>=2.6.0",
    "paddleocr>=2.9.0",
]
```

**移除**：
```toml
# 移除以下直接依赖（由 LiteLLM 统一管理）
# "openai>=1.0.0",
# "anthropic>=0.20.0",
# "google-generativeai>=0.5.0",
```

---

## 附录 B：现有代码结构参考

### B.1 当前项目结构

```
src/markit/
├── cli/
│   ├── commands/
│   │   ├── convert.py      # 单文件转换命令
│   │   ├── batch.py        # 批量转换命令
│   │   ├── config.py       # 配置管理命令
│   │   ├── provider.py     # Provider 管理命令
│   │   └── model.py        # Model 管理命令
│   ├── callbacks.py        # CLI 回调函数
│   ├── main.py             # CLI 入口
│   └── shared/             # 共享工具
├── config/
│   ├── settings.py         # Pydantic 配置模型
│   ├── constants.py        # 常量定义
│   └── prompts/            # 内置 Prompt 模板
├── converters/
│   ├── base.py             # BaseConverter 抽象类
│   ├── markitdown.py       # MarkItDown 转换器
│   ├── pandoc.py           # Pandoc 转换器
│   ├── office.py           # Office 预处理器
│   └── pdf/
│       ├── pymupdf4llm.py  # PyMuPDF4LLM 转换器
│       ├── pymupdf.py      # PyMuPDF 转换器
│       └── pdfplumber.py   # PDFPlumber 转换器
├── core/
│   ├── pipeline.py         # 主管道
│   ├── router.py           # 格式路由器
│   └── state.py            # 状态管理
├── image/
│   ├── analyzer.py         # LLM 图片分析
│   ├── compressor.py       # 图片压缩
│   ├── converter.py        # 格式转换
│   └── extractor.py        # 图片提取
├── llm/
│   ├── base.py             # BaseLLMProvider, LLMMessage, LLMResponse
│   ├── manager.py          # ProviderManager
│   ├── enhancer.py         # MarkdownEnhancer
│   ├── queue.py            # LLM 任务队列
│   ├── openai.py           # [将删除]
│   ├── anthropic.py        # [将删除]
│   ├── gemini.py           # [将删除]
│   ├── ollama.py           # [将删除]
│   └── openrouter.py       # [将删除]
├── markdown/
│   ├── chunker.py          # 文本分块
│   ├── formatter.py        # Markdown 格式化
│   └── frontmatter.py      # Frontmatter 处理
├── services/
│   ├── image_processor.py  # 图片处理服务
│   ├── llm_orchestrator.py # LLM 协调服务
│   └── output_manager.py   # 输出管理服务
└── utils/
    ├── logging.py          # 结构化日志
    ├── concurrency.py      # 并发工具
    ├── fs.py               # 文件系统工具
    ├── stats.py            # 统计工具
    └── adaptive_limiter.py # 自适应限流
```

### B.2 关键现有接口

**BaseLLMProvider** (`src/markit/llm/base.py`):
- `complete(messages, temperature, max_tokens)` → `LLMResponse`
- `stream(messages, temperature, max_tokens)` → `AsyncIterator[str]`
- `analyze_image(image_data, prompt, image_format)` → `LLMResponse`
- `validate()` → `bool`

**ProviderManager** (`src/markit/llm/manager.py`):
- `initialize(required_capabilities, preload_all, lazy)`
- `complete_with_fallback(messages, required_capability, prefer_capability)`
- `analyze_image_with_fallback(image_data, prompt, image_format)`
- `complete_with_concurrent_fallback(messages, timeout, required_capability)`
- `has_capability(capability)` → `bool`
- `calculate_cost(provider_name, response)` → `float | None`

**PyMuPDF4LLMConverter** (`src/markit/converters/pdf/pymupdf4llm.py`):
- 构造函数参数: `extract_images`, `image_dpi`, `write_images`, `embed_images`, `table_strategy`, `force_text`
- `convert(file_path)` → `ConversionResult`
