# MarkIt - 智能文档转 Markdown 工具

## 产品需求文档 (PRD)

**版本**: 1.0.0  
**日期**: 2026-01-06  
**状态**: Draft  

---

## 1. 概述

### 1.1 产品愿景

MarkIt 是一个高效、健壮的命令行工具，用于将各种格式的办公文档批量转换为高质量的 Markdown 文件。通过深度集成大语言模型（LLM）能力，实现智能化的格式优化、内容理解和图片描述生成。

### 1.2 目标用户

- 技术文档工程师
- 知识库管理员
- 内容迁移工程师
- 需要批量处理文档的开发者

### 1.3 核心价值

| 价值点 | 描述 |
|--------|------|
| **高效转换** | 支持 10+ 种文档格式批量转换，并发处理提升效率 |
| **智能优化** | LLM 驱动的格式清洗、内容理解和图片描述 |
| **健壮可靠** | 多引擎回退、断点续传、完善的错误处理 |
| **灵活配置** | 多 LLM Provider 支持，丰富的配置选项 |

---

## 2. 系统架构

### 2.1 高层架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLI Layer (Typer)                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                         Configuration (pydantic-settings)                   │
├──────────────────┬──────────────────┬──────────────────┬────────────────────┤
│   File Scanner   │  Format Router   │  Progress Track  │   State Manager    │
│                  │                  │    (rich/tqdm)   │  (断点续传/日志)     │
├──────────────────┴──────────────────┴──────────────────┴────────────────────┤
│                          Core Processing Pipeline                           │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐             │
│  │  Converter │  │   Image    │  │    LLM     │  │  Markdown  │             │
│  │   Engine   │──│ Processor  │──│  Enhancer  │──│   Writer   │             │
│  └────────────┘  └────────────┘  └────────────┘  └────────────┘             │
├──────────────────┬──────────────────┬───────────────────────────────────────┤
│  Conversion      │  Image Pipeline  │          LLM Providers                │
│  Engines         │                  │                                       │
│  ┌──────────┐    │  ┌──────────┐    │  ┌────────┐ ┌────────┐ ┌──────────┐   │
│  │MarkItDown│    │  │ Pillow   │    │  │ OpenAI │ │ Gemini │ │ Ollama   │   │
│  ├──────────┤    │  │ -simd    │    │  └────────┘ └────────┘ └──────────┘   │
│  │ Pandoc   │    │  ├──────────┤    │  ┌──────────┐                         │
│  ├──────────┤    │  │ oxipng   │    │  │OpenRouter│                         │
│  │PDFPlumber│    │  ├──────────┤    │  └──────────┘                         │
│  ├──────────┤    │  │ mozjpeg  │    │                                       │
│  │ PyMuPDF  │    │  └──────────┘    │                                       │
│  └──────────┘    │                  │                                       │
├──────────────────┴──────────────────┴───────────────────────────────────────┤
│                    Concurrency Layer (anyio + asyncio)                      │
│         AsyncIO + httpx (LLM)    │    ProcessPoolExecutor (Image/CPU)       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 技术选型

| 组件 | 技术方案 | 备选方案 | 选型理由 |
|------|----------|----------|----------|
| **CLI 框架** | Typer | Click, argparse | 类型安全，自动生成帮助文档，现代化 API |
| **配置管理** | pydantic-settings | python-dotenv | 类型验证，环境变量支持，嵌套配置 |
| **主转换引擎** | MarkItDown (fork) | - | 微软出品，质量高，可二次开发 |
| **回退转换** | Pandoc | - | 格式支持广泛，社区成熟 |
| **PDF 处理** | PyMuPDF + pdfplumber | pdf2image | PyMuPDF 速度快，pdfplumber 表格友好 |
| **图片处理** | Pillow | - | 12.x 版本已内置 SIMD 优化 |
| **图片压缩** | oxipng + mozjpeg | pngquant | 压缩率高，质量好 |
| **异步框架** | anyio | asyncio 原生 | 后端无关，便于测试 |
| **HTTP 客户端** | httpx | aiohttp | 同步异步统一 API，类型支持好 |
| **进度显示** | rich | tqdm | 功能丰富，视觉效果好 |
| **日志** | structlog | loguru | 结构化日志，便于分析 |
| **Office 转换** | MS Office (优先) / LibreOffice | - | MS Office 格式兼容最佳，LibreOffice 作为跨平台回退 |
| **LLM SDK** | openai / google-genai / ollama | httpx 原生 | 使用官方 SDK，维护性好，功能完整 |

### 2.3 目录结构

```
markit/
├── pyproject.toml              # 项目配置 (使用 uv/poetry)
├── README.md
├── markit/
│   ├── __init__.py
│   ├── __main__.py             # python -m markit 入口
│   ├── cli/
│   │   ├── __init__.py
│   │   ├── main.py             # Typer app 定义
│   │   ├── commands/
│   │   │   ├── convert.py      # convert 命令
│   │   │   ├── batch.py        # batch 命令
│   │   │   └── config.py       # config 命令
│   │   └── callbacks.py        # CLI 回调函数
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py         # pydantic-settings 配置
│   │   └── constants.py        # 常量定义
│   ├── core/
│   │   ├── __init__.py
│   │   ├── pipeline.py         # 主处理管道
│   │   ├── router.py           # 格式路由器
│   │   └── state.py            # 状态管理 (断点续传)
│   ├── converters/
│   │   ├── __init__.py
│   │   ├── base.py             # 转换器基类
│   │   ├── markitdown.py       # MarkItDown 封装
│   │   ├── pandoc.py           # Pandoc 封装
│   │   ├── pdf/
│   │   │   ├── __init__.py
│   │   │   ├── pymupdf.py      # PyMuPDF 实现
│   │   │   └── pdfplumber.py   # pdfplumber 实现
│   │   └── office.py           # LibreOffice 转换
│   ├── image/
│   │   ├── __init__.py
│   │   ├── extractor.py        # 图片提取
│   │   ├── converter.py        # 格式转换 (emf/wmf/tiff)
│   │   ├── compressor.py       # 图片压缩
│   │   └── analyzer.py         # LLM 图片分析
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base.py             # LLM Provider 基类
│   │   ├── openai.py
│   │   ├── gemini.py
│   │   ├── ollama.py
│   │   ├── openrouter.py
│   │   └── enhancer.py         # Markdown 增强器
│   ├── markdown/
│   │   ├── __init__.py
│   │   ├── formatter.py        # 格式化器
│   │   ├── frontmatter.py      # YAML Frontmatter 处理
│   │   └── chunker.py          # 大文件切块
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── fs.py               # 文件系统操作
│   │   ├── concurrency.py      # 并发工具
│   │   └── logging.py          # 日志配置
│   └── exceptions.py           # 自定义异常
├── tests/
│   ├── conftest.py
│   ├── unit/
│   ├── integration/
│   └── fixtures/               # 测试文件
└── scripts/
    └── install_deps.sh         # 安装外部依赖脚本
```

---

## 3. 核心功能模块

### 3.1 文档转换引擎

#### 3.1.1 支持格式

| 格式 | 扩展名 | 主引擎 | 回退引擎 | 备注 |
|------|--------|--------|----------|------|
| 纯文本 | `.txt` | MarkItDown | - | 直接读取 |
| Word 新版 | `.docx` | MarkItDown | Pandoc | 优先 MarkItDown |
| Word 旧版 | `.doc` | MS Office → MarkItDown | LibreOffice → MarkItDown | Windows 优先用 MS Office，跨平台用 LibreOffice |
| PPT 新版 | `.pptx` | MarkItDown | Pandoc | 优先 MarkItDown |
| PPT 旧版 | `.ppt` | MS Office → MarkItDown | LibreOffice → MarkItDown | Windows 优先用 MS Office，跨平台用 LibreOffice |
| Excel 新版 | `.xlsx` | MarkItDown | Pandoc | 表格保留 |
| Excel 旧版 | `.xls` | MS Office → MarkItDown | LibreOffice → MarkItDown | Windows 优先用 MS Office，跨平台用 LibreOffice |
| CSV | `.csv` | MarkItDown | Pandoc | 表格格式 |
| PDF | `.pdf` | PyMuPDF / pdfplumber | MarkItDown | 可配置 |
| PNG | `.png` | 图片分析管道 | - | LLM 描述 |
| JPEG | `.jpg/.jpeg` | 图片分析管道 | - | LLM 描述 |
| GIF | `.gif` | 图片分析管道 | - | LLM 描述 |
| WebP | `.webp` | 图片分析管道 | - | LLM 描述 |
| BMP | `.bmp` | 图片分析管道 | - | LLM 描述 |

#### 3.1.2 格式路由逻辑

```python
class FormatRouter:
    """根据文件格式路由到对应的转换器"""
    
    def route(self, file_path: Path) -> ConversionPlan:
        """
        返回转换计划，包含:
        - primary_converter: 主转换器
        - fallback_converter: 回退转换器 (可选)
        - pre_processors: 预处理步骤 (如 LibreOffice 转换)
        - post_processors: 后处理步骤 (如 LLM 优化)
        """
```

#### 3.1.3 转换器接口

```python
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass

@dataclass
class ConversionResult:
    markdown: str
    images: list[ExtractedImage]
    metadata: dict
    success: bool
    error: str | None = None

class BaseConverter(ABC):
    """转换器基类"""
    
    @abstractmethod
    async def convert(self, file_path: Path) -> ConversionResult:
        """转换文件为 Markdown"""
        pass
    
    @abstractmethod
    def supports(self, extension: str) -> bool:
        """检查是否支持该格式"""
        pass
```

### 3.2 图片处理管道

#### 3.2.1 处理流程

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   图片提取    │───▶│  格式转换     │───▶│   图片压缩     │───▶│  LLM 分析    │
│  (从文档)     │    │(emf/wmf/tiff)│    │(oxipng/moz)  │    │  (描述生成)   │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
  ExtractedImage     ConvertedImage     CompressedImage     AnalyzedImage
```

#### 3.2.2 图片提取

```python
@dataclass
class ExtractedImage:
    original_path: Path          # 原始路径/文档内位置
    data: bytes                  # 图片数据
    format: str                  # 原始格式
    source_document: Path        # 来源文档
    position: int                # 在文档中的位置

class ImageExtractor:
    """从文档中提取图片"""
    
    async def extract_from_docx(self, docx_path: Path) -> list[ExtractedImage]:
        """从 Word 文档提取图片"""
        
    async def extract_from_pptx(self, pptx_path: Path) -> list[ExtractedImage]:
        """从 PPT 提取图片"""
        
    async def extract_from_pdf(self, pdf_path: Path) -> list[ExtractedImage]:
        """从 PDF 提取图片"""
```

#### 3.2.3 格式转换

需要转换的格式:
- EMF (Enhanced Metafile) → PNG
- WMF (Windows Metafile) → PNG  
- TIFF → PNG/JPEG (根据是否有透明通道)

```python
class ImageFormatConverter:
    """转换不兼容的图片格式"""
    
    CONVERTIBLE_FORMATS = {'emf', 'wmf', 'tiff', 'tif'}
    
    async def convert(self, image: ExtractedImage) -> ConvertedImage:
        """使用 Pillow-simd 转换图片格式"""
        # EMF/WMF 可能需要额外库如 pyemf 或 Inkscape
```

#### 3.2.4 图片压缩

```python
@dataclass
class CompressionConfig:
    png_optimization_level: int = 2      # oxipng: 0-6
    jpeg_quality: int = 85               # mozjpeg: 0-100
    max_dimension: int = 2048            # 最大边长
    skip_if_smaller_than: int = 10240    # 小于 10KB 跳过

class ImageCompressor:
    """图片压缩处理"""
    
    async def compress(
        self, 
        image: ConvertedImage,
        config: CompressionConfig
    ) -> CompressedImage:
        """
        压缩流程:
        1. 检查是否需要压缩 (大小/格式)
        2. 缩放 (如果超过 max_dimension)
        3. PNG 使用 oxipng, JPEG 使用 mozjpeg
        4. 返回压缩后的图片
        """
```

#### 3.2.5 LLM 图片分析

```python
@dataclass
class ImageAnalysis:
    alt_text: str              # 用于 Markdown alt 的简短描述
    detailed_description: str  # 详细描述，用于生成 .md 文件
    detected_text: str | None  # OCR 识别的文字 (如有)
    image_type: str            # 类型: diagram, photo, screenshot, chart, etc.

class ImageAnalyzer:
    """使用 LLM 分析图片"""
    
    async def analyze(
        self, 
        image: CompressedImage,
        context: str | None = None  # 可选的上下文信息
    ) -> ImageAnalysis:
        """
        调用 LLM Vision API 分析图片
        返回 JSON 格式:
        {
            "alt_text": "一句话描述",
            "detailed_description": "详细描述...",
            "detected_text": "图中文字",
            "image_type": "diagram"
        }
        """
```

#### 3.2.6 图片输出

```python
class ImageOutputManager:
    """管理图片输出"""
    
    async def save(
        self,
        image: CompressedImage,
        analysis: ImageAnalysis,
        output_dir: Path
    ) -> ImageOutput:
        """
        输出结构:
        output/
        ├── document.md
        └── assets/
            ├── image_001.png
            ├── image_001.png.md    # 详细描述文档
            ├── image_002.jpg
            └── image_002.jpg.md
        """
```

### 3.3 LLM 集成

#### 3.3.1 Provider 抽象

```python
from abc import ABC, abstractmethod
from typing import AsyncIterator

@dataclass
class LLMMessage:
    role: str  # "system", "user", "assistant"
    content: str | list[ContentPart]  # 支持多模态

@dataclass
class LLMResponse:
    content: str
    usage: TokenUsage
    model: str
    finish_reason: str

class BaseLLMProvider(ABC):
    """LLM Provider 基类"""
    
    @abstractmethod
    async def complete(
        self,
        messages: list[LLMMessage],
        **kwargs
    ) -> LLMResponse:
        """同步完成"""
        
    @abstractmethod
    async def stream(
        self,
        messages: list[LLMMessage],
        **kwargs
    ) -> AsyncIterator[str]:
        """流式输出"""
        
    @abstractmethod
    async def analyze_image(
        self,
        image: bytes,
        prompt: str
    ) -> LLMResponse:
        """图片分析"""
```

#### 3.3.2 Provider 实现

优先使用大模型厂商提供的官方 SDK 或社区优秀的库：

| Provider | SDK/库 | 说明 |
|----------|--------|------|
| OpenAI | `openai` | 官方 Python SDK，支持异步 |
| Anthropic | `anthropic` | 官方 SDK，Claude 系列模型 |
| Google Gemini | `google-genai` | Google 新统一 SDK，替代旧的 `google-generativeai` |
| Ollama | `ollama` | 官方 Python 库，本地模型 |
| OpenRouter | `openai` | 兼容 OpenAI API，复用 openai SDK |

```python
# OpenAI Provider - 使用官方 SDK
from openai import AsyncOpenAI

class OpenAIProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
    
    async def complete(self, messages: list[LLMMessage], **kwargs) -> LLMResponse:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            **kwargs
        )
        return LLMResponse(
            content=response.choices[0].message.content,
            usage=TokenUsage(
                prompt=response.usage.prompt_tokens,
                completion=response.usage.completion_tokens
            ),
            model=response.model,
            finish_reason=response.choices[0].finish_reason
        )

# Anthropic Provider - 使用官方 SDK
from anthropic import AsyncAnthropic

class AnthropicProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model
    
    async def complete(self, messages: list[LLMMessage], **kwargs) -> LLMResponse:
        # 分离 system message
        system = next((m.content for m in messages if m.role == "system"), None)
        user_messages = [{"role": m.role, "content": m.content} 
                         for m in messages if m.role != "system"]
        
        response = await self.client.messages.create(
            model=self.model,
            system=system,
            messages=user_messages,
            max_tokens=kwargs.get("max_tokens", 4096),
            **kwargs
        )
        return LLMResponse(
            content=response.content[0].text,
            usage=TokenUsage(
                prompt=response.usage.input_tokens,
                completion=response.usage.output_tokens
            ),
            model=response.model,
            finish_reason=response.stop_reason
        )

# Google Gemini Provider - 使用官方 SDK
from google import genai

class GeminiProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        self.client = genai.Client(api_key=api_key)
        self.model = model
    
    async def complete(self, messages: list[LLMMessage], **kwargs) -> LLMResponse:
        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=[m.content for m in messages if m.role == "user"],
            **kwargs
        )
        return LLMResponse(...)

# Ollama Provider - 使用官方库
import ollama

class OllamaProvider(BaseLLMProvider):
    def __init__(self, model: str = "llama3.2-vision", host: str = "http://localhost:11434"):
        self.client = ollama.AsyncClient(host=host)
        self.model = model
    
    async def complete(self, messages: list[LLMMessage], **kwargs) -> LLMResponse:
        response = await self.client.chat(
            model=self.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            **kwargs
        )
        return LLMResponse(...)

# OpenRouter Provider - 复用 OpenAI SDK
class OpenRouterProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str):
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        self.model = model
```

#### 3.3.3 Provider 管理器与 Fallback

默认使用配置文件中第一个**有效**的 Provider，支持自动 fallback：

```python
class ProviderManager:
    """LLM Provider 管理器，支持 fallback"""
    
    def __init__(self, configs: list[LLMProviderConfig]):
        self.configs = configs
        self._providers: dict[str, BaseLLMProvider] = {}
        self._valid_providers: list[str] = []
    
    async def initialize(self):
        """初始化并验证所有 Provider"""
        for config in self.configs:
            try:
                provider = self._create_provider(config)
                # 验证 Provider 是否有效 (检查 API Key、连接等)
                await self._validate_provider(provider, config)
                self._providers[config.provider] = provider
                self._valid_providers.append(config.provider)
                log.info(f"Provider {config.provider} 初始化成功")
            except Exception as e:
                log.warning(f"Provider {config.provider} 初始化失败: {e}")
        
        if not self._valid_providers:
            raise LLMError("没有可用的 LLM Provider")
    
    def get_default(self) -> BaseLLMProvider:
        """获取默认 Provider (第一个有效的)"""
        if not self._valid_providers:
            raise LLMError("没有可用的 LLM Provider")
        return self._providers[self._valid_providers[0]]
    
    async def complete_with_fallback(
        self, 
        messages: list[LLMMessage],
        **kwargs
    ) -> LLMResponse:
        """带 fallback 的请求"""
        errors = []
        
        for provider_name in self._valid_providers:
            provider = self._providers[provider_name]
            try:
                return await provider.complete(messages, **kwargs)
            except Exception as e:
                log.warning(f"Provider {provider_name} 请求失败: {e}")
                errors.append((provider_name, e))
                continue
        
        # 所有 Provider 都失败
        raise LLMError(f"所有 Provider 都失败: {errors}")
    
    async def _validate_provider(
        self, 
        provider: BaseLLMProvider, 
        config: LLMProviderConfig
    ) -> bool:
        """验证 Provider 是否有效"""
        # 检查必要配置
        if config.provider in ("openai", "anthropic", "openrouter"):
            if not config.api_key:
                raise ValueError(f"{config.provider} 需要 api_key")
        
        # 可选: 发送测试请求验证连接
        # await provider.complete([LLMMessage(role="user", content="test")])
        return True
```

#### 3.3.4 Markdown 格式优化 (LLM 增强)

对于 markitdown/pandoc 处理后的 Markdown 文本，可选支持传递到大模型进行格式优化。

**优化内容：**

| 优化项 | 说明 |
|--------|------|
| **插入 Frontmatter** | 在文档头部插入 YAML 元数据 |
| **清洗垃圾内容** | 移除页眉页脚、水印、无意义字符等 |
| **标题层级修复** | 确保标题从 h2 (##) 开始，避免多个 h1 |
| **空行规范化** | 标题行与内容行之间保留一行空行 |
| **GFM 规范遵循** | 确保输出符合 GitHub Flavored Markdown 规范 |
| **列表格式化** | 统一列表标记，修复缩进 |
| **代码块规范** | 确保代码块有语言标识 |
| **链接检查** | 修复格式错误的链接 |

```python
@dataclass
class EnhancementConfig:
    remove_headers_footers: bool = True   # 移除页眉页脚
    fix_heading_levels: bool = True       # 修复标题层级 (从 h2 开始)
    normalize_blank_lines: bool = True    # 标题与内容间留一行空行
    follow_gfm: bool = True               # 遵循 GFM 规范
    add_frontmatter: bool = True          # 添加 YAML Frontmatter
    generate_summary: bool = True         # 生成一句话总结
    chunk_size: int = 4000                # 大文件切块大小 (tokens)

class MarkdownEnhancer:
    """使用 LLM 优化 Markdown 格式"""
    
    # LLM 优化 Prompt
    ENHANCEMENT_PROMPT = """
请优化以下 Markdown 文档的格式，遵循以下规则：

1. **清洗垃圾内容**：移除页眉、页脚、水印、重复的无意义字符
2. **标题层级**：确保标题从 ## (h2) 开始，不要有多个 # (h1)
3. **空行规范**：
   - 标题行上方留一个空行
   - 标题行下方留一个空行再写正文
   - 段落之间用一个空行分隔
4. **遵循 GFM 规范**：
   - 列表使用 `-` 作为无序列表标记
   - 代码块使用 ``` 并标注语言
   - 表格格式正确对齐
5. **保持内容完整**：不要删除或修改实际内容，只优化格式

原始 Markdown:
```markdown
{content}
```

请输出优化后的 Markdown（不要包含 ```markdown 标记）：
"""
    
    def __init__(self, provider_manager: ProviderManager, config: EnhancementConfig):
        self.provider_manager = provider_manager
        self.config = config
        self.chunker = MarkdownChunker(ChunkConfig(max_tokens=config.chunk_size))
    
    async def enhance(
        self,
        markdown: str,
        source_file: Path
    ) -> EnhancedMarkdown:
        """
        增强流程:
        1. 检查文档大小，必要时切块
        2. 调用 LLM 清洗和标准化
        3. 合并结果
        4. 注入 Frontmatter
        """
        # 切块处理
        chunks = self.chunker.chunk(markdown)
        
        # 并发处理各块
        enhanced_chunks = await asyncio.gather(*[
            self._process_chunk(chunk) for chunk in chunks
        ])
        
        # 合并
        enhanced_markdown = self.chunker.merge(enhanced_chunks)
        
        # 生成摘要
        summary = await self._generate_summary(enhanced_markdown) if self.config.generate_summary else ""
        
        # 注入 Frontmatter
        if self.config.add_frontmatter:
            enhanced_markdown = self._inject_frontmatter(
                enhanced_markdown, source_file, summary
            )
        
        return EnhancedMarkdown(content=enhanced_markdown, summary=summary)
    
    async def _process_chunk(self, chunk: str) -> str:
        """处理单个块"""
        prompt = self.ENHANCEMENT_PROMPT.format(content=chunk)
        response = await self.provider_manager.complete_with_fallback([
            LLMMessage(role="user", content=prompt)
        ])
        return response.content
    
    async def _generate_summary(self, markdown: str) -> str:
        """生成一句话总结"""
        # 取前 2000 字符用于生成摘要
        preview = markdown[:2000]
        response = await self.provider_manager.complete_with_fallback([
            LLMMessage(
                role="user", 
                content=f"请用一句话（不超过100字）总结以下文档的主要内容：\n\n{preview}"
            )
        ])
        return response.content.strip()
    
    def _inject_frontmatter(
        self,
        markdown: str,
        source_file: Path,
        summary: str
    ) -> str:
        """注入 YAML Frontmatter"""
        frontmatter = f"""---
title: "{source_file.stem}"
processed: "{datetime.now().isoformat()}"
description: "{summary}"
source: "{source_file.name}"
---

"""
        return frontmatter + markdown
```

#### 3.3.5 Frontmatter 格式

保持简洁，仅记录 4 个核心字段：

```yaml
---
title: "文档标题 (来自文件名)"
processed: "2026-01-06T12:00:00Z"
description: "LLM 生成的一句话总结"
source: "original_document.docx"
---
```

| 字段 | 说明 | 来源 |
|------|------|------|
| `title` | 文档标题 | 文件名 (去除扩展名) |
| `processed` | 处理时间 | ISO 8601 格式 |
| `description` | 一句话总结 | LLM 生成 (启用 `--llm` 时) |
| `source` | 源文件名 | 原始文件名 |

### 3.4 大文件切块处理

#### 3.4.1 成熟社区库选型

手写切块逻辑复杂且容易出错，推荐使用成熟的社区库：

| 库 | 特点 | 推荐场景 |
|-----|------|----------|
| **LangChain Text Splitters** | 功能全面，支持多种切分策略 | 首选，社区活跃 |
| **LlamaIndex Node Parser** | 与 LlamaIndex 生态集成好 | 已使用 LlamaIndex 的项目 |
| **semantic-text-splitter** | Rust 实现，性能好 | 高性能需求 |
| **tiktoken** | OpenAI 官方 tokenizer | 配合切块计算 token |

**推荐方案：LangChain Text Splitters**

```python
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
import tiktoken

@dataclass
class ChunkConfig:
    max_tokens: int = 4000
    overlap_tokens: int = 200
    model: str = "gpt-4o"  # 用于计算 token

class MarkdownChunker:
    """基于 LangChain 的大文档切块器"""
    
    def __init__(self, config: ChunkConfig):
        self.config = config
        self.encoding = tiktoken.encoding_for_model(config.model)
        
        # 按 Markdown 标题切分
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
            ]
        )
        
        # 递归字符切分 (处理超长段落)
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name=config.model,
            chunk_size=config.max_tokens,
            chunk_overlap=config.overlap_tokens,
        )
    
    def chunk(self, markdown: str) -> list[str]:
        """
        切块策略：
        1. 先按 Markdown 标题结构切分
        2. 对超长块再用递归切分
        """
        # 按标题切分
        header_chunks = self.header_splitter.split_text(markdown)
        
        # 处理超长块
        final_chunks = []
        for chunk in header_chunks:
            token_count = len(self.encoding.encode(chunk.page_content))
            if token_count > self.config.max_tokens:
                # 超长，继续切分
                sub_chunks = self.text_splitter.split_text(chunk.page_content)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk.page_content)
        
        return final_chunks
    
    def merge(self, chunks: list[str]) -> str:
        """合并处理后的块"""
        return "\n\n".join(chunks)
```

#### 3.4.2 备选方案：semantic-text-splitter

如果对性能要求高，可使用 Rust 实现的 `semantic-text-splitter`：

```python
from semantic_text_splitter import MarkdownSplitter
import tiktoken

class FastMarkdownChunker:
    """高性能切块器 (Rust 实现)"""
    
    def __init__(self, max_tokens: int = 4000):
        tokenizer = tiktoken.encoding_for_model("gpt-4o")
        self.splitter = MarkdownSplitter.from_tiktoken_model(
            "gpt-4o", 
            capacity=max_tokens
        )
    
    def chunk(self, markdown: str) -> list[str]:
        return self.splitter.chunks(markdown)
```

---

## 4. CLI 接口设计

### 4.1 命令结构

```bash
markit [OPTIONS] COMMAND [ARGS]

Commands:
  convert   转换单个文件
  batch     批量转换目录
  config    配置管理
  status    查看处理状态 (断点续传)
  resume    恢复中断的任务
```

### 4.2 convert 命令

```bash
markit convert [OPTIONS] INPUT_FILE

Arguments:
  INPUT_FILE    输入文件路径

Options:
  -o, --output PATH           输出目录 [default: ./output]
  --llm                       启用 LLM Markdown 格式优化 (默认不启用)
  --analyze-image             启用图片 LLM 分析 (默认不启用)
  --no-compress               禁用图片压缩
  --pdf-engine [pymupdf|pdfplumber|markitdown]
                              PDF 处理引擎 [default: pymupdf]
  --llm-provider [openai|anthropic|gemini|ollama|openrouter]
                              LLM 提供商 (默认使用配置中第一个有效的)
  --llm-model TEXT            LLM 模型名称
  -v, --verbose               详细输出
  --dry-run                   仅显示处理计划，不实际执行
```

#### LLM 相关参数说明

| 参数 | 作用 | 默认 |
|------|------|------|
| `--llm` | 启用 Markdown 格式优化：插入 Frontmatter、清洗垃圾内容、标题层级修复、空行规范化、遵循 GFM 规范等 | 不启用 |
| `--analyze-image` | 启用图片智能分析：生成 alt 文本和详细描述文档 | 不启用 |
| `--llm-provider` | 指定 LLM 提供商，覆盖配置文件 | 配置中第一个有效的 |
| `--llm-model` | 指定模型名称，覆盖配置文件 | Provider 默认模型 |

#### LLM Provider 选择逻辑

```
优先级：命令行参数 > 环境变量 > 配置文件
选择逻辑：使用第一个**有效**的 Provider（有配置且可连接）
Fallback：如果当前 Provider 请求失败，自动尝试下一个有效 Provider
```

### 4.3 batch 命令

```bash
markit batch [OPTIONS] INPUT_DIR

Arguments:
  INPUT_DIR    输入目录路径

Options:
  -o, --output PATH           输出目录 [default: INPUT_DIR/output]
  -r, --recursive             递归处理子目录
  --include PATTERN           包含的文件模式 (glob)
  --exclude PATTERN           排除的文件模式 (glob)
  --file-concurrency INT      文件处理并发数 [default: 4]
  --image-concurrency INT     图片处理并发数 [default: 8]
  --llm-concurrency INT       LLM 请求并发数 [default: 5]
  --on-conflict [skip|overwrite|rename]
                              输出文件冲突处理 [default: skip]
  --resume                    从上次中断处继续
  --state-file PATH           状态文件路径
  [... 其他 convert 命令的选项]
```

### 4.4 config 命令

```bash
markit config [OPTIONS] COMMAND

Commands:
  show      显示当前配置
  set       设置配置项
  init      初始化配置文件
  validate  验证配置
```

### 4.5 示例用法

```bash
# 转换单个文件 (不使用 LLM，仅基础转换)
markit convert document.docx -o ./output

# 转换并启用 LLM 格式优化 (Frontmatter、清洗、GFM规范等)
markit convert document.docx --llm -o ./output

# 转换并启用图片智能分析
markit convert document.docx --analyze-image -o ./output

# 同时启用 LLM 格式优化和图片分析
markit convert document.docx --llm --analyze-image -o ./output

# 批量转换目录
markit batch ./documents -o ./markdown --recursive

# 使用指定的 LLM Provider
markit convert doc.pdf --llm --llm-provider anthropic --llm-model claude-sonnet-4-20250514

# 使用本地 Ollama
markit convert doc.pdf --llm --llm-provider ollama --llm-model llama3.2-vision

# 恢复中断的任务
markit batch ./docs --resume --state-file ./markit-state.json

# 查看处理状态
markit status --state-file ./markit-state.json
```

---

## 5. 配置管理

### 5.1 配置优先级

1. 命令行参数 (最高)
2. 环境变量
3. 配置文件 (`~/.config/markit/config.toml` 或项目下 `markit.toml`)
4. 默认值 (最低)

### 5.2 配置模型

```python
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class LLMProviderConfig(BaseModel):
    """单个 LLM Provider 配置"""
    provider: str                       # openai, gemini, ollama, openrouter
    model: str                          # 模型名称
    api_key: str | None = None
    base_url: str | None = None
    timeout: int = 60
    max_retries: int = 3

class LLMConfig(BaseModel):
    """LLM 配置 - 支持多 Provider"""
    providers: list[LLMProviderConfig] = []  # 按优先级排序，第一个为默认
    default_provider: str | None = None       # 可指定默认，否则用 providers[0]

class ImageConfig(BaseModel):
    """图片处理配置"""
    enable_compression: bool = True
    png_optimization_level: int = 2
    jpeg_quality: int = 85
    max_dimension: int = 2048
    enable_analysis: bool = False       # 默认不启用 LLM 图片分析

class ConcurrencyConfig(BaseModel):
    """并发配置"""
    file_workers: int = 4
    image_workers: int = 8
    llm_workers: int = 5

class PDFConfig(BaseModel):
    """PDF 处理配置"""
    engine: str = "pymupdf"  # pymupdf, pdfplumber, markitdown
    extract_images: bool = True
    ocr_enabled: bool = False

class EnhancementConfig(BaseModel):
    """Markdown 增强配置"""
    enabled: bool = False               # 默认不启用 LLM 增强
    remove_headers_footers: bool = True
    fix_heading_levels: bool = True
    add_frontmatter: bool = True
    generate_summary: bool = True
    chunk_size: int = 4000

class OutputConfig(BaseModel):
    """输出配置"""
    default_dir: str = "output"
    on_conflict: str = "skip"  # skip, overwrite, rename
    create_assets_subdir: bool = True
    generate_image_descriptions: bool = True

class MarkitSettings(BaseSettings):
    """主配置类"""
    model_config = SettingsConfigDict(
        env_prefix="MARKIT_",
        env_nested_delimiter="__",
        toml_file="markit.toml",
    )
    
    llm: LLMConfig = Field(default_factory=LLMConfig)
    image: ImageConfig = Field(default_factory=ImageConfig)
    concurrency: ConcurrencyConfig = Field(default_factory=ConcurrencyConfig)
    pdf: PDFConfig = Field(default_factory=PDFConfig)
    enhancement: EnhancementConfig = Field(default_factory=EnhancementConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    
    log_level: str = "INFO"
    log_file: str | None = None
    state_file: str = ".markit-state.json"
```

### 5.3 配置文件示例

```toml
# markit.toml

log_level = "INFO"
log_file = "markit.log"
state_file = ".markit-state.json"

# LLM 配置 - 支持多 Provider，使用第一个有效的作为默认
# 有效 = 配置完整且可连接
[[llm.providers]]
provider = "openai"
model = "gpt-4o"
timeout = 60
max_retries = 3
# api_key 通过环境变量 OPENAI_API_KEY 设置

[[llm.providers]]
provider = "anthropic"
model = "claude-sonnet-4-20250514"
# api_key 通过环境变量 ANTHROPIC_API_KEY 设置

[[llm.providers]]
provider = "gemini"
model = "gemini-2.0-flash"
# api_key 通过环境变量 GOOGLE_API_KEY 设置

[[llm.providers]]
provider = "ollama"
model = "llama3.2-vision"
base_url = "http://localhost:11434"
# 本地服务，无需 api_key

[image]
enable_compression = true
png_optimization_level = 2
jpeg_quality = 85
max_dimension = 2048
enable_analysis = false    # 默认不启用

[concurrency]
file_workers = 4
image_workers = 8
llm_workers = 5

[pdf]
engine = "pymupdf"
extract_images = true
ocr_enabled = false

[enhancement]
enabled = false            # 默认不启用 LLM 增强
remove_headers_footers = true
fix_heading_levels = true
add_frontmatter = true
generate_summary = true
chunk_size = 4000

[output]
default_dir = "output"
on_conflict = "skip"
create_assets_subdir = true
generate_image_descriptions = true
```

### 5.4 环境变量

```bash
# LLM API Keys
export MARKIT_LLM__API_KEY="sk-..."
export MARKIT_LLM__PROVIDER="openai"
export MARKIT_LLM__MODEL="gpt-4o"

# OpenRouter
export MARKIT_LLM__PROVIDER="openrouter"
export MARKIT_LLM__BASE_URL="https://openrouter.ai/api/v1"

# Ollama
export MARKIT_LLM__PROVIDER="ollama"
export MARKIT_LLM__BASE_URL="http://localhost:11434"

# Gemini
export MARKIT_LLM__PROVIDER="gemini"
export GOOGLE_API_KEY="..."

# 并发控制
export MARKIT_CONCURRENCY__FILE_WORKERS=4
export MARKIT_CONCURRENCY__IMAGE_WORKERS=8
export MARKIT_CONCURRENCY__LLM_WORKERS=5
```

---

## 6. 并发模型

### 6.1 并发架构

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Main Process (asyncio)                        │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                    Orchestrator (anyio)                        │  │
│  │                                                                │  │
│  │  ┌─────────────────────┐    ┌──────────────────────────────┐   │  │
│  │  │   File Queue        │    │     Progress Tracker         │   │  │
│  │  │   (asyncio.Queue)   │    │     (rich.Progress)          │   │  │
│  │  └─────────────────────┘    └──────────────────────────────┘   │  │
│  │                                                                │  │
│  │  ┌─────────────────────────────────────────────────────────┐   │  │
│  │  │              Task Semaphores                            │   │  │
│  │  │  file_sem(4)  │  image_sem(8)  │  llm_sem(5)            │   │  │
│  │  └─────────────────────────────────────────────────────────┘   │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌───────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │
│  │  Async IO Tasks   │  │  Async IO Tasks  │  │  Process Pool    │   │
│  │  (LLM Requests)   │  │  (File Convert)  │  │  (Image Process) │   │
│  │                   │  │                  │  │                  │   │
│  │  httpx.AsyncClient│  │  MarkItDown      │  │  ProcessPool     │   │
│  │                   │  │  Pandoc          │  │  Executor        │   │
│  └───────────────────┘  └──────────────────┘  └──────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

### 6.2 并发控制实现

```python
import anyio
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager

class ConcurrencyManager:
    """并发管理器"""
    
    def __init__(self, config: ConcurrencyConfig):
        self.config = config
        self._file_semaphore: anyio.Semaphore | None = None
        self._image_semaphore: anyio.Semaphore | None = None
        self._llm_semaphore: anyio.Semaphore | None = None
        self._process_pool: ProcessPoolExecutor | None = None
    
    async def __aenter__(self):
        self._file_semaphore = anyio.Semaphore(self.config.file_workers)
        self._image_semaphore = anyio.Semaphore(self.config.image_workers)
        self._llm_semaphore = anyio.Semaphore(self.config.llm_workers)
        self._process_pool = ProcessPoolExecutor(
            max_workers=self.config.image_workers
        )
        return self
    
    async def __aexit__(self, *args):
        if self._process_pool:
            self._process_pool.shutdown(wait=True)
    
    @asynccontextmanager
    async def file_slot(self):
        """获取文件处理槽位"""
        async with self._file_semaphore:
            yield
    
    @asynccontextmanager
    async def image_slot(self):
        """获取图片处理槽位"""
        async with self._image_semaphore:
            yield
    
    @asynccontextmanager
    async def llm_slot(self):
        """获取 LLM 请求槽位"""
        async with self._llm_semaphore:
            yield
    
    async def run_in_process(self, func, *args):
        """在进程池中运行 CPU 密集型任务"""
        return await anyio.to_thread.run_sync(
            lambda: self._process_pool.submit(func, *args).result()
        )
```

### 6.3 图片处理并发

```python
class ImagePipeline:
    """图片处理管道"""
    
    def __init__(
        self, 
        concurrency: ConcurrencyManager,
        compressor: ImageCompressor,
        analyzer: ImageAnalyzer
    ):
        self.concurrency = concurrency
        self.compressor = compressor
        self.analyzer = analyzer
    
    async def process_batch(
        self,
        images: list[ExtractedImage],
        progress: Progress
    ) -> list[ProcessedImage]:
        """批量处理图片"""
        
        async def process_one(image: ExtractedImage) -> ProcessedImage:
            # 图片压缩 (CPU 密集，使用进程池)
            async with self.concurrency.image_slot():
                compressed = await self.concurrency.run_in_process(
                    self.compressor.compress_sync,
                    image
                )
            
            # LLM 分析 (IO 密集，使用异步)
            async with self.concurrency.llm_slot():
                analysis = await self.analyzer.analyze(compressed)
            
            return ProcessedImage(compressed, analysis)
        
        async with anyio.create_task_group() as tg:
            results = []
            for image in images:
                tg.start_soon(
                    lambda img=image: results.append(
                        await process_one(img)
                    )
                )
        
        return results
```

---

## 7. 状态管理与断点续传

### 7.1 状态模型

```python
from datetime import datetime
from enum import Enum

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class FileTask:
    """单个文件任务状态"""
    file_path: str
    status: TaskStatus
    output_path: str | None = None
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    images_processed: int = 0
    images_total: int = 0

@dataclass 
class BatchState:
    """批处理状态"""
    batch_id: str
    input_dir: str
    output_dir: str
    started_at: datetime
    updated_at: datetime
    tasks: dict[str, FileTask]  # file_path -> FileTask
    config_hash: str  # 配置哈希，用于检测配置变更
    
    @property
    def progress(self) -> tuple[int, int]:
        """返回 (已完成, 总数)"""
        completed = sum(
            1 for t in self.tasks.values() 
            if t.status in (TaskStatus.COMPLETED, TaskStatus.SKIPPED)
        )
        return completed, len(self.tasks)
```

### 7.2 状态管理器

```python
import json
import fcntl
from pathlib import Path

class StateManager:
    """状态管理器，支持断点续传"""
    
    def __init__(self, state_file: Path):
        self.state_file = state_file
        self._state: BatchState | None = None
        self._lock_file: Path = state_file.with_suffix('.lock')
    
    async def load(self) -> BatchState | None:
        """加载状态"""
        if not self.state_file.exists():
            return None
        
        async with await anyio.open_file(self.state_file, 'r') as f:
            data = json.loads(await f.read())
            return BatchState(**data)
    
    async def save(self, state: BatchState):
        """保存状态 (原子写入)"""
        temp_file = self.state_file.with_suffix('.tmp')
        
        async with await anyio.open_file(temp_file, 'w') as f:
            await f.write(json.dumps(asdict(state), default=str))
        
        temp_file.rename(self.state_file)
    
    async def update_task(
        self, 
        file_path: str, 
        status: TaskStatus,
        **kwargs
    ):
        """更新单个任务状态"""
        if self._state and file_path in self._state.tasks:
            task = self._state.tasks[file_path]
            task.status = status
            for k, v in kwargs.items():
                setattr(task, k, v)
            task.updated_at = datetime.now()
            await self.save(self._state)
    
    def get_pending_tasks(self) -> list[FileTask]:
        """获取待处理的任务"""
        if not self._state:
            return []
        return [
            t for t in self._state.tasks.values()
            if t.status in (TaskStatus.PENDING, TaskStatus.FAILED)
        ]
```

### 7.3 中断处理

```python
import signal

class InterruptHandler:
    """优雅中断处理"""
    
    def __init__(self, state_manager: StateManager):
        self.state_manager = state_manager
        self._interrupted = False
        self._current_tasks: set[str] = set()
    
    def setup(self):
        """设置信号处理"""
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)
    
    def _handle_interrupt(self, signum, frame):
        """处理中断信号"""
        if self._interrupted:
            # 二次中断，强制退出
            raise SystemExit(1)
        
        self._interrupted = True
        print("\n⚠️  收到中断信号，正在保存状态...")
        print("再次按 Ctrl+C 强制退出")
    
    @property
    def should_stop(self) -> bool:
        return self._interrupted
    
    async def mark_in_progress(self, file_path: str):
        """标记任务进行中"""
        self._current_tasks.add(file_path)
        await self.state_manager.update_task(
            file_path, 
            TaskStatus.IN_PROGRESS,
            started_at=datetime.now()
        )
    
    async def cleanup(self):
        """清理：将进行中的任务标记为待处理"""
        for file_path in self._current_tasks:
            await self.state_manager.update_task(
                file_path,
                TaskStatus.PENDING  # 重置为待处理
            )
```

---

## 8. 日志系统

### 8.1 日志配置

```python
import structlog
from rich.console import Console
from rich.logging import RichHandler

def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
    json_format: bool = False
):
    """配置 structlog"""
    
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]
    
    if json_format or log_file:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    if log_file:
        # 添加文件 handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter('%(message)s')
        )
        logging.root.addHandler(file_handler)
```

### 8.2 日志使用示例

```python
import structlog

log = structlog.get_logger()

async def convert_file(file_path: Path):
    log = log.bind(file=str(file_path))
    
    log.info("开始转换文件")
    
    try:
        result = await converter.convert(file_path)
        log.info(
            "转换完成",
            images_count=len(result.images),
            markdown_length=len(result.markdown)
        )
    except Exception as e:
        log.error("转换失败", error=str(e), exc_info=True)
        raise
```

### 8.3 日志输出格式

**控制台输出 (开发模式):**
```
2026-01-06 12:00:00 [info     ] 开始转换文件                   file=document.docx
2026-01-06 12:00:05 [info     ] 提取图片完成                   file=document.docx images=5
2026-01-06 12:00:10 [info     ] LLM 增强完成                   file=document.docx tokens=1234
2026-01-06 12:00:12 [info     ] 转换完成                       file=document.docx
```

**JSON 格式 (生产/日志文件):**
```json
{"timestamp": "2026-01-06T12:00:00Z", "level": "info", "event": "开始转换文件", "file": "document.docx"}
{"timestamp": "2026-01-06T12:00:05Z", "level": "info", "event": "提取图片完成", "file": "document.docx", "images": 5}
```

---

## 9. 错误处理

### 9.1 异常层次

```python
class MarkitError(Exception):
    """基础异常类"""
    pass

class ConversionError(MarkitError):
    """转换错误"""
    def __init__(self, file_path: Path, message: str, cause: Exception | None = None):
        self.file_path = file_path
        self.cause = cause
        super().__init__(f"转换 {file_path} 失败: {message}")

class ConverterNotFoundError(ConversionError):
    """找不到合适的转换器"""
    pass

class FallbackExhaustedError(ConversionError):
    """所有回退方案都失败"""
    pass

class LLMError(MarkitError):
    """LLM 相关错误"""
    pass

class RateLimitError(LLMError):
    """速率限制错误"""
    def __init__(self, retry_after: int | None = None):
        self.retry_after = retry_after
        super().__init__(f"速率限制，{retry_after}秒后重试" if retry_after else "速率限制")

class ImageProcessingError(MarkitError):
    """图片处理错误"""
    pass

class StateError(MarkitError):
    """状态管理错误"""
    pass
```

### 9.2 重试策略

```python
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

class RetryConfig:
    llm_max_attempts: int = 3
    llm_min_wait: int = 1
    llm_max_wait: int = 60
    
    conversion_max_attempts: int = 2

def llm_retry():
    """LLM 调用重试装饰器"""
    return retry(
        stop=stop_after_attempt(RetryConfig.llm_max_attempts),
        wait=wait_exponential(
            min=RetryConfig.llm_min_wait,
            max=RetryConfig.llm_max_wait
        ),
        retry=retry_if_exception_type((RateLimitError, httpx.TimeoutException)),
        before_sleep=lambda retry_state: log.warning(
            "LLM 请求重试",
            attempt=retry_state.attempt_number,
            wait=retry_state.next_action.sleep
        )
    )
```

### 9.3 回退处理

```python
class ConversionOrchestrator:
    """转换编排器，处理回退逻辑"""
    
    async def convert_with_fallback(
        self,
        file_path: Path,
        plan: ConversionPlan
    ) -> ConversionResult:
        """带回退的转换"""
        
        errors = []
        
        # 尝试主转换器
        try:
            log.info("尝试主转换器", converter=plan.primary_converter.name)
            return await plan.primary_converter.convert(file_path)
        except ConversionError as e:
            log.warning("主转换器失败", error=str(e))
            errors.append(e)
        
        # 尝试回退转换器
        if plan.fallback_converter:
            try:
                log.info("尝试回退转换器", converter=plan.fallback_converter.name)
                return await plan.fallback_converter.convert(file_path)
            except ConversionError as e:
                log.warning("回退转换器失败", error=str(e))
                errors.append(e)
        
        # 所有方案都失败
        raise FallbackExhaustedError(
            file_path,
            f"所有转换方案都失败: {[str(e) for e in errors]}"
        )
```

---

## 10. 处理管道

### 10.1 完整处理流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Processing Pipeline                               │
└─────────────────────────────────────────────────────────────────────────────┘

输入文件
    │
    ▼
┌─────────────┐     ┌─────────────┐
│  格式检测    │────▶│  格式路由    │
└─────────────┘     └─────────────┘
                          │
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
    ┌─────────┐     ┌─────────┐      ┌─────────┐
    │ 需要预   │     │ 直接可   │      │  图片   │
    │ 处理    │     │ 转换     │      │  文件    │
    │(doc/ppt)│     │(docx等) │      │         │
    └─────────┘     └─────────┘      └─────────┘
         │                │                │
         ▼                │                │
    ┌─────────┐           │                │
    │LibreOff │           │                │
    │转换格式  │           │                │
    └─────────┘           │                │
         │                │                │
         └────────┬───────┘                │
                  ▼                        │
         ┌─────────────┐                   │
         │ MarkItDown  │                   │
         │  或 Pandoc  │                   │
         └─────────────┘                   │
                  │                        │
                  ▼                        │
         ┌─────────────┐                   │
         │  提取图片    │                   │
         └─────────────┘                   │
                  │                        │
                  └──────────┬─────────────┘
                             ▼
                  ┌─────────────────────┐
                  │     图片处理管道      │
                  │  ┌───────────────┐  │
                  │  │  格式转换      │  │
                  │  │ (emf→png等)   │  │
                  │  └───────────────┘  │
                  │          │          │
                  │          ▼          │
                  │  ┌───────────────┐  │
                  │  │  图片压缩      │  │
                  │  │ (oxipng/moz)  │  │
                  │  └───────────────┘  │
                  │          │          │
                  │          ▼          │
                  │  ┌───────────────┐  │
                  │  │  LLM 分析     │  │
                  │  │ (可选)        │  │
                  │  └───────────────┘  │
                  └─────────────────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │   Markdown 处理     │
                  │  ┌───────────────┐  │
                  │  │  大文件切块    │  │
                  │  └───────────────┘  │
                  │          │          │
                  │          ▼          │
                  │  ┌───────────────┐  │
                  │  │  LLM 优化     │  │
                  │  │ (可选)        │  │
                  │  └───────────────┘  │
                  │          │          │
                  │          ▼          │
                  │  ┌───────────────┐  │
                  │  │ Frontmatter   │  │
                  │  │   注入        │  │
                  │  └───────────────┘  │
                  └─────────────────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │       输出          │
                  │  output/            │
                  │  ├── document.md    │
                  │  └── assets/        │
                  │      ├── img1.png   │
                  │      ├── img1.png.md│
                  │      └── ...        │
                  └─────────────────────┘
```

### 10.2 管道实现

```python
from dataclasses import dataclass
from typing import Callable, Awaitable

@dataclass
class PipelineContext:
    """管道上下文"""
    file_path: Path
    output_dir: Path
    config: MarkitSettings
    state_manager: StateManager
    concurrency: ConcurrencyManager
    progress: Progress

class Pipeline:
    """处理管道"""
    
    def __init__(self, ctx: PipelineContext):
        self.ctx = ctx
        self.router = FormatRouter()
        self.image_pipeline = ImagePipeline(...)
        self.enhancer = MarkdownEnhancer(...)
    
    async def process(self, file_path: Path) -> ProcessingResult:
        """处理单个文件"""
        
        # 1. 路由到转换器
        plan = self.router.route(file_path)
        
        # 2. 预处理 (如需要)
        if plan.pre_processors:
            for processor in plan.pre_processors:
                file_path = await processor.process(file_path)
        
        # 3. 转换
        conversion_result = await self._convert(file_path, plan)
        
        # 4. 处理图片
        images = await self.image_pipeline.process_batch(
            conversion_result.images,
            self.ctx.progress
        )
        
        # 5. 增强 Markdown
        if self.ctx.config.enhancement.enabled:
            markdown = await self.enhancer.enhance(
                conversion_result.markdown,
                file_path
            )
        else:
            markdown = conversion_result.markdown
        
        # 6. 输出
        return await self._write_output(markdown, images)
```

---

## 11. 输出管理

### 11.1 输出结构

```
output/
├── document1.md                    # 转换后的 Markdown
├── document1/                      # 或者使用子目录存放资源
│   └── assets/
│       ├── image_001.png
│       ├── image_001.png.md        # 图片详细描述
│       ├── image_002.jpg
│       └── image_002.jpg.md
├── document2.md
├── document2/
│   └── assets/
│       └── ...
└── .markit-state.json              # 状态文件 (可选)
```

### 11.2 冲突处理

```python
from enum import Enum

class ConflictStrategy(Enum):
    SKIP = "skip"           # 跳过已存在的文件
    OVERWRITE = "overwrite" # 覆盖已存在的文件
    RENAME = "rename"       # 重命名新文件 (添加后缀)

class OutputManager:
    """输出管理器"""
    
    def __init__(self, strategy: ConflictStrategy):
        self.strategy = strategy
    
    async def resolve_path(self, output_path: Path) -> Path:
        """解析输出路径，处理冲突"""
        
        if not output_path.exists():
            return output_path
        
        match self.strategy:
            case ConflictStrategy.SKIP:
                raise FileExistsError(f"文件已存在: {output_path}")
            case ConflictStrategy.OVERWRITE:
                return output_path
            case ConflictStrategy.RENAME:
                return self._generate_unique_path(output_path)
    
    def _generate_unique_path(self, path: Path) -> Path:
        """生成唯一路径"""
        stem = path.stem
        suffix = path.suffix
        parent = path.parent
        
        counter = 1
        while True:
            new_path = parent / f"{stem}_{counter}{suffix}"
            if not new_path.exists():
                return new_path
            counter += 1
```

### 11.3 图片描述文档格式

`assets/image_001.png.md`:

```markdown
---
source_image: image_001.png
image_type: diagram
generated_at: 2026-01-06T12:00:00Z
---

# 图片描述

## 简短描述

系统架构图，展示了前端、后端和数据库之间的交互关系。

## 详细描述

这是一张系统架构图，采用分层设计：

1. **前端层**：包含 React 应用和移动端 App
2. **API 网关**：负责请求路由和认证
3. **后端服务**：
   - 用户服务
   - 订单服务
   - 支付服务
4. **数据层**：PostgreSQL 主数据库 + Redis 缓存

各服务之间通过 REST API 和消息队列进行通信。

## 识别的文字

- "Frontend"
- "API Gateway"
- "Backend Services"
- "Database"
```

---

## 12. 依赖清单

### 12.1 Python 依赖

```toml
# pyproject.toml

[project]
name = "markit"
version = "1.0.0"
requires-python = ">=3.12"

dependencies = [
    # CLI
    "typer>=0.21.0",
    "rich>=14.2.0",
    
    # 配置
    "pydantic>=2.12.0",
    "pydantic-settings>=2.12.0",
    
    # 转换引擎
    "markitdown>=0.1.4",          # 微软转换库 (最新稳定版)
    "pypandoc>=1.15",             # Pandoc Python 封装
    
    # PDF 处理
    "pymupdf>=1.26.0",            # PDF 处理 (速度优先)
    "pdfplumber>=0.11.4",         # PDF 处理 (表格优先)
    
    # 图片处理
    "pillow>=12.0.0",             # 图片处理 (12.x 已内置 SIMD 优化，无需 pillow-simd)
    
    # 异步 & HTTP
    "anyio>=4.12.0",
    "httpx>=0.28.0",
    
    # LLM SDK - 使用官方库
    "openai>=2.14.0",             # OpenAI 官方 SDK
    "google-genai>=1.56.0",       # Google 新统一 SDK，替代旧的 google-generativeai
    "ollama>=0.6.1",              # Ollama 官方库
    "anthropic>=0.75.0",          # Anthropic 官方 SDK (可选)
    
    # 文本切块
    "langchain-text-splitters>=1.1.0",  # 成熟的切块库
    "tiktoken>=0.12.0",           # OpenAI tokenizer
    
    # 日志
    "structlog>=25.5.0",
    
    # 工具
    "tenacity>=9.1.0",            # 重试
    "python-magic>=0.4.27",       # 文件类型检测 (Linux/macOS)，项目近年无更新
    "python-magic-bin>=0.4.14",   # Windows 兼容，项目近年无更新
]

[project.optional-dependencies]
# 高性能切块 (可选)
performance = [
    "semantic-text-splitter>=0.22.0",  # Rust 实现的高性能切块
    # 注: pillow-simd 已移除，Pillow 12.x 已内置 SIMD 优化
]

dev = [
    "pytest>=8.3.0",
    "pytest-asyncio>=1.3.0",
    "pytest-cov>=6.0.0",
    "ruff>=0.14.0",
    "mypy>=1.19.0",
]
```

### 12.2 系统依赖

#### Linux (Ubuntu/Debian)

```bash
# install_deps_linux.sh

#!/bin/bash

# Pandoc
apt-get install -y pandoc

# LibreOffice (headless) - 作为 MS Office 的跨平台替代
apt-get install -y libreoffice --no-install-recommends

# 图片压缩工具
apt-get install -y oxipng
# mozjpeg 需要从源码编译或使用预编译包
# 或使用 jpegtran 作为替代
apt-get install -y libjpeg-turbo-progs

# python-magic 依赖
apt-get install -y libmagic1

# 可选: EMF/WMF 支持
apt-get install -y libwmf-bin inkscape
```

#### macOS

```bash
# install_deps_macos.sh

#!/bin/bash

# 使用 Homebrew 安装

# Pandoc
brew install pandoc

# LibreOffice - 作为 MS Office 的替代
brew install --cask libreoffice

# 图片压缩工具
brew install oxipng
brew install mozjpeg

# python-magic 依赖
brew install libmagic

# 可选: EMF/WMF 支持
brew install libwmf
brew install --cask inkscape
```

#### Windows

```powershell
# install_deps_windows.ps1

# 使用 winget 或手动安装

# Pandoc
winget install --id JohnMacFarlane.Pandoc

# MS Office - Windows 优先使用 (如已安装)
# 或安装 LibreOffice 作为替代
winget install --id TheDocumentFoundation.LibreOffice

# 图片压缩工具
# oxipng: 从 GitHub releases 下载
# https://github.com/shssoichiro/oxipng/releases

# mozjpeg: 从预编译包安装
# https://mozjpeg.codelove.de/binaries.html

# 可选: EMF/WMF 支持
# Windows 原生支持 EMF/WMF，无需额外安装

# 可选: Inkscape (用于矢量图处理)
winget install --id Inkscape.Inkscape
```

#### Windows 特殊说明

1. **MS Office COM 自动化**：如果系统已安装 MS Office，优先使用 COM 接口进行格式转换
   ```python
   # 需要安装 pywin32
   pip install pywin32
   ```

2. **python-magic**：Windows 需要额外的 DLL 文件
   ```bash
   pip install python-magic-bin  # 包含预编译的 libmagic
   ```

3. **路径处理**：注意 Windows 路径分隔符差异，代码中使用 `pathlib.Path`

#### 系统依赖检测

工具会在启动时检测系统依赖，并给出友好提示：

```python
class DependencyChecker:
    """系统依赖检测"""
    
    def check_all(self) -> list[DependencyStatus]:
        checks = [
            self._check_pandoc(),
            self._check_office(),        # MS Office 或 LibreOffice
            self._check_image_tools(),   # oxipng, mozjpeg
            self._check_libmagic(),
        ]
        return checks
    
    def _check_office(self) -> DependencyStatus:
        """检测 Office 套件"""
        if sys.platform == "win32":
            # 优先检测 MS Office
            if self._has_ms_office():
                return DependencyStatus("MS Office", True, "COM automation")
        # 回退检测 LibreOffice
        if self._has_libreoffice():
            return DependencyStatus("LibreOffice", True)
        return DependencyStatus("Office Suite", False, "需要安装 MS Office 或 LibreOffice")
```

### 12.3 Docker 支持

```dockerfile
# Dockerfile

FROM python:3.12-slim

# 系统依赖
RUN apt-get update && apt-get install -y \
    pandoc \
    libreoffice --no-install-recommends \
    oxipng \
    libjpeg-turbo-progs \
    libwmf-bin \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖
WORKDIR /app
COPY pyproject.toml .
RUN pip install uv && uv pip install .

COPY . .

ENTRYPOINT ["python", "-m", "markit"]
```

### 12.4 跨平台兼容性矩阵

| 功能 | Linux | macOS | Windows |
|------|-------|-------|---------|
| 基础转换 (docx/pptx/xlsx) | ✅ | ✅ | ✅ |
| 旧格式转换 (doc/ppt/xls) | LibreOffice | LibreOffice | MS Office (优先) / LibreOffice |
| PDF 处理 | ✅ | ✅ | ✅ |
| 图片压缩 (oxipng) | apt | brew | 手动下载 |
| 图片压缩 (mozjpeg) | apt/编译 | brew | 手动下载 |
| EMF/WMF 支持 | libwmf | libwmf | 原生支持 |
| python-magic | libmagic1 | libmagic | python-magic-bin |

---

## 13. 测试策略

### 13.1 测试层次

| 层次 | 覆盖内容 | 工具 |
|------|----------|------|
| 单元测试 | 各模块独立功能 | pytest |
| 集成测试 | 模块间交互、管道流程 | pytest + fixtures |
| E2E 测试 | CLI 命令完整流程 | pytest + subprocess |
| 性能测试 | 并发性能、大文件处理 | pytest-benchmark |

### 13.2 测试 fixtures

```
tests/fixtures/
├── documents/
│   ├── simple.docx          # 简单 Word 文档
│   ├── complex.docx         # 含图片表格的文档
│   ├── legacy.doc           # 旧版 Word
│   ├── slides.pptx          # PPT
│   ├── spreadsheet.xlsx     # Excel
│   └── sample.pdf           # PDF
├── images/
│   ├── photo.jpg
│   ├── diagram.png
│   ├── vector.emf
│   └── old_format.wmf
└── expected/                 # 预期输出
    ├── simple.md
    └── ...
```

### 13.3 Mock 策略

```python
# LLM Mock
@pytest.fixture
def mock_llm():
    """Mock LLM Provider"""
    with patch("markit.llm.openai.OpenAIProvider") as mock:
        mock.return_value.complete.return_value = LLMResponse(
            content="Mocked response",
            usage=TokenUsage(prompt=100, completion=50),
            model="gpt-4o",
            finish_reason="stop"
        )
        yield mock
```

---

## 14. 性能指标

### 14.1 目标指标

| 指标 | 目标值 | 说明 |
|------|--------|------|
| 单文件转换 (无 LLM) | < 5s | 中等复杂度 docx |
| 单文件转换 (含 LLM) | < 30s | 包含 LLM 优化 |
| 批量处理吞吐量 | > 100 文件/分钟 | 并发处理，无 LLM |
| 图片处理吞吐量 | > 50 张/分钟 | 含压缩和 LLM 分析 |
| 内存占用 | < 2GB | 处理 1000 文件批次 |

### 14.2 性能优化策略

1. **异步 IO**: 所有 IO 操作使用 asyncio
2. **进程池**: CPU 密集型图片处理使用 ProcessPoolExecutor
3. **批处理**: LLM 请求适当批处理减少开销
4. **流式处理**: 大文件流式读取，避免全量加载
5. **缓存**: 相同图片内容复用分析结果

---

## 15. 安全考虑

### 15.1 API Key 管理

- 支持环境变量存储 API Key
- 配置文件权限检查 (600)
- 日志中脱敏处理

### 15.2 文件处理安全

- 路径遍历防护
- 文件大小限制
- 格式验证 (magic number)

### 15.3 外部工具调用

- LibreOffice/Pandoc 沙箱执行
- 命令注入防护
- 超时控制

---

## 16. 里程碑规划

### Phase 1: MVP

- [ ] 项目脚手架搭建
- [ ] 基础 CLI (convert 命令)
- [ ] MarkItDown 集成
- [ ] 基础图片提取
- [ ] 单 LLM Provider (OpenAI)

### Phase 2: 核心功能

- [ ] Pandoc 回退支持
- [ ] LibreOffice 预处理
- [ ] 图片压缩管道
- [ ] LLM 图片分析
- [ ] Markdown 增强

### Phase 3: 批处理 & 健壮性

- [ ] batch 命令实现
- [ ] 并发控制
- [ ] 断点续传
- [ ] 多 LLM Provider

### Phase 4: 优化 & 完善

- [ ] 性能优化
- [ ] 完整测试覆盖
- [ ] 文档完善
- [ ] Docker 镜像

---

## 17. 附录

### A. 参考资料

- [MarkItDown GitHub](https://github.com/microsoft/markitdown)
- [Pandoc User's Guide](https://pandoc.org/MANUAL.html)
- [PyMuPDF Documentation](https://pymupdf.readthedocs.io/)
- [pdfplumber Documentation](https://github.com/jsvine/pdfplumber)
- [Typer Documentation](https://typer.tiangolo.com/)
- [pydantic-settings Documentation](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)

### B. 术语表

| 术语 | 说明 |
|------|------|
| Frontmatter | Markdown 文件头部的 YAML 元数据块 |
| EMF/WMF | Windows 矢量图格式 |
| OCR | 光学字符识别 |
| Token | LLM 处理的最小文本单位 |

### C. 变更日志

| 版本 | 日期 | 变更内容 |
|------|------|----------|
| 1.0.0 | 2026-01-06 | 初始版本 |
| 1.0.1 | 2026-01-06 | 优化：旧格式转换优先用 MS Office；LLM 使用官方 SDK；简化 Frontmatter；添加成熟切块库；CLI 默认不启用 LLM；多平台依赖说明 |
| 1.0.2 | 2026-01-06 | 优化：LLM Provider fallback 机制；新增 Anthropic Provider；明确 --llm 用于 Markdown 格式优化；详细描述格式优化内容；更新依赖版本 |
| 1.0.3 | 2026-01-06 | 依赖更新：google-genai 替代旧 google-generativeai；Pillow 12.x 内置 SIMD 优化移除 pillow-simd；全部依赖更新到最新版本 |
