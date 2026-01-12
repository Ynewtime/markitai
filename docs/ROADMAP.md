# ROADMAP


## 任务批次 2026011202 - v0.2.0

### 目标

本版本聚焦于**本地化能力增强**与**底层架构升级**。
1. **本地 OCR 支持**：引入本地 OCR 引擎，支持离线环境下的扫描件/图片文字提取，提升中文识别准确率，降低对昂贵的多模态大模型的依赖。
2. **LLM 架构重构调研**：深度评估 `litellm` 生态，探索用其统一接口替代当前自研的 Provider/Model 适配层，以获得更广泛的模型支持、更精准的成本计算及更强大的路由能力。

### 任务

#### 1. 本地 OCR 能力支持 (Local OCR)

基于 `docs/reference/ocr.md` 的调研结论，分阶段落地本地 OCR 能力。

*   **Phase 1: PyMuPDF4LLM 内置 OCR (MVP)**
    *   **依赖集成**：更新 `pyproject.toml`，引入 `ocr` 可选依赖组（含 `opencv-python`）。
    *   **配置适配**：在 `markit.yaml` 的 `pdf` 节增加 OCR 开关及语言选项（默认 `chi_sim+eng`）。
    *   **系统适配**：在 `markit check` 命令中增加对系统级依赖（Tesseract）的检查与提示。
*   **Phase 2: 高精度中文 OCR (RapidOCR)**
    *   **引擎集成**：引入 `rapidocr-onnxruntime`，封装统一的 OCR 引擎接口。
    *   **混合策略**：支持在配置中选择 OCR 引擎（`pymupdf` vs `rapidocr`），为纯中文文档提供更高精度的选择。

#### 2. LLM 架构演进调研 (LiteLLM Analysis)

基于 `docs/reference/litellm.md`，开展 `litellm` 库与 MarkIt 现有架构的差异分析与重构预研。

*   **能力映射与差距分析**：
    *   **接口标准化**：对比 MarkIt `Provider` 抽象类与 LiteLLM `completion()` 统一接口，评估迁移复杂度。
    *   **成本/Token管理**：评估 LiteLLM 的 Cost Tracking 机制是否能覆盖 MarkIt v0.1.2 的自定义成本估算需求。
    *   **高级特性**：分析 LiteLLM 的 Router（负载均衡）、Fallback（故障转移）与 MarkIt 现有实现的优劣。
*   **重构可行性验证 (POC)**：
    *   创建一个 POC 分支，尝试用 `litellm` 替换 `markit/llm/openai.py` 等底层实现。
    *   **关键决策点**：
        1. 是否完全废弃现有的 `ProviderManager`？
        2. 如何保持现有的 CLI 交互体验（进度条、流式输出、Rich 渲染）不变？
        3. 配置文件结构是否需要随 LiteLLM 规范调整？

### 进展

待开始


---


# 归档

## 任务批次 2026010901

### 新特性

1. 特性一：任务级别的日志记录，每次任务生成对应时间戳的日志，同时在日志头部打印当此任务的详细配置，执行完成或者被打断时在该次任务日志尾部记录最终报告
2. 特性二：大模型资源池，支持配置/使用多个 Provider/Models，形成一个可用的负载均衡资源池，从而大大提高 llm 相关任务的并发数支持

### 可靠性

1. 根据 git 未提交的最新修改，进一步提高单元测试覆盖率
2. 审视 ruff 和单元测试，修复相关报错

### 进展

已完成

## 任务批次 2026010902

### 新特性

1. 特性一：新增 `markit provider select` 命令行功能，支持用户从已配置的 provider 中选取模型直接写入 markit.toml 配置。细分功能点：
   1. 需要优化当前的配置文件设计，区分 Provider 和 Model，`provider test` 和 `provider models` 命令应该针对 Provider 相关配置进行
   2. 对于 select/models 命令，要支持获取模型的 capabilities（如是否支持 text/vision/推理 等）

### 进展

已完成

## 任务批次 2026010903

### 重构

1. [x] 配置格式迁移：将配置文件格式从 TOML 迁移至 YAML
   1. [x] 引入 PyYAML 依赖
   2. [x] 更新配置加载逻辑 (YamlConfigSettingsSource)
   3. [x] 转换配置文件模板 (markit.example.toml -> markit.example.yaml)
   4. [x] 重构 CLI `provider select` 命令的配置写入逻辑
      - [x] 引入 `ruamel.yaml` 解决配置写入时的缩进和注释保留问题，防止配置错乱
      - [x] 修正 CLI 成功后的提示信息，更准确引导用户
   5. [x] 更新相关文档和默认值
   6. [x] 移除旧版 TOML 支持 (Hard Switch)

### 进展

已完成

## 任务批次 2026011001

### 新特性

1. 特性一：新增 `markit provider add` 命令，支持用户添加新的 provider credential
   1. 交互方式：无参数时启动交互式向导，有参数时直接使用参数值
   2. 支持的 provider 类型：openai, anthropic, gemini, ollama, openrouter，以及 OpenAI Compatible（复用 openai 类型，但强制要求填写 base_url）
   3. 添加成功后询问用户是否要立即运行 `provider select` 选择模型
   4. 使用 ruamel.yaml 保持配置文件格式和注释

### 进展

已完成

## 任务批次 2026011002

### 新特性

1. 特性一：优化 `markit provider` 命令
  1. -h 帮助菜单中，`add` 移到 `test` 命令前面
  2. 新增 `markit provider list` 命令，仅列出所有 Provider，不做测试
  3. 将 `markit provider models` 重命名为 `markit provider fetch`，功能不变
  4. 将 `markit provider select` 改为下方特性二中的 `markit model add`
2. 特性二：新增 `markit model` 命令
  1. 全部子命令为: `add` / `list`
  2. `add`：同 `markit provider select`
  3. `list`：仅列出所有 model
3. 可靠性：补充对应单元测试，刷新已有单元测试，验证 ruff 和已有单元测试，刷新文档

### 进展

已完成

## 任务批次 2026011003 - v0.1.0

### 背景

基于 `markit batch input --llm --analyze-image-with-md --verbose` 命令的日志分析，发现以下问题：

1. **模型初始化效率低**：初始化 5 个模型耗时 8 秒，但部分模型可能未被使用
2. **能力路由缺失**：文本任务在所有模型间轮询，浪费 vision 模型配额
3. **验证失败静默处理**：`_validate_provider` 验证异常时返回 `True`，导致无效模型被标记为可用
4. **日志信息不完整**：缺少耗时统计、token 用量、成本估算等关键信息
5. **缺少执行模式区分**：无法在速度和可靠性间灵活选择

另外，

1）当前系统缺少一次大批量文件输入的 batch 命令测试，如何验证程序在超长处理时间、并发拉满、触发模型限额中断等场景的能力？
2）结合工具当前的能力，对于工具的后续规划，你有什么建议吗？

### 改进计划

#### 1. 基于能力的模型路由（P0）

**目标**：文本任务优先使用 text-only 模型，vision 任务使用 vision 模型

**实现要点**：
- `complete_with_fallback` 添加 `required_capability` 和 `prefer_capability` 参数
- 向后兼容：参数为 None 时保持当前轮询行为
- 优先选择只具备所需能力的模型，减少高成本模型使用

#### 2. 懒加载模型初始化（P1）

**目标**：按需初始化模型，减少启动时间

**实现要点**：
- `initialize()` 方法支持 `required_capabilities` 参数
- 单文件转换：懒加载，首次使用时初始化
- batch 模式：预分析任务类型，提前初始化所需能力的模型
- 添加 `preload_all` 选项兼容当前行为

#### 3. 模型成本配置（P1）

**目标**：支持用户配置模型成本，用于成本统计和优化选择

**配置格式**（可选，不影响基本使用）：
```yaml
models:
  - name: xiaomi/mimo-v2-flash:free
    # 成本配置为可选，默认不需要配置
    # cost:
    #   input_per_1m: 0        # USD per 1M input tokens
    #   output_per_1m: 0       # USD per 1M output tokens
    #   cached_input_per_1m: 0 # USD per 1M cached input tokens (可选)
  - name: gpt-5.2
    cost:
      input_per_1m: 2.50
      output_per_1m: 10.00
```

**实现要点**：
- 成本配置完全可选，不配置不影响任何功能
- 配置后用于：日志统计中的成本估算、未来的成本优化路由
- 在 `markit config init` 生成的配置模板中说明配置方式

#### 4. 执行模式支持（P1）

**目标**：支持极速模式（`--fast`），默认为增强模式

| 特性 | 默认（增强模式） | 极速模式 (`--fast`) |
|------|-----------------|---------------------|
| 模型验证 | 完整验证 + 重试 | 跳过 |
| 错误处理 | 重试 + 详细报错 | 跳过失败，继续 |
| 日志级别 | DEBUG 全量 | ERROR + 统计 |
| LLM fallback | 尝试所有模型 | 最多1次 |

**配置方式**：
- 命令行：`markit batch input --fast`
- 配置文件：`execution: { mode: fast }`

#### 5. 验证策略优化（P2）

**目标**：支持配置验证行为（注意兼容旧配置，默认不需要配置，同时要兼容 convert/batch/--fast 不同场景）

**配置格式**：
```yaml
llm:
  validation:
    enabled: true        # 是否验证
    retry_count: 2       # 重试次数
    on_failure: warn     # warn | skip | fail
```

#### 6. 增强日志统计（P2）

**目标**：增强模式下提供详尽统计

**输出示例**：
```
Complete: 6 success, 0 failed
Total: 152s | LLM: 112s | Convert: 8s | Init: 8s
Tokens: 28,543 | Est. cost: $0.12
Models used: xiaomi(3), deepseek(1), gemini(2), gpt-5.2(8), claude(4)
```

#### 7. 超时触发并发 fallback（P1）

**背景**：日志显示 Gemini 服务端超时返回 504 耗时约 90 秒，导致整体任务阻塞

**目标**：主模型超时后启动备用模型并发执行，不打断主模型，谁先返回用谁

**核心原则**：
- 给主模型充足时间（180s）
- 超时不打断，而是启动并发备模型
- 没有备模型时继续等待主模型

**实现方案**：
```python
async def complete_with_concurrent_fallback(self, messages, timeout=180):
    """主模型超时后启动备用模型并发，不打断主模型"""
    primary_task = asyncio.create_task(primary.complete(messages))

    try:
        return await asyncio.wait_for(shield(primary_task), timeout=timeout)
    except asyncio.TimeoutError:
        # 超时不打断主模型，检查是否有备用模型
        if not fallback_provider:
            log.warning(f"Primary model exceeded {timeout}s, no fallback available, continuing to wait...")
            return await primary_task  # 继续等待主模型

        # 启动备用模型并发执行，主模型继续运行
        log.warning(f"Primary model exceeded {timeout}s, starting fallback concurrently")
        fallback_task = asyncio.create_task(fallback.complete(messages))

        # 谁先返回用谁
        done, pending = await asyncio.wait(
            [primary_task, fallback_task],
            return_when=FIRST_COMPLETED
        )

        # 取消未完成的任务
        for task in pending:
            task.cancel()

        return done.pop().result()
```

**配置**：
```yaml
llm:
  timeout: 180    # 主模型等待时间，超时后启动并发 fallback（默认 180s）
```

**效果**：
- 主模型 < 180s：正常返回
- 主模型 > 180s 且有备模型：启动并发，谁快用谁
- 主模型 > 180s 且无备模型：继续等待主模型直到完成或失败

#### 8. `markit config` 命令优化

1）`markit config show` 改名为 `markit config list`
2）`markit config validate` 改名为 `markit config test`
3）调整 `markit config` 命令顺序为 `init/test/list/locations`

### 进展

已完成


## 任务批次 2026011101 - v0.1.1

### 角色
请担任高级 Python 软件架构师和 DevOps 工程师。

### 上下文
你正在分析 `markit` 代码库，这是一个智能文档转 Markdown 的工具。该项目使用了 `typer`、`rich`、`pydantic`、`anyio` 以及多种 LLM SDK。目前的核心逻辑位于 `markit/core/pipeline.py` 中，该文件已演变成一个职责混乱的“上帝类（God Class）”。

### 目标
重构代码库以提高可维护性、性能和配置管理规范，深度优化 LLM 提示词以适配中文场景，同时确保不改变外部 CLI 的行为。

### 任务

#### 第一阶段：架构重构（简化与解耦）
1.  **拆解 `ConversionPipeline`**：
    *   分析 `markit/core/pipeline.py`。
    *   将图像处理逻辑（约第 796-911 行）提取到新的服务 `markit/services/image_processor.py` 中，建立 `ImageProcessingService` 类。
    *   将 LLM 协调逻辑（约第 512-635 行）提取到 `markit/services/llm_orchestrator.py` 中，建立 `LLMOrchestrator` 类。
    *   将文件输出逻辑（约第 1084-1238 行）提取到 `markit/services/output_manager.py` 中，建立 `OutputManager` 类。
    *   更新 `ConversionPipeline` 以注入并使用这些新服务，使其仅作为高层协调者存在。
2.  **标准化接口**：
    *   确保所有新服务都实现由抽象基类或 Protocol 定义的清晰异步接口。

#### 第二阶段：配置管理（最佳实践）
1.  **移除硬编码值**：
    *   定位 `markit/core/pipeline.py` 中的 `_get_default_model` 方法。
    *   将默认模型字典（如 OpenAI: "gpt-5.2" 等）移动到 `markit/config/constants.py` 或 `markit/config/defaults.py` 中。
    *   更新 `markit/config/settings.py` 以引用这些常量，避免在业务逻辑中硬编码。
    *   尽可能识别并提取其他硬编码值，尤其是配置相关的。

#### 第三阶段：性能优化
1.  **优化 LibreOffice 转换（配置池模式）** (`markit/converters/office.py`)：
    *   **关键要求**：为了避免 LibreOffice 的并发锁文件冲突，必须实现**互斥访问**的配置池。
    *   创建一个 `LibreOfficeProfilePool` 类，管理 N 个独立的配置目录（N = 最大并发数）。
    *   使用 `asyncio.Queue` 或 `contextlib.asynccontextmanager` 来实现“借出/归还”机制。确保在同一时刻，**一个配置目录只能被一个转换任务占用**。
    *   不要在每次转换后删除目录，而是保留复用。仅在连续失败或处理 X 次后进行重置（清理并重建）。
2.  **并行图像处理**：
    *   在新的 `ImageProcessingService` 中，将 CPU 密集型的图像压缩/转换任务（PIL 操作）从 `asyncio.to_thread`（多线程）切换为 `ProcessPoolExecutor`（多进程），以绕过 Python 的全局解释器锁（GIL）并提升性能。

#### 第四阶段：大模型提示词工程（Prompt Engineering）
1.  **优化文档增强提示词** (`markit/llm/enhancer.py`)：
    *   **语言规则**：在 `ENHANCEMENT_PROMPT` 和 `SUMMARY_PROMPT` 中明确增加规则，**要求默认使用中文**输出摘要和处理说明（除非文档内容完全是其他语言）。
    *   **细节完善**：
        *   增加对技术文档中代码块（Code Blocks）保留的强调，防止误删。
        *   增加对复杂表格（Tables）格式化的具体指令，确保 Markdown 表格对齐。
        *   增加去除“扫描件水印”、“页眉页脚干扰字符”的具体示例。
2.  **优化图像分析提示词** (`markit/image/analyzer.py`)：
    *   **语言规则**：修改 `IMAGE_ANALYSIS_PROMPT`，明确要求返回的 JSON 字段中 `alt_text`（替代文本）和 `detailed_description`（详细描述）**必须使用中文**。
    *   **细节完善**：
        *   增强 `image_type` 的分类描述，使其更精准。
        *   增加对 OCR 文本 (`detected_text`) 的处理指示：如果是乱码则忽略，如果是关键信息则提取。
3.  **知识图谱支持**：
    *   **元数据提取**：新增规则，尽可能提取有利于后续知识图谱改造相关的 meta 元属性信息。

#### 第五阶段：测试与质量（覆盖率）
1.  **更新测试**：
    *   在 `tests/unit/services/` 中为新的 `ImageProcessingService`、`LLMOrchestrator` 和 `OutputManager` 创建单元测试。
    *   重构 `tests/unit/test_pipeline.py`，通过 Mock 这些新服务来仅验证协调逻辑。
2.  **类型安全**：
    *   确保所有新代码都能通过 `mypy --strict` 检查。

### 约束条件
*   **严禁**破坏现有的 CLI 命令（`markit convert`, `markit batch`）及其参数行为。
*   **严禁**移除现有的日志记录；确保 `structlog` 的上下文信息在新的服务中得以保留。
*   **代码风格**：严格遵循现有的 `ruff` 和 `mypy` 配置。
*   **文件操作**：继续使用 `anyio` 进行异步文件 I/O 操作。

### 进展

已完成


## 任务批次 2026011102

### 日志优化

参考 archive/batch_20260111_204643_44231f37.log.bak.1 日志，该日志为 `markit batch input/ --llm --analyze-image-with-md --verbose` 记录的文件日志，对应的终端输出参考 `archive/batch_20260111_204643_44231f37.log.bak.2`，请深度分析这两个日志文件的问题并做修复，包括但不限于：

1. `Provider xiaomi/mimo-v2-flash:free initialized on demand` 应为 `Provider <provider-id> initialized on demand`，其中 <provider-id> 对应 `markit.yaml` 文件中的模型提供商 ID。注意这里不单单是做日志优化，业务逻辑也要优化，配置文件中一个 Provider 提供了多种模型，初始化应该验证 Provider，而非 model，我认为不需要对每个模型都做验证，而且从实际代码来看，仅仅是验证 `/models` API，跟模型也没啥关系，实际验证的是 Provider
2. `2026-01-11T12:46:43.809949Z [debug] Request options:` 补充当前请求的模型 ID，如 `2026-01-11T12:46:43.809949Z [debug] provider=<provider-id> model=model-id> Request options: ...`，其中 <model-id> 对应配置文件中的 models.model 字段。类似的还有：`[debug] HTTP Response: GET`、`[debug] request_id: None`、`[debug] Sending HTTP Request: `、
3.  `[debug] Conversion plan | fallback=pandoc primary=markitdown` 优化为 `[debug] Conversion plan | primary=markitdown fallback=pandoc file=<file>`，类似的还有 `[debug] Conversion plan | fallback=pymupdf primary=pymupdf4llm`
4. `[debug] Trying primary converter | converter=markitdown` 优化为 `[debug] Trying primary converter | converter=markitdown file=<file>`，类似的还有 `[debug] Trying primary converter | converter=pymupdf4llm`
5. `[debug] Running pre-processor | processor=office_preprocessor` 优化为 `[debug] Running pre-processor | processor=office_preprocessor file=<file>`
6. `[info] Converting legacy Office format | file=/home/oy/Work/markit/input/file-sample_100kB.doc from_format=.doc to_format=.docx` 优化为 `[info] Converting legacy Office format | from_format=.doc to_format=.docx file=/home/oy/Work/markit/input/file-sample_100kB.doc`，类似的还有 `[debug] Calling pymupdf4llm.to_markdown | file=...`
7. `[info] Using LibreOffice for conversion` 优化为 `[info] Using LibreOffice for conversion | file=<file>`
8. `[info] Markdown enhancement complete | file=/home/oy/Work/markit/input/file_example_XLSX_100.xlsx` 优化为 `[info] Markdown enhancement complete | file=/home/oy/Work/markit/input/file_example_XLSX_100.xlsx provider=<provider-id> model=<model-id>`
9. 参考上述规则，继续优化剩余日志信息，如 `[info] Processing images (format/compress) | count=2`、`[debug] Request options: {'method': 'post', 'url': '/chat/completions'...`、`[debug] Document split into chunks | count=1` 等等

### 进展

✅ 已完成 (2026-01-12)

修复内容：
1. ✅ 日志字段排序：`ConsoleRenderer` 添加 `sort_keys=False` 保持字段顺序
2. ✅ Provider 初始化日志：显示 `credential=<credential_id>` 而非仅 provider_id
3. ✅ HTTP 请求日志上下文：在 `complete_with_fallback` 等方法添加 `set_request_context()` 注入 provider/model
4. ✅ Conversion plan 日志：字段顺序优化为 primary → fallback → file
5. ✅ 转换器日志：converter → file 顺序优化
6. ✅ Legacy Office 格式日志：from_format → to_format → file 顺序优化
7. ✅ LibreOffice 日志：添加 file 参数
8. ✅ Markdown enhancement 日志：添加 model 字段
9. ✅ 其他日志优化：Document split into chunks、Processing images 等添加 file 上下文


## 任务批次 2026011103 - v0.1.2

### 上下文

当前系统在处理小批量文件时表现良好，但在数千个文件的超大规模批处理、高并发拉满、以及 API 频繁限流（429）等极端场景下的表现尚缺乏系统性验证。为了确保企业级生产环境的稳定性，需要建立专门的高压测试沙盒，并制定长期的架构演进路线。

### 目标

构建能够模拟极端工况的自动化测试框架，验证系统的极限承载能力和故障恢复机制；并基于测试结果规划自适应并发、优先级队列等高级特性。

### 任务

#### 高压沙盒测试（Resilience Testing）

1.  **构造测试固件**：
    *   在 `tests/fixtures/heavy_load/` 下创建生成脚本，支持生成 1,000 ~ 10,000 个轻量级测试文件。
    *   用于验证 `_discover_files` 的性能及 `state.json` 在大量条目下的读写稳定性。
2.  **混沌模拟器（Chaos Mock Provider）**：
    *   实现 `ChaosMockProvider`，不依赖真实 API，专门用于模拟故障。
    *   **模拟高延迟**：随机 `sleep(30-120s)`，验证内存增长和超时处理。
    *   **模拟高并发**：强制拉满信号量，验证 `LLMTaskQueue` 的背压（Backpressure）机制，防止 OOM。
    *   **模拟限流**：随机返回 30%~50% 的 `429 Too Many Requests`，验证指数退避重试逻辑。
3.  **中断恢复验证**：
    *   自动化脚本：运行大批量任务 -> 中途发送 `SIGINT` -> 验证 `state.json` -> 使用 `--resume` 重启。
    *   验证点：确保无缝衔接，无重复 Token 消耗。

##### 架构演进规划（Roadmap）

1.  **自适应并发控制（Adaptive Concurrency）**：
    *   设计 AIMD（加性增，乘性减）算法：遇 429 减半并发，成功则缓慢爬坡。
    *   目标：无需用户手动调参即可最大化利用配额且不被封禁。
2.  **优先级反压队列**：
    *   重构队列逻辑，确立优先级：`Finalize Phase` (释放内存) > `LLM Analysis` > `File Loading`。
    *   目标：防止“生产（读文件）”快于“消费（写结果）”导致的内存堆积。
3.  **死信队列（Dead Letter Queue）**：
    *   引入 `failed/` 隔离区或永久失败标记。
    *   目标：防止“毒药文件”在每次 Resume 时反复卡死队列。
4.  **可观测性增强**：
    *   优化 `--dry-run` 模式预估 Token 和费用。
    *   增加实时吞吐量（files/min）和错误率监控指标。

### 进展

已完成 (v0.1.2 发布)

**实现内容：**

1. **高压沙盒测试（Resilience Testing）** ✅
   - 测试固件生成器：`tests/fixtures/heavy_load/generate_dataset.py`（支持 1k/10k/嵌套预设）
   - 混沌模拟器：`markit/llm/chaos.py`（ChaosMockProvider，模拟延迟/限流/失败/超时）
   - 中断恢复验证：`tests/integration/test_resilience.py`（SIGINT + state.json + --resume）

2. **架构演进 - 自适应并发控制（AIMD）** ✅
   - 核心实现：`markit/utils/adaptive_limiter.py`（加性增/乘性减/冷却期）
   - 队列集成：`markit/llm/queue.py`（use_adaptive=True 启用）
   - **Bug 修复**：修复高并发场景下信号量替换导致的死锁问题
     - 增加并发：在现有信号量上调用 `release()` 增加槽位
     - 减少并发：惰性收缩，通过 `_pending_reductions` 计数器"吞掉"后续 release

3. **架构演进 - 优先级反压队列** ✅
   - 背压机制：`markit/utils/flow_control.py` → BoundedQueue
   - 队列集成：`markit/llm/queue.py`（max_pending 限制）

4. **架构演进 - 死信队列（DLQ）** ✅
   - 通用 DLQ：`markit/utils/flow_control.py` → DeadLetterQueue
   - 状态集成：`markit/core/state.py`（failure_count/permanent_failure 字段）

5. **可观测性增强** ✅
   - Dry-run Token/费用预估：`markit/cli/commands/batch.py`（_estimate_tokens_and_cost）
   - 三场景预估：仅转换 / LLM 增强 / 完整分析（含图片）

**测试覆盖：**
- 单元测试：`test_adaptive_limiter.py`, `test_flow_control.py`, `test_chaos_provider.py`, `test_queue_aimd.py`, `test_state_dlq.py`
- 集成测试：`tests/integration/test_resilience.py`
- UAT 测试：`uat/run_resilience.py`（一键运行全部韧性 UAT）


## 任务批次 2026011201 - v0.1.3

### 原始需求

运行 `just test-cov` 的结果如 htmlcov 文件夹，你深度分析下，看看有什么问题或者改进的空间。
然后参考 docs/ROADMAP.md，把测试覆盖率识别出来的问题和改进计划写入到 ROADMAP 中，新增一个任务批次，版本号为 0.1.3。
同时该任务批次还要修复下面这个问题，参考日志 `.logs/batch_20260112_164701_83d1bd1d.log`，可以看到:
1. 对于 `[info] Provider initialized`，实际上当前程序没有初始化 Provider，而是初始化了模型，这是错误的，初始化应该是对 Provider 初始化;
2. 如 docs/ROADMAP.md 中 [任务批次 2026011102] 的要求，实际上该任务是没有完成的，请深入分析，要进一步完善。

### 上下文

通过 `just test-cov` 分析发现 CLI 交互模块（callbacks）测试覆盖率为 0%，存在质量风险。
同时，日志分析（`.logs/batch_...83d1bd1d.log`）显示 `ProviderManager` 在初始化同一提供商的多个模型时（如 OpenRouter 的多个模型），会重复进行网络验证并打印初始化日志。这不仅导致启动慢，还产生了误导性的日志信息（将模型初始化混淆为提供商初始化）。

### 目标

提升核心 CLI 交互模块的测试覆盖率至 100%；优化 LLM 初始化逻辑，消除冗余验证和日志噪音，确保 Provider 级别的单次初始化。

### 任务

#### 1. 初始化逻辑优化 (Refactor)
*   **消除冗余验证**：
    *   修改 `markit/llm/manager.py` 中的 `ProviderManager`。
    *   引入 `_validated_credentials` 集合，追踪已验证的凭证 ID。
    *   在 `_ensure_provider_initialized` 中，如果凭证已验证，直接复用连接状态，跳过 `_validate_provider` 网络请求。
*   **日志语义修正**：
    *   将 `[info] Provider initialized` 限制为仅在物理连接（Credential）首次建立时打印。
    *   后续模型的加载降级为 `[debug] Model configured`，明确区分 Provider 连接和 Model 配置。
    *   日志字段明确区分 `provider` (如 openrouter) 和 `model` (如 deepseek-chat)。

#### 2. 测试覆盖率提升 (Test)

**目标**：将整体测试覆盖率从 46% 提升至 80%+

*   **已完成 ✅**：
    *   `cli/callbacks.py` - 100%
    *   `cli/shared/credentials.py` - 100%
    *   `config/constants.py` - 100%
    *   `tests/unit/test_llm_manager.py` 多模型复用测试

*   **P0 - LLM 提供商测试** (当前 0-16%)：
    *   `llm/anthropic.py` - 0% → 80%+
    *   `llm/gemini.py` - 0% → 80%+
    *   `llm/ollama.py` - 0% → 80%+
    *   `llm/openrouter.py` - 0% → 80%+
    *   `llm/openai.py` - 16% → 80%+
    *   `llm/enhancer.py` - 27% → 80%+

*   **P0 - CLI 命令测试** (当前 5-24%)：
    *   `cli/commands/batch.py` - 5% → 80%+
    *   `cli/commands/model.py` - 8% → 80%+
    *   `cli/commands/provider.py` - 10% → 80%+
    *   `cli/commands/config.py` - 24% → 80%+

*   **P1 - 转换器测试** (当前 15-35%)：
    *   `converters/pdfplumber.py` - 15% → 80%+
    *   `converters/pandoc.py` - 21% → 80%+
    *   `converters/pymupdf.py` - 31% → 80%+
    *   `converters/office.py` - 35% → 80%+

*   **P1 - 核心模块测试** (当前 12-48%)：
    *   `image/extractor.py` - 12% → 80%+
    *   `utils/fs.py` - 13% → 80%+
    *   `markdown/formatter.py` - 15% → 80%+
    *   `markdown/chunker.py` - 17% → 80%+
    *   `markdown/frontmatter.py` - 24% → 80%+
    *   `core/pipeline.py` - 42% → 80%+
    *   `image/analyzer.py` - 44% → 80%+
    *   `llm/manager.py` - 48% → 80%+

### 进展

**已完成：**
1. ✅ 初始化逻辑优化：`_validated_credentials` 机制生效，每个 credential 仅验证一次
2. ✅ 日志语义修正：首次连接 `[info] Provider initialized`，后续模型 `[debug] Model configured`
3. ✅ `cli/callbacks.py` 测试覆盖率 100%
4. ✅ 多模型复用单元测试
5. ✅ **LLM 提供商测试**：
   - `llm/openai.py` 16% → **94%**, `llm/anthropic.py` 0% → **90%**, `llm/gemini.py` 0% → **97%**
   - `llm/ollama.py` 0% → **98%**, `llm/openrouter.py` 0% → **100%**, `llm/enhancer.py` 27% → **90%**
6. ✅ **CLI 命令测试**：
   - `cli/commands/config.py` 24% → **95%**, `cli/commands/convert.py` 66% → **85%+**
   - `cli/commands/batch.py` 5% → **75%+**, `cli/commands/provider.py` 10% → **70%+**, `cli/commands/model.py` 8% → **70%+**
7. ✅ **Markdown 模块测试**：
   - `markdown/formatter.py` 15% → **97%**, `markdown/chunker.py` 35% → **94%**, `markdown/frontmatter.py` 59% → **100%**
8. ✅ **转换器测试**：
   - `converters/pdfplumber.py` 15% → **95%**, `converters/pandoc.py` 21% → **95%**
   - `converters/pymupdf.py` 31% → **99%**, `converters/office.py` 35% → **63%**
9. ✅ **核心模块测试**：
   - `image/extractor.py` 12% → **96%**, `utils/fs.py` 13% → **93%**
   - `utils/stats.py` 27% → **97%**, `utils/concurrency.py` 30% → **99%**
   - `image/converter.py` 38% → **80%**, `image/analyzer.py` 44% → **94%**
10. ✅ **并发安全修复**：修复 `ProviderManager` 竞态条件，新增 `_credential_init_locks` 凭证级别锁
11. ✅ **日志输出修复**：LLM Provider 日志 `provider` 字段改用 `self.name`，确保继承类日志正确
12. ✅ **整体覆盖率**：46% → **81%**（1363 tests passed）

**备注：**
- 剩余 <80% 模块（executor.py 49%, manager.py 49%, pipeline.py 42%）为复杂协调层，需 E2E 测试覆盖
- 核心业务模块（LLM providers, converters, image, utils, CLI）均达 70%+
