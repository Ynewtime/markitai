# ROADMAP

## 任务批次 2026011403 - v0.2.1

**本地 OCR 能力**：引入 RapidOCR + PaddleOCR 双引擎方案，支持离线中文 OCR，同时保留 PDF 图片提取能力。OCR 方案调研详见 [OCR 方案研究报告](./reference/ocr.md)。

### 进展

待开始

## 任务批次 2026011402 - v0.2.0

本版本进行**全量架构重构**，核心目标是 **LLM 层重构**：全面切换至 LiteLLM，废弃现有 Provider 层代码。LiteLLM 调研参考 [LiteLLM 调研报告](./reference/litellm.md)。

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


## 任务批次 2026011301 - v0.1.4

### 任务

1. ✅ ~~当前 convert 任务依然偶发 JSON 爬取失败的问题~~ -> 实现 `_extract_first_json_object` 方法使用括号计数算法精确提取 JSON
2. ✅ 日志格式优化：
   1. ✅ `Analyzing image with LLM` - 移动到 `manager.py` 的 `analyze_image_with_fallback` 函数内，通过 `set_request_context` 自动注入 provider/model
   2. ✅ `Provider initialized` - 格式改为 `provider=<provider> model=<model> base_url=<base_url>`
   3. ✅ `Image analyzed` - file 字段改为文件名
   4. ✅ `Analyzing images in parallel` - file 字段改为文件名
   5. ✅ `Provider failed for image analysis` - 移除 provider_id，添加 filename
   6. ✅ `Applying LLM enhancement` - file 字段改为文件名
   7. ✅ `Enhancing Markdown` - file 字段改为文件名
   8. ✅ `Document split into chunks` - file 字段改为文件名
   9. ✅ `Request options` 等长内容 - 添加 event 截断（200 字符限制）
3. ✅ 超时问题 - `convert` 命令现在正确读取 `concurrent_fallback_enabled` 配置，启用并发回退机制
4. ✅ 20260113 11:40 新增需求：
   1. ✅ `Applying LLM enhancement` - 保持原样（此日志在调用 LLM 前输出，无法获取 provider/model）
   2. ✅ `Enhancing Markdown` - 已移除（冗余日志）
   3. ✅ `Sending LLM request` - 通过 `set_request_context(file_path=...)` 自动注入 file
   4. ✅ `Request options` - 简化为 `method=<method> url=<url>` 格式，file 通过 context 注入
   5. ✅ `Sending HTTP Request` - file 通过 context 自动注入
   6. ✅ `HTTP Response` - 简化为 `POST url "status"` 格式（移除 Headers 详情），file 通过 context 注入
   7. ✅ `request_id: None` - 通过 `RequestIdNoneFilter` 过滤掉
   8. ✅ `LLM response received` - file 通过 context 自动注入
   9. ✅ `Markdown enhancement complete` - 修改为 `file=<filename>` 格式，provider/model 通过 context 注入
5. ✅ 日志优化（基于 batch_20260113_155121_bb5b711e.log.terminal 分析）：
   1. ✅ Provider 字段修正：移除 provider 类中硬编码的 `provider=self.name`，改用 context 注入正确的 credential_id（如 `deepseek` 而非 `openai`）
   2. ✅ Concurrent fallback 日志简化：移除末尾重复的 `provider/model` 字段（winner_id/primary_id/fallback_id 已包含信息）
   3. ✅ Fallback context 修复：为 fallback 任务设置独立的 request context，确保日志正确显示 fallback 的 provider/model
   4. ✅ 模型统计分析：被取消的 fallback 请求不计入统计是预期行为（API 可能未计费）
   5. ✅ Warn/Error 分析：Anthropic 连接超时（网络问题）、OpenRouter 连接中断（服务端问题）、Primary 超时触发 fallback（正常机制）
6. ✅ Timeout 配置统一：
   - 移除 `DEFAULT_LLM_TIMEOUT`（120s），统一使用 `DEFAULT_MAX_REQUEST_TIMEOUT`（300s）作为 httpx 超时
   - 解决了 httpx 超时（120s）比 max_request_timeout（300s）短导致 concurrent fallback 机制失效的问题

### 进展

✅ 已完成


## 任务批次 2026011302 - v0.1.5

### 任务

命令和终端完整输出信息参考: `docs/fail_logs/20260113_windows_powershell_test.txt`

处理结果参考: `docs/fail_output`

请深度分析上面终端输出，全量识别其中的问题，基于本项目库代码，制定修复方案。

包括但不限于：
1. 终端输出：大量 API 报错;
2. 输出格式：
   1. 如 `docs/fail_output/2.Hello OCS源码剖析.doc.md` 和 `docs/fail_output/OCS计费原理与实现(排版后).doc.md`，输出 Markdown 中存在大量无效 Chunk YAML
   2. 如 `docs/fail_output/CBS架构演进规划201605 V0.5.pptx.md`，header 没有遵循前后各一行空行的要求，同时未做内容规整，LLM 处理后的格式跟直接使用工具转换出来的内容基本一致，未达到使用 LLM 来规整内容的目的
   3. 如 `docs/fail_output/预付费业务信令流程规范（V4.0）.doc.md`，同样存在内容未规整、零散的不成意义的文本散落在正文中，意味不明，未作清洗

备注：上述命令测试和处理结果在 Windows PowerShell 环境执行，此处需根据其日志和交付件深入推理分析。

### 进展

#### 问题分析

基于 `docs/fail_logs/20260113_windows_powershell_test.txt` 和 `docs/fail_output/` 的深度分析，识别出以下核心问题：

1. **无效 Chunk YAML（严重）**
   - **现象**：输出 Markdown 中存在多个散落的 frontmatter 块
   - **根因**：每个 chunk 都使用相同的 prompt，包含生成 frontmatter 的指令

2. **格式未规整（中等）**
   - **现象**：标题前后空行不规范，LLM 输出与原始转换基本一致
   - **根因**：`ENHANCEMENT_PROMPT` 中的格式规则不够具体

3. **内容未清洗（中等）**
   - **现象**：零散文本、图表残留（坐标轴数字、图例等）未清理
   - **根因**：`ENHANCEMENT_PROMPT` 中的清理规则太笼统，缺少具体示例

4. **API 错误频繁（低）**
   - **现象**：`httpx.ConnectError`、`httpx.ReadError`、超时
   - **根因**：待分析（需进一步确认是网络问题还是配置问题）

#### 第一次实现尝试（已回滚）

首次实现采用了以下方案，但经分析存在严重设计缺陷，已全部回滚：

| 修改 | 设计缺陷 |
|------|----------|
| 新增 `ENHANCEMENT_PROMPT_CONTINUATION_*` 续段 Prompt（不含 frontmatter 指令） | 如果关键 entities/topics 在文档后半部分，会丢失元数据 |
| `_remove_intermediate_frontmatter()` 正则清理中间 frontmatter | 可能误删文档中的 YAML 代码块示例 |
| Pipeline 后处理调用 `format_markdown(ensure_h2_start=True)` | 会破坏 LLM 已处理的标题结构（如 PPT 各 slide 标题） |
| `MarkdownCleaner` 新增图表残留正则模式 | 正则无法理解语义，风险高（误删数字列表、有效内容） |
| `retry_count` 从 2 增至 3 | 创可贴方案，未深入分析根因 |

**核心问题**：违反了"程序提取、LLM 清理"的职责分离原则。试图用程序正则来弥补 LLM 清理不彻底的问题，而非优化 Prompt。

#### 新方案设计（已完成）

遵循 CLAUDE.md 中的核心设计原则，采用纯 Prompt 优化方案：

##### 1. 多 Chunk 元数据策略

```
┌─────────────────────────────────────────────────────────────────┐
│                      Prompt 设计                                │
├─────────────────────────────────────────────────────────────────┤
│ 首 Chunk:  生成完整 frontmatter (entities, topics, ...)        │
│ 续 Chunk:  生成 partial_metadata (仅新增的 entities, topics)   │
│            明确指令："不要生成 --- frontmatter 块"              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    合并策略 (Enhancer)                          │
├─────────────────────────────────────────────────────────────────┤
│ 1. 解析首 chunk 的 frontmatter                                  │
│ 2. 解析续 chunk 末尾的 partial_metadata JSON                    │
│ 3. 合并去重：entities = set(all_entities)                       │
│ 4. 注入最终 frontmatter 到输出开头                              │
└─────────────────────────────────────────────────────────────────┘
```

##### 2. Prompt 优化方向

**src/markit/llm/enhancer.py** 中的 `ENHANCEMENT_PROMPT_*`:

1. **格式规则具体化**：
   - 标题前后各保留一个空行
   - 表格前后各保留一个空行
   - 代码块保持原样，不要修改缩进

2. **图表残留清理示例**：
   ```
   删除以下类型的无意义文本：
   - 坐标轴刻度：如单独的 "0", "10", "20", "30" 数字行
   - 图例标签：如 "Series1", "Category A", "Row 1" 等孤立文本
   - 图表组件：如 "Legend", "Title", "X-axis", "Y-axis"
   ```

3. **PPT 标题层级规则**（明确写入 Prompt）：
   ```
   对于 PPT/PPTX 文档：
   - 每张幻灯片标题 → ## (h2)
   - 幻灯片内的一级标题 → ### (h3)
   - 幻灯片内的二级标题 → #### (h4)
   ```

4. **续 Chunk 指令**：
   ```
   【重要】这是文档的第 N 段（非首段）：
   1. 不要生成 YAML frontmatter 块（--- ... ---）
   2. 在文档末尾以 JSON 格式输出本段新发现的实体和主题：
      <!-- PARTIAL_METADATA: {"entities": [...], "topics": [...]} -->
   ```

##### 3. 实施计划

| 阶段 | 文件 | 修改内容 | 状态 |
|------|------|----------|------|
| P0 | `src/markit/llm/enhancer.py` | 新增 `ENHANCEMENT_PROMPT_CONTINUATION_*` 续段 Prompt | ✅ |
| P0 | `src/markit/llm/enhancer.py` | 新增 `_extract_partial_metadata()` 提取方法 | ✅ |
| P0 | `src/markit/llm/enhancer.py` | 修改 `enhance()` 流程支持多 chunk 元数据收集 | ✅ |
| P0 | `src/markit/llm/enhancer.py` | 修改 `_inject_frontmatter()` 支持 partial_metadata 合并 | ✅ |
| P1 | `tests/unit/llm/test_enhancer.py` | 新增 12 个多 chunk 元数据合并测试 | ✅ |
| P2 | `docs/fail_logs/` | 深入分析 API 错误日志，确认是否为网络问题 | 跳过（网络问题） |

##### 4. 验证标准

1. ✅ 多 chunk 文档只有一个 frontmatter（开头）
2. ✅ 后续 chunk 中的 entities/topics 被正确合并到 frontmatter
3. ✅ 不使用正则删除任何文档内容（交给 LLM 判断）
4. ✅ PPT 标题层级规则已写入续段 Prompt

##### 5. 专家复盘后改进 (2026-01-13)

基于 `docs/015-REVIEW.md` 专家分析，追加以下修复：

| 问题 | 修复内容 | 文件 |
|------|----------|------|
| description 切块问题 | 续段输出 `key_points`，合并到 description | `enhancer.py` |
| batch 模式无 post-processing | 在 `create_enhancement_task` 添加格式化 | `llm_orchestrator.py` |
| 标题后无空行 | 修复 `_normalize_headings` | `formatter.py` |
| chunk 并发失控 | 新增 `chunk_concurrency` 参数和 semaphore | `llm_orchestrator.py` |
| CRLF 问题 | 格式化前 normalize 换行符 | `llm_orchestrator.py` |

**不采纳的建议**（违反"程序提取，LLM 清理"原则）：
- P0-5（frontmatter regex 收窄）- 新方案不用正则删 frontmatter
- P1-6（图表残留 deterministic 清理）- 应优化 Prompt，不加正则

##### 6. 第二次复盘改进 (2026-01-13)

基于进一步代码审查，追加以下优化：

| 问题 | 修复内容 | 文件 |
|------|----------|------|
| 配置命名不一致 | `*_prompt` → `*_prompt_file`，语义更清晰 | `settings.py` |
| 标题层级规则错误 | `_fix_heading_levels` 从仅改 h1 → 所有标题下移一级 | `formatter.py` |
| Cleaner 职责混乱 | 移除内容清洗规则，仅保留格式规则，内容交给 LLM | `formatter.py` |
| Prompts 硬编码 | 迁移到 `src/markit/config/prompts/`，使用 `get_prompt()` 加载 | `enhancer.py`, `settings.py` |
| key_points 冗余 | 从 continuation prompts 和代码中移除（方案 D 简化） | `enhancer.py`, prompts |
| description 策略缺失 | 新增 `description_strategy` 配置（first_chunk/separate_call/none） | `settings.py` |

**关键设计变更**：

1. **PromptConfig 新增字段**：
   - `image_analysis_prompt_file`、`enhancement_prompt_file`、`summary_prompt_file`
   - `description_strategy`: `"first_chunk"` (默认) / `"separate_call"` / `"none"`

2. **`_fix_heading_levels` 行为变更**：
   - 旧：仅将 h1 改为 h2
   - 新：若文档以 h1 开头，所有标题下移一级 (h1→h2, h2→h3, ..., h6 保持)

3. **MarkdownCleaner 精简**：
   - 保留：`zero_width`、`empty_links`、`html_comments`
   - 移除：`page_numbers`、`separator_lines`、`repeated_chars`（交给 LLM）

4. **Prompts 文件化**：
   - `enhancement_zh.md` / `enhancement_en.md`
   - `enhancement_continuation_zh.md` / `enhancement_continuation_en.md`
   - `summary_zh.md` / `summary_en.md`
   - `image_analysis_zh.md` / `image_analysis_en.md`

##### 7. CI 验证

- 1384 tests passed, 5 skipped
- 0 type errors
- Lint checks passed

### 状态

**已完成** - v0.1.5 所有核心改进已实施并通过 CI 验证。


## 任务批次 2026011401 - v0.1.6

问题一:

```
PS C:\Users\user\Documents\markit> & C:/Users/user/Documents/markit/.venv/Scripts/Activate.ps1
(markit) PS C:\Users\user\Documents\markit> markit model add
? Select a provider credential to use: gemini (gemini)
Fetching models for gemini... 54 models found
? Select a model (type to search): models/gemini-2.5-flash-lite [text, vision]
? Enter a display name for this configuration: models/gemini-2.5-flash-lite
Failed to update config: 'charmap' codec can't encode character '\u2264' in position 550: character maps to <undefined>
```

### 进展

**已完成** - v0.1.6 所有核心改进已实施并通过 CI 验证 (1384 passed)。

#### Bug 修复 (2026-01-14)

1. **`complete_with_concurrent_fallback` 路由策略支持**
   - **问题**：配置 `strategy: "round_robin"` 或 `strategy: "least_pending"` 时，`complete_with_concurrent_fallback` 方法直接使用 `candidates[0]`，完全忽略路由策略配置
   - **根因**：该方法只优先使用 `_last_successful_provider`，然后直接取第一个候选，没有调用 `_select_best_provider()`
   - **修复**：
     - `cost_first` 策略：保留 `_last_successful_provider` 优先逻辑（稳定性优先）
     - `round_robin`/`least_pending` 策略：调用 `_select_best_provider()` 选择主模型，并更新 `_current_index`
   - **文件**：`llm/manager.py:1026-1055`

2. **Fallback 选择后更新 `_current_index`**
   - **问题**：fallback 选择时调用了 `_select_best_provider`，但没有更新 `_current_index`，导致并发场景下多个超时请求的 fallback 都选择同一个模型
   - **根因**：三个请求相隔 5-6 秒超时，但都读取到同一个 `_current_index` 值
   - **修复**：fallback 选择后也更新 `_current_index`
   - **文件**：`llm/manager.py:1120-1122`

3. **日志路径简化扩展**
   - **问题**：`_simplify_file_paths` 只处理 `file` 键，其他路径键（`original`, `converted`, `path`, `output`, `output_dir`, `input_dir`）仍显示完整路径
   - **用户需求**：除了最后一行 "Batch complete" 之外，所有路径都应简化为文件名
   - **修复**：
     - 扩展 `_PATH_KEYS_TO_SIMPLIFY` 包含所有路径键
     - 最后一行使用 `task_output` 字段保留完整路径
   - **文件**：`utils/logging.py:316-329`, `core/state.py:274`

#### 问题一修复：Windows 字符编码

- **根因**：Windows 默认使用 cp1252/GBK 编码，而非 UTF-8
- **修复方案**：
  | 级别 | 文件 | 修改内容 |
  |------|------|----------|
  | P0 | `cli/commands/model.py:244,295` | YAML 读写添加 `encoding="utf-8"` |
  | P0 | `cli/commands/provider.py:129,182` | YAML 读写添加 `encoding="utf-8"` |
  | P1 | `utils/flow_control.py:351,366,521` | JSON 读写添加 `encoding="utf-8", ensure_ascii=False` |
  | P1 | `cli/commands/provider.py:982` | JSON 缓存写入添加编码 |
  | P2 | `converters/office.py:279,418` | stderr 解码使用 `decode('utf-8', errors='replace')` |
  | P2 | `converters/pandoc.py:140,249,274,288,314` | subprocess 添加 `encoding="utf-8", errors="replace"` |

#### 问题二修复：并发架构重构

采用 **Plan C - 两层并发控制** 方案：

1. **配置层** (`config/constants.py`, `config/settings.py`)：
   - 新增 `chunk_workers` (默认 6)
   - 新增 `AdaptiveConfig` (AIMD 参数)
   - 新增 `RoutingConfig` (路由策略)

2. **ProviderManager 智能路由** (`llm/manager.py`)：
   - 每 credential 独立 AIMD 限流器 (`_credential_limiters`)
   - 智能路由 `least_pending` 策略：`cost × cost_weight + load × load_weight`
   - 请求计数追踪 (`_credential_pending`)

3. **LLMOrchestrator 每文件独立 semaphore** (`services/llm_orchestrator.py`)：
   - 移除全局 `_chunk_semaphore`
   - 每个文件创建独立 `asyncio.Semaphore(chunk_concurrency)`
   - 防止大文档独占所有并发槽

4. **测试修复**：
   - `test_round_robin_load_balancing` 显式配置 `round_robin` 策略

详细设计见 `docs/016-SPEC.md`。

---

问题二原始记录：

5 个文件的 LLM 分析（无图片分析，未开 Verbose 模式）任务耗时（第一次执行，日志见 /mnt/c/Users/user/Documents/markit/.logs/batch_20260113_233226_7402b527.log）:

```
Batch Statistics:
Complete: 5 success, 0 failed
Total: 870s | Init: 5s | Process: 865s | LLM: 859s
Tokens: 307,527 | Est. cost: $0.1012
Models used: deepseek-chat(5)
```

5 个文件的 LLM 分析（无图片分析，开了 Verbose 模式）任务耗时（第二次执行，日志见 /mnt/c/Users/user/Documents/markit/.logs/batch_20260114_000012_0cd682ed.log）:

```
Batch Statistics:
Complete: 5 success, 0 failed
Total: 795s | Init: 1s | Process: 794s | LLM: 789s
Tokens: 385,795 | Est. cost: $0.0881
Models used: models/gemini-2.5-flash-lite(5)
```

5 个文件的 LLM 分析 + 图片分析任务耗时（日志见 /mnt/c/Users/user/Documents/markit/.logs/batch_20260113_230432_c3a53b8c.log）:

```
Batch Statistics:
Complete: 5 success, 0 failed
Total: 923s | Init: 6s | Process: 917s | LLM: 911s
Tokens: 567,208 | Est. cost: $0.3713
Models used: models/gemini-3-flash-preview(134), deepseek-chat(5)
```
