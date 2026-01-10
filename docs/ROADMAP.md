# ROADMAP

## 任务批次 2026011003

### 背景

基于 `markit batch input --llm --analyze-image-with-md --verbose` 命令的日志分析，发现以下问题：

1. **模型初始化效率低**：初始化 5 个模型耗时 8 秒，但部分模型可能未被使用
2. **能力路由缺失**：文本任务在所有模型间轮询，浪费 vision 模型配额
3. **验证失败静默处理**：`_validate_provider` 验证异常时返回 `True`，导致无效模型被标记为可用
4. **日志信息不完整**：缺少耗时统计、token 用量、成本估算等关键信息
5. **缺少执行模式区分**：无法在速度和可靠性间灵活选择

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
- 在 `markit.example.yaml` 注释中说明配置方式

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
