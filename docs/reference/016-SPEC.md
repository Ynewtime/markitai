# v0.1.6 实施规格书

本版本聚焦 **并发优化与资源池充分利用**，解决大文档处理性能瓶颈和 Windows 兼容性问题。

---

## 1. 问题分析

### 1.1 问题一：Windows 字符编码错误

**现象**：
```
markit model add
Failed to update config: 'charmap' codec can't encode character '\u2264' in position 550
```

**根因**：
- `ruamel.yaml` 在 `model.py:295` 和 `provider.py:182` 中直接接收 `Path` 对象
- Windows 默认编码是 `cp1252`（charmap），不支持 `≤` 等 Unicode 字符
- 模型元数据可能包含 Unicode 字符（如 "context_length ≤ 128k"）

**涉及代码**：
```python
# model.py:295, provider.py:182
yaml.dump(config_data, config_path)  # 使用系统默认编码
```

### 1.2 问题二：大文档处理性能瓶颈

**现象**：5 个文件处理耗时 795-923 秒，其中单文件 `2.Hello OCS源码剖析.doc`（40+ chunks）耗时约 650 秒。

**根因分析**：

| 问题 | 描述 | 影响 |
|------|------|------|
| Chunk Semaphore 跨文件共享 | `_chunk_semaphore` 容量 4，所有文件竞争 | 大文件的 chunks 被严重限制 |
| `chunk_concurrency` 硬编码 | 默认值 4，无配置项 | 用户无法调整 |
| 串行 Fallback | 只有当前模型失败才尝试下一个 | 高并发时，便宜模型成为瓶颈 |
| LLM 资源池未充分利用 | 优先便宜模型，贵模型闲置 | 配置的多模型没有并行利用 |

**当前架构问题**：

```
┌─────────────────────────────────────────────────────────────┐
│                    当前架构问题                              │
├─────────────────────────────────────────────────────────────┤
│  LLMTaskQueue (max_concurrent=10)                           │
│       │                                                     │
│       ▼                                                     │
│  [Task: File A enhance] [Task: File B enhance] ...          │
│       │                      │                              │
│       ▼                      ▼                              │
│  enhancer.enhance()     enhancer.enhance()                  │
│       │                      │                              │
│       ├──► Chunk 1 ─┐       ├──► Chunk 1 ─┐                │
│       ├──► Chunk 2 ─┼─► _chunk_semaphore(4) ◄─┼─ Chunk 1   │
│       └──► Chunk 3 ─┘  (跨文件共享!!)        └─ Chunk 2    │
│              │                                              │
│              ▼                                              │
│  provider_manager.complete_with_fallback()                  │
│  (绕过了 LLMTaskQueue，无全局并发限制)                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 主流 LLM Provider Rate Limits

| Provider | Tier | RPM | TPM | 备注 |
|----------|------|-----|-----|------|
| **Anthropic** | Tier 1 | 50 | ~48k | 最严格 |
| **Gemini** | Free | 5-15 | - | 免费层极低 |
| **Gemini** | Tier 1 (付费) | 300 | 1M | 付费后大幅提升 |
| **OpenAI** | Tier 1 | 500-3500 | 10k-200k | 因模型而异 |
| **DeepSeek** | - | 无限制 | 无限制 | 通过响应延迟调节 |

**结论**：
- 不同 provider 的 rate limit 差异巨大
- 需要 **每 credential 独立的限制器**，而非全局统一
- AIMD 算法可自动适应不同 provider 的限制

---

## 3. 解决方案

### 3.1 问题一修复：跨平台编码兼容性

#### 3.1.1 问题背景

Windows 和 Linux 的默认编码不同：
- **Linux**: 默认 UTF-8
- **Windows**: 默认 `locale.getpreferredencoding()`，中文系统是 GBK/CP936，西文系统是 CP1252

Python 的以下操作在 Windows 上会使用系统默认编码：
- `Path.write_text()` / `Path.read_text()` 不指定 encoding
- `open()` 不指定 encoding
- `subprocess.run(text=True)` 不指定 encoding
- `bytes.decode()` 不指定 encoding

#### 3.1.2 全面扫描结果

**P0 - 必须修复（已确认会导致错误）**

| 文件 | 行号 | 问题代码 | 风险说明 |
|------|------|----------|----------|
| `cli/commands/model.py` | 244 | `yaml.load(config_path)` | ruamel.yaml 直接读取 Path，含中文注释或 Unicode 会失败 |
| `cli/commands/model.py` | 295 | `yaml.dump(config_data, config_path)` | ruamel.yaml 直接写入 Path，模型元数据含 Unicode |
| `cli/commands/provider.py` | 129 | `yaml.load(config_path)` | 同上（读取问题） |
| `cli/commands/provider.py` | 182 | `yaml.dump(config_data, config_path)` | 同上（写入问题） |

**P1 - 建议修复（潜在风险）**

| 文件 | 行号 | 问题代码 | 风险说明 |
|------|------|----------|----------|
| `utils/flow_control.py` | 351 | `.read_text()` | DLQ 读取，无 encoding，中文文件名/错误信息 |
| `utils/flow_control.py` | 366 | `.write_text(json.dumps(...))` | DLQ 保存，无 encoding |
| `utils/flow_control.py` | 519 | `.write_text(json.dumps(...))` | DLQ 报告导出，无 encoding |
| `cli/commands/provider.py` | 979 | `.write_text(json.dumps(...))` | 模型缓存，无 encoding |

**P2 - 建议修复（subprocess 编码）**

| 文件 | 行号 | 问题代码 | 风险说明 |
|------|------|----------|----------|
| `converters/office.py` | 279 | `e.stderr.decode()` | LibreOffice 错误信息，无 encoding |
| `converters/office.py` | 418 | `e.stderr.decode()` | 同上 |
| `converters/pandoc.py` | 140-145 | `subprocess.run(..., text=True)` | Pandoc 输出，使用系统编码 |
| `converters/pandoc.py` | 247-251 | `subprocess.run(..., text=True)` | Pandoc 版本检查 |
| `converters/pandoc.py` | 270-276 | `subprocess.run(..., text=True)` | CSV 转换 |
| `converters/pandoc.py` | 284-290 | `subprocess.run(..., text=True)` | HTML 转换 |
| `converters/pandoc.py` | 306-310 | `subprocess.run(..., text=True)` | 版本检查 |

**已正确处理（无需修改）**

| 文件 | 行号 | 备注 |
|------|------|------|
| `config/settings.py` | 239, 245, 258 | ✅ `read_text(encoding="utf-8")` |
| `cli/commands/config.py` | 143 | ✅ `write_text(..., encoding="utf-8")` |
| `services/output_manager.py` | 87, 122, 277 | ✅ `anyio.open_file(..., encoding="utf-8")` |
| `image/analyzer.py` | 278 | ✅ `anyio.open_file(..., encoding="utf-8")` |
| `core/state.py` | 204, 323, 478 | ✅ `open(..., encoding="utf-8")` |
| `utils/fs.py` | 282 | ✅ `atomic_write` 默认 `encoding="utf-8"` |
| `converters/pandoc.py` | 149 | ✅ `read_text(encoding="utf-8")` |
| `utils/logging.py` | 188-213 | ✅ `SafeStreamHandler` 处理终端编码 |
| `utils/logging.py` | 466 | ✅ 文件日志 `encoding="utf-8"` |

#### 3.1.3 修复方案

**P0 - YAML 读写**

```python
# cli/commands/model.py:244, cli/commands/provider.py:129
# 修改前（读取）
config_data = yaml.load(config_path)

# 修改后（读取）
with config_path.open('r', encoding='utf-8') as f:
    config_data = yaml.load(f)

# cli/commands/model.py:295, cli/commands/provider.py:182
# 修改前（写入）
yaml.dump(config_data, config_path)

# 修改后（写入）
with config_path.open('w', encoding='utf-8') as f:
    yaml.dump(config_data, f)
```

**P1 - JSON 读写**

```python
# utils/flow_control.py, cli/commands/provider.py
# 修改前
data = path.read_text()
path.write_text(json.dumps(data, indent=2))

# 修改后
data = path.read_text(encoding='utf-8')
path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
```

**P2 - subprocess 输出**

```python
# converters/office.py:279, 418
# 修改前
f"LibreOffice error: {e.stderr.decode() if e.stderr else 'Unknown error'}"

# 修改后
f"LibreOffice error: {e.stderr.decode('utf-8', errors='replace') if e.stderr else 'Unknown error'}"

# converters/pandoc.py - 所有 subprocess.run 调用
# 修改前
result = subprocess.run(cmd, capture_output=True, text=True, check=True)

# 修改后
result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', check=True)
```

#### 3.1.4 改动清单

| 文件 | 改动内容 | 优先级 |
|------|----------|--------|
| `cli/commands/model.py` | YAML 读写指定 UTF-8（行 244, 295） | P0 |
| `cli/commands/provider.py` | YAML 读写指定 UTF-8（行 129, 182）+ JSON 写入（行 979） | P0/P1 |
| `utils/flow_control.py` | JSON 读写指定 UTF-8（行 351, 366, 519） | P1 |
| `converters/office.py` | stderr.decode 指定 UTF-8（行 279, 418） | P2 |
| `converters/pandoc.py` | subprocess 指定 encoding（行 140, 247, 270, 284, 306） | P2 |

### 3.2 问题二修复：并发架构重构

#### 3.2.1 目标架构

```
┌──────────────────────────────────────────────────────────────┐
│                    目标架构                                   │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │   每 Credential 的 AIMD Limiter                        │ │
│  │   gemini:   current=15/max=30, pending=10              │ │
│  │   openai:   current=20/max=30, pending=5               │ │
│  │   deepseek: current=25/max=50, pending=8               │ │
│  └────────────────────────────────────────────────────────┘ │
│                          │                                   │
│                          ▼                                   │
│  ┌────────────────────────────────────────────────────────┐ │
│  │   智能路由（Least-Pending 策略）                        │ │
│  │   综合成本权重 + 负载权重选择最优 credential            │ │
│  └────────────────────────────────────────────────────────┘ │
│                          │                                   │
│         ┌────────────────┼────────────────┐                 │
│         ▼                ▼                ▼                 │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │
│  │   File A     │ │   File B     │ │   File C     │        │
│  │  sem(6)      │ │  sem(6)      │ │  sem(6)      │        │
│  │  (独立)      │ │  (独立)      │ │  (独立)      │        │
│  └──────────────┘ └──────────────┘ └──────────────┘        │
│                                                              │
│  效果：                                                       │
│  - 每 credential 独立 AIMD，自动适应 rate limit             │
│  - 每文件独立 chunk semaphore，保证公平性                    │
│  - 智能路由充分利用所有配置的模型                            │
└──────────────────────────────────────────────────────────────┘
```

#### 3.2.2 两层并发控制

| 层级 | 控制器 | 作用域 | 配置项 |
|------|--------|--------|--------|
| **全局** | 每 Credential 的 AIMD | 限制单个 API key 的总并发 | `llm.adaptive.*` |
| **文件级** | 每文件独立 Semaphore | 限制单文件内 chunks 的并发 | `concurrency.chunk_workers` |

#### 3.2.3 智能路由策略

**当前策略（cost_first）**：
- 按成本排序，优先便宜模型
- 串行 fallback，只有失败才尝试下一个
- 高并发时便宜模型成为瓶颈

**新策略（least_pending）**：
- 综合成本权重 + 负载权重选择最优
- 便宜模型繁忙时自动分流到其他模型
- 充分利用所有配置的模型

**评分公式**：
```
score = cost_weight × cost_score + load_weight × load_score

其中：
- cost_score = 1 - (候选列表中的位置 / 候选总数)    # 越前面越便宜
- load_score = 1 - min(pending / capacity, 1.0)    # pending 越少越好
```

**配置示例**：
```yaml
llm:
  routing:
    strategy: "least_pending"  # 或 "round_robin", "cost_first"
    cost_weight: 0.6
    load_weight: 0.4
```

#### 3.2.4 AIMD 配置

复用现有 `AdaptiveRateLimiter`（`utils/adaptive_limiter.py`），每 credential 独立实例。

**配置示例**：
```yaml
llm:
  adaptive:
    enabled: true
    initial_concurrency: 15    # 付费用户可更激进
    max_concurrency: 50
    min_concurrency: 3
    success_threshold: 15      # 连续成功多少次后提升
    multiplicative_decrease: 0.5  # 遇 429 减半
    cooldown_seconds: 5.0
```

**AIMD 算法流程**：
```
                    ┌─────────────────┐
                    │  初始并发: 15    │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
    ┌─────────────────────┐      ┌─────────────────────┐
    │  连续 15 次成功      │      │  遇到 429 错误      │
    │  → 并发 +1          │      │  → 并发 ×0.5       │
    │  (加性增)           │      │  (乘性减)          │
    └─────────────────────┘      └─────────────────────┘
              │                             │
              │                             │
              │    ┌─────────────────┐     │
              └────►  冷却期 5 秒    ◄─────┘
                   │  防止震荡       │
                   └─────────────────┘
```

---

## 4. 配置变更

### 4.1 新增配置项

**`ConcurrencyConfig`**（`settings.py`）：
```python
class ConcurrencyConfig(BaseModel):
    file_workers: int = Field(default=4, ge=1)
    image_workers: int = Field(default=8, ge=1)
    llm_workers: int = Field(default=20, ge=1)     # 从 10 调高
    chunk_workers: int = Field(default=6, ge=1)    # 新增：每文件 chunk 并发
```

**`LLMConfig`**（`settings.py`）：
```python
class AdaptiveConfig(BaseModel):
    """AIMD 自适应并发配置"""
    enabled: bool = True
    initial_concurrency: int = 15
    max_concurrency: int = 50
    min_concurrency: int = 3
    success_threshold: int = 15
    multiplicative_decrease: float = 0.5
    cooldown_seconds: float = 5.0

class RoutingConfig(BaseModel):
    """LLM 路由策略配置"""
    strategy: Literal["cost_first", "least_pending", "round_robin"] = "least_pending"
    cost_weight: float = 0.6
    load_weight: float = 0.4

class LLMConfig(BaseModel):
    # ... 现有字段 ...
    adaptive: AdaptiveConfig = Field(default_factory=AdaptiveConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
```

### 4.2 配置文件示例

```yaml
concurrency:
  file_workers: 4
  image_workers: 8
  llm_workers: 20        # 全局上限
  chunk_workers: 6       # 每文件 chunk 并发

llm:
  credentials:
    - id: gemini
      provider: gemini
      api_key_env: GOOGLE_API_KEY
    - id: deepseek
      provider: openai
      api_key_env: DEEPSEEK_API_KEY
      base_url: https://api.deepseek.com

  models:
    - name: gemini-flash
      model: models/gemini-2.0-flash
      credential_id: gemini
      capabilities: [text, vision]
    - name: deepseek-chat
      model: deepseek-chat
      credential_id: deepseek
      capabilities: [text]

  # AIMD 自适应并发（每 credential 独立）
  adaptive:
    enabled: true
    initial_concurrency: 15
    max_concurrency: 50
    min_concurrency: 3

  # 智能路由策略
  routing:
    strategy: least_pending
    cost_weight: 0.6
    load_weight: 0.4
```

---

## 5. 代码改动

### 5.1 改动清单

| 文件 | 改动内容 | 优先级 |
|------|----------|--------|
| **编码兼容性修复** | | |
| `cli/commands/model.py` | YAML 读写指定 UTF-8（行 244, 295） | P0 |
| `cli/commands/provider.py` | YAML 读写指定 UTF-8（行 129, 182）+ JSON 缓存（行 979） | P0 |
| `utils/flow_control.py` | JSON 读写指定 UTF-8（行 351, 366, 519） | P1 |
| `converters/office.py` | stderr.decode 指定 UTF-8（行 279, 418） | P2 |
| `converters/pandoc.py` | subprocess 指定 encoding（行 140, 247, 270, 284, 306） | P2 |
| **并发架构重构** | | |
| `config/settings.py` | 添加 `chunk_workers`, `AdaptiveConfig`, `RoutingConfig` | P0 |
| `llm/manager.py` | 每 credential AIMD，智能路由逻辑 | P0 |
| `services/llm_orchestrator.py` | 每文件独立 chunk semaphore | P0 |
| `llm/enhancer.py` | 支持两层限制，report rate_limit | P1 |
| `image/analyzer.py` | 接入 AIMD（统一管理） | P1 |
| `core/pipeline.py` | 传递 `chunk_workers` 参数 | P1 |

### 5.2 核心代码示例

#### 5.2.1 编码兼容性修复

**P0 - YAML 读写**
```python
# cli/commands/model.py:244, 295
# cli/commands/provider.py:129, 182
from ruamel.yaml import YAML

yaml = YAML()
yaml.preserve_quotes = True
yaml.indent(mapping=2, sequence=4, offset=2)

# 修改前（读取）
config_data = yaml.load(config_path)

# 修改后（读取）
with config_path.open('r', encoding='utf-8') as f:
    config_data = yaml.load(f)

# 修改前（写入）
yaml.dump(config_data, config_path)

# 修改后（写入）
with config_path.open('w', encoding='utf-8') as f:
    yaml.dump(config_data, f)
```

**P1 - JSON 读写**
```python
# utils/flow_control.py:351, 366, 519
# cli/commands/provider.py:979

# 修改前
data = self._storage_path.read_text()
self._storage_path.write_text(json.dumps(data, indent=2))

# 修改后
data = self._storage_path.read_text(encoding='utf-8')
self._storage_path.write_text(
    json.dumps(data, indent=2, ensure_ascii=False),
    encoding='utf-8'
)
```

**P2 - subprocess 编码**
```python
# converters/office.py:279, 418
# 修改前
f"LibreOffice error: {e.stderr.decode() if e.stderr else 'Unknown error'}"

# 修改后
f"LibreOffice error: {e.stderr.decode('utf-8', errors='replace') if e.stderr else 'Unknown error'}"

# converters/pandoc.py:140, 247, 270, 284, 306
# 修改前
result = subprocess.run(cmd, capture_output=True, text=True, check=True)

# 修改后
result = subprocess.run(
    cmd,
    capture_output=True,
    text=True,
    encoding='utf-8',
    errors='replace',
    check=True,
)
```

#### 5.2.2 ProviderManager 智能路由

```python
# llm/manager.py
class ProviderManager:
    def __init__(self, config: LLMConfig):
        self.config = config

        # 每 credential 一个 AIMD limiter
        self._credential_limiters: dict[str, AdaptiveRateLimiter] = {}
        self._credential_pending: dict[str, int] = {}

        # 初始化 AIMD limiters
        if config.adaptive.enabled:
            for cred in config.credentials:
                self._credential_limiters[cred.id] = AdaptiveRateLimiter(
                    AIMDConfig(
                        initial_concurrency=config.adaptive.initial_concurrency,
                        max_concurrency=config.adaptive.max_concurrency,
                        min_concurrency=config.adaptive.min_concurrency,
                        success_threshold=config.adaptive.success_threshold,
                        multiplicative_decrease=config.adaptive.multiplicative_decrease,
                        cooldown_seconds=config.adaptive.cooldown_seconds,
                    )
                )

    def _get_credential_id(self, provider_name: str) -> str:
        """获取 provider 对应的 credential ID"""
        state = self._provider_states.get(provider_name)
        return state.credential_id if state else provider_name

    def _select_best_provider(
        self,
        candidates: list[str],
    ) -> str:
        """智能选择最优 provider"""
        strategy = self.config.routing.strategy

        if strategy == "cost_first":
            return candidates[0]

        elif strategy == "least_pending":
            cost_weight = self.config.routing.cost_weight
            load_weight = self.config.routing.load_weight

            scores = []
            for i, provider_name in enumerate(candidates):
                cred_id = self._get_credential_id(provider_name)
                limiter = self._credential_limiters.get(cred_id)
                pending = self._credential_pending.get(cred_id, 0)
                capacity = limiter.current_concurrency if limiter else 10

                # 成本评分（越前面越便宜）
                cost_score = 1 - (i / len(candidates)) if len(candidates) > 1 else 1
                # 负载评分（pending 越少越好）
                load_score = 1 - min(pending / capacity, 1.0)

                score = cost_weight * cost_score + load_weight * load_score
                scores.append((provider_name, score))

            return max(scores, key=lambda x: x[1])[0]

        else:  # round_robin
            idx = self._current_index % len(candidates)
            self._current_index += 1
            return candidates[idx]

    async def complete_with_fallback(
        self,
        messages: list[LLMMessage],
        required_capability: str | None = None,
        **kwargs,
    ) -> LLMResponse:
        """带智能路由和 AIMD 的请求方法"""
        await self._load_configs()

        candidates = self._filter_providers_by_capability(required_capability, ...)
        if not candidates:
            raise ProviderNotFoundError(...)

        errors = []
        provider_count = len(candidates)

        # 智能选择起始 provider
        best = self._select_best_provider(candidates)
        start_index = candidates.index(best)

        for i in range(provider_count):
            idx = (start_index + i) % provider_count
            provider_name = candidates[idx]
            cred_id = self._get_credential_id(provider_name)
            limiter = self._credential_limiters.get(cred_id)

            # 增加 pending 计数
            self._credential_pending[cred_id] = self._credential_pending.get(cred_id, 0) + 1

            try:
                # AIMD 限制
                if limiter:
                    await limiter.acquire()

                try:
                    provider = self._providers[provider_name]
                    response = await provider.complete(messages, **kwargs)

                    # 记录成功
                    if limiter:
                        await limiter.record_success()

                    return response

                except RateLimitError as e:
                    # 记录 rate limit
                    if limiter:
                        await limiter.record_rate_limit()
                    raise

                finally:
                    if limiter:
                        limiter.release()

            except Exception as e:
                errors.append((provider_name, e))

            finally:
                self._credential_pending[cred_id] -= 1

        raise LLMError(f"All providers failed: {errors}")
```

#### 5.2.3 `complete_with_concurrent_fallback` 路由策略支持

**注意**：`complete_with_concurrent_fallback` 方法也需要支持路由策略，而不是直接使用 `candidates[0]`。

```python
# llm/manager.py - complete_with_concurrent_fallback 中的关键修改
async def complete_with_concurrent_fallback(self, messages, ...):
    # 获取候选列表
    candidates = self._filter_providers_by_capability(required_capability, ...)

    # 根据策略选择主模型
    capability_key = required_capability or "text"
    routing_strategy = self.config.routing.strategy

    if routing_strategy == "cost_first" and capability_key in self._last_successful_provider:
        # cost_first: 优先使用上次成功的 provider（稳定性优先）
        preferred = self._last_successful_provider[capability_key]
        if preferred in candidates:
            candidates.remove(preferred)
            candidates.insert(0, preferred)
        primary_name = candidates[0]
    else:
        # round_robin/least_pending: 使用智能路由策略
        primary_name = self._select_best_provider(candidates)
        # 更新 round-robin 索引
        provider_count = len(candidates)
        self._current_index = (self._current_index + 1) % max(provider_count, 1)

    # ... 创建 primary_task ...

    # 超时后选择 fallback 时也要更新索引
    fallback_candidates = [c for c in candidates if c != primary_name]
    fallback_name = self._select_best_provider(fallback_candidates)
    # 更新索引，确保下一个 fallback 选择不同模型
    fallback_count = len(fallback_candidates)
    self._current_index = (self._current_index + 1) % max(fallback_count, 1)
```

**关键点**：
- `cost_first` 策略：保留 `_last_successful_provider` 优先逻辑，保证稳定性
- `round_robin`/`least_pending` 策略：使用 `_select_best_provider()` 选择，实现负载均衡
- Fallback 选择后也要更新 `_current_index`，避免并发场景下多个 fallback 选择同一个模型

#### 5.2.4 LLMOrchestrator 每文件独立 Semaphore

```python
# services/llm_orchestrator.py
class LLMOrchestrator:
    def __init__(
        self,
        llm_config: LLMConfig,
        chunk_workers: int = 6,
        **kwargs,
    ):
        self.chunk_workers = chunk_workers
        # 移除全局 _chunk_semaphore

    async def enhance_markdown(
        self,
        markdown: str,
        source_file: Path,
        return_stats: bool = False,
    ):
        """增强 Markdown，每文件独立的 chunk 并发控制"""
        # 每文件独立的 chunk semaphore
        per_file_semaphore = asyncio.Semaphore(self.chunk_workers)

        enhancer = await self.get_enhancer()
        result = await enhancer.enhance(
            markdown,
            source_file,
            chunk_semaphore=per_file_semaphore,
            return_stats=return_stats,
        )
        # ... 后续处理 ...
```

#### 5.2.5 Enhancer 支持 Chunk Semaphore

```python
# llm/enhancer.py
async def enhance(
    self,
    markdown: str,
    source_file: Path,
    chunk_semaphore: asyncio.Semaphore | None = None,
    return_stats: bool = False,
) -> EnhancedMarkdown | LLMTaskResultWithStats:
    """增强 Markdown 文档"""
    chunks = self.chunker.chunk(markdown)

    async def process_with_limit(chunk: str, chunk_index: int):
        is_first = chunk_index == 0
        # 每文件独立的 chunk semaphore
        if chunk_semaphore:
            async with chunk_semaphore:
                return await self._process_chunk_with_stats(chunk, is_first_chunk=is_first)
        return await self._process_chunk_with_stats(chunk, is_first_chunk=is_first)

    # 并行处理所有 chunks
    chunk_results = await asyncio.gather(
        *[process_with_limit(chunk, i) for i, chunk in enumerate(chunks)]
    )
    # ... 后续处理 ...
```

---

## 6. 测试计划

### 6.1 单元测试

| 测试文件 | 测试内容 |
|----------|----------|
| **编码兼容性测试** | |
| `test_encoding.py` | 跨平台编码兼容性测试（新增） |
| `test_model_cli.py` | YAML 读写含 Unicode 字符（中文注释、特殊符号 ≤≥→） |
| `test_provider_cli.py` | YAML 读写含 Unicode 字符 |
| `test_flow_control.py` | JSON 读写含中文文件名、中文错误信息 |
| `test_pandoc.py` | subprocess 输出含非 ASCII 字符 |
| **并发架构测试** | |
| `test_manager_routing.py` | 智能路由策略（cost_first, least_pending, round_robin） |
| `test_manager_aimd.py` | 每 credential AIMD 独立性 |
| `test_orchestrator_chunk.py` | 每文件独立 chunk semaphore |

### 6.1.1 编码兼容性测试用例（test_encoding.py）

```python
"""Cross-platform encoding compatibility tests."""
import json
import tempfile
from pathlib import Path

import pytest
from ruamel.yaml import YAML


class TestYAMLEncoding:
    """Test YAML read/write with Unicode characters."""

    def test_yaml_write_unicode_characters(self, tmp_path: Path):
        """Test writing YAML with Unicode characters (≤, ≥, →, Chinese)."""
        yaml = YAML()
        yaml.preserve_quotes = True

        config = {
            "description": "模型上下文长度 ≤ 128k",
            "name": "测试模型",
            "symbols": "→ ← ↑ ↓ ≤ ≥ ≠",
        }

        config_path = tmp_path / "test.yaml"
        with config_path.open("w", encoding="utf-8") as f:
            yaml.dump(config, f)

        # Verify content
        with config_path.open("r", encoding="utf-8") as f:
            loaded = yaml.load(f)

        assert loaded["description"] == "模型上下文长度 ≤ 128k"
        assert loaded["name"] == "测试模型"

    def test_yaml_read_unicode_comments(self, tmp_path: Path):
        """Test reading YAML with Chinese comments."""
        config_path = tmp_path / "test.yaml"
        content = '''# 这是中文注释
llm:
  # 配置说明：使用 ≤ 128k 的模型
  models:
    - name: "测试"
      model: "test-model"
'''
        config_path.write_text(content, encoding="utf-8")

        yaml = YAML()
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.load(f)

        assert config["llm"]["models"][0]["name"] == "测试"


class TestJSONEncoding:
    """Test JSON read/write with Unicode characters."""

    def test_json_write_chinese_content(self, tmp_path: Path):
        """Test writing JSON with Chinese content."""
        data = {
            "filename": "中文文件名.pdf",
            "error": "转换失败：文件格式不支持",
            "path": "/path/to/中文目录/file.pdf",
        }

        json_path = tmp_path / "test.json"
        json_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        # Verify content
        loaded = json.loads(json_path.read_text(encoding="utf-8"))
        assert loaded["filename"] == "中文文件名.pdf"
        assert "转换失败" in loaded["error"]

    def test_json_read_chinese_content(self, tmp_path: Path):
        """Test reading JSON with Chinese content."""
        content = '{"name": "中文名称", "desc": "描述 → 说明"}'
        json_path = tmp_path / "test.json"
        json_path.write_text(content, encoding="utf-8")

        loaded = json.loads(json_path.read_text(encoding="utf-8"))
        assert loaded["name"] == "中文名称"


class TestSubprocessEncoding:
    """Test subprocess output encoding."""

    def test_decode_stderr_with_unicode(self):
        """Test decoding stderr that may contain Unicode."""
        # Simulate stderr bytes with Unicode
        stderr_bytes = "错误：文件不存在 → /path/中文".encode("utf-8")

        # Should not raise exception
        decoded = stderr_bytes.decode("utf-8", errors="replace")
        assert "错误" in decoded
        assert "→" in decoded

    def test_decode_stderr_with_invalid_bytes(self):
        """Test decoding stderr with invalid UTF-8 bytes."""
        # Invalid UTF-8 sequence
        stderr_bytes = b"Error: \xff\xfe invalid bytes"

        # Should replace invalid bytes, not raise exception
        decoded = stderr_bytes.decode("utf-8", errors="replace")
        assert "Error:" in decoded
        assert "�" in decoded  # Replacement character
```

### 6.2 集成测试

| 场景 | 验证点 |
|------|--------|
| 大文件多 chunks | 并发数符合预期，不再串行 |
| 多文件并行 | 每文件公平获得 chunk 并发 |
| 触发 429 | AIMD 自动降速，不影响其他 credential |
| 多模型配置 | least_pending 策略正确分流 |

### 6.3 性能验收

| 指标 | 改动前 | 目标 |
|------|--------|------|
| 5 文件处理时间 | ~800s | < 400s |
| 40 chunks 文件 | ~650s | < 200s |
| 资源池利用率 | 便宜模型 100%，其他闲置 | 均衡利用 |

---

## 7. 兼容性

### 7.1 向后兼容

- 所有新配置项有默认值，不需要用户修改配置
- 默认 `routing.strategy: least_pending`，行为略有变化但更优
- 原有 `llm_workers` 配置仍生效

### 7.2 配置迁移

无需迁移，新配置项为可选。

---

## 8. 风险评估

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| **并发相关** | | |
| 新路由策略增加贵模型使用 | 成本增加 | `cost_weight: 0.6` 默认偏重成本 |
| AIMD 参数不当导致过度限流 | 性能下降 | 保守默认值，可配置调整 |
| **编码相关** | | |
| Windows 编码问题遗漏 | 程序报错 | 已全面扫描代码库，见 3.1.2 清单 |
| 第三方库编码行为变化 | 回归问题 | 添加编码兼容性测试，CI 验证 |
| 中文文件名在 Windows 上处理 | 路径错误 | `pathlib.Path` 自动处理 Unicode，已验证 |

### 8.1 已验证安全的代码（无需修改）

以下代码已正确处理编码问题：

| 位置 | 原因 |
|------|------|
| `config/settings.py` 所有 `read_text()` | ✅ 已指定 `encoding="utf-8"` |
| `cli/commands/config.py` 配置写入 | ✅ 已指定 `encoding="utf-8"` |
| `services/output_manager.py` 异步文件写入 | ✅ `anyio.open_file(..., encoding="utf-8")` |
| `image/analyzer.py` Markdown 写入 | ✅ `anyio.open_file(..., encoding="utf-8")` |
| `core/state.py` 状态文件读写 | ✅ `open(..., encoding="utf-8")` |
| `utils/fs.py` `atomic_write()` | ✅ 默认 `encoding="utf-8"` |
| `converters/pandoc.py:149` 输出文件读取 | ✅ `read_text(encoding="utf-8")` |
| `utils/logging.py` SafeStreamHandler | ✅ 处理终端编码错误 |
| `converters/markitdown.py:232,244` zipfile | ✅ `decode("utf-8", errors="ignore")` |
| `pathlib.Path` 字符串转换 | ✅ 自动处理 Unicode |

---

## 9. 参考资料

- [Rate limits | OpenAI API](https://platform.openai.com/docs/guides/rate-limits)
- [Rate limits | Gemini API](https://ai.google.dev/gemini-api/docs/rate-limits)
- [Rate Limits for LLM Providers | Requesty Blog](https://www.requesty.ai/blog/rate-limits-for-llm-providers-openai-anthropic-and-deepseek)
- AIMD 算法实现：`src/markit/utils/adaptive_limiter.py`
