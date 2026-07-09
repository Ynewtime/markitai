# StageList 多阶段清单式进度 — 设计文档

日期:2026-07-09
状态:已批准(设计阶段)
范围:单 URL 与单文件转换路径的进度反馈重构

## 背景与问题

`mkai <url> -p standard --no-cache`(stdout 模式,不带 `-o`)在整个
fetch + LLM 增强期间(1~3 分钟)终端完全静默,用户无法区分"在干活"
和"挂死"。

根因是三重 gating 叠加出的盲区:

- `ConversionStatus`(`cli/ui.py`)在 stdout 模式被禁用
  (`processors/url.py`:`enabled=not stdout_mode and ...`),理由是
  "stdout 模式由 OutputManager 接管 stderr"
- 但 `ProgressReporter` 在 stdout 模式同样 `enabled=False`
  (`url.py`),其 `start_spinner` 在转发给 OutputManager 之前就短路
  返回(`utils/progress.py`)
- `OutputManager` 虽在 stdout 模式启用(`main.py`),但除 auth 登录
  流程外没有任何代码调用它的 spinner/print

结果:带 `-o` 时有单行 spinner;不带 `-o` 时什么都没有。

同时,现状存在两套并行的 stderr 进度设施(`ConversionStatus` 的
loguru 阶段桥 + 耗时后缀,`ProgressReporter`/`OutputManager` 的手动
ANSI 行追踪擦除),职责重叠、盲区互补不成。

## 决策记录

经用户确认:

1. **修复档位**:不止接通 spinner,升级为多阶段清单式进度
2. **适用范围**:单 URL(`processors/url.py`)与单文件
   (`processors/file.py`)都升级;批量模式已有 rich Progress 不动
3. **清单去留**:`-o` 文件模式完成后保留清单(含各阶段耗时);
   stdout 模式完成后擦除,stdout 只留 markdown
4. **技术路线**:新建多行清单组件 StageList(rich Live),统一并退役
   两套旧设施

## 1. 组件与渲染

新组件 `StageList`(放在 `cli/ui.py`,替代 `ConversionStatus`),
基于 `rich.live.Live` 渲染到 stderr:

```
  ✓ Fetched via fxtwitter (2.1s)      ← 已完成行(定格,含耗时)
  ✓ Downloaded 3 images (1.4s)
  ⠙ Enhancing with LLM... (23s)       ← 活跃行(spinner + >5s 显示已耗时)
```

- **stdout 模式**:`Live(transient=True)` —— `stop()` 时 rich 自动
  擦除整个区域,stdout 只留 markdown。不再需要 OutputManager 的手动
  ANSI 行数追踪。
- **-o 文件模式**:`Live(transient=False)` —— `stop()` 保留最后一
  帧,清单留在屏幕上,最后照旧追加 `✓ <输出路径>`。
- 活跃行耗时后缀沿用现有 asyncio ticker 机制(阶段运行 >5s 才显示,
  避免短阶段噪音;阈值常量沿用 `ELAPSED_SUFFIX_THRESHOLD_S`);
  完成行始终显示耗时。
- spinner 字符沿用 `STATUS_SPINNER = "line"`(纯 ASCII,任意终端
  可渲染)。

## 2. 阶段模型与事件源

组件状态 = `已完成行列表 + 至多一个活跃阶段 (key, text, started_at)`。

### 显式 API(URL 路径的主驱动)

- `advance(key, text, pin=False)` —— 开启新阶段,自动把上一活跃阶段
  定格为 ✓ 行;`pin=True` 时 loguru bridge 不得推进或改写该阶段
- `update_text(text)` —— 更新活跃行文本,不重置阶段计时
- `finalize(text=None)` —— 用完成态文本定格活跃阶段
  (如 `"Fetched via fxtwitter"`);缺省用活跃文本去掉尾部 `...`
- `note(text)` —— 插入一条信息行(如 screenshot 提示),不影响活跃
  阶段
- `fail(text=None)` —— 把活跃行定格为 ✗(错误路径);缺省用活跃
  文本
- `stop()` —— 停止 Live;幂等,支持上下文管理器用法

### loguru bridge(file 路径的主驱动)

沿用 `stage_from_log_record` 机制,映射表每条目标注 stage key:

- fetch 内部策略跳转(static → playwright → fxtwitter / defuddle /
  jina / cloudflare)全部映射到 key `"fetch"`——**同 key 只更新活跃
  行文本**,不新增清单行
- `"[LLM]"` → key `"llm"`,`"Analyzing "` → key `"images"`——
  **异 key 自动推进阶段**(定格旧的、开新的)

这让 `convert_document_core` 内部零改动就能驱动 file 路径的清单。

### 并行 LLM 场景

URL 路径 document + images 并行时,显式
`advance("llm", "Enhancing with LLM (document + images)...", pin=True)`
锁定阶段,期间 bridge 收到的交错 `"[LLM]"` / `"Analyzing "` 日志不会
导致阶段来回跳。

### 取舍说明

bridge 的"异 key 自动推进"让 file 路径零侵入,代价是阶段边界依赖
日志前缀映射表——新增 fetch 策略时需同步加映射。现状
`ConversionStatus` 已是如此,无新增负担。

## 3. 调用点迁移与设施退役

| 现有设施 | 去向 |
|---|---|
| `ConversionStatus` | 退役删除;`commands/init.py` 的两处单行用法改用 StageList 单阶段模式 |
| `ProgressReporter` | 退役删除(仅 url.py / file.py 在用);`progress.start_spinner/log/clear_and_finish` 调用点一对一改写为 `stages.advance/finalize/note/stop` |
| `OutputManager` | 保留但瘦身:`start_spinner`/`stop_spinner` 职责移除,继续服务 auth 登录流程(`attempt_login`,发生在转换开始前,与 StageList 无时序重叠);`process_url`/`process_file` 不再接收 `output_manager` 参数 |

gating 统一为一处:`enabled = not quiet and not verbose`
(**去掉 `not stdout_mode`** —— 本次的根因修复),stderr 非 TTY 时
自动降级(见第 4 节)。

`markitai.utils` / `markitai.cli` 包级导出中的 `ProgressReporter`
一并移除,CHANGELOG 注明。

## 4. 边界行为

- **非 TTY**(`2>file`、CI):无 Live/spinner;文件模式下 StageList
  内部降级为在 `finalize`/`note`/`fail` 时直接向 stderr 打印静态完成
  行(调用方无感知,与现状 `progress.log` 行为一致),stdout 模式
  完全静默(不污染管道场景)
- **`--quiet` / `-v`**:维持现状——quiet 全静默;verbose 走日志流、
  不显示清单
- **错误路径**:活跃行定格为 `✗ <阶段> (Ns)`;即使 stdout 模式
  (transient)也在擦除前把已完成 + 失败行持久打印到 stderr,保留
  "死在哪一步"的上下文
- **cache 命中**:`✓ Fetched via static (cached, 0.2s)`
- **已知限制**(继承自 ConversionStatus v1):fetch 链内的 lazy
  remote-consent prompt(`click.confirm`)可能与 Live 帧短暂交错,
  下一次重绘后恢复干净;修复需改动 fetch.py 的 prompt 所有权,
  不在本次范围

## 5. 完成态文本规范

- `✓ Fetched via fxtwitter (2.1s)`
- `✓ Fetched via static (cached, 0.2s)`
- `✓ Downloaded 3 images (1.4s)`;0 张时降为 note 行
  `No images to download`
- `✓ LLM enhanced (48.3s)`
- 失败:`✗ Enhancing with LLM (12s)`

## 6. 测试

- StageList 单测:阶段推进/定格/pin/bridge 映射/transient 两模式/
  非 TTY 降级/幂等 stop/上下文管理器
- 更新 url/file processor 现有集成测试断言(涉及
  `test_cli_main.py`、`test_cli_framework.py` 等)
- 手动端到端:`mkai <url> -p standard` 分别验证 stdout 与 `-o`
  两种模式、Ctrl-C 中断后终端状态干净
