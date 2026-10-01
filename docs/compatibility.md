# 兼容性基线与迁移验收

本文件从 Rust 产品的验收角度定义迁移边界。它是对参考实现的只读审计结果，不表示 Rust 实现已经完成对应功能，也不把 Python 内部模块结构作为 Rust 架构约束。

审计日期：2026-09-28。参考目录：`/Users/example-user/work/markitai`。参考提交：`ba374322f884b0e720b45466cc1196f4574a3da5`，审计时工作区干净；公开版本为 `1.2.0`。下文源码路径均相对于参考仓库，行号对应此提交。

## 1. 兼容性原则

- 保留命令名、现有参数拼写、配置键、输出命名、JSON 字段、Python 的公开函数与结果属性；新增 Node.js/Go 接口围绕相同的核心结果模型设计。
- 对尚未实现的能力明确报错并记录迁移状态，不能接受选项后悄悄忽略，也不能用空 Markdown 宣告成功。
- Markdown 正文首先要求语义与信息保真，再评估字节兼容。文件名、JSON schema、默认选项和退出码需要直接对比；时间戳、持续时间、绝对路径等动态字段可规范化。
- 已发布接口、已经移除的接口和内部实现细节分开处理。参考实现已移除的参数无需重新引入。
- 默认本地转换、无配置、无网络、单文件 stdout 路径必须先形成完整闭环；这不是缩减最终迁移范围。

## 2. CLI 契约

入口为 `markitai`，短别名为 `mkai`；另有 `markitai-mcp`，等价功能经 `markitai mcp` 暴露。来源：`packages/markitai/pyproject.toml:91`、`src/markitai/cli/framework.py:23`（本节起 `src/`、`tests/` 均位于 `packages/markitai/` 下）。

命令形式为 `markitai [OPTIONS] [INPUT] [COMMAND]`。INPUT 是单个文件、目录、HTTP(S) URL 或 `.urls` 列表。选项可在 INPUT 前后出现；INPUT 和子命令同时出现应拒绝。与子命令同名的本地文件通过 `./config` 一类路径消歧。没有 INPUT 和子命令时输出帮助，退出 0。来源：`src/markitai/cli/framework.py:288`、`src/markitai/cli/main.py:605`。

### 转换参数清单

| 分类 | 参数 | 约束与行为 |
| --- | --- | --- |
| 输出 | `-o, --output PATH` | 单项可指目录或 `.md` 文件；批量必须目录。已存在且名为 `out.md` 的目录仍按目录处理 |
| 输出 | `--json` | stdout 仅一个 JSON 文档，必须给 `-o`，与 `--dry-run`、`--llm-batch-collect` 互斥 |
| 配置 | `-c, --config PATH` | 转换与只读子命令要求文件存在；`config set/edit` 可创建 |
| 配置 | `--config-json JSON` | 必须为 JSON object，递归合并在文件配置之上，不持久化 |
| 配置 | `-p, --preset NAME` | 内建 `rich/standard/minimal`，也支持配置中的命名 preset |
| 输出 | `--profile NAME` | `rag/obsidian/okf`，大小写不敏感，与 preset 正交 |
| 增强 | `--llm/--no-llm` | 三态，省略保留配置值 |
| 图像 | `--alt/--no-alt`、`--desc/--no-desc` | 三态，依赖 LLM |
| OCR | `--ocr/--no-ocr` | 无 LLM 为本地 OCR，有 LLM 则通过视觉模型读取页图 |
| 截图 | `--screenshot/--no-screenshot` | PDF/PPTX 页图、URL 长截图 |
| 截图 | `--screenshot-only/--no-screenshot-only` | 开启时隐含 screenshot，不隐含 LLM；URL 无 LLM 时仅保存截图，不写 Markdown |
| 图像 | `--no-compress/--compress` | 三态，反向映射 `image.compress` |
| 缓存 | `--no-cache/--cache` | 禁止读取缓存仍写入新结果 |
| 缓存 | `--no-cache-for PATTERNS` | 逗号分隔文件名/glob |
| 增强 | `--pure/--no-pure` | 跳过 frontmatter 与后处理；LLM 模式原始 Markdown 直接送模型 |
| 增强 | `--keep-base` | LLM 模式保留 `.md` 与 `.llm.md` 两份 |
| 并发 | `-j, --batch-concurrency N`、`--url-concurrency N`、`--llm-concurrency N` | 整数至少 1，三种限制独立 |
| 批量 | `--resume` | 从持久化状态恢复 |
| 批量 | `-g, --glob PATTERN` | 可重复，`!` 前缀排除，按相对输入目录路径匹配 |
| 批量 | `--max-depth N` | 至少 0；0 仅扫描输入目录 |
| Batch API | `--llm-batch` | 仅目录批量，单模型 OpenAI/Anthropic 池；仅提交本轮实际转换的文件 |
| Batch API | `--llm-batch-timeout N` | 默认 3600 秒，至少 60 秒 |
| Batch API | `--llm-batch-collect BATCH_ID` | 无需 INPUT，必须 `-o` 指向原批量目录 |
| 获取 | `-s, --strategy NAME` | `auto/static/playwright/defuddle/jina/cloudflare` |
| 转换 | `-b, --backend NAME` | `native/cloudflare` |
| 获取 | `--no-remote-fetch` | 全局禁止远程抽取服务，包括显式选中的远程策略 |
| 观察 | `-v, --verbose`、`-q, --quiet` | 进度和诊断不能污染 Markdown/JSON stdout；单项默认安静 |
| 日志 | `--log-level LEVEL` | `DEBUG/INFO/WARNING/ERROR/CRITICAL`，仅影响配置启用的文件日志 |
| 预览 | `--dry-run` | 预览，不写转换文件 |
| 历史 | `--record-history/--no-record-history` | 覆盖环境和配置；stdout 模式不记录 |
| 交互 | `-I, --interactive` | 引导选择配置和转换 |
| 信息 | `-V, --version`、`-h, --help` | 无转换的快速路径 |

来源：`src/markitai/cli/main.py:262-530`；输出目标判定 `src/markitai/runs/output.py`，测试 `tests/unit/test_output_target.py`。

已移除的 `--playwright/--defuddle/--static/--jina/--cloudflare` 应报错并提示 `-s`；`--kreuzberg` 已移除，RTF 为原生转换。来源：`src/markitai/cli/framework.py:143`。

### 子命令清单

| 命令 | 子命令与选项 |
| --- | --- |
| `config` | `list [-f,--format json/yaml/table] [--show-secrets]`；`path`；`validate [CONFIG_FILE]`；`get KEY [--show-secrets]`；`set KEY VALUE [--show-secrets]`；`edit` |
| `cache` | `stats [--json] [-v,--verbose] [--limit N=20]`；`clear [--include-spa-domains] [-y,--yes]`；`spa-domains [--json] [--clear]` |
| `auth` | `copilot/claude/chatgpt` 各含 `status [--json]`、`login` |
| `init` | `[-y,--yes] [-o,--output PATH] [--local]`；本地配置文件名 `markitai.json` |
| `doctor` | `[--json] [--fix] [--suggest-extras]`；Rust 中诊断内容与安装建议需要重新设计，命令保留 |
| `serve` | `[--host 127.0.0.1] [--port 3600] [--no-open] [--no-auth] [--allowed-host HOSTNAME ...]` |
| `mcp` | stdio MCP 服务 |

来源：`src/markitai/cli/commands/{config,cache,auth,init,doctor,serve,mcp}.py`。根级 `-c` 与 `--config-json` 也应用于子命令；秘密默认脱敏，只有 `--show-secrets` 显式展示。

### stdout、JSON 和退出状态

单文件/URL 未给 `-o` 时输出 Markdown 到 stdout。单文件明确不以 `output.dir` 配置替代 `-o`；目录批量可以从配置选择输出目录。默认普通转换加 frontmatter；`--pure` 才是原始正文路径。库调用则始终不能向 stdout 写内容。

stdout 图片持久化（`image.stdout_persist`，默认开启）与参考一致：被引用的图片和页面截图存入 `stdout_persist_dir/blobs/`，引用改为绝对 `file://` URI，包括生成的页面截图注释。有意的差异：默认目录跟随 `MARKITAI_HOME`（参考按真实 HOME 展开并忽略 `MARKITAI_HOME`）；文件名取 SHA-256 前 24 位（参考 16 位），已有同名不同内容的文件不改写而改用完整摘要；不建立参考的 `refs/<来源>/<图片>` 符号链接索引；关闭或保存失败时保留原相对引用与 alt 并警告（参考改为丢失 alt 的 `![image: 名称]()` 占位符）；无终端内联图片，`stdout_fetch_external` 无作用。对照记录见[stdout 中的图片](images.md#images-on-stdout)。

转换退出矩阵：成功（包括 dry-run）为 0；单项失败/运行级输入失败为 1；Click 参数、枚举、缺失 `-c` 等 usage 错误为 2；目录或 URL 批量有任意失败为 10。中断当前由 Click 接收 `KeyboardInterrupt`，现有测试只要求非零；不要在未做进程级基线试验前声称中断固定为 130。来源：`src/markitai/runs/report.py:209`、`src/markitai/cli/main.py:197,1472`、`tests/unit/test_json_output.py:620`。

JSON envelope `version="1.0"`，顶层字段为 `version, ok, error, batch, items, totals`。`ok` 仅在无 failed、无 pending、无运行级 error 时为真。条目按照完成顺序排列；总耗时是条目耗时和，不是墙钟时间。

每条 item 固定字段：`kind, source, status, output, error, warnings, skip_reason, images, screenshots, cost_usd, duration_s, cache_hit, fetch_cache_hit, llm_cache_hit, fetch_strategy, source_file, llm_usage`。`kind` 为 `file/url`；status 为 `completed/failed/skipped/pending`；images/screenshots 为数量；cost 四舍五入 6 位，duration 3 位。totals 为 `total, completed, failed, skipped, pending, cost_usd, duration_s`。Batch API 异步交接的 batch 为 `id/status/collect_command` 对象，否则 null。输出漂亮缩进 JSON 并以换行结束，非 ASCII 可转义。usage 错误只在 stderr，无 JSON；运行失败应尽可能仍输出 envelope。来源：`src/markitai/runs/json_output.py:26-167`。

## 3. 配置与环境

配置文件选择顺序：显式 `-c` > `MARKITAI_CONFIG` > 当前目录 `markitai.json` > `~/.markitai/config.json` > 内建默认。只选择一个文件，不合并多个文件；`--config-json` 在选中文件上深度合并，preset 再覆盖功能位，显式 CLI 参数最后覆盖。`MARKITAI_CONFIG` 指向缺失文件时当前实现警告并用默认值；CLI 显式缺失 `-c` 则是 usage 错误。来源：`src/markitai/config.py:1004-1099`、`src/markitai/cli/main.py:814-889`。

注意参考 API docstring 曾写 `.markitai.json`，实际常量和代码是 `markitai.json`，迁移按代码和测试执行。配置格式为 UTF-8 JSON object，支持 `env:VAR_NAME` 密钥引用；非法 JSON、错误类型、范围错误要给可操作诊断。未知键的保留/忽略策略需通过契约测试明确，不能假定所有类都 `deny_unknown_fields`。`config set` 支持 `llm.model_list[0].model_name` 一类数组路径；保存应为最小差异，不把临时 override 写入。

CLI 在启动时按当前 `.env`、用户 `~/.markitai/.env` 顺序补充环境，已经存在的进程环境优先。关键环境名：`MODEL`、`MARKITAI_CONFIG`、`MARKITAI_PURE`、`MARKITAI_RECORD_HISTORY`、`MARKITAI_NO_REMOTE_FETCH`、`MARKITAI_NO_VLM_OCR`、`MARKITAI_LOG_DIR`、`MARKITAI_LOG_FORMAT`、`MARKITAI_LANG`、`MARKITAI_SERVE_TOKEN`、`MARKITAI_PDF_WORKERS`、`MARKITAI_STATIC_HTTP`、`PLAYWRIGHT_BROWSERS_PATH`。代理服从标准 HTTP(S)/ALL_PROXY 和 NO_PROXY 系列变量。以上是参考版本的清单；本构建实际读取的变量见[配置](configuration.md#环境变量)，其中 `MARKITAI_PDF_WORKERS`、`MARKITAI_STATIC_HTTP` 不读取；`MARKITAI_LANG` 的范围见 [CLI 终端语言](cli.md#终端语言)。

云服务凭据名：`OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, DEEPSEEK_API_KEY, OPENROUTER_API_KEY, JINA_API_KEY, CLOUDFLARE_API_TOKEN, CLOUDFLARE_ACCOUNT_ID`。Copilot 另支持 `COPILOT_GITHUB_TOKEN/GH_TOKEN/GITHUB_TOKEN`。LLM 模型解析顺序是已有 model_list > MODEL > 可检测的供应商；可能形成跨供应商池，须保留对应通知。测试使用用户环境时，只读取必要值传给隔离子进程；严禁输出值、复制进 Git，或写入用户现有缓存/历史/登录文件。

默认行为：LLM/OCR/alt/desc/screenshot/pure 关闭；冲突 rename；压缩开启，JPEG quality=75，最大 1920×99999；图像过滤最小宽高 50/50、面积 5000；LLM/file/URL 并发 10/10/5；扫描深度 5、最多 10000 文件；fetch auto，本地优先，remote_consent 默认 always（允许策略链中的远程回退）；fetch 缓存 TTL 86400 秒；缓存限制 512 MiB；输入文件限制 500 MiB；历史/文件日志默认关闭。preset minimal 全关闭，standard 开启 llm/alt/desc，rich 再加 screenshot，均不自动开启 OCR。来源：`src/markitai/constants.py`、`src/markitai/config.py:861-891`。

### 配置字段全集索引

下表列出配置路径下的直接字段；数组项、动态键以 `[]`、`{name}` 表示。类型、边界和默认值的参考证据是 `src/markitai/config.schema.json` 与 `src/markitai/config.py`；Rust 应生成自己的 schema 与文档，不能将参考 schema 当作已完成实现的声明。

| 路径 | 直接字段 |
| --- | --- |
| `output` | dir, on_conflict, allow_symlinks, report, profile, wikilinks |
| `llm` | enabled, pure, keep_base, on_failure, model_list, providers, router_settings, concurrency, max_requests_per_document, max_cost_per_document_usd, max_vision_pages_per_document |
| `llm.model_list[]` | model_name, litellm_params, model_info |
| `llm.model_list[].litellm_params` | model, api_key, api_base, weight, api_version, max_tokens |
| `llm.model_list[].model_info` | id, provider_id, supports_vision, max_tokens, max_input_tokens |
| `llm.providers[]` | id, provider, api_key, api_base |
| `llm.router_settings` | routing_strategy, num_retries, timeout, fallbacks |
| `image` | alt_enabled, desc_enabled, compress, quality, format, max_width, max_height, filter, stdout_persist, stdout_persist_dir, stdout_fetch_external |
| `image.filter` | min_width, min_height, min_area, deduplicate |
| `ocr` | enabled, lang, per_page_routing |
| `office` | macos_fallback |
| `screenshot` | enabled, viewport_width, viewport_height, quality, max_height, tile_height, screenshot_only |
| `prompts` | dir，以及 cleaner/image_caption/image_description/image_analysis/document_process/document_vision/url_enhance 各自的 `_system`、`_user` 字段 |
| `batch` | concurrency, url_concurrency, state_flush_interval_seconds, scan_max_depth, scan_max_files, heavy_task_limit |
| `log` | level, dir, rotation, retention, format |
| `cache` | enabled, no_cache, no_cache_patterns, fetch_ttl_seconds, max_size_bytes, global_dir |
| `fetch` | strategy, remote_consent, defuddle, playwright, jina, cloudflare, policy, domain_profiles, fallback_patterns |
| `fetch.policy` | enabled, max_strategy_hops, strategy_priority, local_only_patterns, inherit_no_proxy |
| `fetch.domain_profiles.{name}` | wait_for_selector, wait_for, extra_wait_ms, prefer_strategy, strategy_priority, skip_auto_scroll, reject_resource_patterns |
| `fetch.playwright` | timeout, wait_for, extra_wait_ms, session_mode, session_ttl_seconds, wait_for_selector, cookies, reject_resource_patterns, extra_http_headers, user_agent, http_credentials |
| `fetch.defuddle` | timeout, rpm |
| `fetch.jina` | api_key, timeout, rpm, no_cache, target_selector, wait_for_selector |
| `fetch.cloudflare` | api_token, account_id, timeout, wait_until, cache_ttl, reject_resource_patterns, user_agent, cookies, wait_for_selector, http_credentials, convert_enabled |
| `security` | pdf_sanitize (`off/warn/remove`) |
| `presets.{name}` | llm, ocr, alt, desc, screenshot |
| `history` | record |

## 4. 输出和文件系统契约

- 默认文件名保留原扩展名：`report.pdf.md`；LLM 版 `report.pdf.llm.md`。LLM 成功默认仅保留增强版，`keep_base` 同时保留基础版。`llm.on_failure=fallback` 默认写基础版并带 warning，仍算成功；`fail` 同样保留基础版但将该项判为失败。
- 冲突策略 `skip/overwrite/rename`；rename 从 `.v2` 开始，例 `report.pdf.v2.md`、`report.pdf.v2.llm.md`。基础版和增强版共用命名空间；同批并发命名冲突必须预留，不能两个任务各自检查不存在后互相覆盖。大小写不敏感文件系统也需保护。来源：`src/markitai/utils/output.py:17-209`。
- 批量输出保持输入的相对目录；忽略 `.markitai` 与自身输出目录，rag/obsidian 的资产目录不能被再次当成输入。恢复必须复用对应项的已分配文件名，避免重新计费和重复版本文件。
- 默认资产/截图/报告/状态分别在 `<output>/.markitai/assets`、`screenshots`、`reports`、`states`。报告文件 `markitai.<task_hash>.report.json`，重名为 `.v2.report.json`。历史在用户 `.markitai/serve/jobs`；SQLite 缓存 `cache.db` 与 `fetch_cache.db`。
- 基础 Markdown frontmatter 按序含 `title, source, markitai_processed`；完整增强 frontmatter 在 source 后含 `description`，有标签时再含 `tags`。URL 可含 `fetch_strategy` 与可信附加元数据。不要传播抽取器错误 language，也不要让附加元数据覆盖这些规范字段。正文页标记为 `<!-- Page number: N -->`。来源：`src/markitai/workflow/helpers.py:313`、`src/markitai/utils/frontmatter.py:370-449`、`src/markitai/constants.py:89`。
- `rag` 将资产移到可见 `assets/`，页标记改为 `<!-- page: N -->`，检查管道表格列数；`obsidian` 同样移资产，`output.wikilinks=true` 时输出 `![[assets/name]]`；`okf` 注入 `type: Document`，source→resource，markitai_processed→generated.at，并含 generated.by。来源：`src/markitai/output_profiles.py`。
- `images.json` schema 1.0 顶层固定 `version, created, updated, images`；图像项字段 `path, alt, desc, text, created, source`；不得泄漏内部 `llm_usage`。资产按路径合并，每个输出目录分别维护。来源：`src/markitai/workflow/helpers.py:476`、`tests/unit/test_images_json_schema.py:19`。
- 已接受差异（用户于 2026-09-30 决定）：资产文件名采用内容哈希（例 `93e62bacfce62672e95d86d1.png`），不沿用参考的 `<文档名>.<NNNN>.<ext>` 序号命名。相同字节只保存一份、重复运行名称稳定；引用、`images.json` 路径与各输出配置同步使用该名称。依赖参考序号文件名的下游需按 Markdown 引用或 `images.json` 定位资产。
- 输出采用原子写入；默认拒绝不安全 symlink；路径穿越、归档炸弹、HTML 外部引用、远程抓取与视觉上传同意策略不能因 Rust 改写而退化。来源：`src/markitai/security.py`、`fetch_policy.py`、`fetch_consent.py`、`vision_consent.py`。

`.urls` 是 UTF-8（允许 BOM）文本：空行/`#` 注释忽略，每行 `URL [自定义输出名]`，名称可带引号；也支持字符串 JSON 数组或 `{"url": "...", "output_name": "..."}` 对象数组。非法条目当前警告并跳过，格式损坏整体报错。URL 路径末段用于文件名；带 query 时加域名前缀，不能把凭据写进文件名或日志。来源：`src/markitai/urls.py:35-177`、`src/markitai/utils/cli_helpers.py:59`。

## 5. Python 与新增 bindings

公开根模块导出：`ConversionOutput, ConversionUsage, MarkitaiConfig, __version__, aconvert, convert, enable_worker_processes`。`markitai.api` 还导出 `ConversionError, FetchError, NoModelConfiguredError, OutputProfileName`。来源：`src/markitai/__init__.py:32`、`src/markitai/api.py:46`。

```python
convert(source: str | Path, *, output_dir=None, config=None,
        llm=None, ocr=None, screenshot=None, alt=None, desc=None,
        profile=None) -> ConversionOutput
async aconvert(source: str | Path, *, output_dir=None, config=None,
               llm=None, ocr=None, screenshot=None, alt=None, desc=None,
               profile=None) -> ConversionOutput
```

参数中布尔值为三态，不得把 None 当成 False；config 必须私有复制，调用不修改传入对象；API 只接受单文件或 URL，目录抛 `IsADirectoryError`。无 output_dir 为内存模式：使用私有临时目录，返回正文，最终所有持久路径为 None、assets/screenshots 空，可能保留相对图像引用。写盘时 path 属性使用 `pathlib.Path`。来源：`src/markitai/api.py:531-704`。

`ConversionOutput` 属性：`source: str, markdown: str, llm_markdown: str|None, frontmatter: dict, output_path: Path|None, llm_output_path: Path|None, assets: list[Path], screenshots: list[Path], images: list[dict], usage: ConversionUsage, skip_reason: str|None, duration: float, warnings: list[str]`。markdown 与 llm_markdown 不含 frontmatter；frontmatter 取最丰富的输出，优先增强版。skip_reason 当前为 `exists` 或 None。warnings 必须调用隔离，不能跨并发污染。

`ConversionUsage` 属性：`cost_usd, requests, input_tokens, output_tokens, by_model`；by_model 每个模型对应 `requests/input_tokens/output_tokens/cost_usd`。

保留异常类型与可判定分类：不存在文件 `FileNotFoundError`；目录 `IsADirectoryError`；转换失败/无内容 `ConversionError`；抓取 `FetchError`；无模型 `NoModelConfiguredError(ValueError)`。同步 convert 在已有 Python event loop 内抛 RuntimeError，引导使用 await aconvert；async 不应阻塞 loop，Rust bindings 应释放 GIL。原 enable_worker_processes 在 Rust 中可以成为兼容性 no-op，前提是写明其语义变更和并行机制。

Node.js 与 Go 没有既存接口需要逐字兼容；围绕同一 options/result/error 模型新增，保持 null/None、路径、usage、warning 等意义一致。C ABI 内存所有权、错误字符串释放、取消和线程安全应独立设计，避免直接暴露 Rust 内存布局。Python 原有 Pydantic MarkitaiConfig 的构造、嵌套属性、model_copy/model_dump 使用也属于需审计的适配面，不能只暴露 dict 后称完全兼容。

## 6. 格式和集成范围

参考实现扩展名注册共 39 项（别名计入）：`docx/doc/pptx/ppt/xlsx/xls/pdf/txt/md/markdown/jpeg/jpg/png/webp/svg/csv/xml/ods/odt/numbers/gif/bmp/tiff/tif/heic/heif/avif/tsv/epub/rtf/rst/org/tex/html/htm/xhtml/eml/msg/ipynb`。检测按小写扩展名。来源：`src/markitai/converter/base.py:109-149`。

单独图片没有 LLM/OCR 时没有可提取正文，单项应失败并提供开启建议；不能写空文档报告成功。Office 图表、公式、合并表格、页/幻灯片几何布局、PDF 隐藏文字与扫描判断、邮箱 MIME/附件、EPUB 阅读顺序是语义验收重点。HEIF/SVG、旧 Office、本地 OCR、浏览器是参考实现的可选能力，Rust 单 binary 的核心体积预算与完整能力的外部运行时/模型依赖需明确区分。

EML 采用参考结构（`## Attachments`：图片附件为图片，其它附件带大小，附带邮件引用展开一层；空正文仅保留 `## Content` 标题），并保留两项增强（用户于 2026-09-30 选择）：每个附件（含附带邮件）保留下载链接，正文内 Content-ID 图片绑定到本地资产而非保留 `cid:`。细节见 [EML](eml.md)。

`serve` 还有既存 HTTP/OpenAPI/SSE 契约与 Web UI；定义位于 `src/markitai/serve/schemas.py`、路由 `serve/app.py:1846+`，前端镜像 `webapp/src/api/types.ts`，同步测试 `tests/unit/serve/test_contract_sync.py`。MCP 公开工具是 `convert_document, convert_url, batch_convert, job_status`（`src/markitai/mcp/server.py:258,320,490,574`）。这些接口须登记为后续兼容工作，不能因 CLI 首发而从迁移清单消失。

## 7. 测试资产与优先次序

静态 AST 统计：278 个 `test_*.py` 文件，5976 个测试函数定义，其中 unit 5680、integration 296；参数化后的实际用例数不同。本次审计没有运行旧项目测试，不声称全部通过。

| 验收面 | 现有证据 |
| --- | --- |
| CLI 语法/配置/退出码 | `tests/unit/test_cli_framework.py`, `test_cli_main.py`, `test_config_json_option.py`, `test_root_config_subcommands.py`, `test_self_explanatory_errors.py` |
| JSON 和库契约 | `tests/unit/test_json_output.py`, `test_api.py`, `test_json_order.py` |
| 路径/输出/竞争 | `tests/unit/test_output_target.py`, `test_output_reservations.py`, `test_batch_thread_safety.py`, `test_batch_resume_state.py` |
| frontmatter/图像/profile | `tests/unit/test_frontmatter.py`, `test_images_json_schema.py`, `test_output_profiles.py`, `tests/integration/test_output_format.py` |
| 文件转换 | `tests/unit/test_converter_*.py`, `test_docx_plain.py`, `test_xlsx_plain_tables.py`, `test_structured_text_parity.py` |
| 抓取与安全 | `tests/unit/test_fetch_*.py`, `test_remote_consent.py`, `test_vision_consent.py`, `test_security.py`, `tests/unit/serve/test_serve_security.py` |
| 网页抽取保真 | `tests/unit/webextract/`, `tests/integration/test_defuddle_parity*.py`, `tests/fixtures/web/`, `benchmarks/local_fixtures/` |
| LLM 可靠性 | `tests/unit/test_llm_request_budget.py`, `test_cost_circuit_breaker.py`, `test_url_llm_failure_paths.py`, `test_llm_batch_api.py` |
| 性能/质量基线 | `scripts/benchmarks/compare_cli.py`, `audit_performance.py`, `audit_paired.py`, `packages/markitai/benchmarks/docs_snapshots/expected/`, `benchmarks/guardrails.json` |

建议迁移验收顺序：

1. P0：真实 CLI 可执行文件、帮助/版本、配置优先级、纯文本/结构化文本、输出目标/原子写入/冲突、JSON envelope、Python 既存 API 与 Node/Go 最小调用。以隔离环境构造新的测试，建立功能支持表。
2. P1：静态 HTML/URL、本地 Office/EPUB/邮件、目录与 `.urls` 批量、glob/恢复/报告、profile、图像资产；对同一夹具执行 Python/Rust 配对测试。
3. P2：PDF 高保真、OCR/截图、浏览器获取、多供应商 LLM、成本/请求预算、缓存及 Batch API；将需要大型运行时的能力和单 binary 交付界限写入 ADR。
4. P3：auth/init/doctor 完整体验、serve/MCP、跨平台发行签名、三类 bindings 制品；关闭所有兼容缺口才可宣告整体迁移完成。

每个里程碑同时记录冷启动、转换墙钟、峰值 RSS、并发吞吐、压缩/未压缩制品大小及质量对比。不能用少做转换、跳过图片、截断正文或仅比较热缓存来证明 Rust 性能收益。逐项标记支持/部分支持/未实现；外部服务响应时间须与核心提取耗时分开。

## 8. 尚待基线实验的边界

Rust 下载 PDF 的媒体处理是明确的能力扩展：参考 URL 转换器未向 PDF
转换器传递 OCR/截图配置。静态/自动抓取识别 PDF 后可复用下载字节进入本地
逐页处理，原 URL 身份、命名及 URL `pure` 优先级保持。PDF 的仅截图输出
保留 Markdown 和页图引用；普通网页仍采用只保存截图的语义。显式浏览器或
远程策略不隐式替换。未知 URL 的仅截图内存请求需要先判别 HTTP 内容，若为
网页则在启动浏览器前报缺少输出目录；网络错误可能先于该目录错误返回。

- CLI 中断进程退出码、各非 ASCII/窄终端场景。
- 参考 API Config 对象被下游直接使用到何种程度；现有文档的“provisional/0.x”文字过时，不能据此忽略 1.2.0 兼容性。
- 全格式语义和字节差异；当前清单确认可见注册与覆盖位置，不证明每个可选后端在本机可用。
- serve/OpenAPI 和 MCP 的完整逐字段验收需单独展开。当前列出其契约位置和范围，不声称完成 HTTP 全面审计。
- 旧 SQLite 缓存、批量状态和运行历史是否原位兼容，还是需要显式可回退迁移；开发默认使用 Rust 独立状态目录，不能改写用户旧状态来试验。
