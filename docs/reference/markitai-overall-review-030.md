我看到的严重问题（按风险/影响排序）
- 依赖不可复现（高风险）：项目使用 uv workspace，但 uv.lock 被忽略且未纳入版本控制（.gitignore:32，且当前 git status 显示 !! uv.lock）。同时 Python 依赖大量使用 >=（packages/markitai/pyproject.toml:6-20），这会导致不同机器/不同时间安装出不同依赖树，线上/CI 很容易“昨天能跑今天炸”，也增加供应链风险。
- 质量门禁缺失（高风险）：仓库只有网站部署工作流（.github/workflows/deploy-website.yml），没有 Python 的单测/类型检查/lint 的 CI 强制；本地虽有 pre-commit（.pre-commit-config.yaml），但缺少 CI 兜底会导致回归直接进 main。
- 工具配置可能“不生效/不一致”（高风险）：ruff/pyright 配置在 packages/markitai/pyproject.toml，但 pre-commit 从仓库根运行（.pre-commit-config.yaml），根 pyproject.toml 没有 [tool.ruff]/[tool.pyright]（pyproject.toml:1-30）。这类布局很容易出现“本地/CI/IDE 用的不是同一套规则”，导致误报漏报。
- License 与发布元数据不完整（高风险，尤其开源/分发场景）：README 写 MIT（README.md:81-83），但仓库未发现 LICENSE 文件；同时包元数据里也没声明 license/readme 等（packages/markitai/pyproject.toml）。
- 文档里大量 “sk-xxxx/sk-1234” 形态示例（中-高风险）：docs/reference/litellm*.md、docs/spec.md 等包含大量 sk-...（示例）。这很容易触发 GitHub Secret Scanning/企业 DLP 的误报，造成告警噪音甚至阻断流程。
- 跨平台与稳定性隐患（中风险）：website/package.json 用 cp（website/package.json:7-9），在 Windows 本地开发会直接失败；同时 VitePress 依赖是 ^2.0.0-alpha.15（website/package.json:12）且 CI 安装未使用 --frozen-lockfile（.github/workflows/deploy-website.yml:41-47），存在“alpha 版本 + 非冻结安装”导致构建漂移的风险。
- 生态兼容性风险（中风险）：Python 版本被锁死为 ==3.13.*（pyproject.toml:4、packages/markitai/pyproject.toml:5），会显著缩小可用用户面，也可能踩到部分三方库/平台 wheels 的兼容坑（尤其 OCR/Office/PDF 这类重依赖栈）。

对 .llm.md 最终输出影响最大的高风险逻辑问题
- 文档清理/元数据“合并调用”会截断输入，直接导致 .llm.md 丢内容：_process_document_combined() 构造 prompt 时把全文做了 self._smart_truncate(markdown, 8000)（packages/markitai/src/markitai/llm.py:3861，实现见 packages/markitai/src/markitai/llm.py:1723）。这个截断是“保留开头”的硬截断，不是分段/合并，所以只要正文超过 ~8000 字符，就可能生成“看起来成功但后半段全没了”的 .llm.md。
- Base64 图片“过滤/去重”会打乱替换序列，导致图片引用错位（严重污染后续 LLM 增强）：base64 图片先在 process_and_save() 里按 idx 遍历，但遇到过滤/去重直接 continue（packages/markitai/src/markitai/image.py:543-547），而替换时 replace_base64_with_paths() 用 iter(saved_images) 顺序消费（packages/markitai/src/markitai/image.py:249-259）。只要出现：
  - 小图标被过滤（默认阈值 50x50 / 5000，packages/markitai/src/markitai/constants.py:69-71），或
  - 重复 logo 被去重（默认 deduplicate=True，packages/markitai/src/markitai/config.py:164）
  就会出现“第二张图引用到第三张图文件”“后面图片整体错位/残留 base64”的结果（发生点：packages/markitai/src/markitai/workflow/core.py:233-287）。
- _unprotect_content() 有“短 slide 自动塞图”的启发式，存在误判篡改内容风险：当 slide 段落很短（<10 字符）且无标题/无图片时，会从 protected["images"] 抽一张图强行替换该 slide 内容（packages/markitai/src/markitai/llm.py:1950-1996）。像 PPT 里只有 “Agenda”“Thanks” 这种短文本页，可能被错误替换成某张图片，属于“静默破坏内容结构”的高危逻辑。
- .llm.md 的 alt 文本回填存在竞态/超时导致不一致：图片分析并行跑时会轮询等待 .llm.md 出现，最多 120s，超时就直接放弃更新（packages/markitai/src/markitai/workflow/single.py:282-305）。当文档 LLM 清理慢、重试多或卡住时，.llm.md 可能缺少 alt 更新（你以为开了 --alt，但最终文件没更新）。
- 截图清理规则可能误删真实内容：format_llm_output() 会调用 _remove_uncommented_screenshots() 清理 screenshots/ 引用（packages/markitai/src/markitai/llm.py:3970-3979），如果用户原文/模板里本来就有合法的 screenshots/... 图片引用，可能被当成“页面截图”误删（这类误删很难被测试覆盖到，属于规则过宽的风险）。

主要性能瓶颈（对吞吐/延迟影响最大）
- 在 async 流程里做同步图片处理，容易阻塞事件循环：process_embedded_images() 是 async，但普通分支直接调用同步 image_processor.process_and_save()（packages/markitai/src/markitai/workflow/core.py:264），遇到图片多/大时会卡住整体并发。
- SQLite cache 每次操作新建连接 + set 时全表 SUM 统计：SQLiteCache._get_connection() 每次 get/set 都开新连接（packages/markitai/src/markitai/llm.py:327-335），set() 还会先 SUM(size_bytes) 再循环淘汰（packages/markitai/src/markitai/llm.py:414-431）。在“多图 + 批量”场景下 cache 读写非常频繁，会成为明显热点。
- PDF 截图渲染为线程安全每页重复 open 文档：每个线程内 pymupdf.open(input_path)（packages/markitai/src/markitai/converter/pdf.py:604），大 PDF 会有显著 I/O+CPU 开销（可理解，但就是瓶颈）。
- LibreOffice 子进程被用于 EMF/WMF 转 PNG：每张图最多 30s 超时（packages/markitai/src/markitai/image.py:153-194），一旦文档里有这类矢量图，会出现非常不稳定的长尾延迟。
- I/O 并发控制存在“形同虚设”的实现：LLMProcessor.io_semaphore 在无 runtime 时每次访问都返回一个新的 Semaphore（packages/markitai/src/markitai/llm.py:1107-1115），如果有代码路径依赖它做限流，会出现实际不生效的并发飙升风险（进而放大磁盘/内存/网络抖动）。
