# 更新日志

本项目的所有重要变更都会记录在此文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)，
版本号遵循[语义化版本](https://semver.org/spec/v2.0.0.html)规范。

## [1.3.0]

### 新增

- **能识别存在却读错的 PDF 文字层。** 以前只能识别辅音乱码；现在还能识别把字母映射到私用区、替换字符或 `(cid:NN)` 编码的字体，被当成 Windows-1252 读出的 UTF-8（`cafÃ©`），以及按相反顺序存放的从右到左的希伯来文或阿拉伯文。这些页面并入扫描页提示（`[PDF] plan.pdf: 2 of 40 pages look scanned/garbled (pages 7–8); consider re-running with --ocr`），`--ocr` 会对它们做 OCR，而不是提取文字层。在 `--ocr` 下，还会把每种字体的几小段文字裁出来交给识别器读，揪出编码代表的字母和画出来的不一样的旧式字体（用拉丁字母编码的中日韩字形、整体移位的字母表）：每个文档每种字体两段，每页最多四段。页面所用的文字是配置的 `ocr.lang` 读不了的，就保留文字层，并提示哪个 `ocr.lang` 能读（`… right-to-left text in reverse order; consider re-running with --ocr and ocr.lang set to ar`）。中文、日文、英文、代码、表格和公式页面不受影响，Markdown 不变。这种检查文字层的思路来自 [Indra OCR](https://github.com/datalab-to/indra)，代码由 markitai 自己实现。
- **PDF 每页都说明文字来自哪里。** `page_sources` 把各页列成 `{"page", "source", "reason"}`：`source` 是 `text`（文字层）、`ocr` 或 `mixed`（文字层加上对其中图片的 OCR，或在 `--ocr --llm` 下视觉模型对照文字层读页面图片），`reason` 是一句简短说明，比如 `text layer trusted`、`no text layer`、`text layer failed the script check: …` 或 `OCR forced by --ocr (ocr.per_page_routing is off)`。它出现在 `--json` 条目、Python API 的 `ConversionOutput`、记录的历史和 `markitai serve` 的条目里（workbench 会忽略这个新增字段）；`--verbose` 会在日志里列出没有被直接信任的页面。

### 变更

- **长 PDF 用上所有核提取，也不再拖住同时转换的其他 PDF。** 提取用的工作进程（CLI、`markitai serve`、MCP）原来是 CPU 核数减二：4 核的机器只用两个核提取，主进程只是等待工作进程，另外两个核就空着。现在每个本进程可用的核一个工作进程（taskset 和容器的 CPU 集合都算在内），仍然最多 12 个，合计不超过可用内存的四分之一，所以 14 核及以上的机器不变。文档以每任务四页交给工作进程，同一时刻每个工作进程一个任务（外加一个排队），原来是每个工作进程一整段、一次全部提交：同时转换的 PDF（例如工作台的第二个任务、批处理的下一个文件）只需等一个任务，而不是等整篇文档。页面截图（`--screenshot`，以及 `--ocr --llm` 发给模型的页面）在工作进程运行时由它们渲染；PyMuPDF 渲染时持有 Python 的 GIL，原来在主进程里渲染的多个线程只能轮流进行。输出逐字节相同。在 4 核 Linux 机器上（409 页的文字 PDF，冷启动 CLI）：53–55 秒 → 29–32 秒，CPU 时间不变，约 100 秒，峰值内存 0.8 GB → 1.3 GB（四个工作进程，每个约 380 MB）。在它开始 3 秒后启动的 5 页 PDF 原来 55–65 秒后才完成，现在 1.4–4.7 秒。166 页、带图片并开启 `--screenshot` 的 PDF 在主进程里只花 1.0 秒 CPU，原来是 6.3 秒。

### 修复

- **OpenDocument 文本保留格式、脚注、链接和图片，只输出真正的正文。** `.odt` 保留粗体和斜体，编号列表输出为编号列表（原本全部变成项目符号），图片像 DOCX 的图片一样保存为条目的资源文件（原本被丢掉）。脚注和尾注变成 `[^n]` 引用，内容放在文末（原本夹在句子中间）；链接保留地址；按大纲编号的标题仍是标题；批注和修订中被删除的文字不再输出（原本被当成正文读出，`.ods` 单元格里也是如此）。
- **Excel 单元格显示的内容与 Excel 一致。** 现在会应用单元格的数字格式：`0.00%` 格式下的 `0.256` 显示为 `25.60%`，固定小数位、千位分隔符、货币、科学计数法和文本节都按 Excel 的方式显示（四舍五入，`5.255` → `5.26`）。日期和时间保持类 ISO 的写法，只显示格式中写出的部分（`m/d/yy h:mm` 显示为 `2024-03-05 14:30`，`h:mm` 显示为 `09:05`），经过时间显示为时长（`27:03:00`），用英文月份名并带四位年份的格式按格式原样输出（`March 5, 2024`）。分数和特定区域的格式仍显示原始数值。`.xlsx` 和 `.xls` 都是如此，与 markitai-rust 的输出一致。
- **隐藏的工作表会被标出，空工作表不再输出。** 隐藏工作表的标题现在注明 `(hidden sheet)`，没有内容的工作表不再留下一个光秃秃的标题，与 markitai-rust 一致。
- **RTF 的脚注和尾注得以保留。** 它们原本被丢掉；现在每条都在标记处变成 `[^n]` 引用，内容放在文末，Word、LibreOffice 和 pandoc 生成的文件都是如此。
- **Word 和 LibreOffice 生成的 RTF 列表不再重复编号。** 这些程序为不支持列表的阅读器保留的编号文字（`\listtext`）原本被当成正文读出，列表项变成 `1. 1.    First` 和 `- ·    One`。编号子列表现在缩进到上级列表项的文字处，Markdown 阅读器才会把它当作嵌套列表；按大纲编号的标题仍是标题。
- **幻灯片中的每个形状各自成为一个 Markdown 块。** PPTX 幻灯片上紧跟在表格后的图片或文本框原本粘在表格最后一行上，被读成表格多出来的一行。
- **带 UTF-8 BOM 的 CSV 文件**（Excel 的“CSV UTF-8”）不再把 BOM 放在第一个列名前面。
- **本地 HTML 文件所在文件夹的名字含 `#` 或 `%` 时，相对路径的图片和链接不再失效。** 它们的基准地址原本是手工拼出来的，遇到 `#` 就断了。
- **本地 HTML 文件中的图片保存为资源文件。** 它们原本保留绝对的 `file://` 地址，Markdown 换个地方读就失效。现在页面所在文件夹内的图片像 DOCX 图片一样随条目的资源文件保存；文件夹以外的图片（经 `..`、绝对路径或符号链接）、超过 32 MiB 的图片，以及内容不是 PNG、JPEG、GIF、WebP、BMP 或 TIFF 的文件保留原引用，也不会被读取。
- **目录批量转换跳过隐藏文件夹、`node_modules` 和 Office 锁文件。** 原本会扫描 `.git`、`.venv` 等所有点目录（虚拟环境里的 `.html` 或 `.md` 也会被转换），点文件和 `~$report.docx` 锁文件也一样。现在与 markitai-rust 一致，除非 glob 写出了这类名字（`-g '.notes/**'`），否则一律跳过；CLI 指南已写明。
- **批量目录里的符号链接有自己的续跑记录。** 批量状态原本按解析后的真实路径记文档，`link.docx -> report.docx` 与目标共用一条记录：状态里只记下两份输出中的一份，`--resume` 把两份文档算成一份（失败后也只重做一份）。现在每个名字各有一条记录。
- **用户令牌用浏览器抓取私有地址时说明拒绝原因。** 在 `markitai serve` 中，用户令牌提交私有或本机地址（`http://127.0.0.1:…`、局域网主机）的 URL 时，`playwright` 策略只报 Chromium 的 `net::ERR_BLOCKED_BY_CLIENT`（错误码为 `conversion_error`），而 `static` 会说 `Fetch target refused: private_or_local_host`。现在浏览器路径给出同样的原因，错误码为 `fetch_error`。
- **MCP `job_status` 的计数与 `markitai serve` 一致。** `done` 原本把失败的条目也算进去，三项中一项失败时显示 `done: 3, failed: 1`。现在 `done` 只计成功的条目（`done: 2, failed: 1`），新增的 `finished`（`done` + `failed`）给出 `done` 原来的数，可用于显示进度。MCP 指南已列出这些计数。
- **目录批量转换时，`--glob` 按符号链接自己的名字匹配**，而不是按它指向的文件名。
- **UTF-8 文本网址不再变成乱码。** `.md`、`.txt`、`.csv`、`.json` 地址先按 `Content-Type` 解码，之后又根据开头的字节重新猜一次编码，于是较短或以 ASCII 为主的文件会被读成 Windows-1252 或 GB18030（raw.githubusercontent.com 上 defuddle 的 README 变成 `deÂ·â€‹fudÂ·dle`）。
- **不经浏览器抓到的人机验证页面不再被当成网页内容。** PyPI 的 “Client Challenge” 转换后只剩一句普通的话（“A required part of this site couldn't load …”），`auto` 把它当作内容返回。现在会检查抓到的 HTML 是否是整页拦截页（Cloudflare、DataDome、PerimeterX、Incapsula、AWS WAF、Fastly）或含有验证用语：`auto` 改用浏览器，`--strategy static` 报告遇到了验证页。表单页上的 reCAPTCHA 或 Turnstile 不算拦截页。
- **网页正文提取保留数据表格的表头行和列表项内嵌套的列表。** `<tr class="header">`（pandoc 就这样写）原本被当成页面装饰删掉，表格因此没有表头；列表项内嵌套的短列表也可能被当成元数据列表丢掉。
- **浏览器启动期间按 Ctrl-C 不再让 `markitai serve` 卡住**（任何正在用 Playwright 抓取的运行也是如此）；以前需要再强制结束一次。
- **只有已安装的 Playwright 实际会启动的 Chromium 版本才算已安装。** `markitai doctor`、`serve` 和抓取流程原本把任何 `chromium-*` 文件夹都当作已安装，于是其他 Playwright 版本留下的浏览器能通过检查，启动时却失败。
- **`{"operation": "enhance"}` 可以增强未经 LLM 转换的条目。** `POST /api/jobs/{job_id}/items/{item_id}/retry` 原本沿用条目保存的选项，以 `minimal` 预设或 `"llm": false` 转换的条目会得到 409 `llm_unavailable`；现在除非请求的选项写明 `"llm": false`，enhance 都会开启 LLM。

## [1.2.1] - 2026-10-18

### 新增

- **新的 `markitai serve` 工作台。** 网页界面换成了 markitai Rust 实现同样使用的工作台，此后两者共用一套界面。侧栏有**转换**、**转换任务**和**系统设置**（模型、外观、访问令牌；手机上改为底部栏，系统设置是一个包含文档和 GitHub 的列表）。文件和文件夹先暂存并显示上传进度，按下**转换**才提交（手机上选项、上传和文件夹是紧凑的图标），可以和粘贴的 URL 合成一次提交。**转换任务**是一个列出所有条目的列表，按提交分组，以来源文件夹或 `.urls` 列表命名，可按名称和状态筛选，**新建转换**按钮回到转换页。名称太长放不下时，从中间按文件夹边界省略，保留主机名和扩展名（`/slow/annual-reports/2026/…/s5.html`），完整名称在提示中。点开一个转换会在列表上方居中的窗口中打开（点窗口的放大按钮或按 F 可全屏并还原，由各浏览器记住选择；手机上始终全屏），标签页排成一行，↑/↓ 切换到上一个或下一个条目：**详情**与预览使用同一个居中的阅读栏，显示条目的状态、原件与输出大小、模型、费用、选项、提示、保留期限和结果开头。**预览**（它的视图：渲染、源码、差异、文件，以及 Base | LLM 位于标签行末尾，放不下时合成一个紧凑的切换按钮，手机上就是如此）会绘制 Markdown 中内嵌的 SVG 图形（经过清理，其中的内容不会运行脚本，也不会加载任何资源）；**对照**会把原件放在 Markdown 旁边：PDF 页面（鼠标指向的标题或表格会在页面上框出）、图片、HTML 和文本文档，或转换时保存的页面截图。网页的“对照”显示服务器抓取到的页面本身，放在沙箱框架中：带有网站自己的图片、样式表和字体（由浏览器经 HTTPS 从网站加载），不运行它的任何脚本，链接在新标签页打开；保存了截图时可与截图切换，标题栏有“打开网页”。标题栏的“下载”会打开预览的“文件”：先列出原件（标注“原件”，网页标注“抓取的页面”），再列出每个结果文件，各自可下载，并显示总大小和“全部下载（.zip）”，压缩包包含结果文件和原件。提示是窗口底部的单行提示条（手机上位于底部栏之上），会自行消失；对单个转换的操作（重试、删除、下载等）失败时，原因显示在该转换的详情里；持续存在的情况（未使用模型完成的转换、未能保存的历史、缺少访问令牌）显示在相关位置，不再以横跨页面的横幅出现。**添加模型**打开服务商选择器：一个搜索框覆盖服务商名称和模型前缀（输入 `ollama_chat` 可找到 Ollama），常用服务商以带品牌标志的卡片列出，其余按分组筛选（云平台、推理服务、网关、本地、订阅、模型厂商）。选择器列出 markitai 能经已安装的 LiteLLM 调用的所有服务商，约 90 个，原来只有 8 个：Mistral、Groq、Together AI、Fireworks、Amazon Bedrock、Hugging Face、LM Studio、vLLM 等，各自标明所用地址和 LiteLLM 读取的 key 变量；服务商公开模型列表的从列表中挑选，不公开的手动输入模型 ID。LiteLLM 自带的默认地址过时或缺失时，新建的连接会保存服务商文档给出的地址：否则 `ai21/`、`lm_studio/`、`xinference/` 模型会把 key 发到 api.openai.com，Perplexity 模型会发往 Perplexity 宣布于 2026-09-27 停用的 Sonar chat completions 接口（现在改用它的 Router API）。GradientAI 和 Morph 不在其中：LiteLLM 会拒绝 markitai 发给它们的请求。每个服务商只列一次：服务器运行环境里设置了密钥变量的任何服务商（不只 OpenAI、Anthropic、Gemini、DeepSeek 和 OpenRouter）都会出现在“已保存”中，并合并进已经使用该变量的连接；每个已保存的行说明它如何认证（`使用 DEEPSEEK_API_KEY`、`已保存密钥`、`无密钥`），常用服务商卡片会标出已添加的服务商，密钥输入框会提示可以改用哪个环境变量。“设置 → 访问令牌”可以重新生成令牌：旧令牌立即失效（在其他所有标签页和设备上，包括本机，连同尚未使用的下载链接），新令牌只显示一次，附带登录链接和在页面内绘制的二维码。预览、对照和详情里的结果开头会显示转换自身提取出的图片（原来显示 "[Image: preview unavailable]"），图片经一个接口读取，它只提供该转换列出的位图，以内联、沙盒方式返回。界面在中英文之外新增日文。大结果依然流畅：两万行表格的预览约 0.15 秒显示前 500 行（**显示全部行**加载其余），而 1.2 的界面要 3 秒，期间页面卡住 2.5 秒，并生成 16 万个页面元素。长文档分段绘制，只排版屏幕附近的部分，切换标签页时保留已绘制的内容：一个 460 KB、含 160 张图片的 EPUB 结果在手机级 CPU 速度下不到 0.1 秒显示开头，页面不会卡住；差异视图能比较这么大的文档，并折叠未改动的段落。有四个及以上标题的文档在旁边显示目录（手机上用按钮以底部面板打开），标出正在阅读的章节。页面首次绘制约 0.1 秒，原来约 0.2 秒。预览里的公式（`$…$`、`$$…$$`）用 KaTeX 绘制，只在文档含公式时才加载。**重新转换**（选择栏和单个转换的 ⋯ 菜单）会把保留的原件按你选的选项再转换一次，结果成为新批次。版本号只占一行，通知的文件列表前带数量，警告徽标用提示框显示通知，转换窗口可用返回键或 Esc 关闭，手机顶栏有各个按钮，编辑配置在 JSON 检查下显示问题行，通知标题译成中文和日文。模型页会说明订阅类服务商（Claude、Copilot、ChatGPT）为什么没有模型列表：缺少程序、版本过旧，或尚未登录。
- **停止剩余**：正在运行的提交里尚未开始的条目可以停下；已在转换的条目照常完成，被停下的条目之后可以重试（`POST /api/jobs/{job_id}/cancel`）。在手机上，它一直是盖住列表的全屏窗口（一次提交、一个等待中的条目或“汇总”）标题栏里的按钮；在**转换任务**里，它单独占标题和状态下面的一行。
- **网页界面支持 `.urls` 列表**：上传的 `.urls` 文件会展开为 URL 条目，读法与命令行一致（每行一个 URL，可附输出名，`#` 注释，或 JSON 列表）。此前 `markitai serve` 会把它当作不支持的格式拒绝。
- 为工作台新增的 serve API：条目原件（`GET /api/jobs/{job_id}/items/{item_id}/original`、它的 `/file`、用 PyMuPDF 和 Pillow 渲染的页面图片 `/pages/{n}?width=`，以及 PDF 页面带位置的文字行 `/pages/{n}/text`）、一次性下载票据（`POST /api/download-tickets`）、`GET /api/history?include=items`、结果节选（`?max_bytes=`）、单个条目连同原件的 ZIP（`GET /api/jobs/{job_id}/items/{item_id}/archive`）、供工作台沙箱框架显示的已保存网页（`…/original/view`；URL 条目现在会保存转换所用的页面，最多 10 MiB，仅限 markitai 自己抓取的页面，Jina、Defuddle 和抓取缓存命中不保存）、`POST /api/access-token/regenerate`、任务 `label`，以及每个条目上的 `error_code`、`pricing`、`models`、`source_size` 和保存的 `options`。`GET /api/capabilities` 会列出服务实现的 `features`、`docs_url` 与 `source_url`、历史保留期限，以及能否使用 Cloudflare。浏览器打开工作台没有的地址时，会看到它的「页面不存在」视图（状态码 404），而不是一段 JSON 错误。
- **`markitai serve` 可以一次删除多个转换，新增用户令牌和“编辑配置”**。**转换**列表的每一行和每个提交的标题左侧都有一个勾选框（空格勾选当前行，Shift 扩展选择，Ctrl/⌘+A 勾选所有列出的行，Esc 清除；手机上用**选择**模式），底部栏显示数量、**全选**和**删除**，确认一次即可；服务端用一个请求删除它们。**设置 → 访问令牌**可以按名称创建用户令牌：令牌只显示一次，附带复制、登录链接和二维码；列表按末尾几位字符显示，并显示最近使用时间和**撤销**。用户令牌可以转换文件和 URL，但只能查看、下载、停止、重试和删除自己的转换；模型、访问令牌和编辑配置需要管理员令牌（服务器自己的令牌，或像以前一样本机不带令牌访问），管理员能看到所有人的转换，包括已撤销用户的。撤销会立即让该令牌失效，连同它的下载链接和实时进度，本机也一样。每个用户令牌只保存 SHA-256 摘要，存放在 `~/.markitai/serve/access.json`（仅所有者可读写，从不写入配置文件），因此重启和重新生成管理员令牌后依然有效；服务器日志只记令牌名称，从不记录令牌本身。**设置 → 编辑配置**以文本形式显示配置文件，其中的密钥按 `markitai config show` 的脱敏规则遮盖（`sk…7890`；`env:` 引用照原样显示），输入时检查 JSON（报告行和列），以逐行差异查看修改，并原样保存这段文本，保留排版、键的顺序和换行符（Windows 上写的文件仍是 CRLF，没有末尾换行的仍然没有）；留在原处的遮盖值会保留已保存的密钥，期间被改动过的文件会被拒绝，并提供**加载当前文件**。模型和连接立即生效；其他改动需要重启时页面会说明。网络图片（Markdown 中的绝对 `https://` 地址）现在会在预览、对照和详情节选中显示，不再是占位符；对照视图会把网页与 Markdown 逐块关联：在任一侧悬停标题、段落、列表项、表格或图片，两侧都会描边。
- 工作台契约 v1.13 的 serve API：`POST /api/history/delete`（`{jobs?, items?}`，共 1 到 1000 项，每项按对应的单个 `DELETE` 路由处理，应答列出每项结果）、`GET`/`POST /api/access-tokens` 和 `DELETE /api/access-tokens/{id}`（仅管理员）、`GET`/`PUT /api/settings/config`（`{expected_revision, text}`；409 `stale_revision` 或 `config_changed`，422 `invalid_config` 附行列号或校验器的说明），以及 `GET /api/capabilities` 中调用者的 `access`（`admin` 或 `user`）和功能 `batch_delete`、`user_tokens`、`config_edit`。用户令牌访问设置和令牌路由会得到 403 `settings_forbidden`，访问不是自己创建的转换会得到 404 `job_not_found`；历史、历史归档、批量删除和下载票据都只限于自己的转换。
- **对照视图知道网页抓取到的是什么，并能逐块关联截图**（契约 v1.14）。网址条目现在记录保留的网页和截图在抓取时是否可用：是网页内容、空白页（空白，或只要求启用 JavaScript 的页面），还是代替内容显示的验证、验证码、机器人检查或登录页面（与 Rust 相同的规则，外加 Python 自己的标记）。对照视图只提供显示了网页的视图；两者都不可用时说明原因（"抓取时网站显示的是验证页面，而不是网页本身……"、"抓取时网页是空白的……"）并提供打开原网页。浏览器截图前会先用工作台附带的脚本（`capture/page-blocks.js`，原样运行）记录网页的标题、段落、列表项、表格和图片的绘制位置，因此在截图上或 Markdown 中悬停某块会框出另一侧的对应部分。浏览器为提取正文加载网页而未要求截图时，也会保留对照截图（JPEG 质量 70，按截图分块高度切块，最多 20 块、5000 万像素；文字页面每 2000 px 一块约 130–160 kB）：对照视图显示它们，它们不是输出，不进入结果或压缩包，也不会发送给模型或 OCR。显示图片或媒体播放器的短网页算作内容，不算空白页（只有一行说明文字的图片帖子会正常显示）；跟踪像素、隐藏的图片或占位图片不算。重试会替换它们，删除条目或其转换会删除它们；没有加载网页的重试，或浏览器看到验证页、空白页的重试，会保留之前的文件。
- 工作台契约 v1.14 的 serve API：原件描述的 `capture`（`{page, screenshot}`，每项为 `{state, text_length, title, reason}` 或 null；上传文件为 null）和 `blocks`、`GET /api/jobs/{job_id}/items/{item_id}/original/blocks`（区块框；没有时 404 `original_unavailable`），以及作为网址条目 `pages` 的对照截图。它们与网页一起保存在任务的 `snapshots/` 文件夹中：`<item>.capture.json`、`<item>.blocks.json` 和 `<item>.shots/<n>.jpg`，仅所有者可读；之前记录的转换在请求时评估其保留的网页。
- **`markitai serve` 支持用户自己的模型和管理员共享的模型**（契约 v1.15）。用户令牌的**设置 → 模型**现在就是管理员的模型页面，用于管理自己的模型：用服务商选择器添加模型、测试、编辑、删除和设置权重，下方是只读的**管理员共享**（模型、服务商、权重）。管理员保存的每个模型都有**私有 | 共享**开关（默认私有，在配置文件中保存为 `"model_info": {"share": "shared"}`，因此编辑配置里能看到）。用户的转换使用一个模型池：先是自己的模型，再是共享的模型，按权重选择，请求失败时换用池中另一个模型；提示词、预算和缓存仍是管理员的设置，任务会显示用了哪个模型及其费用。两者都没有的用户看不到 LLM 选项（“在设置 → 模型中添加模型……”），请求 LLM 会被拒绝；服务器的环境变量密钥和自动检测到的模型从不为用户服务。共享模型的密钥和地址从不离开服务器：用户只看到它是什么模型，看不到它在哪里提供服务。用户的模型只属于自己：保存在 `~/.markitai/serve/user-models/<令牌 id>.json`（仅所有者可读写，从不写入配置文件），密钥只以遮盖形式显示；只接受带用户自己密钥的模型，或不需要密钥的服务器地址，不接受用服务器自身账号登录的服务商（订阅、Vertex AI、watsonx、GigaChat、SageMaker、不带密钥的 Bedrock），也不接受指向服务器本机的地址。**访问令牌**列表显示每个用户有几个模型，撤销令牌会连同密钥删除这些模型。
- 工作台契约 v1.15 的 serve API：供用户令牌使用的 `/api/me/models*`（其他调用者得到 403 `user_token_required`）：`GET /api/me/models`（`UserModelsView`：用户自己的部署、`shared` 为 `{id, model, provider, weight}`、`limit` 为 50）、`GET …/providers`、`GET …/providers/{id}/credentials`、`POST …/model-discovery`、`POST …/test`、`POST …/deployments/batch`、`PATCH`/`DELETE …/deployments/{id}` 和 `…/providers/{id}`，请求体、版本规则和状态码与设置路由相同，在保存或发送任何内容之前以 422 `server_credentials`、`key_or_address_required`、`server_address` 或 409 `model_limit` 拒绝；`PATCH /api/settings/llm/deployments/{id}` 接受 `share: "private" | "shared"`，每个已保存的部署都带 `share`；`GET /api/access-tokens` 增加 `model_count`；`GET /api/capabilities` 增加 `features.user_models`，对用户而言 `llm` 描述其模型池。
- **用户可以为管理员共享的模型设置自己的权重并测试，也能以文本形式编辑自己的模型**（契约 v1.16）。**管理员共享的模型**下的每一行有 **测试** 和一个属于该用户自己的权重：点击权重、输入数字，按 Enter（或离开输入框）保存，Esc 取消。这个权重只作用于该用户自己的转换；设为 0 表示该用户不再使用这个模型（"Off for your conversions (weight 0). Other users are not affected."），**Use the admin's N** 恢复跟随管理员的权重。测试在服务器上用管理员的设置进行，只告诉用户是否通过以及未通过的原因，绝不透露服务商的原话、地址或密钥；每个用户同一时间只能测试一个共享模型，每分钟最多六次（费用由管理员的密钥承担）。模型页上管理员和用户自己的模型权重也可以这样直接修改。用户令牌获得 **设置 → 编辑配置**，只针对它自己的模型：一份包含其模型、连接和共享模型权重的 JSON 文档，密钥以掩码显示，校验、审阅和冲突处理与管理员的页面相同；服务器的配置文件绝不会向用户显示，也不会被用户修改。侧边栏（以及手机上的设置列表）显示当前应答的构建：`1.2.1 1a2b3c4 10-15 09:30`，即版本、提交、构建含未提交改动时的 `-dirty`，以及按查看者所在时区显示的构建时间（源码检出没有构建时间）。以带链接的封面图写出的视频（`[![…](poster)](https://…/clip.mp4)`）可以在预览和对照中播放，播放前不加载任何内容，也不发送来源信息。警告中的页码以区间列出；不少于九成页面没有可读文字的 PDF 会提示 "This PDF is scanned: turn on OCR and convert again"，被分页校验搁置的模型回答也会用平实的话说明原因。被行分隔符或段分隔符打断的表格重新显示为表格。
- 工作台契约 v1.16 的 serve API：`PATCH /api/me/models/shared/{id}`（`{expected_revision, weight}`，`weight` 为整数或 null；不再共享的模型返回 404 `settings_entry_missing`，版本过期返回 409 `stale_revision`；保存在用户自己的文件中，字段为 `shared_weights`），`POST /api/me/models/test` 只带 `{shared_id}`（返回 `{ok, detail}` 或 `{ok: false, reason, detail}`；超出限制时返回 429 `provider_busy` 和 `Retry-After`），`GET`/`PUT /api/me/config`（`ConfigDocument`，其中 `path: null`、`origin: "user"`；文本不符合文档结构时返回 422 `invalid_config`，用户不能保存的内容按模型路由各自的拒绝原因返回，`restart_required` 恒为 false），每个 `SharedModel` 的 `default_weight`，以及 `GET /api/capabilities` 中对所有调用者返回的 `build: {version, commit, built_at, dirty}` 和 `features.user_config`。wheel 和 sdist 带有 `markitai/_build_info.py`（短提交号、构建时间或 `SOURCE_DATE_EPOCH`、被跟踪文件是否有改动），由 `hatch_build.py` 中的构建钩子写入，绝不写进源码树；源码检出则读取自己的 git 状态。
- **新建转换的选项从服务器默认值开始，PDF 自带书签填入 Compare 的目录**（契约 v1.17）。新转换的选项（模型处理、OCR、替代文字、图片描述、截图、仅截图、纯净模式、不用缓存、不压缩图片、输出配置、抓取策略）改为从 `config.json` 叠加内置默认值得出（命令行读取的也是这些键），而不是浏览器上次的选择。命令行上方有一行写着“服务器的默认值”、“你的默认值”或“已偏离默认值”，并提供 **恢复默认值** 和 **存为默认值**：管理员的选择写入 `config.json`（只写文件里已有的键，或与内置值不同的值；带版本检查和设置锁），同时更新运行中的配置；用户令牌的选择写入它自己的模型文件，并可用 **使用服务器的默认值** 撤销。Cloudflare 永远不会成为默认值。`GET`/`PUT /api/convert-defaults`（`features.convert_defaults`；409 `stale_revision`，422 `invalid_convert_defaults`，`--no-auth` 服务器上的匿名调用者 403）。`GET /api/jobs/{job_id}/items/{item_id}/original/outline`（`features.pdf_outline`）返回 PDF 的书签 `{items: [{title, page, level}], truncated}`（最多 2000 条，标题为一行；没有指向页面的条目指向下一条的页面；没有目录或无法读取的 PDF 返回空）。

### 变更

- **扫描版 PDF 的 OCR 用满所有核。** 一本 491 页的扫描书在工作台里用了 9 分 17 秒。它的页面以四页一组（文件超过 50 MB 时两页）在共用一个识别器的线程中识别，而每一页又把识别器铺到所有核上，这样做收益很小：四页争抢同一批核，大部分 CPU 时间都在互相等待。现在 CLI、`markitai serve` 和 MCP 服务器在 PDF 提取工作进程里识别页面，每个进程一页：每个工作进程只加载一次 OCR 模型，并在分给它的核上运行（工作进程与核一样多时各用一个核）；同时进行的页数不超过工作进程数和内存允许的数量（可用内存的一半，每页约 400 MB），所以长扫描件既不会把所有页面排在其他文档前面，也不会占用超出机器容量的内存。在一本虚构的 490 页扫描书（中文夹杂英文和代码，六个空白页）上，18 核机器同时被其他任务占用（负载 18 到 42）的情况下，工作台任务用时 294 秒（之前 497 秒，含“修复”中的链接文字修复），CLI 用时 288 秒（之前 510 秒），一次 CLI 运行的 CPU 时间从 6655 秒降到约 2600 秒。 每一页的文字与之前逐字节相同（字符错误率前后都是 1.75%）。工作进程合计约占 6 GB 内存，之前为 2.1 GB。页面仍以 150 dpi 渲染：无论分辨率如何，识别器都占每页 80% 以上的时间，而 120 或 200 dpi 对这些虚构页面的识别并不更好。没有调用 `enable_worker_processes()` 的脚本最多同时识别八页（每两个核一页），之前为四页。OCR 未找到文字的页面提示改为与扫描页提示相同的写法：`PDF pages 166, 360 (2 of 491): OCR found no text; they are blank in the output.`（与 Rust 相同的写法；契约 v1.17）。
- **对照视图中框入的网页与工作台同源**（`…/original/view` 的策略为 `sandbox allow-same-origin allow-popups allow-popups-to-escape-sandbox; default-src 'none'; …`），这样工作台可以读取页面的各个块并与 Markdown 关联。页面仍然什么都不执行：没有脚本、内联事件处理器、`javascript:` 链接、meta 刷新、表单、框架或对象（由 `scripts/sandbox_probe.py` 在 Chromium 中验证）。工作台自身的页面允许 `https:` 图片（`img-src 'self' blob: https:`）和 `https:` 视频（`media-src https:`）。
- **警告中的页码以区间列出，扫描页提示给出文档总页数**：`[PDF] plan.pdf: 4 of 120 pages look scanned/garbled (pages 3–5, 9); consider re-running with --ocr`（此前逐页列出：`pages 3, 4, 5, 9`），隐藏文字、OCR 和截图提取的提示也同样列出页码。被分页校验清空的模型回答现在说明为 `the model's answer did not match the document page by page, so the page drift guard set it aside`，不再是 `LLM answer text was discarded by the page drift guard`。
- **React 网页应用退役。** `webapp/` 及其在 `markitai/serve/static/` 的构建产物已从仓库和安装包中移除；工作台在 markitai-rust 仓库构建，连同记录每个文件摘要的清单一起放在 `markitai/serve/workbench/`（`scripts/sync_workbench.py` 负责更新和校验），构建 markitai 不需要 Node 或 Rust。你会注意到：上面描述的新布局；设置是独立页面（`/settings/models` 等），不再是弹窗；转换的预览在列表上方的窗口中打开（可切换为全屏），与详情和对照放在一起；1.2 界面的 `/assets/…` 地址不再存在。wheel 从 1.51 MB 变为 1.64 MB：工作台的每个脚本同时附带原文和 gzip 压缩版，服务端直接发送压缩版，不用在请求时压缩。
- **没有可用模型时拒绝 LLM 请求**：`POST /api/jobs` 要求 LLM 处理（`llm: true`，或开启 LLM 的预设）而没有任何已配置模型可用时，现在返回 422 `llm_unavailable`，不保存任何内容。此前会不用模型直接转换并报告成功。服务端配置开启了 LLM 却没有模型时，仍和以前一样不用模型转换。
- **`serve` 中的 Cloudflare 需要逐次确认**：选用 Cloudflare 抓取策略或文件后端的任务（或重试），必须在同一请求的选项里带上新选项 `"remote_processing": "cloudflare"`；工作台会先征得你的确认，缺少时返回 422 `remote_processing_confirmation_required`。服务端不信任的调用方（没有令牌的其他机器）会被拒绝（403 `remote_processing_forbidden`），因为费用记在服务端的 Cloudflare 账户上；Cloudflare 未配置时返回 422 `cloudflare_unavailable`（服务端设置了 `MARKITAI_NO_REMOTE_FETCH=1` 或 `fetch.remote_consent: never` 时为 `remote_processing_disabled`）。确认不会被保存，所以重试这类条目要再次确认。
- `markitai serve` 的错误在 `detail`、`code` 之外带上稳定的 `reason`（`llm_unavailable`、`file_too_large`、`ticket_invalid` 等），工作台据此显示本地化的说明。
- **任务选项严格校验**：`POST /api/jobs` 和重试请求里，未知的选项名或类型不对的值会以 422 拒绝（任务为 `invalid_options`，重试为 `invalid_retry_body`），不再被忽略或自动转换；此前 `"llm": "yes"` 会被当成 true。URL 在提交时就校验（422 `invalid_url`、`unsupported_url_scheme`），不再变成失败条目；超过 120 个字符或含控制字符的标签会被拒绝（422 `invalid_label`）。
- **422 用一句话说明原因**：被请求结构拒绝的请求，`detail` 是一句逐个点名字段及其问题的说明（如 `invalid options: colour: Extra inputs are not permitted; llm: Input should be a valid boolean`，最多五项），不再是 pydantic 的对象列表，也绝不回显提交的值（它可能是 API key）。
- **超过 64 MiB 的结果不再以 JSON 返回**：`GET /api/jobs/{job_id}/items/{item_id}/result` 对超过 64 MiB 的 Markdown 结果返回 413 `result_too_large`，不再拼出这么大的响应；请改为下载文件，或用 `?max_bytes=` 取开头部分。
- 仍在等待提供商批处理（`--llm-batch`）的条目不能再从网页重试或增强（409 `batch_pending`），否则同一份增强会付两次费。
- 凭据接口在 `api_key_masked` 之外，也以工作台读取的 `api_key` 字段返回同一个打码后的 key。
- PyYAML、beautifulsoup4、soupsieve、defusedxml、PyMuPDF、openpyxl、numpy 和 olefile 改为声明的依赖。markitai 直接导入它们，此前只是随其他包间接安装，上游某次发布去掉其中一个就可能让 `import markitai` 失败。
- **DeepSeek 默认模型改为 `deepseek/deepseek-flash`**（DeepSeek V4.1-Flash），`markitai init` 和检测到 `DEEPSEEK_API_KEY` 时都用它。DeepSeek 已于 2026-09-10 下线 V4-Flash，`deepseek-v4-flash` 和 `deepseek-v4-flash-vision-exp` 只是暂时转到 V4.1-Flash。配置里仍写着它们的（1.2.0 的 `markitai init` 写入的是 `deepseek/deepseek-v4-flash`）会在启动时收到警告，并给出新的默认模型；markitai 不会改写你的配置。
- **其他提供商的默认模型更新为当前型号**：只找到某个提供商的 key 或订阅时，`markitai init` 和自动检测现在选用 OpenAI `openai/gpt-6-luna`（原为 `gpt-5.6-luna`；每百万 token $0.10/$0.50，原为 $0.20/$1.20）、Anthropic `anthropic/claude-haiku-5-5`（原为 `claude-haiku-4-5`；提示不超过 10 万 token 时 $0.10/$0.50，原为 $1/$5）、ChatGPT `chatgpt/gpt-6-luna`（原为 `gpt-5.6`）、GitHub Copilot `copilot/auto`（原为 `claude-haiku-4.5`；即 Copilot 的自动选择模型 Auto，自 2026 年 6 月起是 Copilot Free 和 Student 套餐唯一可用的选项，按它选中的模型计费）、OpenRouter `openrouter/~anthropic/claude-haiku-latest`（原为 `google/gemini-3.1-flash-lite`；即 OpenRouter 指向最新 Claude Haiku 的别名。它的目录写的是 Claude Haiku 5.5，$0.10/$0.50，但 2026-10-10 发往它的请求实际仍由 Claude Haiku 4.5 应答并按 $1/$5 计费；要固定使用更便宜的模型，请设 `openrouter/anthropic/claude-haiku-5.5`）和 Claude Code `claude-agent/haiku`（原为 `claude-agent/sonnet`；即 Claude Code 指向其当前最便宜模型的别名，Claude Code 2.1.29x 中为 Claude Haiku 5.5，也更省订阅额度）。Gemini 和 DeepSeek 的默认模型不变。配置里写着旧默认模型的照常可用，也不会收到警告，因为它们都没有被弃用。LiteLLM 1.104.2 没有 `claude-haiku-5-5` 的价格，所以在 LiteLLM 收录之前由 markitai 自带这一条：它的请求会按价格计费并计入 `max_cost_per_document_usd`，不再记为价格未知。OpenRouter 的请求按 OpenRouter 自己报告的金额计费，因为别名背后实际是哪个模型要看回答才知道，而 LiteLLM 1.104.2 没有 `anthropic/claude-haiku-5.5` 的价格。
- **`markitai doctor` 直接说明是否就绪。** 它原来以 "! Core check passed (3 required/configured checks passed, 2 non-blocking warnings)" 结尾："passed" 旁边是警告符号，而且满是术语。现在最后一行是 "✓ Ready. Not available (optional): Playwright, LibreOffice"、"✓ Ready. Everything is available." 或 "✗ Not ready: fix LLM API first (see above)"。每条修复方法都是适用于这台机器的命令，单独成行：Playwright 的浏览器用 markitai 自己的 Python 安装（`python -m playwright` 用的是 PATH 上恰好有的 `python`，而 `setup.sh` 和 `uv tool install` 把 markitai 装在别的环境里），Linux 上带 `--with-deps`；LibreOffice 用检测到的 Linux 发行版的包管理器安装，不再是一块丢了缩进的 apt、dnf、pacman 命令。提供商 key 没有设置的模型（没有 `DEEPSEEK_API_KEY` 的 `deepseek/deepseek-flash`）不再显示通过：doctor 写明缺哪个变量、运行时会略过几个模型，一个可用模型都不剩时判为失败。已设置、但没有任何已配置模型用到的 key 会被列出（`--json` 里为 `llm-api.unused_api_keys`）。Claude Code 一行说明 Claude Code 用的是订阅还是按 `ANTHROPIC_API_KEY` 计费（原来写 "claude doctor passed"，markitai 从没运行过它）；RapidOCR 一行说明默认的 `ocr.lang: en` 能读中文、英文、日文和拉丁字母文字：RapidOCR 的 PP-OCRv6 识别模型是一个多语言模型，`en`、`zh`、`ja` 读中文和日文扫描件的效果一样，"lang: en (English)" 让人以为读不了。
- **`markitai init` 说明哪些 key 没用上、为什么。** 设置了所有提供商 key 并登录了 Claude Code 时，它配置了 Claude Code、DeepSeek、Gemini 和 OpenAI，对 `ANTHROPIC_API_KEY` 和 `OPENROUTER_API_KEY` 只字未提。略过它们是有意的（Claude Code 订阅提供的 Claude Haiku 也是它们的默认模型，加进来只会把请求分给通往同一模型的付费途径）；现在 init 会为每个略过的 key 打印一行并说明原因。更新已有配置时，只有配置里还没有某个提供商的模型，才会加入这个提供商，所以写着旧默认 `claude-agent/sonnet` 的配置不会再多出一条 Claude Code。
- MCP Registry 条目运行它所标明的版本（`uvx --from "markitai[mcp]==1.2.1"`），不再运行 PyPI 上的最新版。
- 移除未使用的 Python API：`LLMProcessor.clean_markdown`、`get_usage`、`get_total_cost` 和 `reset_call_counter`（按文档统计的用量仍由 `get_context_usage`/`get_context_cost` 提供）、`LLMProcessor.enhance_url_with_vision`、从未被赋值的 `FetchResult.static_content`/`browser_content` 字段、`BatchProcessor.process_batch`（连同 `markitai.batch.ProcessFunc`）、`ImageProcessor.process_and_save_async`、`OCRProcessor.is_scanned_pdf`、`BaseConverter.can_convert`、`PromptManager.list_prompts`、`collect_user_notices`、`markitai.llm.MarkitaiLLMLogger`（它被注册为全局 LiteLLM 回调），以及从未抛出过的 `QuotaError`、`ProviderTimeoutError` 和 `SDKNotAvailableError`。`SingleFileWorkflow` 现在必须传入 `processor=`，不再接受 `no_cache`/`no_cache_patterns`。`prompts.url_enhance_system`/`url_enhance_user` 设置及其提示词文件也已移除：没有任何代码路径会发送它们，仍设置了它们的配置照常加载。
- `markitai serve` 的 LLM 设置接口（`/api/settings/llm…`，包括已存凭据）对所有客户端都要求访问令牌，本机也不例外：本机任何程序、另一个 `localhost` 端口上的网页同样经回环地址连入。启动时打开的浏览器和打印出的令牌 URL 都带有令牌；使用 `--no-auth` 时，本机仍可不带令牌访问设置。没有客户端地址的请求不再被当作来自本机。
- **`markitai serve --no-auth` 只为本机抓取 URL**：其他机器提交带 URL 的任务或重试会收到 403 `remote_url_forbidden`，公网地址和内网地址一样，与共享工作台的 API 约定一致。此前其他机器只是限于公网地址，任何能连上服务器的人仍可借它抓取网页。本机仍可提交任意 URL，其他机器仍可上传文件；启用令牌认证（默认）时，持有令牌的机器保有全部权限。
- 升级须知：LLM 缓存键现在覆盖全文和图片字节，所以已有的 `cache.db` 条目在升级后首次运行时都不会命中，已处理过的文档第一次重跑会重新调用 LLM。`markitai serve` 的历史改为从任务完成时间起算，不再看目录修改时间，所以完成超过七天的历史会在首次启动时删除，即使其目录最近被改动过。

### 修复

- **中文 `markitai doctor` 报告也覆盖了登录检查的错误**（`providers/auth.py` 报告的缺少凭据文件、令牌过期等），并有测试保证。
- **命令行的几处小修复。** 没有安装 `mcp` 可选依赖时，`markitai-mcp` 会给出安装方法而不是打印错误堆栈；选择终端语言时 `LC_ALL` 优先于 `LANG`（POSIX 顺序）；`doctor --help` 把 LibreOffice 描述为 PPTX 幻灯片渲染。
- **`markitai serve` 的几处小修复。** “编辑配置”会拒绝带重复键的文件（两个 `api_key` 中的第一个原本会明文显示）；`PUT /api/convert-defaults` 读到 64 KiB 就停止（返回 413）；作业压缩包的每次下载使用独立文件，同一作业同时下载两次不会在 Windows 上失败；被中断的下载留下的压缩包会被清理；`.msg` 和 `.ipynb` 原件使用各自的媒体类型。
- **发布流程现在逐字节检查要发布的 wheel。** `scripts/check_wheel.py` 会把随包的每个工作台文件与清单中的摘要和大小比对（`.gz` 与对应的明文文件也比对），并在 CI 和发布工作流中对实际上传的构建产物运行。
- **重试一个任务不会再让刚被删除的作业“复活”。** 重试现在会先取得作业的删除锁并重新检查，再排队；重试等待期间到达的删除请求会生效。此前该任务可能被再次转换（并再次计费），结果写进已不存在的文件夹。
- **中文 `markitai doctor` 报告更完整。** 摘要行用“、”连接名称，并翻译 `1 LLM model` 与 `Playwright (slow to start)`；进度文字、`--fix` 的失败详情、Claude 云服务商的修复说明、`# or:` 和括号补充说明也是中文，并有测试保证 `doctor.py` 新增的消息一定带中文形式。
- **`markitai doctor` 不再为运行时会被忽略的模型打勾“视觉模型”和“VLM OCR”。** 没有设置 API 密钥的视觉模型不再计入，该行会说明它在等哪个环境变量。
- **Claude、Copilot 和 ChatGPT 的模型发现更接近 Rust 的 API。** 失败原因不再缓存五分钟（登录后重新加载即可消除）；失败时不再返回过期的空列表，也不再错误地返回 `no_key`；Claude 登录过期会返回 `not_signed_in`；命令行过旧或没有响应会返回 `runtime_unsupported`（附 `runtime.installed`）或 `runtime_failed`，而不是“未登录”；安装命令固定到测试过的命令行版本，缺少 SDK 可选依赖时使用与 markitai 安装方式相符的安装命令；ChatGPT 从一开始就有连接卡片，“未登录”的帮助随时可达。
- **URL 的文本阶段和图片阶段重新共用同一个请求预算。** 它们此前各开一个预算，一个页面在 serve、命令行和 API 中可能花掉两倍的 `llm.max_requests_per_document`（默认 100 次请求）。
- **已付费的 LLM 请求不再从合计中漏掉。** 无法解析或校验的 Batch API 答案仍会计入（它已被计费）；URL 列表的报告包含失败 URL 的花费；`--llm-batch-collect` 会打印该批次的成本（部分请求无价格时显示“至少”）。
- **用 `--record-history` 记录的命令行运行，在 `markitai serve` 中不再把未知成本显示为 $0。** 记录的任务现在带有 `pricing` 和 `models`，没有价格的模型不再显示成 $0 或“没有使用模型”。
- **`markitai serve` 现在在每个含无价格请求的任务上都提示成本不完整**，用工作台能识别的句式 `Cost is incomplete: 2 requests to <model> have no reviewed price…`（会被翻译，并带价格帮助链接）。以前只有最先回答的那个任务带一条英文提示。
- **中文终端下 `markitai doctor` 全部显示中文。** 各项检查的说明和修复方法（`RapidOCR：v3.9.2，可识别中文、英文…`、`Playwright：已安装，但尚未下载其 Chromium 浏览器`，以及 LLM、视觉模型和登录各行）与 Rust 版 doctor 语气一致；检查名称、命令和 `--json` 仍为英文。
- **`markitai doctor` 的标题下不再空一行。**
- **模型把页面/截图引用并进前一段文字时，现在会恢复成独立的一块。** LLM 的回答把 `<!-- ![Page 1](…) -->`（或“Page images for reference”标题）粘在段落末尾时，还原步骤以前会让它留在段落里；现在会在前后补上空行，与 Rust 一致。模型本来就单独成行的注释不变。
- **由视觉模型读过的扫描版 PDF，不再提示“页面看起来是扫描件，请加 --ocr 重新运行”。** 使用 LLM 和页面截图时，该提示会等模型回答之后再决定：回答里有文字就不再提示；模型只返回图片或这一步失败时仍会提示。
- **订阅类服务商（Claude、Copilot、ChatGPT）取不到模型列表时，模型发现现在会说明原因**：`reason` 为 `runtime_missing`（程序或其 Python 可选依赖未安装）、`not_signed_in`、`runtime_failed` 或 `timeout`，并附 `runtime` 和 `next_step`（安装命令，或 `markitai auth <provider> login`），与 Rust API 一致（契约 v1.18）。工作台会显示原因，并在必然失败时关闭按 ID 添加模型。
- 批量转换时多个线程同时首次导入 markitdown，偶尔让某个文件（常见是 `.msg`）失败，报 `_DeadlockError: deadlock detected by _ModuleLock('markitdown.converters')`：现在 markitdown 在任何转换器使用前、加锁只导入一次。
- **在提供商页面勾选一个模型、取消勾选另一个，不再因 `settings_entry_missing` 失败**（契约 v1.17）。页面原来先删除，这会让手写条目重新编号，或删掉提供商卡片所代表的模型，随后的保存就找不到它的卡片。`POST /api/settings/llm/deployments/batch` 和 `POST /api/me/models/deployments/batch` 现在也接受 `remove`（部署 id），并对页面读取的版本同时应用添加和删除，要么全部成功要么全部不做（视图中 `batch_remove: true`）。路由权重是 0 到 100 的整数（0 表示关闭，模型仍保留），所有写入权重的路由都按此检查，其他值返回 422 `invalid_weight`，Edit config 会指出具体条目。文件里已有的同一条目的权重可以保留，所以旧文件照常保存。
- **长文档转换后不再让 `markitai serve` 卡住一分钟。** Markdown 清理中有一步会把跨行断开的链接文字接回去（`[Title\n\nDescription](/url)` → `[Title](/url)`）：对每个后面没有 `]` 的 `[`，它把后文的每个换行都试作断点，并从每个断点一直读到文末。一本 490 页、带三个孤立 `[` 的 OCR 书籍在这一步花了 50 秒；这期间它占着 Python 解释器锁，工作台和 API 都停止响应（一次状态请求等了 80 秒），CLI 则多花 50 秒。现在只把文本读一遍，结果相同，耗时以毫秒计。
- **2027 年起的 Gemini 3.8 Flash 和 3.6 Flash**：Google 在 Gemini API 上对 Gemini 3.8 Flash、3.6 Flash 和 `gemini-flash-latest` 的首发优惠价只到 2026 年 12 月 31 日，2027 年 1 月 1 日起价格翻倍。LiteLLM 1.104.2 只有优惠价，所以从 2027 年起 markitai 报出的费用只有实际的一半，`max_cost_per_document_usd` 也只计入一半。现在 2027-01-01（UTC）起的应答记为未定价，与 markitai Rust 实现一致。
- **DeepSeek 与中国法定节假日**：DeepSeek 在工作日北京时间 09:00–12:00 和 14:00–18:00 按高峰价计费，其余时段（包括周末和法定节假日）按半价计费。markitai（经由 LiteLLM）原先把法定节假日的工作时段也按高峰价计算。现在采用与 markitai Rust 实现相同的规则：请求的发出和应答在同一时段内时，按该时段的价格计费；工作日高峰时段的请求，如果日期尚未对照国务院节假日安排核对过（已核对 2026-10-08 至 2026-12-31），则记为未定价，而不是可能多算；跨越两个时段的请求同样记为未定价。
- **Claude Batch 费用计入提示缓存**：Anthropic 的 `--llm-batch` 结果原先只按输入和输出 token 计价，而 Anthropic 的输入数不含缓存部分，所以缓存读取和缓存写入都没算进批次的费用（使用缓存的典型提示低估 1%–3.5%，缓存用得多时更多）。现在与实时请求一样计价，包括 5 分钟和 1 小时缓存写入，按标价的一半计。
- **Claude Sonnet 5.5 和 GPT-6.1 Sol 有了价格**：markitai 依赖的 LiteLLM 1.104.2 没有这两个模型，它们的请求都被计为未定价：摘要显示费用未知，`max_cost_per_document_usd` 也无法计入。现在 markitai 像 Claude Haiku 5.5 一样自带它们的价格行，直到某个 LiteLLM 正式版收录为止。这些行取自 LiteLLM 1.106.0.dev2，与服务商 2026-10-11 公布的价格一致（包括 Batch 和提示缓存价格），也是 markitai Rust 实现使用的价格。
- **`serve` 不再把未知的 LLM 费用显示为 `$0`**：提供商没有给模型定价时，网页界面在每个条目和合计里显示「$0」，而条目自己的提示却说费用未知。现在条目和历史带有 `pricing`（有价格和无价格的请求数、`cost_status`），工作台显示「未知」或「至少 $x」。
- **网页下载不再把访问令牌放进 URL**：「全部下载」等浏览器下载打开的是 `…?token=…`，会留在浏览历史和下载列表里。现在改用一分钟内有效、只能用一次的链接（`?ticket=`，与 `?token=` 一样在请求日志中遮蔽）。
- **`markitai serve` 可以从环境变量卡片添加模型了。** 设置了 `OPENAI_API_KEY`（或 Anthropic、Gemini、DeepSeek、OpenRouter 的 key）时，添加模型里会出现“使用 OPENAI_API_KEY”卡片，但加载它的模型或保存模型都返回“This saved provider connection no longer exists”（HTTP 404），保存服务器根据这个 key 检测到的模型也一样：服务器只在已保存的连接里找这张卡片。现在这张卡片代表该服务商自己的 key 变量和该服务商自己的端点，与 API 约定和 Rust 实现一致：用这个 key 列出模型，保存的模型只记下引用（`api_key: "env:OPENAI_API_KEY"`，不带地址），从不写入 key 本身，并归入一个已保存的连接，之后由它代替这张卡片。这个 key 不会发往别处：随卡片一起发来的地址、与模型服务商不同的卡片，以及指向某个变量或没有这种卡片的服务商的卡片 id（`env:AWS_SECRET_ACCESS_KEY`、`env:groq`）都会被拒绝，返回 HTTP 422。
- **`markitai serve` 测试模型失败时会说明原因。** 在没有 `DEEPSEEK_API_KEY` 的服务器上测试 DeepSeek 模型，请求照样发了出去（LiteLLM 找不到 key 时就不带 key 发送），报出的是 "AuthenticationError: litellm.AuthenticationError: AuthenticationError: DeepseekException - Authentication Fails (governor)"，看起来像是 key 填错了。现在测试会在发送任何东西之前找出没有设置的 key 变量，工作台能连接的所有服务商都是如此；工作台会说明原因和处理办法：**未设置 API 密钥**（“服务器上未设置 DEEPSEEK_API_KEY。请在运行 markitai serve 的环境中设置它并重启服务，或为该模型的连接添加密钥。”）、**密钥被拒绝**（HTTP 401/403）、**找不到模型**、**请求受限或额度不足**、**无法连接服务商**、**响应超时**或**服务商返回错误**，服务商自己的原话（去掉了 LiteLLM 的类名）放在“服务商原文”下（`Model connection returned HTTP 401: Authentication Fails (governor)`）。API 的回答与 markitai 的 Rust 实现一致：`reason`（`no_key`、`key_rejected`、`model_not_found`、`rate_limited`、`network`、`timeout`、`provider_error`），缺少 key 时还有 `key_variable`（变量名，不含它的值）。
- `serve` 曾为无法转换的文件（`unknown.bin`）提供重试；条目现在带 `error_code`（`unsupported`、`fetch_error`、`cancelled` 等），工作台只在重试有用时提供。没有令牌的标签页现在会提示输入令牌，不再显示一段原始的请求头语法说明。
- **EPUB 的图片随 Markdown 一起保存。** 以前 Markdown 保留的是书内各章节自己的引用（`../Images/cover.jpg`），在输出旁边无处可寻，所以 Markdown 里缺了这些图片，`markitai serve` 预览里也只显示占位。现在书中的位图（PNG、JPEG、GIF、WebP）会保存为输出的资源（`book.epub.0001.jpg`），像 DOCX 的图片一样链接，`--alt` 和 `--desc` 也会描述它们；远程、SVG、有歧义或过大的图片保留原引用。包在 SVG 里的封面图（Calibre、Sigil、古登堡计划生成的封面页）和只在 OPF 里登记的封面也会保留；OPF 中文件名经过百分号编码的章节（如 Sigil 写入的中文文件名）不再连同全部正文被跳过。
- **没有模型成功时不再写 `.llm.md`。** 以前即使没有可用的模型回答，文件也可能得到 `.llm.md` 并算作已增强：模型拒绝描述图片时（"I'm sorry, I can't help with that"），这句话会成为图片的描述和替代文字，而且拒绝的回答会被缓存；被服务商内容过滤截断的回答也会被接受；仅截图模式下读出为空的页面会被悄悄丢掉；在 `markitai serve` 中，整段回答还可能被提取出的原文替换（PPTX 的 "Slide" 标记与识别结果的 "Page" 标记永远对不上），条目却仍显示已增强。现在只有完整的回答才算数（没有在输出上限处截断，也没有被内容过滤中止），而且不能是拒绝，也不能是被放回的原文；拒绝的回答不会被缓存。一个文件的所有请求都没有得到这样的回答时，只写基础 `.md`，条目提示的最后一条是 "No model succeeded for this document; only the base Markdown was kept."（工作台把它显示为一条提示，措辞与 Rust 实现相同）；随失败的文档一起得到的图片回答会一并舍弃。读出为空的截图页会在提示中列出。在 `markitai serve` 中，条目只列出结果实际用到其回答的模型（只有基础结果时一个也不列，但费用照计），重试或增强失败时保留之前的结果并记录原因（`rerun_failure`，重启后仍在；增强成功后清除），写明真正的原因，不再是 "LLM enhancement did not produce an enhanced Markdown result"。
- **Claude Code 一轮失败不再被当成模型的回答。** Claude Code 无法回答时（用量达到上限、登录过期、API 出错），仍会返回错误文字，markitai 把这段文字当成了结果：图片会得到一个 `.llm.md`，描述和替代文字都是 "Claude AI usage limit reached"，而且没有任何提示；文档则以 "Invalid JSON" 失败，看不出原因。现在这一轮会像其他被拒绝的请求一样失败（用量上限按限流处理，登录或余额被拒按需要用户处理的错误处理），条目只保留基础 `.md`，并在提示里说明原因，单文件、批量、URL 和 `markitai serve` 都是如此。GitHub Copilot 一轮结束却没有回答时，也不再返回文字 "None"。
- **Claude Code 的请求不再按 `ANTHROPIC_API_KEY` 计费而不走你的订阅。** Agent SDK 以非交互模式运行 Claude Code，只要设置了 `ANTHROPIC_API_KEY`，即使已登录 Claude 订阅，Claude Code 也会用这个 key，所以加载了各提供商 key 时，每个 `claude-agent/` 请求都按 token 计入这个 key，而且是按当时默认的 `claude-agent/sonnet` 的价格。现在已登录订阅时（`claude login` 或 `CLAUDE_CODE_OAUTH_TOKEN`），markitai 运行 Claude Code 时会对它隐藏这个变量。只靠这个 key 登录的 Claude Code 不再被 `markitai init` 和自动检测当成订阅，也不再因此略过 Anthropic API 模型：`anthropic/claude-haiku-5-5` 直接用同一个 key，不必每个请求都启动一个 Claude Code 进程。
- **终端输出折行时缩进对齐正文。** 较长的状态、警告或摘要行会在短语中间折回第 0 列（`markitai doctor` 的 VLM OCR 一行、预算警告 "…leaves 10 of 10. Nothing" / "was sent: …"）；现在按词折行并悬挂缩进到正文下方，不会拆开路径、URL 或命令。不在终端时，`doctor` 和 `init` 的输出也不再按 80 列硬折行（每条消息一行，和 stderr 一样）。输出流不是 UTF-8 时（`PYTHONIOENCODING=ascii`、Latin-1 locale），每个命令都会在第一个状态符号处以 `UnicodeEncodeError` 失败；现在改用 ASCII 符号（`+ x ! - * > o`）；`TERM=dumb`、无法处理转义序列的 Windows 控制台（Windows 10 1511 之前的版本，或控制台主机开启了“使用旧版控制台”），以及 Windows 上控制台输出代码页不是 UTF-8 时的管道和文件，也是如此：Windows PowerShell 5.1 按这个代码页解码程序经管道输出的内容，所以在代码页 437 下，`markitai doctor | Out-File -Encoding utf8 x.txt` 会把 `✓` 写成 `Γ£ô`、把 `•` 写成 `ΓÇó`。执行 `[Console]::OutputEncoding = [Text.Encoding]::UTF8`（或在 cmd 中执行 `chcp 65001`）之后，管道和文件使用 Unicode 符号；没有附加控制台的输出（服务、脱离控制台的进程）没有代码页可依，也使用 Unicode 符号。其余情况在所有平台上都使用 Unicode 符号 `✓ ✗ ! • ◆ → ○`，与 markitai-rust 一致：能处理转义序列的控制台（未开启时 markitai 会先开启），以及 mintty（Git Bash）。一次运行只用一套符号：stdout 或 stderr 任一需要 ASCII 时，两者都用 ASCII。文本本身仍是 UTF-8。`NO_COLOR` 只关闭颜色。进度条、帮助面板、边框和旋转指示与符号采用同一选择（在一个经典控制台窗口中，markitai 输出了 `✓ •`，进度条却画成 ASCII 短横线，因为 Rich 自己的判断与 markitai 的规则不一致）：符号为 Unicode 时用圆角面板和实心进度条，符号为 ASCII 时一律用 ASCII。
- **`markitai serve` 的登录 URL 完整打印**：80 列时局域网 URL 会在访问令牌中间折行，复制下来的链接无法登录。
- **缺少 API key 时直接点名要设置的变量，不再发出请求**：没设 `DEEPSEEK_API_KEY` 时，`--llm` 用 `deepseek/deepseek-flash` 仍会把请求发出去，把 DeepSeek 的 "Authentication Fails (governor)" 打印四遍却不说该设哪个变量，文件还记为 `completed`。现在开始转换前会检查每个模型的 key：条目里的 `api_key`（明文或 `env:`），或该提供商的环境变量，包括 LiteLLM 同样会读的 `GOOGLE_API_KEY`、`OR_API_KEY`、`ANTHROPIC_AUTH_TOKEN`。没有 key 的模型不参与这次运行，一条警告写明要设置的变量；所有模型都没有 key 时，运行只输出这一行并以 `1` 退出（`--json` 为 `ok: false`），Python API 和 MCP 抛出 `NoModelConfiguredError`。自带 `api_base` 的 DeepSeek 或 OpenRouter 模型（本地服务、网关）不受影响。`markitai doctor` 用的是同一套检查。
- **key 被拒绝时只报一次，并写明该修哪个变量**：提供商返回 HTTP 401 时，每个条目只打印一行，例如 `deepseek/deepseek-flash rejected the API key in DEEPSEEK_API_KEY with HTTP 401; set a valid one with: export DEEPSEEK_API_KEY=your_key. Provider message: …`，不再每一层都重复同一个错误。不会换别的结构化模式重试；所有模型的 key 都被拒绝后，本次运行不再发出请求。以前 `.urls` 批量里后面的 URL 会在约 9 秒重试后报 "No deployments available"。批量和单文件的报告文件现在也像 `--json` 一样列出各条目的 `warnings`，保留了未增强输出的条目不会再像完全成功。
- **缺少 extra 时的安装提示不再破坏现有安装**：`uv tool install "markitai[legacy]" --force`（`doctor` 和需要某个 extra 的报错都会给出）会把工具整个替换成这一项，丢掉 `setup.sh` 装好的所有 extra，并把从源码目录安装的 markitai 换成 PyPI 版本。现在这条命令会根据 uv 自己记录的安装信息，列出已装的 extra 加上缺少的那个，并保持原来的来源（`uv tool install "markitai[browser,legacy,ocr,serve] @ /path/to/markitai" --force`）。缺少多个 extra 时（例如没有终端时运行 `setup.sh` 或 `setup.ps1`，只会装核心），`doctor` 给出一条命令一次装齐，不再每个 extra 各重装一次；安装摘要也会说明没有终端时如何装全（设置 `MARKITAI_INSTALL_OPTIONAL=1` 后重新运行安装脚本）。
- **Windows on Arm 上装在 x64 Python 中**：markitai 无法装进原生 ARM64 Python：每个 PDF 转换都要用的 pymupdf-layout 根本没有 ARM64 版本，LLM 库依赖的 fastuuid 和 tiktoken 也得用 Rust 编译。`setup.ps1` 以前用 uv 最先找到的 Python，装了 python.org ARM64 版 Python 的机器会停在 `pymupdf-layout ... has no wheels with a matching platform tag (e.g., win_arm64)`。现在它在 ARM64 Windows 上选用 x64 CPython 3.11–3.13，没有就用 uv 安装 CPython 3.13 x64，并用一行说明原因。`doctor` 和报错给出的安装命令在这类机器上会带 `--python cpython-3.13-windows-x86_64`，重装时不会换掉解释器；入门指南写明了直接用 uv 安装时的命令。markitai 在这类机器上由 Windows 模拟运行，和 uv 默认安装的 Python 一样。
- **Windows 上不再每次运行都查询 WMI**：Python 3.12 起，`platform.system()` 会通过 Windows Management Instrumentation 查询系统版本和 CPU，而 markitai 每次运行都会调用它（确定转换线程池大小、抓取前读取系统代理）。每次查询都要调用另一个进程；在无权查询 WMI 的会话里（例如 SSH 登录），每次查询要等约 5 秒才失败。现在 markitai 改从 `sys.platform` 判断操作系统。
- **安装只装一次 markitai，装完后的第一次运行也不再慢**：在 Windows 11 on Arm（x64 Python 由系统模拟运行）上，`setup.ps1` 用了十多分钟，装完后第一次 `markitai doctor` 也要几分钟。以前安装脚本先装 markitai，再用新装的 markitai 询问要加哪些 extra（`markitai doctor --suggest-extras`，即新安装的第一次运行），然后带 `--force` 再装一遍，137 个包全部重建；两次都让 uv 把整个环境的每个文件编译成字节码（`UV_COMPILE_BYTECODE`，约 13,400 个文件，大部分 markitai 根本不会导入，而且每个新文件都要经 Windows Defender 扫描）。现在 `setup.sh` 和 `setup.ps1` 先问完所有问题（Web UI、OCR、Playwright 浏览器、Claude Code 和 Copilot CLI），再按回答需要的全部 extra 只装一次 markitai；已有的安装原地更新，uv 只增删有变化的部分，不固定版本的 PyPI 安装像 `uv tool upgrade` 一样取最新版本。装好之后，`python -m markitai.warmup` 在后台以低于正常的优先级用四个进程编译 markitai 代码实际导入的模块（约 6,300 个文件），安装脚本同时继续其他步骤；浏览器下载完后它还会启动一次 Chromium（杀毒软件扫描时第一次启动很慢）。脚本在显示摘要前等它完成。你自己设置的 `UV_COMPILE_BYTECODE` 仍然生效。每一步都有带已用时间和该步最新一行输出（uv 的 `Downloading …`）的进度动画，结束时显示用时，最后显示 `安装用时 …`；设置 `MARKITAI_SETUP_TIMINGS=1` 会列出每一步的用时，包括 uv 自己的解析、下载和安装时间。以前第一次运行要编译它导入的每个模块：在一台 ARM64 测试机上，第一次转换 URL 用了 44 秒，之后是 4.6 秒。
- **`markitai doctor` 和 `markitai --version` 启动更快**：doctor 以前为了判断每个已配置的模型能不能读图，要导入 LiteLLM（约 2,400 个模块，在一台 Windows 11 on Arm 机器上每次 8 秒）；现在直接读取 LiteLLM 的模型表文件（对 LiteLLM 1.104.2 表中全部 6,978 个名称，结果与 LiteLLM 相同）。各项检查同时进行，包括会启动 `claude` CLI 的 Claude Code 登录检查，终端里会显示还在进行哪些检查；Chromium 成功启动过一次后，对同一个 Chromium、Playwright 和 Python 记住一周，不再每次运行都启动（`--fix` 总会启动）。`markitai --version` 在加载 CLI 之前就给出答案（导入 50 个模块，以前 430 个）。`doctor --timings`，或对任何命令设置 `MARKITAI_TIMINGS=1`，会列出时间花在哪里。在 Linux 上 doctor 从 1.47 秒降到 0.17 秒（模块数从 2,945 降到 537）。
- **网页转换快了约一倍**：从网页 HTML 中提取正文以前占了 URL 转换的大部分时间，在 Windows 上尤其明显（一台 x64 模拟运行的 ARM64 机器上，转换一个 Wikipedia 页面共 7.6 秒，其中提取占 4.2 秒）。现在提取以更少的遍历完成同样的工作：CSS 判断复用同一个匹配器，每个元素只和它可能匹配的规则比较；判断元素是否已被移除时不再每次搜索整棵子树；第一次尝试前也不再复制整个页面。在 Linux 上对 235 个页面测量：总计从 6.0 秒降到 3.2 秒，Wikipedia 页面从 0.40 秒降到 0.20 秒。生成的 Markdown 逐字节相同。
- **脚注引用交错的页面重新能正常提取**：在部分页面上（包括 Wikipedia 的“Python (programming language)”），一个引用标记按链接匹配到一个脚注、又按编号匹配到另一个脚注，脚注处理随之报错，整页退回到普通 HTML 转换器，连导航也一并输出。现在第二次匹配会被忽略，与 markitai 所遵循的 defuddle 提取器一致。
- **维基条目从正文开始**：MediaWiki 当前皮肤的页面（Wikipedia、Wiktionary、Wikivoyage、ArchWiki）以前以 `[ ]` 和该条目所有语言版本的列表开头，正文第一句之前有 25 到 80 行。现在根据链接上 HTML 的 `hreflang` 标记（而不是某个网站的类名）识别语言切换器，并连同它自己的控件（下拉框的复选框、“33 种语言”标签、“编辑链接”）一起移除；提到几个译本的正文、外语文章的阅读列表，以及内容本身就是语言列表的页面都会保留。
- **保留文章自己的侧栏和摘要**：文章正文里的插入框以前会被针对页面侧栏的规则（所有 `<aside>` 和 `.sidebar`）一并删除，被识别为页面装饰的区块也会把文章摘要一起带走。martinfowler.com 的“Microservices”因此少了约 1,100 词：摘要和“How big is a microservice?”等七个侧栏。现在，位于文章段落之间、以正文写成（30 词以上、链接很少）的 aside 会保留；以链接为主的页面侧栏和简短的推广框仍会删除。第一个标记为摘要的元素（class `abstract`、`itemprop="abstract"`、`role="doc-abstract"`）在外层区块被删除时会保留下来，作者简介等区块其余部分照常删除。
- **不再把词粘在一起**：有三处会丢掉两段文字之间的空格或换行。图片说明以前被压平成纯文本，行内代码和链接两侧的词会粘在一起，标记也丢了（react.dev：“Initially,MyApp’scountstate is0and”）；现在保留说明本身的行内内容，输出为 `` *Initially, `MyApp`’s `count` state is `0` and …* ``。只含一个空格的元素（例如排版插件或 MediaWiki 生成的 `&nbsp;` span）以前被当作空元素删除（“beforehashing”“December1989”，Python 文档里的“*class*asyncio.TaskGroup”）；现在保留为空格。代码块中的 `<br>` 以前会丢失，每行一个 `<span>` 的代码被压成一行（react.dev 的全部代码示例、Hexo 博客）；现在是换行。
- **MediaWiki 页面不再有编辑链接和打印页脚**：Wikipedia、Wikivoyage、Wiktionary、ArchWiki 和 mediawiki.org 的输出以前在每个标题后带一个 `[edit]` 链接（`[Bearbeiten]`、`[編集]` 等），结尾还有一行 `Retrieved from "…"`。现在按 MediaWiki 的类名（`mw-editsection`、`printfooter`）删除，与语言无关。
- **Windows 上的 `markitai doctor`**：刚运行完 `setup.ps1` 时，Windows Defender 会扫描新下载的 `chrome.exe`，Chromium 首次启动可能超过 30 秒，doctor 便报告浏览器"不可用"，而下一次启动只用了 0.7 秒。现在启动检查允许 60 秒，超时后再试一次；仍然很慢时，终端和 `--json` 都报告为"启动慢，请再运行一次 doctor"。所有 Playwright 提示（包括 URL 转换时的报错）都改为写出 markitai 自己的 Python，不再是 `python`（在 Windows 上可能只是 Microsoft Store 的别名）、`playwright` 或 `uv run playwright`，这些在 `uv tool` 或 `setup.ps1` 安装中都用不了；两项检查需要同一个修复方法时只打印一次；在 Windows 上，换行比窗口宽度少一列，以免占满整行的文字被控制台再折一次。
- **批量汇总把失败标为错误**：失败的文件和 URL 以前和跳过的文件、提示混在一起，用黄色警告符号列出；有失败的运行仍以 "✓ Done" 结尾；"Files: 9/11 ✓" 即使有条目失败也带绿色对勾。现在失败项先列出、用红色 ✗ 标记；"Done" 一行计入失败数，没有失败标 ✓、部分失败标 !、全部失败标 ✗；计数改为文字（"Files: 9 of 11 converted, 2 skipped, 1 failed"）。单文件运行中，转换期间记录的警告（扫描页、LLM 回退）标为 `!`（错误标 `✗`），不再是第 0 列的一段黄色文字。
- **续跑的批量统计整个批量的费用**：按 Ctrl-C 后用 `--resume` 续跑，报告里第一次运行完成的条目显示 `cost_usd: 0.0`、没有 `llm_usage`、也没有时间戳，摘要里的费用和请求数只算续跑这一次。现在批量状态会记下每个已完成条目的费用、用量和时间戳。摘要给出整个批量的费用和这次运行的部分（`$0.094 total, $0.067 this run`），报告的 `llm_usage` 是整个批量的合计，这次运行的部分在 `llm_usage.this_run`。之前完成的条目标上 `earlier_run: true`，`--json` 里也一样：`--json` 现在会列出这些条目，并新增 `totals.this_run_cost_usd`。旧版本写的状态照样能续跑：其中已完成条目的费用记为未知（`null`，计入 `llm_usage.usage_unknown_items`），摘要写 "cost of earlier runs unknown"，不再给出漏掉它们的合计。按 Ctrl-C 时还在进行中的请求不计入，尽管提供商通常会对它们计费。
- **PDF 用 `--llm` 增强时，模型漏掉页码标记不再白白丢掉整份清理结果**：`default-table.pdf` 在 Gemini Flash-Lite 和 OpenRouter 的 Claude Haiku 别名上失败（"LLM answer dropped 1/1 structural placeholders"）：回答漏掉了单页 PDF 开头、标题之前的页码标记，条目只保留了未增强的输出。现在，单独占一行的页码、幻灯片标记或图片若被漏掉，只要它前后的两行在回答中仍然相邻（后一行在回答中只出现一次，前一行或文档开头仍紧挨在它上面），或回答在同一位置把它写成了粗体或转义文本，就放回原位；放不回去的再发一次请求并点名；第二次仍缺失时条目照旧失败，错误信息写明是哪一页或哪张图（"page 2 marker"）。标记旁的文字被改写、合并或删掉的回答仍会被拒绝。这个问题不是 PDF 表格修复引起的：实测中 Gemini 在该修复之前的提交上同样漏掉了标记，OpenRouter 在没有表格的 `cjk-text.pdf` 上也漏掉了；是否发生取决于模型怎么回答。修复后，五个默认模型都完成了 `default-table.pdf`、`cjk-text.pdf` 以及一份 75 页 PDF 的前两页。
- **被拒绝的 LLM 回答不再记作免费**：markitai 拒绝了的回答（漏掉占位符、拒答、无法解析的 JSON、回答里混入提示词、截断或空回答）照样计费，但条目显示 `cost_usd: 0.0`、没有 `llm_usage`，无论是退回未增强输出还是失败；被拒后重问的回答根本不计，也都不计入 `max_cost_per_document_usd`。现在每个回答一到就计入条目的费用和用量（`--json`、批处理状态与报告、`serve`）以及费用上限；`llm_usage.requests` 按每个得到回答的请求计数，含重试。
- **推理模型清理文档前不再先"思考"**：`deepseek/deepseek-flash` 每次回答前都先推理，`large-sample-x15.pdf`（约 3.3 万 token 的 Markdown）用了 50,770 个输入、70,393 个输出 token，$0.049，接近 $0.05 上限。现在对 DeepSeek 推理模型和 Claude Haiku 5.5 发送 `thinking: disabled`，对接受该值的 OpenAI 模型（`gpt-6-luna`）发送 `reasoning_effort: none`；Gemini 和 OpenRouter 的默认模型本就不思考。实测同一 PDF 现在用 44,710 个输出 token（没有推理），原为 70,393 个；按同样的闲时价格为 $0.034，原为 $0.049（按 DeepSeek 标准价格为 $0.069，原约 $0.10）。在 `cjk-text.pdf` 上，DeepSeek 以前把整个 2,048 token 的回答额度都花在推理上，以 "output is incomplete" 失败；现在 214 个 token 就答完。
- **URL 转换不再为普通请求就能取到的页面启动浏览器**：静态抓取原先发送 httpx 默认的 `python-httpx/0.28.1` User-Agent，维基百科、Wikimedia Commons、Ars Technica、CNBC、豆瓣等站点会返回 403，`auto` 只好回退到 Chromium。现在以 `markitai/1.2.1 (+https://github.com/Ynewtime/markitai)` 表明身份，遇到 401/403 时先用桌面浏览器的请求头重试一次，再启动浏览器。example.com 加维基百科 Markdown 条目的列表从 6.6 秒降到 1.1 秒，Ars Technica 从 9.3 秒降到 2.0 秒，CNBC 从 10.2 秒降到 2.0 秒，每日邮报首页改为本地抓取，不再经由远程 Defuddle 服务。`MARKITAI_STATIC_HTTP=curl_cffi` 仍发送所模拟 Chrome 自己的 User-Agent。
- **浏览器抓取不再每页固定睡 4.4 秒**：`fetch.playwright.extra_wait_ms`（3 秒）原先总是等满，自动滚动之后还要再固定停 0.8 秒。现在页面稳定（0.5 秒内没有进行中的脚本、XHR 或 fetch 请求，DOM 也没有变化）即结束等待，`extra_wait_ms` 只作上限；滚动后的停顿只在截图时保留。`-s playwright` 抓 example.com 从 5.4 秒降到 2.2 秒，抓维基百科从 6.5 秒降到 4.1 秒，产出的 Markdown 不变。
- **静态抓取的页面保留真实标题、去掉图标代码**：维基百科页面的标题原先取自 JSON-LD 里的条目简介（"a lightweight markup language …" 而不是 "Markdown"），内联 SVG 图标和 logo 也被原样写进 Markdown（cnbc.com 首页因此有 1.5 MB，大部分是一个包着 base64 PNG 的 SVG）。带文字标签的 SVG 图表和示意图照旧保留。
- **`setup.ps1` 可在 Windows PowerShell 5.1 下运行**：Windows PowerShell 5.1 按系统代码页读取不带 BOM 的脚本，中文提示变成乱码，`powershell -File scripts\setup.ps1` 报 48 个解析错误（中文代码页下 18 个）；通过 `irm https://markitai.dev/setup.ps1 | iex` 运行时中文提示显示为乱码。脚本现在只含 ASCII 字符，中文在运行时生成，`-File`、`irm | iex` 和复制粘贴在 PowerShell 5.1 与 7、任意代码页下都能正常使用。无法显示中文的旧版控制台（代码页 437 或 1252）改用英文。
- **在 markitai 源码仓库中运行安装脚本时默认安装 markitai**：此前在仓库根目录运行 `setup.sh` / `setup.ps1` 会静默改为设置开发环境（uv sync、pre-commit hooks、`uv run` 提示），而不是安装。现在会先询问，默认安装；并且默认安装这个源码仓库中的 markitai，除非在下一个问题中回答否（或设置 `MARKITAI_SETUP_SOURCE=pypi`、或指定 `MARKITAI_VERSION`），那样就安装 PyPI 发布版；没有终端可询问时同样安装源码仓库中的版本。设 `MARKITAI_SETUP_MODE=install` 或 `dev`、`MARKITAI_SETUP_SOURCE=local` 或 `pypi` 可跳过这两个问题。
- **安装脚本能找到 uv 安装的 markitai**：`UV_TOOL_BIN_DIR`（或 `XDG_BIN_HOME`）指向 `~/.local/bin` 以外时，脚本找不到刚装好的 markitai，于是静默跳过其余 extra（旧版 `.doc`/`.ppt`、MCP、SVG、HEIF、extra-fetch）、`markitai init` 和版本检查。现在改为向 uv 查询该目录，显示 extra 安装步骤及结果，并在该目录不在 PATH 中时给出提示（`uv tool update-shell`）。
- 安装脚本的输出统一为一种样式：章节标题不再着色（洋红色横线容易被误读为错误），绿色只表示成功，黄色表示警告，红色表示错误；终端中长行在正文下方缩进换行，管道或日志输出不带颜色代码、每条消息一行，支持 `NO_COLOR`，符号遵循 markitai 本身的规则：`TERM=dumb`、非 UTF-8 的 locale（`setup.sh`）、无法处理转义序列的 Windows 控制台，以及控制台代码页（脚本把它切换为 UTF-8 之前的那个，也就是 Windows PowerShell 解码时用的代码页）不是 UTF-8 时 `setup.ps1` 输出到管道或文件的内容，使用 ASCII 标记。`setup.ps1` 向管道或文件写入 UTF-8，读取这类日志时请按 UTF-8 解码（Windows PowerShell 5.1 中用 `Get-Content -Encoding UTF8`）。
- 以 root 或管理员身份运行安装脚本时会说明原因（markitai 按用户安装，从不需要这些权限），且在没有终端可询问时（如容器构建或 SSH 命令）不再中止。
- **运行结束后不再出现 "leaked semaphore objects" 警告**：包含多个 PDF 的批量任务、或 12 页及以上的 PDF 会启动 PDF 提取工作进程，而 markitai 可能在进程池释放其队列之前就退出，Python 的 resource tracker 随后会在汇总之后（有时在 shell 提示符已经返回之后）打印 `UserWarning: resource_tracker: There appear to be 3 leaked semaphore objects to clean up at shutdown`，看起来像输出被截断（11 个文件的批量任务约每 7 次出现 1 次）。现在 CLI、`markitai mcp` 和 `markitai serve`（用 Ctrl-C 或 SIGTERM 停止时）会等进程池释放后再退出。Windows 不受影响。
- 批量汇总中跳过文件的那一行不再把提示接在最后一个文件名后面（`english.png, sample.jpg Use --llm or --ocr …`），现在写作 `2 files skipped (image_only): english.png, sample.jpg; use --llm or --ocr for content extraction`，超过两个文件时以 `and 3 more` 结尾。扫描页提示改为 `2 of 12 pages look scanned/garbled (pages 1–2)`，不再是 `1 page(s) look scanned/garbled (pages 1)`；OCR 空白页提示写作 `[OCR] scan.pdf: 1 of 12 pages has no text (page 7); it is left blank in the output`。
- **`--ocr --llm` 保留扫描页的识别文字**：模型在一页提取文字为空的页面上写出超过 200 个字符时，页面漂移防护会把该页还原成提取出的文字，而扫描页的提取文字总是空的，所以扫描 PDF 虽经模型识别并计费，结果却报告为已完成、正文只有页码标记。现在视觉路径会把空页上的文字作为该页的识别结果保留；如果模型回答中的文字会被防护全部丢弃，该条目会失败，不再以空内容报告完成。
- **`llm.max_requests_per_document` 限制整份文档**：此前一份文档的文本增强和图片分析各有一份预算，某个阶段结束时还会退回它用掉的请求数，所以上限为 3 时，`sample.pptx --llm --alt --desc` 发出了 6 个请求。现在同一个文件或 URL 的所有请求共用一份预算，`max_cost_per_document_usd` 也一样。
- **模型价格未知时不再报告为免费**：LiteLLM 没有价格的模型，或者配置里只用 `model_info` 描述的模型（这会把它注册成没有价格的条目），此前都报告为 `cost_usd: 0.0`，像是免费的。现在这些请求在 `llm_usage` 里计入 `unpriced_requests`（`--json`、报告文件以及 Python API 的 `ConversionUsage.by_model` 都有），批处理摘要显示 "cost unknown"，每次运行还会对该模型给出一次警告。`cost_usd` 仍是数字，只含有价格的请求。这类请求不计入 `max_cost_per_document_usd`。价格表里标为 $0 的模型（本地模型）仍按免费计算。
- **请求使用配置的 `max_tokens`**：构建模型池时丢掉了 `litellm_params.max_tokens`，所以每个请求要的都是模型自身的输出上限（配置为 `max_tokens: 8192` 时，`deepseek/deepseek-flash` 发出 `max_tokens: 128000`，Gemini 发出 `max_output_tokens: 65535`），抬高了每个请求的最坏情况费用。现在实时的文本、文档和图片请求都会发送配置的值（同一个池里有多个模型时取最小值），`--llm-batch` 也会发送：Anthropic 批处理用它代替模型上限，OpenAI 批处理以 `max_completion_tokens` 发送；配置值高于模型的输出上限时，两者都与实时路径一样降到该上限。`model_info.max_tokens` 和 `max_input_tokens` 现在也会按文档所述替代 LiteLLM 报告的上限。未设置时，请求仍使用模型的输出上限，最多 128000 个 token。
- **DeepSeek 默认模型不再先发出一个失败请求**：默认的 `deepseek/deepseek-flash` 以思考模式回答，而 DeepSeek 在思考模式下拒绝强制工具调用（HTTP 400 "Thinking mode does not support this tool_choice"）。现在 markitai 对 DeepSeek 的思考模型（`deepseek-flash`，以及旧配置里的 `deepseek-v4-flash` 和 `deepseek-v4-pro`）改为在提示词里要求输出 JSON，不再强制工具调用，每份文档一个请求即可，不会先浪费一个失败请求。另外，请求失败（服务商返回 4xx，或会被重试的错误）而请求上限已不允许再试时，条目会报告服务商的错误，不再只有 "budget exhausted"。
- **OpenAI 默认模型不再先发出一个失败请求**：OpenAI 在 chat completions 接口上拒绝默认推理强度下的函数工具（HTTP 400 "Function tools with reasoning_effort are not supported for gpt-5.6-luna in /v1/chat/completions"），所以对 `openai/gpt-5.6-luna`，以及文档注明有同样限制的新默认模型 `openai/gpt-6-luna`，每次结构化调用都要先浪费一个失败请求才回退，较小的 `llm.max_requests_per_document` 也因此被用掉一半。现在对 OpenAI 推理模型的实时请求改为要求 JSON schema 格式的回答（`response_format`），不再使用函数调用；`--llm-batch` 仍使用函数调用，并关闭推理。
- **订阅制提供商的费用估算对应正确的模型**：Copilot 或 ChatGPT 请求的等价 API 费用是用模型名去匹配 LiteLLM 的价格表得出的，名字里出现两次的数字可能匹配到只出现一次的名字，所以 `copilot/claude-haiku-5.5` 被按 `claude-haiku-4-5` 计价，是实际价格的十倍。
- **`serve` 在 LiteLLM 加载期间也能响应最初的请求**：配置了模型时，`serve` 启动后会在后台导入 LiteLLM，而 LiteLLM 会在 Uvicorn 和 httpx 的日志记录器上安装日志过滤器，这些过滤器要等导入完成后才能正常工作。此间应答的请求会触发其中一个过滤器，报错 "cannot import name 'RouteContext' from partially initialized module" 并断开连接（Windows 上为 `ConnectionAbortedError` 10053），导入本身也可能失败，使 LLM 功能在重启前一直不可用。CLI 或 MCP 运行时若在 LiteLLM 加载期间往这些记录器写了警告，也会遇到同样的问题。现在 markitai 会在导入完成前让这些过滤器不起作用。`serve` 启动后一秒左右内打开**系统设置 → 模型**也会造成同样的损坏：服务商列表在另一个线程导入 LiteLLM 的一部分，而后台导入还在进行，两个导入互相等待，Python 让其中一个失败，于是选择器一个服务商都不列，所有模型测试和 LLM 转换都报 "partially initialized module 'litellm'"，直到重启。现在那里会先导入 LiteLLM 本身。
- **Claude Agent 和 Copilot 提供方不再开放任何工具**：不再自动批准 Bash、写文件和网页抓取，被注入提示词的文档或网页无法以你的身份执行命令。
- **以相对路径输入目录 `--resume`**：子目录中重试的文件（`docs/sub/a.pdf`）写回 `out/sub/`，不再落到输出根目录（那里可能与其他文件冲突，图片链接也会失效）。
- **`--resume` 状态文件跨平台通用**：各系统上文档都以 POSIX 风格路径（`sub/a.txt`）为键。此前 Windows 上的键是 `sub\a.txt`，在 Windows 写出的状态无法在 Linux 或 macOS 上续跑；在 Linux 或 macOS 上自上次完整保存以来的进度，到 Windows 上续跑时也会丢失。在 Windows 上，旧版本写出的状态仍可续跑。
- **`serve` 拒绝来自本机其他端口的写请求**：此前另一个 `localhost` 端口上的页面（Jupyter、开发服务器、用 `python -m http.server` 打开的下载 HTML）可以发起任务、重试和设置操作；现在 Origin 必须与服务器的主机和端口一致，与文档的说明相符。
- **ODS 表格保留空隙**：连续的空单元格（`1, <空 ×2>, 4`）不再合并成一个，此前这会让后面的值错到别的列；连续的空行也会保留。
- **没有表格线的 PDF 表格仍输出为表格**：PDF 提取未识别出的无边框表格（浏览器默认表格样式，单元格之间只隔几个点）此前输出为一行文字，例如 `Language Year Typing Primary Use Python 1991 ...`。现在 markitai 根据每一行共有的空白间隙划分列，输出 Markdown 表格，但只替换文字与表格内容完全一致的段落；正文、列表、分栏文字、对齐的代码和等宽字体文本保持原样。在 39 个 PDF（论文、报告、表单、RFC、日文统计资料）上，它只改动了两个被压平的表格，每页约 3 毫秒。
- **ChatGPT 提供方**：重试后仍然限流（429）或服务端出错（5xx）时，只让这一次请求失败，不再跳过文档的其余部分；流式回答被截断或失败时会如实报告，不再当作完整结果接受并缓存。
- stderr 不是终端时（管道、日志文件、CI），错误和警告不再按 80 列硬换行，每条消息保持在一行，便于 grep。用法错误（未知选项、`markitai mcp` 搭配 `--config-json` 等）也包括在内：它们原先画在边框里，消息被边框字符拆成几行，现在在终端之外输出为 `Usage: …` 和 `Error: <消息>` 两行。终端里的输出不变。
- **`serve` 的 LLM 设置**：设置接口的调用方不能再把任意主机环境变量（`env:NAME`）或已保存的 key 发往自己指定的服务器。Web API 传入的 `env:` 引用仅限各服务商的 key 变量（`env:OPENAI_API_KEY` 等）且只能配合服务商的默认端点，或者是配置文件里已有的 key 与 `api_base` 组合。已保存的 key 或 `env:` 引用始终绑定在它保存时的 `api_base` 上，编辑模型或服务商时也一样（否则返回 422，要换端点请直接提供 key）。
- **LLM 错误不再显示 API 密钥或文档内容**：服务商的错误信息如果复述了请求，markitai 会原样输出：复述密钥的 HTTP 401 会显示完整密钥，引用请求内容的 HTTP 400 会显示文档文字，出现在 stderr、日志文件、`--json` 输出、批处理报告和状态文件，以及 `serve` 和 MCP 的错误里。现在配置的密钥、环境变量里的服务商密钥，以及形似密钥或令牌的内容（`sk-…`、`Bearer …`）都替换为 `[REDACTED]`，超过 40 个字符的引文替换为 `'…'`，复述请求的信息整条隐去。错误类型、HTTP 状态码和服务商信息的其余部分保留。`--llm-batch` 的错误行也按同样方式清理。
- **崩溃的 traceback 不再显示变量值**：带 traceback 记录的意外错误（转换步骤的 DEBUG 日志、`serve` 任务和请求的崩溃、LiteLLM 和 Uvicorn 等依赖库报出的错误）会把每一帧的局部变量打印到 stderr 和日志文件，其中可能有 API 密钥和文档文字。现在 CLI、`serve` 和 `markitai-mcp` 的所有日志输出都只打印 traceback 本身、不打印变量值，异常信息里的密钥和 URL 凭据也替换为 `[REDACTED]`。
- **`serve` 请求日志**：网页工作台加在事件流、文件和归档 URL 上的访问令牌（`?token=`）在终端和日志文件里会被遮蔽。
- **URL 抓取**：超过 500 MB（与本地文件相同的上限）的响应会让抓取失败，不再无上限地读进内存；`Content-Length` 已超限时在读取正文前就失败。
- **URL 中的凭据**：带 `user:password@` 的 URL 不会再把这部分写进截图文件名或 `~/.markitai/learned_spa_domains.json`。
- **Windows 上的 URL 截图**：长 URL 的截图文件名可达 200 个字符，放在普通输出目录下时整条路径会超过 Windows 的 260 个字符上限；未开启长路径支持（默认关闭）时截图会丢失。现在 Windows 上这类文件名会缩短到能放下，保留开头并附上一段短哈希，不同 URL 仍对应不同文件。
- **网页提取的清理**：内联 SVG 里的 `javascript:` 链接（`xlink:href`）、`srcset`/`poster`/`data` 中的此类链接，以及被制表符、换行或控制字符拆开的写法（`java&#9;script:`）都会被移除，不再进入 Markdown。
- **`markitai-mcp` 的日志**与 `markitai mcp` 一致：stderr 上只有错误，URL 中的凭据会被去掉。此前它会打印调试信息和完整 URL，包括其中的 `user:password@` 或令牌。
- **PDF 隐藏文字**：`security.pdf_sanitize = "remove"` 不再删除与隐藏片段相同的可见文字（以前一个隐藏的 `e` 会删掉整页所有的 `e`），深色背景上的白字不再被当作隐藏文字，`--ocr --llm` 也会像其他路径一样检查发给模型的文字。
- **页眉页脚清理**：对所有带页面或幻灯片标记的输出执行的清理，改用 PDF 转换器的规则：只看每页开头和结尾各两行，一行至少要在 60% 的页面（且不少于 3 页）重复出现，不足 4 页的文档不处理，标题和表格行永远不删。此前它会把正文中任意位置出现的同一短行删掉，只在一半页面出现的行也会被删，且漏掉页眉。页面和幻灯片都会再处理一轮，所以三行的页脚（如 PowerPoint 的日期、文字、页码）仍会完整删除。`markitai.utils.text.clean_ppt_headers_footers` 由 `markitai.utils.page_chrome.strip_repeated_marker_lines` 取代。
- **`.numbers`**：不加 `-b cloudflare` 时，Apple Numbers 文件现在会转换失败并说明原因（导出为 `.xlsx`/`.csv`，或使用 `-b cloudflare`），不再报告为转换成功、内容却只有包内的元数据。
- **转换器**：旧编码（Latin-1、GBK、带 `<meta charset>` 的网页）的 `.txt`/`.md`、HTML、XML、reStructuredText、Org 和 LaTeX 文件能正确读取，不再报错或变成替换字符；检测不出编码的 CSV 保留原有行，不再变成 `None`；带嵌入二进制数据（`\bin`）的 RTF 可以转换；LaTeX 的 `\href{url}{text}` 转成链接；选取标题时，正文里的两条水平分隔线之间的内容不再被当作 frontmatter；含反斜杠的图片描述（`C:\temp`）按原样写入。
- LLM 限流报错里出现 "402"（例如 token 数 "Used 29402"）时仍会重试，`1.204s`、`750ms`、`6m0s` 这类等待时间也能正确识别；只有 HTTP 402 才算计费错误。
- LLM 缓存不再给前 50,000 个字符之后被改过的长文档、或图片内容变了但文件名没变的页面和截图返回旧结果：缓存键现在覆盖全文和图片字节。
- `cache.enabled: false` 现在会关闭 LLM 结果缓存（不再读写 `cache.db`），`cache.max_size_bytes` 也会对它生效。
- 模型返回的空图片说明或空截图提取结果不再被永久缓存（视为分析失败：保留原 alt 文本，下次运行重新请求）；发送前被重新编码为 JPEG 的图片也不再被标成 PNG。
- `.urls` 批量处理共用一个 LLM 处理器，`--llm-concurrency` 现在限制整个列表而不是每个 URL；同一批量中的同名文件（`a/report.pdf`、`b/report.pdf`）也不再共用或互相清掉 LLM 费用、用量和 `max_requests_per_document` 预算。
- `--resume` 不再跳过在其他输出设置下完成的条目：`--pure`、`--keep-base`、`--screenshot-only`、`output.profile` 和 `-s` 现在各自使用独立的批量状态。`--llm-batch` 遇到 URL 或 `.urls` 文件时直接报错，不再按全价逐个请求。
- MCP `batch_convert` 启用 LLM 时整个任务共用一个 LLM 处理器，`llm.concurrency` 现在限制整个批量而不是每个条目（此前 10 个条目的批量可能同时发出约 100 个请求）。
- 批量抓取同时包含走代理和 NO_PROXY/本机地址的 URL 时不再随机失败：静态 HTTP 客户端为每个代理各保留一个连接池，不再在其他请求仍在使用时关闭共用的那个。
- 显式的 `-s jina`、`-s defuddle` 或 `-s cloudflare` 只代表对该服务的同意：它拒绝请求、markitai 回退到自动链时，由 `fetch.remote_consent` 决定其他远程服务（defuddle.md、Cloudflare、FxTwitter）能否收到该 URL。
- macOS 系统代理自动检测恢复正常（`scutil --proxy` 先列端口后列主机，此前总是检测不到代理）；`NO_PROXY` 或 `local_only_patterns` 中的 `corp.example` 这类条目现在也覆盖其子域名（`api.corp.example`），与 curl、urllib 和 Go 一致。
- Cloudflare 验证页（"Just a moment..."、"Ray ID"）、验证码页面和仅仅是内容较短的页面不再写入 30 天的 SPA 缓存，一次人机验证不会再让该站点之后的所有请求都走浏览器。声明需要 JavaScript 或只显示加载占位的页面仍会写入。
- 网页提取按主机名选择站点提取器：`dropbox.com/status/…`、`linux.com/news/article/…`、`docs.github.com/…/issues/…` 这类页面不再被当作 X 帖子或 GitHub 讨论处理，内容不再丢失。
- `--record-history` 只复制本次记录的输出所用的图片和截图，不再把共享输出目录里历来生成的全部资源都复制一遍；`markitai serve` 运行期间也会按任务完成时间删除超过 7 天的历史（此前只在启动时按目录修改时间清理）。
- `markitai -c FILE mcp` 现在使用 `FILE` 提供服务（此前读取的是默认配置），`markitai mcp` 搭配 `--config-json` 会报错而不是被悄悄忽略，错误信息里的 "Config file:" 提示也会给出本次运行实际使用的 `-c` 文件。
- 文件批处理在每个文件结束后释放 LLM 图片缓存，不再把已完成文件的页面图片和截图（最多 500 MB）一直留到批处理结束。
- 文档与代码保持一致：命令行用法错误以 `2` 退出，在转换开始前被拒绝的输入（文件不存在或格式不支持、`--llm-batch` 未配 `--llm`）以 `1` 退出；配置文件的优先级高于环境变量，环境变量只作后备，`MARKITAI_RECORD_HISTORY`、`MARKITAI_PURE` 和 `MARKITAI_NO_REMOTE_FETCH` 除外；`batch.heavy_task_limit: 0` 按内存自动检测（不是"不限制"）；`batch.scan_max_files` 统计整个输入目录树；只有 Copilot 会自动伸缩请求超时。
- **Copilot 提供方重新可以配合 GitHub Copilot SDK 使用**：此前 `create_session` 和 `send_and_wait` 收到的是单个字典，回退到 PATH 中 CLI 时传的是 `{"cli_path": ...}` 字典，所有受支持的 SDK 版本（0.2.0 及以上）都不接受，因此每个 `copilot/` 请求都会失败。现在按 SDK 定义的关键字参数调用，并把请求超时传给 `send_and_wait`（其默认超时会在 60 秒时中断请求）。
- `markitai serve` 不再返回已保存的 API key：`GET /api/settings/llm/providers/{id}/credentials` 返回打码后的 `api_key_masked`（`sk…7890`；`env:` 引用原样返回），不再返回原始 `api_key`。提供商编辑器显示这个打码值，只有你改动 key 时才发送，所以保存时不动该字段就保留已保存的 key；清空字段仍会删除 key。
- **`serve` 设置不再显示已保存 API 基础地址里的凭据**：提供商编辑器（`GET /api/settings/llm/providers/{id}/credentials`）按原样返回保存的基础地址，其中的 `user:password@` 或敏感参数（`?api_key=`、`token`、`key`、`sig` 等）会传到浏览器。现在所有设置响应都把它们打码，例如 `https://…@proxy.example/v1?api_key=…&api-version=2024-10-01`，**测试**返回的错误里也不会出现。地址保持显示的样子保存时，沿用已保存的地址；改过但仍含 `…` 的地址会被拒绝（422 `invalid_settings_request`），要修改请输入完整地址。
- `markitai serve` 设置中 key 与非保存时地址配对被拒时，错误信息现在会说明可行的做法：直接输入 key 本身，或在配置文件里配置这一对。serve 和配置指南也解释了这条规则和两种做法。
- 缩小到 512 px 后仍超过 5 MB 视觉输入上限的图片，现在按 512 px 的版本发送，不再发送原始的更大图片。缩放也会恰好停在 512 px，而不是停在高于它的上一档。
- MCP 服务器遇到格式错误或取值无效的配置文件时，会返回具体信息（"Invalid JSON in …"、"Invalid configuration in …"），不再是含糊的 "Error executing tool" 错误。`ConfigFileError` 现在可从 `markitai.api` 导入，`markitai.config.ConfigFileError` 照常可用。
- **EPUB 的元数据进入 front matter，不再写在正文开头。** 基础 Markdown 此前以 `**Title:** … **Authors:** … **Language:** … **Publisher:** … **Date:** … **Description:** …` 这样一段开头。现在书的信息写进 front matter，位于 `title` 和 `source` 之后：`authors`（列表）、`language`、`publisher`、`date`、`description`，均为纯文本（Calibre 简介里的 HTML 会去掉，Calibre 表示“无日期”的 0101-01-01 不写）。`.llm.md` 也保留这些字段；其中 `description` 仍是模型写的摘要，模型没给时用书自带的简介。
- **PDF 的标题是文档的名字，为纯文本，两份输出一致。** 一份保单 PDF 的 `.md` 标题是恰好成为第一个标题的小号加粗说明文字（“面积单位：平方千米”），`.llm.md` 的标题是一个上标标题，带着 HTML 和两个不可见的字体符号，因为 LLM 输出用另一套规则重新取了标题。现在 PDF 的标题依次取：PDF 自带且确实是名字的标题（不是“Untitled”“Document1”“Microsoft Word - report.docx”、文件名或数字），前几页上字号最大的文字（“电子登记单”），第一个是名字的标题，最后是文件名。所有格式的标题都是纯文本（没有标签、强调标记或私用区字形），也不会是单纯的数字、编号或页码；`.llm.md` 与 `.md` 标题相同。
- **PDF 的标题为纯文本，符号字体的字形不再显示为方框。** 一份中文保单在工作台目录中列出了 `▯▯<sup>…</sup>` 这样的标题和整行正文：章节标题前的大号 Wingdings 图标让 PyMuPDF 把标题标成上标，图标本身成了私用区字符，版面分析又把加粗或带下划线的句子当成标题。现在所有 PDF 路径（`.md`、OCR 运行中的原生文本页、视觉模型读取的文本）中，标题不再含 HTML 标签和上标脚注标记（保留 Markdown 强调），Symbol、Wingdings、Wingdings 2 字体的字形按绘制它的字体映射（`−`、`α`、`❖`、`☞`、`✓` 等）或去掉，标题中和列表项开头的装饰符号去掉，私用区字符和非字符（iText 的 `U+FFFF`）在全文中去掉，正文里包住中文的 `<sup>` 去掉标签。不是标题的行恢复为段落，规则与 markitai 的 Rust 实现相同：句子（以 `。！？；` 结尾或含有这些标点，或在 `，` 之后继续），`标签：值` 字段（`计价单位：人民币元`），超过十五个词（三个 CJK 字符算一个词），带引导点的目录项，以及引出列表的句子（`…须符合下列要求：`）。与标题分开的章节号（`2.4` 与其后的 `服务范围`）合为一个标题，一句话被拆成几个标题的行合为一个段落。`.llm.md` 会去掉模型在标题中保留的标签和字形。在所有者的保单上，106 个标题中有 55 个带标签、带字形或其实是正文，另有 26 个字形显示为方框；现在它的 60 个标题都是章节标题，方框全部消失；测试语料中其余 38 个 PDF 只有带这些形态的 3 个有变化。
- **front matter 中的网址变得可读。** URL 条目的 `source` 和 `canonical_url` 此前是百分号编码（`https://ynewtime.com/posts/20251223-ai-%E5%BA%94…`）。现在按字符写出（`…/20251223-ai-应用案例…`），仍是同一个地址：影响地址含义的转义（`%20`、`%2F`、`%3F`、`%25`）保持不变，再次编码即得到原地址。
- **`markitai serve` 中密钥来自服务器环境变量的连接可以加载模型列表。** 在配置里手写的模型（`deepseek/deepseek-flash`）在服务器设置了 `DEEPSEEK_API_KEY` 时能通过模型测试，但“编辑”里显示“API key：未设置”，加载模型列表失败并显示 “model discovery failed (HTTPStatusError)”，因为加载列表时没有带上密钥。现在“编辑”显示“来自环境变量 DEEPSEEK_API_KEY”，模型列表用与转换和测试相同的密钥加载（不会发往浏览器指定的地址），加载失败时说明原因：使用与模型测试相同的原因（`no_key`、`key_rejected`、`rate_limited`、`network`、`timeout`、`provider_error`）和提供商自己的说法，而不是异常类名。完全没有密钥时不发送请求，并说明应设置哪个变量（`DEEPSEEK_API_KEY is not set on the server, so nothing was sent`）。失败说明与 markitai 的 Rust 实现一致（`Model discovery returned HTTP 401: <提供商的消息>`、`Model discovery timed out`、`Model discovery request failed: <原因>`）；404 或不是模型列表的应答算作提供商错误，重定向不会跟随，不公开模型列表的服务商不算失败（手动输入模型 ID）。
- **表格的空单元格显示为空。** `.xlsx` 原先经 pandas 转换：每个空单元格都写成 `NaN`，表头单元格为空的列被命名为 `Unnamed: 1`，文本 `NA` 变成 `NaN`，表格上方合并的标题行（表格标题、"联系人：……"一行）被当成整张表的表头。现在单元格按原样读取（`TRUE`/`FALSE`、`0.1`、`2024-01-02`、错误单元格的 `#DIV/0!`；单元格里的 `|` 会转义），并按一条规则排版：一行中唯一的值位于跨两列及以上的合并单元格时，这一行写成段落，并结束它上方的表格；到下一个这样的行为止的各行组成一张表，第一行是表头，按原样写出（空的表头单元格保持为空）；在这张表中每行都为空的列不输出，表内的空行保留原位置。`.xls` 改用 xlrd 按同样方式读取（原先经 markitdown 和 pandas），`.ods` 表格也采用同样的标题行规则。转换表格不再加载 pandas。
- **PDF 的页眉、页脚和页码不再混入正文。** 一份三页的指南，每页正文里都留着页脚（文档标题）和页码：版面引擎能识别它们，但它们照样被写出，而删除重复行的文本规则要四页起才生效。现在，版面引擎标为页眉或页脚的框，若位于页面顶部或底部五分之一以内，且是单独的页码（`3`、`- 3 -`、`Page 3 of 9`、`第 3 页`）或在两页及以上重复出现（某一页上标题和页码合成一个框也算），就会删除；只出现一次的页眉保留。
- **一行文字不再变成"这行的图片 + 这行文字"。** 版面引擎有时把加粗或彩色的一行（PDF 用 Type3 字形绘制的中日韩文字，Chrome 的"另存为 PDF"就是这样）当成插图；页面随即被裁出这一行的图片，后面再跟一遍同样的文字作为"图片文字"。现在，图片框若不与任何嵌入图片重叠、且大部分被文字覆盖，就只保留文字，不保留裁图；真正的插图保留图片。
- **X 帖子的链接指向真实地址，视频仍是视频。** 帖子里的链接原先保留 X 的 `https://t.co/…` 短链：现在显示并链接 FxTwitter 提供的目标地址；X 页面的 Markdown 中仍剩下的短链，用一次发往 `t.co` 的 `HEAD` 请求解析（每页最多 20 个，同时 4 个，每个 5 秒、合计 10 秒，结果在进程内缓存）。视频原先是封面图加一个单独的 "Video" 链接；现在是链接到视频文件的封面图 `[![Video](poster.jpg)](video.mp4)`，与 Rust 对 `<video>` 的写法一致。
- **对照视图能显示 X 帖子。** x.com 对 Playwright 默认的无头浏览器返回 HTTP 403，所以没有保留网页，截图是一张空白页。现在 X 帖子用 Chromium 的新无头模式加载（`playwright install chromium` 已经安装的完整 Chromium），使用去掉 "HeadlessChrome" 的浏览器 User-Agent，x.com 会正常返回：截图和保留的网页就是帖子本身的页面，正文和链接完整。缺少该版本或 x.com 仍然拒绝时，改用 X 的嵌入帖子页面（无需登录）。两种情况下对照视图都提供"截图 | 网页"切换。在五条公开帖子上，直接加载每条中位耗时 3.1–5.1 秒，嵌入帖子为 4.8–7.0 秒；保留的字节约为四倍（网页 85–150 kB、截图 100–160 kB，嵌入帖子为 23–30 kB 和 40–75 kB）。浏览器还会等待 X 当前的帖子标记（不带属性的 `<article>`），不再每条帖子都空等 10 秒超时。
- **验证页不会被当作网页保留。** 网站给浏览器显示验证码或"正在检查您的浏览器"页面（例如 Bilibili 的按顺序点字验证），而阅读服务仍取回了正文时，验证页的截图曾被当作该网页的截图保留。现在，浏览器绘制页面时就会识别可见文字很少且带验证标记的页面，以及空白页面：浏览器会再加载一次，若仍是验证页，则既不保留其副本也不保留其截图，条目会说明原因（"No usable copy or screenshot of … was kept: the site showed a verification page (CAPTCHA) instead of the page; open the page itself to compare"）。验证页本身另行保存（从不显示），以便对照视图说明网站显示了什么。
- **浏览器加载过的每个网址条目都同时保留网页和截图。** 正文来自 Jina、Defuddle 或 Cloudflare 时，即使浏览器为截图加载过该页面，也不保留网页，对照视图只能显示截图；现在也保留浏览器的网页。保留的是浏览器绘制出的页面（在为提取文字而清理之前）：脚本插入的样式规则（CSS-in-JS）写入副本，脚本不保留。代价是网页副本的大小，几 kB 到几百 kB（一条 X 帖子：27 kB，截图 66 kB）。
- **之前会话中的网页按其自身地址显示。** `markitai serve` 重启后，对照视图以 `<base href="None">` 显示网址条目保留的网页，其样式表、图片和链接按工作台地址解析而不是按网站地址，页面没有样式、图片也损坏。现在 base 重新是条目的网址。
- **Windows 上网址截图的注释使用 `/`。** Markdown 以 `.markitai\screenshots\…` 引用截图，其他系统上的 Markdown 查看器找不到它们，`markitai serve` 也不把它们算作该条目的文件：对照视图没有截图页面，结果不列出它们，删除条目也会留下它们。现在各平台都以 `/` 写入引用，之前写入的任务两种写法都能读取。
- **表格不会再多出一行分隔行。** 模型改写表格时有时会重复 `| --- | --- |` 一行，在 `.llm.md` 中显示为一行破折号数据；共用的 Markdown 清理现在会删除表格已有的分隔行（代码块之外）。
- **文件名与内容不符时会说明实际内容。** 一个实际是 gzip 压缩的 PostScript 文件的 `.epub`，原先报错 "Malformed EPUB: File is not a zip file"；现在提示 "This is not an EPUB file: it holds gzip-compressed data, while an EPUB is a ZIP package. Check the file (it may be misnamed or still compressed)."。OpenDocument 文件同样如此。
- **Chrome 生成的 PDF 中的中日韩文字使用正确的字符。** Chrome 的"另存为 PDF"在文字层里用部首码位表示部分汉字，Markdown 里于是出现 `使⽤`、`右下⻆`（U+2F64、U+2EC6），看起来与"使用""右下角"一样，却无法搜索、排序或匹配。现在 PDF 输出（基础与 LLM 输出，以及 LLM 路径读取的文字）会把康熙部首（U+2F00–U+2FDF）和代表单个汉字的部首补充字符转换为对应的统一汉字，映射与工作台相同（先 NFKC，再加 NFKC 不处理的 28 个部首补充字符）；⺁ 这类部件形状保持不变。
- **表格单元格里的换行不再打断表格**：单元格中的换行（XLSX、XLS、ODS 单元格或带引号的 CSV 字段里的换行，DOCX、HTML、EPUB、PPTX、ODT 单元格里的第二段或 `<br>`，PPTX 单元格里的软换行，以及任何单元格里的 U+2028/U+2029 行或段分隔符、垂直制表符或 NEL）现在写成 `<br>`，这一行仍是一行。此前 CSV 把换行原样写进行里，把这一行拆成两行；U+2028 在把它当作行尾的渲染器里会结束表格；DOCX、HTML 和 PPTX 用空格把各行连起来。连续的换行合并为一个 `<br>`，单元格首尾的换行去掉；CSV、DOCX、HTML、PPTX 单元格里的 `|` 会转义（此前会多出一列）；单元格里作为文字写下的 `<br>` 仍是文字（`&lt;br&gt;`）。模型回答（`.llm.md`）或其他路径留在表格行里的分隔符，也由共用的 Markdown 清理写成 `<br>`。在 Windows 上，lxml 自带的 libxml2 版本较旧，HTML 文档里的垂直制表符、换页符以及文件、组、记录分隔符会被删掉，两侧的词因此连在一起（单元格内和正文中都是如此）；现在每次解析 HTML 都把它们当作换行处理，各平台结果一致。单元格 HTML 源码中的换行（LF、CRLF 或 CR CR LF）仍是空白，显示为一个空格，与浏览器一致。

### 安全

- **服务器自己的 API key 不再发往设置调用方选定的地址。** 通过 `markitai serve` 设置页面保存（或在那里测试）的模型或连接如果没有 key 却指定了地址，LiteLLM 会读取服务器上该提供商的 key 变量（`OPENAI_API_KEY`、`GROQ_API_KEY`、`LM_STUDIO_API_KEY` 等）并发往那个地址：持有访问令牌的人就能让服务器把它的 key 发到自己的主机，而禁止把 `env:` 引用指向调用方所选主机的规则正是为了防止这一点。现在这样的条目会保存占位 key `markitai-no-key`，任何服务器都不会把它当作凭据（不校验 key 的服务器，如 LM Studio、vLLM、llama.cpp，照常可用，无 key 的 OpenAI 兼容端点也终于能用了；界面上显示为没有 key），在调用方指定的地址上做连接测试时也用它。已有 key 的连接不能再改成指向另一个地址的无 key 连接（HTTP 422）。你在配置文件里手写的条目不变：没有 `api_key` 的条目无论 `api_base` 指向哪里，仍使用环境变量中的 key。
- **服务器的 Google、watsonx 和 GigaChat 登录凭据不再发往设置调用方选定的地址。** 无论条目带什么 key，LiteLLM 都会用服务器自己的 Google 凭据（应用默认凭据、`gcloud` 登录或 `GOOGLE_APPLICATION_CREDENTIALS`）签署 Vertex AI 请求，用 `WATSONX_TOKEN` 签署 watsonx 请求，用 `GIGACHAT_ACCESS_TOKEN` 签署 GigaChat 请求，并发往条目的地址：通过 `markitai serve` 设置页面以调用方所选地址保存或测试的 Vertex AI 模型（包括 LiteLLM 会路由到 Vertex AI 的裸名称如 `gemini-2.5-pro`，以及保存在另一个连接地址下的模型），在测试和转换中都会把服务器的 Google 访问令牌带到那里。现在设置页面会拒绝带地址的此类模型（422 `invalid_settings_request`，以英文、中文和日文显示），**添加模型** 也不再为 Vertex AI 提供地址栏，而是用一行短说明告诉你它在 Vertex AI 自己的端点使用服务器的 Google 登录，已保存的 Vertex AI 连接也不再提供 key；不带地址时仍在服务商自己的端点使用服务器的登录凭据，手动写在配置文件中的地址照常使用。Amazon Bedrock 不受影响：带地址而无 key 时，它把占位 key `markitai-no-key` 作为 key 发送，从不发送服务器的 AWS 凭据。
- **演示文稿的文件名不再被 LibreOffice 当作选项。** 渲染幻灯片时会把文件路径原样交给 `soffice`，因此文件名以 `-` 开头的文件（例如 `markitai ./--accept=socket,host=0.0.0.0,port=2002;urp;.pptx --screenshot`：Python 会去掉 `./`）会变成 LibreOffice 的一个选项，比如在转换期间打开网络监听的选项。现在传入的路径总是绝对路径。
- **用户令牌的 URL 只从公网抓取。** 用户令牌的 URL 任务可以指向服务器所在网络中的地址，包括服务器自己的 API：在那里不带令牌的请求算作管理员，因此 `http://127.0.0.1:<端口>/api/history?include=items` 及随后的 `/api/jobs/<id>/files/<名称>` 能读到管理员和其他用户的全部转换。现在用户的 URL、它的重试（包括管理员发起的重试）以及它的文档引用的图片，都按原本用于 `--no-auth` 上传的公网策略抓取：私有、回环或链路本地地址（包括域名解析或重定向得到的此类地址）会让该条目以 `private_or_local_host` 失败。管理员自己的任务不变。
- **用户令牌的模型不再使用服务器的 key 变量。** 用户令牌通过**编辑配置**（`PUT /api/me/config`）保存的模型如果有地址却没有 key，或链接到自己的某个连接却同样没有 key，会在不带 key 的情况下路由，LiteLLM 便读取服务器上该提供商的 key 变量（`OPENAI_API_KEY` 等）并发往用户的地址，或用服务器的 key 在提供商端点运行。现在这类条目会保存占位 key `markitai-no-key`（模型路由早已如此）；链接到连接的模型使用该连接的 key 和地址；此前保存的文件也按同样方式读取。
- **缓存只有所有者可读。** `~/.markitai/cache.db`（文档文本和模型回答）和抓取缓存（带查询令牌的 URL、用你的 Cookie 或凭据抓取的网页）以前按默认权限创建，同一台机器上的其他账户也能读取。现在它们创建时只有你可读，之前版本留下的可读缓存会在下次打开时收紧（macOS 和 Linux；Windows 上沿用所在文件夹的访问规则）。
- **LibreOffice 不再加载文档仅以链接方式引用的内容。** 幻灯片渲染和 EMF/WMF 回退以空白配置启动 LibreOffice，文档中的链接图片会被加载进导出结果：以转换机器上的 `file:` 路径链接的图片会被画进幻灯片图像（这些图像也会发给视觉模型，在 `markitai serve` 中还会返回给上传文档的人），http(s) 链接也会被抓取。现在私有配置会阻止链接内容和宏。
- **文件夹自带的 `.env` 和 `markitai.json` 不再能让转换反过来对付你。** 在下载的文件夹中运行 markitai 时，它的 `.env` 会被加载进环境变量，`markitai.json` 会作为配置：`.env` 可以指定 markitai 作为 Claude Code 或 Copilot 启动的程序、AWS 或 Google 凭据文件、代理、`*_API_BASE` 地址或 `NODE_OPTIONS`/`PYTHONPATH`，配置则可以让模型指向它自带的地址，LiteLLM 会把你环境中该服务商的密钥发过去。现在 `./.env` 仍可设置密钥和选项，但不再能设置程序、路径、地址、代理、主目录、凭据文件、加载器或解释器钩子以及证书包（被忽略的名称会提示一次）。在当前文件夹找到的 `./markitai.json` 不会把环境中的密钥发往它自带的地址：这类模型只使用条目里写明的密钥（否则使用占位密钥 `markitai-no-key`，`env:` 引用同样如此，并会提示），自带地址的 Vertex AI、watsonx 或 GigaChat 模型会被略过。`markitai doctor` 和 `--verbose` 会注明配置文件是在当前文件夹找到的。用 `--config` 指定该文件，或把这些设置放在 `~/.markitai/`，即可照旧使用。
- **按公网策略抓取的页面不再能把浏览器重定向到内部地址。** 让用户令牌的 URL 和不受信任调用方（`--no-auth`）的 URL 只走公网的策略会检查浏览器发出的每个请求，但会把重定向响应交回浏览器，而浏览器跟随重定向时不会再次询问：一个重定向到 `http://127.0.0.1:<端口>/…`、服务器所在网络中的地址或云元数据的公网页面会被加载并转换。现在由 markitai 跟随重定向，每个目标在请求之前都会检查，指向内部地址的重定向会被拒绝；经公网重定向到达的页面按它自己的地址解析相对链接和图片，结果的 URL 也是这个地址。
- **Cloudflare 处理只供管理员使用。** 通过 `fetch.cloudflare` 的远程处理计入服务器的 Cloudflare 账户，但任何受信任的调用方都能请求它，包括用户令牌，而用户令牌在其他方面都无法使用管理员的凭据。现在 `GET /api/capabilities` 对管理员以外的调用方都报告 Cloudflare 为 `client_not_trusted`，请求它的任务或重试会得到 403 `remote_processing_forbidden`。
- `serve`：用户令牌自己的模型只能到达公网地址。原先针对服务器本机地址的检查只看写出的主机名，所以解析到回环、服务器所在网络或云元数据的域名，在该用户的转换、模型发现和测试中都能通过。现在这些连接在建立时解析，地址不在公网上即拒绝；不跟随重定向，会经环境代理发出的模型请求也被拒绝：在这样的代理之后，该用户使用 LLM 的转换会以 `enhancement_failed` 失败并说明原因，不使用 LLM 的转换照常运行。管理员共享的模型不变。
- `serve`/抓取：Playwright 的 `extra_http_headers` 只发送到初始导航来源，不再发送到浏览器上下文发出的每个请求。其中的机密（如 API 令牌）不再送达跨来源子资源主机、favicon 主机或重定向目标；同源子资源和同源重定向仍会收到它们，跳转到其他来源的重定向则不带它们继续跟随。公网守卫也以同样方式携带它们。
- 打包与 `serve` 加固。sdist 不再包含 `tests/` 或 `benchmarks/`（测试套件含有第三方网页抓取，包括他人的电子邮件地址，且体积很大）：大小为 2.3 MB，1.2.0 为 4.1 MB。包元数据中的作者只写名字，不再带电子邮件地址。任务的下载归档不再跟随指向任务输出树之外的符号链接（只加入解析后仍在树内的真实文件）。构建后端（`hatchling`）固定到确切版本。

## [1.2.0] - 2026-09-28

### 新增

- **`llm.on_failure`**：LLM 增强失败（key 无效、服务端报错、超时、模型拒答）时保留未增强的 `.md` 并说明原因；设为 `"fail"` 则把条目记为失败（单个输入以 `1` 退出，批量以 `10` 退出）。
- **值得处理的提示**：默认模式下就会显示，并出现在所有结果里（stderr、`--json`、网页工作台、Python API、MCP），包括未开 `--ocr` 的扫描页、PDF 隐藏文字、OCR 或截图失败、无法渲染的幻灯片、LLM 或图片分析失败。
- **`.urls` 列表支持 `--resume`**，各种模式下按 Ctrl-C 都会保存批处理状态。
- `--json` 描述交接出去的 `--llm-batch` 运行（带收取命令的 `batch` 对象、`pending` 状态）以及逐项警告。
- `cache.fetch_ttl_seconds`（默认 24 小时）让没有 ETag/Last-Modified 的已抓取页面过期；`cache stats`/`cache clear` 覆盖抓取缓存。
- `markitai.enable_worker_processes()` 和 `MARKITAI_PDF_WORKERS` 用于并行提取 PDF；`markitai.api` 导出 `ConversionError`、`FetchError` 和 `NoModelConfiguredError`。

### 变更

- **PDF 转换最高快 4 倍，输出完全不变**：页面交给工作进程处理，每个进程单线程运行版面模型。在 18 核 Mac 上，300 页的 PDF 和 40 个 PDF 的批处理都快 4.1 倍，240 个混合文件快 3.3 倍。
- **LLM 批处理让转换和模型调用重叠进行**（30 个文档加 `--alt`：快 1.5 倍）；一次请求装不下的文档改为分块增强，不再在 32,000 字符处截断。
- **网页工作台与 CLI 对齐**：每个任务最多 1000 个条目（原为 50），240 个文件快 3.2 倍，任务运行期间 API 保持响应。静态资源由浏览器缓存，响应启用压缩。
- **OCR 按阅读顺序输出**：分栏、竖排中日文、倒置扫描页和表格。`--ocr` 保留有真实文字层页面的结构；RapidOCR 最低版本为 `>=3.9`。
- 新写出的文件跟随 umask（通常为 `0644`），不再一律是 `0600`，覆盖已有文件时保留原权限；配置文件和 `.env` 仍保持私有。
- `-c`/`--config-json` 对子命令生效，`config set`/`config edit` 可以创建新的 `-c` 文件。
- Python API 和 MCP 服务器与 CLI 一样确定模型，不再联网获取 LiteLLM 的价格表；`markitai-mcp` 与 `markitai mcp` 一样加载 `.env`。
- 提取的图片和截图按最终输出名命名（`report.pdf.v2-0001-01.jpg`），重跑写出 `report.pdf.v2.md` 时不再覆盖 `report.pdf.md` 引用的图片。
- 网页提取跟进 defuddle 0.19.4：代码块围栏、`<sub>`/`<sup>`、带标签的日期和 Shadow DOM 内容。

### 修复

- **批处理与续跑**：批处理不再把自己的输出或 `.markitai/` 资源当作输入，能识别 `.Docx` 这类扩展名，同名文件和 URL 不再互相覆盖。续跑时条目覆盖自己之前的输出（不再生成 `.v2` 副本），在另一个目录下续跑也一样。
- **LLM 输出**：清洗提示词列出了每一种页、幻灯片和图片占位符，Gemini Flash-Lite 这类模型会保留页面边界，清洗结果不再被整体丢弃。模型拒答不会被接受或写入缓存，图片分析失败时保留原来的 alt 文本，失败的运行留下本轮的 `.md`，不留旧的 `.llm.md`。
- **`--llm-batch`**：缓存命中的图片答案不再丢失，`--resume` 不再重新提交尚未收取的批次（改为打印收取命令），收取时应用 `--alt`/`--desc`，Anthropic 图片请求可用，凭据取自 `llm.model_list`。
- **网页抓取**：按声明的编码（GBK、Shift_JIS、CP1252）解码，个别非法字节不再让整页乱码；重定向后的相对链接解析正确；验证页不会替换正常的缓存副本；本机地址和 NO_PROXY 主机不走代理。
- **截图**：在清理 DOM 之前截取，保留全部分块并有单独的时间预算；页面加载后卡死会超时退出。
- **PDF 与 OCR**：输出路径含空格或括号时图片能正确写出；OCR 失败会让条目失败，不再写进输出；多帧 TIFF 逐帧识别。
- **网页工作台**：Ctrl-C 能及时关闭并保留历史；增强失败或重跑途中关闭服务时恢复之前的结果；同时删除和名字只差一个后缀的条目不再冲突。
- 另外：带 BOM 的 `.urls` 文件、带引号的多行 TSV 单元格、正文只有 HTML 的 `.msg` 邮件、只读输出文件、无法创建的输出目录（只输出一行错误，不再打印堆栈）；MCP 工具会返回失败原因。

## [1.1.0] - 2026-09-14

- **综合提速约 6 倍**：网页 URL **5.8×**、Office/PDF **5.6×**、批处理 **6.6×**，由三类等权的本地基准测得，不计 LLM、OCR 和公网时延。
- **更快得到结果**：启动和重复解析的开销都少了；短文和中日韩正文只要第一次就抓全，就不再回退到浏览器。
- **输出质量不变**：正文、元数据、公式和图片沿用原有处理，复杂文档照旧走成熟的回退路径。
- **修复公网 URL 误判**：Clash 等 Fake-IP 环境下，Defuddle 等远程抓取现在能正常工作，私网保护照常生效。
- **改善 X/Twitter 抓取**：允许远程服务时，匿名推文不用启动浏览器就能拿到；遇到 HTTP 错误及时回退，富文本、链接、图片和发布日期都保留下来。
- **保留域名调优**：自定义域名配置只覆盖显式设置的字段，其余内置浏览器参数照常生效。
- **减少重复提示**：远程抓取和 VLM OCR 的提示每位用户只看一次，授权询问和禁用选项照旧有效。

## [1.0.1] - 2026-09-10

### 新增

- `--json` 输出逐项状态、产物、usage、费用及运行级错误；需配合 `-o`，与 `--dry-run` / `--llm-batch-collect` 互斥。
- `--no-remote-fetch` 禁用远程 URL 抽取，`--log-level` 控制已配置的文件日志。
- MCP 支持输出 profile 和有界批量并发（默认 10）；省略 `llm` 时跟随服务器配置。
- 双语网站对比页、搜索与社交元数据，以及文档和 UI 的 CI 检查。

### 变更

- Serve 登录地址改用 `#token=`；config list/get/set 默认脱敏，包括嵌套请求头，显式 `--show-secrets` 才显示原值。
- 网页工作台增加可取消的上传进度、行内下载、清晰的选项与错误反馈，改善无障碍访问，设置和预览按需加载。移除 Tailwind 与无用 UI 代码。
- CLI 进度写入 stderr；帮助与报错明确输出路径、支持格式、可选依赖和已移除参数。

### 修复

- 防止 MCP 同名输出互相覆盖，以及 API 混入残留的 LLM 文件。
- 保留重试和重启后的任务选项，修复提前完成、恢复幽灵行及并发 ZIP 下载冲突。
- 未认证远程任务逐跳校验并固定公网目标，覆盖重定向与浏览器子请求，隔离抓取缓存和浏览器会话。
- Batch 只增强本次转换的文件并正确回填最终产物、usage 与失败状态；RAG/Obsidian 图片、wikilink 和 alt 在 Batch 与历史中保持完整。
- 统一处理非法配置、缓存替换容量、空 URL 列表及错误退出码；发布检查适配 fragment 登录地址。

## [1.0.0] - 2026-09-09

### 新增

- **markitai 现在既是 CLI 也是库**：`markitai.convert("report.pdf")` 及其异步孪生 `aconvert` 返回带类型的 `ConversionOutput`，里面有 markdown、frontmatter、资源与截图路径、逐图分析和用量合计，配置分层则复用 CLI 自己的那一套。暂定：签名与结果字段仍可能变化
- **`markitai mcp` 从 CLI 启动内置 MCP 服务**，这也是官方 MCP Registry 条目（`io.github.Ynewtime/markitai`）使用的形式：`uvx --from "markitai[mcp]" markitai mcp`
- **MCP agent 通过 `mcp` extra 接入 markitai**：`markitai.mcp` 随主 wheel 发布，经 stdio 暴露 `convert_document`、`convert_url`、`batch_convert`、`job_status`。`claude mcp add markitai -- uvx --from "markitai[mcp]" markitai-mcp`
- **`--profile rag|obsidian|okf` 按下游消费者塑造输出**：`rag` 把图片移到可见的 `assets/`（LlamaIndex 的 `SimpleDirectoryReader` 等摄取器会跳过隐藏路径）并改写 PDF 页标记；`obsidian` 提供可选 wikilink；`okf` 将 frontmatter 映射到 Open Knowledge Format。与 `--preset` 正交；不加它输出逐字节不变
- **`--llm-batch` 让整个目录的 LLM 阶段以半价运行**：在单模型的 OpenAI 或 Anthropic 池上走提供商的 Batch API，最长等待 `--llm-batch-timeout`（默认 1 小时）后转交 `--llm-batch-collect`
- **`--llm-batch` 覆盖图片分析与页截图**：`--alt`/`--desc` 与 `--screenshot` 同批提交、同享折扣。页数超过单次调用上限的文档改为实时增强，因为批处理每多一轮就多一次等待
- **网页端可以选输出格式**：与 Preset 并列，并且刻意不随 LLM 开关隐藏。profile 塑造的是输出形状而非增强，纯本地转换同样能带
- **三个限制单文档成本的断路器**：`llm.max_requests_per_document`（默认 50）限制重试倍增；`llm.max_vision_pages_per_document` 在发送前检查，超限文档一分钱不花；`llm.max_cost_per_document_usd` 每次拿到回答后计费，约束该文档后续还能花多少。触发后跳过剩余增强、保留未增强产出。两个成本上限默认关闭
- **旧版 `.doc`/`.ppt` 转换改由 `markitai[legacy]` 提供**：内置 Rust 的 anydoc 后端以毫秒级处理 Office 97-2003，任何平台都不需要 Microsoft Office 或 LibreOffice
- **`--ocr --llm` 现在有名字、有度量、有开关**：这个组合发送的是页图给视觉模型而非运行 RapidOCR，此前无处说明。help、`doctor` 和中英 CLI 指南统称它 VLM-OCR，metadata 记录实际走了哪条路径，`MARKITAI_NO_VLM_OCR=1` 可彻底禁用
- **超长页面截图改为切片而非压缩**：高于 `screenshot.tile_height`（2000px）的页面切成多张全宽 tile，不再被压成一条读不了的长图
- **数学公式以 LaTeX 形式回来了**：每个会看到公式的阶段都要求输出 `$...$` / `$$...$$`。pymupdf 会把行间公式当图片交出，把行内数学揉成会吞掉旁边正文的 markdown 噪声
- **CLI 转换可以选择进入网页端历史**：`--record-history`（或 `MARKITAI_RECORD_HISTORY`、或 `history.record`）把完成的运行记录为 `~/.markitai/serve/jobs/` 下的一个任务
- **转换质量现在有三种度量**：快照护栏冻结 PDF/DOCX/PPTX/XLSX 样本集的默认输出，可选的 A/B harness 比较 prompt 与模型改动，webextract 语料对照 defuddle 给 HTML 抽取打分
- **serve 的 API 契约由机器校验**：每个 JSON 路由声明 pydantic 响应模型，`scripts/export_openapi.py` 导出带 SSE 载荷的 schema，并有测试拿 `webapp/src/api/types.ts` 与之比对
- **defuddle 移植有了清单与上游追踪**：`PORT_MANIFEST.md` 记录每个 `webextract` 模块跟踪的上游来源和语料 pin 的提交，测试保证两个 pin 一致
- **网站提供 `/llms.txt`**，README 带一张对照 markitdown、docling、anydoc 的表格，明说各自更擅长什么
- **`NOTICE` 记录 MIT 自身覆盖不了的第三方义务**：AGPL-3.0 的 PyMuPDF 技术栈及其对再分发与网络使用的含义、`webextract/` 所基于的 defuddle 移植（MIT © kepano），以及源自 marker 的基准打分器
- **CI 会因非商用或意料之外的 copyleft 依赖而失败**：`scripts/check_licenses.py` 从无 extras 的安装读取许可证元数据，拒绝任何非商用或专有许可，以及白名单之外的 AGPL/GPL 包
- **Substack 文章提取**能读已渲染正文和 `window._preloads` JSON，自定义域名和署名栏日期也算在内；Notes 提取与通用回退照旧
- **Linux 桌面代理发现**读取 GNOME/Unity 和 KDE 手动设置的 HTTP 代理及绕过列表。显式的环境代理仍然优先；PAC、仅 SOCKS 和需要认证的桌面代理设置不会导入
- **可选的基准 LLM 评分**：`score_with_llm_judge` 返回经过校验的内容、结构和噪声分数，复用离线缓存，限制输入长度，也不会自己重试。缓存未命中时必须指定模型并设置 `allow_network=True`；默认基准仍是离线的启发式评分
- **八种格式不再需要 extra**：`.tsv`、`.xml`、`.rst`、`.org`、`.tex`、`.odt`、`.ods`、`.rtf` 现在只靠标准库就能在基础安装里转换。制表符分隔文件本就是一张 Markdown 表格，OpenDocument 文件本就是一包 XML
- **`.rtf` 有了真正的原生解析器**：分词器处理分组、控制字与 `\'xx` 字节，驱动一份按分组进出栈的排版状态，于是 outlinelevel 与样式表名变成标题，`\trowd`/`\cell`/`\row` 变成 Markdown 表格，Word 97 的 `\pntext` 符号与新式 `\ls`/`\ilvl` 列表保住各自的项目符号与层级，`HYPERLINK` 域变成链接，`\ansicpg`/`\fcharset` 正确解码中日韩与 Windows 代码页。不需要 extra，也不需要抽取引擎
- **每个布尔 CLI 开关都有明确的否定形式**：新增 `--cache`、`--compress`、`--no-pure`、`--no-screenshot-only`，配置里的默认值可以在命令行上朝任一方向覆盖

### 变更

- **三个 LLM 路由层合并为一个 `MarkitaiRouter`**：本地提供商经各自的 handler 表分发，标准模型交回 litellm 自己的分组负载均衡、冷却与 fallback，不再手工重实现
- **结构化 LLM 调用采用模型真正支持的最强模式**：先原生 tool calling，再 `response_format` JSON schema（这也顺带启用了 claude-agent 提供商此前从未走过的原生路径），markdown-JSON 降为最后手段
- **四个入口共用一条 URL 管线**：`serve`、Python API、`markitai <url>` 和 URL 批量各自维护着几乎相同的「抓取 → 本地化 → 增强 → 写出」副本，现统一到 `workflow/url.py` 的一条级联
- **Prompt 缓存真正生效了**：五个系统模板把逐请求变量放在靠前位置，稳定前缀短到任何提供商的缓存都匹配不上。变量段移到末尾，指令文本未变
- **转换一个 `.txt` 不再加载 PDF 与 Office 工具链**：转换器此前全部急切注册，每个进程多花约 430ms，且无论输入是什么都会拉起 markitdown、Magika 和 onnxruntime
- **一个提供商只在一张表里描述**：它的默认模型和 API key 变量曾被手抄到凭据检测、安装向导、`serve` 启动候选和 `init` 另外四处，几份抄本早已各自漂移
- **退役告警与成本估算改为推导**：替代模型取自默认表，退役日期取自 litellm 自己的记录，取代了一个已经过时的字面量和一个盖在所有模型上的硬编码日期
- **自解释错误不再泄漏异常类名**：缺少 OCR 后端、文件超限、格式不支持时只打印可操作的那句话
- **对照语料与算法 pin 到同一个上游**：样本从 2026 年 3 月快照重新同步到 defuddle 0.19.3（83 → 208），并补齐了新语料暴露出的六项上游行为
- **webextract 少解析也少复制**：fragment 与 document 统一到一条 lxml 路径，五份 block 标签表收敛，重试不再深拷贝整棵树，`srcset` 只剩一份实现
- **默认安装体积减少 155 MB**，633 MB 降到 477 MB。`opencv-python` 完全移出核心，RapidOCR 移到新的 `ocr` extra。**扫描件与图片 OCR 现在需要 `uv tool install "markitai[ocr]" --force`**
- **去掉 OpenCV 后图片质量反而更好**：在所有实测样本上纯 Pillow 都胜出，PSNR 与 SSIM 更高、文件小 7.7%，因为 OpenCV 的 `INTER_LANCZOS4` 缩放跳过抗锯齿预滤波
- **所有依赖升级到当前版本**：56 项更新中包括 `litellm` 1.91.1 → 1.97.0、`markitdown` 0.1.6 → 0.1.7，并对升级后的依赖树重跑了许可证审计
- **PDF 引擎下限提升到 `pymupdf4llm>=1.28.2`**：1.28.0 引入的 `pymupdf-layout` 采用 Polyform Noncommercial 许可，直接禁止商用；1.28.2 恢复了 AGPL-3.0 双许可
- **安装器不再询问 LibreOffice**：幻灯片渲染是可选的运行时路径，Windows 与 macOS 上本机的 MS Office 已能覆盖，引导安装彻底不再过问；PPTX `--screenshot`/`--ocr` 转换在本机没有任何渲染器时于转换时告警，并给出对应平台的安装命令
- **`--no-pure` 优先于 `MARKITAI_PURE`**：命令行显式给出的开关现在双向覆盖环境变量，之前是环境变量优先

### 移除

- **三平台 Office 自动化整体退役**：约 1.3k 行 Windows COM、macOS AppleScript 和 LibreOffice CLI 驱动代码，连同围绕它的批量预转换机制。它服务的格式改由 `markitai[legacy]` 覆盖，LibreOffice 的文案收窄到 PPTX 幻灯片渲染
- **`kreuzberg` extra 与 `-b kreuzberg` 后端全部移除**：它覆盖的每一种格式现在都原生转换，这个 extra 已经没有任何东西可以解锁。`--backend` 只剩 `native|cloudflare`，`fetch.kreuzberg_convert_enabled` 配置项删除，传入 `--kreuzberg` 会得到一条说明 `.rtf` 已原生转换的用法错误，而不是指向某个替代写法
- **六个已弃用的抓取与后端别名全部移除**（`--playwright`、`--defuddle`、`--static`、`--jina`、`--cloudflare`、`--kreuzberg`），改用 `-s` 与 `-b`。传入已移除的名字现在是会指出替代写法的用法错误，`--help` 的选项从 36 个降到 30 个
- **LLM 路径里两层多余的防御**：一个手写的 JSON 模式策略压在本就做同一件修复的结构化阶梯之下，以及一份实际流量早已不再走到的手抄 Copilot 价格表
- **`llm.prompts.page_content_system` 与 `llm.prompts.page_content_user` 两个配置项**，连同它们配置的那条代码路径：其 docstring 声称在用它的两种模式早已改道，所以设了没有任何效果
- **FFmpeg 不再被检查、宣传或安装**：markitai 从来不支持音视频，`doctor` 却把它呈现为「音视频文件处理」，安装脚本还会主动询问

### 修复

- **OpenAI 系默认模型刷新到当前代**：`gpt-5.6-luna` 取代 `gpt-5.4-nano`，$0.20/$1.20 对 $0.20/$1.25。陈旧的默认值只有在模型开始拒绝服务时才出错，因此新增守护测试：任一默认距 litellm 记录的退役日期不足 120 天即失败
- **Azure OpenAI 部署终于用上配置里写的 `api_version`**：`litellm_params` 从未声明这个字段而未知键默认被忽略，所以文档给出的 Azure 示例里它在解析时就被丢弃，从未到达 litellm
- **`--ocr` 转出来的 PDF 重新按页划分**：两条 OCR 路径都用空行拼接各页，纯截图模式写的标记又无人匹配，于是 OCR 产出到达 LLM 分批、漂移保护和输出 profile 时是一整段没有页边界的文本
- **LLM 调用失败的那篇文档，布局与其余保持一致**：所有失败路径都在管线的 profile 步骤之前返回，`--profile` 运行中会剩下一个仍是默认资源布局的文件
- **`--llm-batch` 拒绝 `--ocr` 时会说清原因**：页图走在文档请求内部，而 batch 第一阶段关闭 LLM 转换，那正是用本地 OCR 读扫描页、而非渲染页图的分支
- **`--llm` 在没有配置文件时形同虚设**：从 `MODEL` 或已检测到的 key 填充模型池的那一步被配置自身的 `llm.enabled` 拦住，且运行在 `--preset` 与 `--llm` 生效之前
- **`--resume` 无法真正续跑**：CLI 批量路径从未写出加载器所需的基准状态文件，中断的运行只留下无法回放的增量 sidecar，于是每篇文档都重新付费
- **`--llm-batch-collect` 现在遵守 `--config-json`**：它构建配置时丢掉了内联覆盖，导致 collect 的实时兜底用的是配置文件里碰巧写的那个池
- **发往推理模型的批请求不再整批失败**：OpenAI 的 batch 部署在推理开启时拒绝 function tools，于是每篇文档都返回 400 并回落到实时重跑。现在对「推理能力模型 + tools」的请求注入 `reasoning_effort="none"`
- **`router_settings.fallbacks` 真正生效**：此前部署是按 id 直接寻址的，litellm 从未看到可供回落的分组。未指定 `default` 分组的配置现在会在启动时明确报错
- **Claude 模型不再在用量报告里显示为 $0**：litellm 对表里没有的名字会返回零价格，读起来像「免费」，还掩盖了模糊匹配本可找到的条目。相关地，`gpt-5.4-codex` 曾被按普通 `gpt-5.4` 计价
- **被截断的 LLM 响应现在计入成本报告**：达到 token 上限被截断的响应在记录用量之前就被丢弃，于是最贵的那些调用恰好是合计里缺失的
- **改进 prompt 现在真的会改变输出**：持久缓存按调用的类别而非 prompt 本身作键，所以 prompt 改动之前转换过的文档会一直返回旧结果
- **失败的 LLM 调用不会再让文档永久空白**：所有重试都返回空时，那个空字符串被写进了没有过期时间的缓存，此后每次运行该文档都「转换」为空
- **已完成的转换不会再报告失败**：onnxruntime 在解释器关闭时的清理偶尔会 abort 进程，把一次已经写出产物的转换变成非零退出
- **没装 OCR extra 时的 `--ocr` 不再报告成功**：它会回落到图片占位符并以 0 退出，文件「转换」完成却什么都没读出来
- **安装提示给出的是真能用的命令**：extras 名被 rich 当作标记吃掉，而 `pip install` / `uv add` 作用于当前项目而非 markitai 所在的隔离环境。现在由一个 helper 统一渲染命令并读取 `sys.prefix`，pipx 装的会被告知用 pipx，`uv tool` 装的会被告知用 `uv tool`
- **wheel 现在带上了 `LICENSE` 与 `NOTICE`**：`license-files` 指向的路径在包目录内并不存在，于是每个已发布的 wheel 都没有许可证
- **对齐 defuddle 0.19.3 的六个抽取缺口**：可关闭的 `aria-hidden` 浮层内正文、CodeMirror 代码块、文中图片行、Substack Notes、SVG 图形、行内相关文章块。resync 还修复了它新暴露的问题：Hugo admonition、lightbox 去重、LaTeX 图片服务、`<noscript>` 回退与带行号的代码
- **Reddit、Hacker News 与 YouTube 页面重新受质量校验**：它们的校验门注册在一个 extractor 从不产出的 profile 名下，三者都落到了通用文章校验
- **PDF 内嵌图片在符号链接与树外输出目录下不再丢失**：pymupdf4llm 可能写出符号链接解析后的路径或相对工作目录的路径，而此前只改写了其中一种写法
- **CMYK JPEG 不再导致图片压缩失败**：来自印刷源 PDF 的图片走到了只处理 RGBA、P、LA 的分支而被丢弃
- **截图视口设置终于起作用**：`screenshot.viewport_width` 与 `viewport_height` 有声明、有描述、也发布在 schema 里，却没有任何代码读取
- **两个看起来可配置、实际从来不可配置的设置**：`auto_proxy` 和截图的 `full_page` 用带默认值的 `getattr` 读取，而没有任何配置模型声明它们
- **每条抓取路径都遵守 `NO_PROXY`**：统一的解析器此前没有任何调用方，Playwright 启动、两条 Cloudflare 路径和批量渲染器都在不查例外列表的情况下解析代理，macOS 与 Windows 的系统列表也不例外
- **代理自动检测不再凭空发明代理**：它此前靠探测常见本地端口的 TCP 连通性判断，而在 TUN 模式 VPN 下任何端口都会成功
- **`markitai --help` 不再以 Batch API 开场**：`--llm-batch` 系列选项不属于任何面板，而未分组选项会被优先渲染。示例里还提供过一个从不存在的 YouTube 转换
- **安装器只在默认索引确实慢时才询问镜像**，不再对所有没配代理的用户发出警告。此前一位法兰克福用户的第一印象，是一条关于中国镜像的提问
- **安装器现在尊重对 OCR 说的「不」**：建议列表被合并回选择结果，导致拒绝了仍会安装
- **测试套件对开发者自己的配置免疫**：真实的 `~/.markitai/config.json` 或 `.env` 曾让部分测试在真正使用 markitai 的人的机器上失败
- **移动端转换采用紧凑的一体输入卡片**：URL 输入区下面是选项、上传和转换共用的一行，取代三层堆叠控件，触控目标保持 44px，移动端 CLI 预览自然换行；桌面端仍是行内输入布局
- **输入卡片的三个操作统一为轻量样式**：选项、上传文件和转换都是无边框、等距内边距的控件，悬停与聚焦表现一致；转换按钮靠更重的字重和浅色强调底色保持主操作的辨识度，不再用实心色块
- **选项面板改为分组**：预设作为首行主选项，保留「已调整」状态与提示；「增强」汇集 LLM、OCR 与图片分析；「输出」放输出规范；URL／文件抓取选择器、内容源开关和缓存／压缩开关收进「高级」折叠区，其中任一项已非默认时自动展开。所有控件、提示、联动规则与翻译都没变
- **预设与转换选项统一联动**：面板、API 请求和 CLI 预览读的都是服务端的预设定义和解析规则。选择预设会重置它的五项功能；图片分析随 LLM／纯净模式暂停和恢复；截图来源与远程后端在三处解析结果一致。CLI 预览默认按服务端预设生成「预设加差异」的精简命令，并写明它假设的是默认配置；本地配置不同时，勾选「包含配置覆盖参数」就能保留截图来源、纯净模式、缓存和压缩的显式关闭开关

### 安全

- **`markitai serve` 现在签发访问令牌**：启动横幅打印登录 URL，来自其它机器的请求必须出示令牌，此前无鉴权的部分访问改为 401
- **`markitai serve` 不再能被指使去探测它所在的网络**：`--host 0.0.0.0` 时，局域网内任何机器都能提交 URL 让服务端去抓取，包括私网地址和云元数据端点
- **绑定到非回环地址时会说明这意味着什么**：`--host 0.0.0.0` 会警告任何能连到该服务的人都可以转换文件，并读取、下载或删除全部转换历史
- **`fetch.remote_consent=ask` 现在在 X/Twitter 富化前也会询问**：那条路径此前不经提示就走向远端，只遵守更早做出的决定

## [0.23.0] - 2026-07-18

### 新增

- **网页工作台的预览可导出为 PDF**：渲染后的 Markdown 会打印成整洁的 A4 文档，四边都有真实页边距，代码块与图片不会在页面中间被截断，跨页表格会重复表头。「PDF 设置」菜单中提供可选的自定义页眉页脚（默认开启）；PDF 始终以浅色打印，因此在 Chrome、Safari 和 macOS 预览中效果一致

### 变更

- **网页工作台现已适配手机与窄屏**：≤780px 采用统一布局，让输入区、选项与任务列表形成一条连贯的单列。顶栏收拢为标识加历史/设置，语言与主题切换移入设置弹窗，外部链接与通知停靠到页面底部，批量下载移到任务列表下方
- **批量下载在桌面端移到转换选项旁边**，不再与工作台顶部的清空操作争抢位置

### 修复

- **本地服务现在会拒绝跨源与 DNS 重绑定请求**：Host/Origin 白名单（回环地址、IP 字面量主机，以及你传入的任意 `--allowed-host`）可阻止恶意网页访问 API，否则它可能读取已保存的提供商凭据，或驱动服务端去抓取 URL。同源使用与局域网绑定不受影响
- **就地重试与 LLM 增强变得可靠**：在同任务其他条目仍在转换时重试某条，不再让它卡在「排队中」；增强失败会保留可用的基础结果，而非把该行降级为无输出的错误；重跑被中断后重启也不再从历史中抹掉整个任务；重跑会覆盖自身此前的输出，而不是累积带版本号的重复文件或返回过期的增强版本
- **扫描文档的 OCR 在整批任务中保持准确**：稀疏页的分块回退不再全局降低检测阈值，后续页面均以完整置信度识别
- **LLM 替代文本可应用于名称含空格或中文的图片**：此前以百分号编码写入的引用无法匹配，导致这些资源的图注被静默丢弃
- **X 的引用推文保留原文**：引用内部的省略号或「显示更多」文本不再对完整内容伪造截断标记
- **仅修改提供商的 API Key 不再固化其默认 API Base**：当连接使用提供商默认值时，编辑框为空（默认值显示为占位符），保存后会继续跟随该默认值
- **整段历史下载会自我清理**：生成的归档在下载完成后即被删除，不再残留为孤儿文件；构建过程中被删除的文件会被跳过，而不是导致下载失败
- **任务全部成功时完成通知显示「N 完成」**，而非「N 完成 · 0 失败」；本地服务在为大型输出目录计算体积时，不再阻塞其他任务的实时进度

## [0.22.0] - 2026-07-14

### 新增

- **`markitai serve` 新增本地网页工作台**：安装 `serve` 附加组件后，可以上传文件或文件夹、提交 URL、查看实时进度、预览和下载结果，并在中英双语的无障碍界面中回看七天内保存在磁盘上的任务历史
- **LLM 设置可直接在网页工作台中完成**：它会找出你机器上可用的本地与 API 提供商，实时列出模型，让你配置带权重的部署、在不暴露已保存凭据的前提下测试连接，转换任务用的也是这同一套设置
- **网页任务可以重试和对比**：失败的文件或 URL 条目可以单独重试，结果多了可以筛选，任务在后台完成时会有通知，等效的 CLI 命令一键复制，基础 Markdown 与 LLM 增强版本并排对比

### 修复

- **文件名包含空格或括号的 PDF 现在也会分析内嵌图片**：pymupdf4llm 写出图片时会清洗源文件名，但此前收集阶段仍按原名匹配，导致 `--alt`、`--desc` 和 `rich` 静默跳过全部内嵌图片，而页面截图不受影响。现在会解析 Markdown 中实际写入的图片引用，保留原有前缀匹配作为后备，并在引用的资源不存在时记录警告
- **Windows 上的网页结果可正常下载嵌套资源**：API 路径统一使用 URL 分隔符，不再泄漏文件系统反斜杠，因此图片及其他生成资源在所有受支持系统上都能通过相同链接打开

## [0.21.1] - 2026-07-12

### 修复

- **macOS 与 Windows 全新安装不再尝试从源码编译 litellm**：litellm 1.92.0 只发布了 Linux 平台的 wheel，其他平台的 `uv tool install markitai` 会退回从 sdist 构建其新增的 Rust 扩展。没有 Rust 工具链时直接失败，构建环境选中不受支持的 Python（如系统默认的 3.14）时则报出大段 pyo3 错误。现已排除该版本，依赖解析会落在各平台均有 wheel 的版本上

## [0.21.0] - 2026-07-12

### 变更

- **旧版 `.xls` 文件改为纯 Python 转换**（经 MarkItDown 的 xlrd 路径，与 `.xlsx` 同引擎）：任何平台都不再启动或依赖 Office 应用与 LibreOffice。单元格内容与此前 Excel 自动化的输出完全一致（真机验证）；转换更快，也不再受应用级锁串行化影响
- **持久 LLM 缓存按模型池划定作用域**：更换模型配置后不再继续使用旧模型产出的结果，而轮换 API key 不会让缓存失效；同一内容的文档改名后清洗结果仍然命中。升级后现有缓存会一次性全部未命中；旧条目本来就读不到（见"修复"）
- **结构化 LLM 调用接入完整的传输层重试**（指数退避、配额/账单错误短路、空响应重试）。此前这类调用完全绕过了纯文本调用一直具备的重试机制。模型持续输出退化内容时最坏情况的尝试次数会相应增加，这是真正执行重试的代价
- **独立 `.urls` 批量在 `--screenshot` 多源场景下获得视觉增强**，与单条 URL 转换同一页面的行为保持一致
- **内部：fetch、LLM 与 CLI 三个子系统在行为不变的前提下完成结构重组**。机器可执行的 import 分层契约进入 CI，fetch.py 从约 3000 行缩至约 1100 行并按策略拆分模块，LLM 管线由五份手写副本收敛为单一引擎，报告/退出码/输出处理在全部输入路径间单源化。针对 0.15–0.20 全部修复条目的 43 项回归矩阵验证零回归

### 移除

- **Python 3.14 支持**：包与安装脚本的 `requires-python` 收窄为 `>=3.11,<3.14`，CI 与发布矩阵不再测试 3.14；仅为让 3.14 可解析而存在的平台条件 ONNX Runtime 约束一并移除
- Excel COM 自动化（Windows）与 Excel AppleScript 自动化（macOS），以及批量预转换、重任务调度中的 `.xls` 条目，`.xls` 已经不再需要它们

### 修复

- **跨会话 LLM 缓存此前从未真正命中**：持久缓存的读取与写入在缓存键的 model 列上不一致，任何一次运行都读不到之前运行写入的结果，重复处理同一文档每次都在全额消耗 LLM 调用。现已修复读写一致（回归测试锁定），重跑未变更的文档近乎即时且零成本
- **macOS 上经 Microsoft Office 转换 `.doc`/`.ppt` 此前整体不可用**：Word 与 PowerPoint 的 AppleScript `open` 没有返回值，脚本的文档绑定从未建立。首个引用即报错，而错误处理器对同一变量的引用又把真实错误掩盖成难以理解的"variable openedItem is not defined"，文档从未被关闭，残留的僵尸文档让应用状态随重试持续恶化。脚本现在按唯一暂存名绑定文档、在保存前后两个名字下分别关闭、并始终抛出原始错误。Excel 的"Parameter error -50"同根同修（`.xls` 已改纯 Python 转换后此路径不再使用，见"移除"，但该修复同样覆盖 PPTX 转 PDF 导出路径）
- **macOS：Office 更新后应用首次被脚本拉起时 `.doc`/`.ppt` 转换失败**（"document never registered" / PowerPoint 错误 -9074）。该状态下应用会静默丢弃 Word 带参数的 `open` 请求但正常响应其他事件；Word 脚本现在会在轮询停滞时改用无参数 `open` 重试（实测可穿透并治愈该状态）并记录恢复日志。停滞与 -9074 的报错文案改为经过验证的补救措施（手动打开该应用一次让其完成初始化），替换掉误导性的"应用卡死或过载，退出后重试"
- **截断或退化的 LLM 输出不再污染缓存**：截图提取与 URL 增强现在会像其他调用点一样拒绝因 max_tokens 截断的响应；URL 增强也不再缓存被截去退化尾部的输出
- **`markitai -I` 遇到无效配置文件不再抛出裸 traceback**：交互向导现在给出与其他命令一致的可操作错误信息
- **目录批量的报告不再丢失 URL 条目的截图数**
- **defuddle HTTP 客户端在代理或超时设置变更后会重建**（此前整个进程沿用首个客户端，而 jina 一直是正确处理的）
- **禁用的模型（`weight: 0`）不再影响缓存作用域**：禁用某个模型后其旧缓存结果不再被使用，增删禁用条目也不再让整个池的缓存失效
- **Copilot 认证状态在新版 Copilot CLI 下能正确读取**：新版 CLI 会写入 JSONC 风格注释并更换了登录状态键名，导致 `doctor` 与 `auth copilot status` 把解析错误当作"未登录"。现在两种格式均可识别，无法解析时报告"无法确定"而非"未认证"；CLI 自身报告成功的登录也不会再被后续的配置读取否决

## [0.20.0] - 2026-07-10

### Added

- **macOS 未安装 LibreOffice 时可使用本机 Microsoft Office**：旧版 `.doc`、`.ppt`、`.xls` 文件可通过 AppleScript 调用 Word、PowerPoint 或 Excel 完成转换，PowerPoint 也能先将 PPTX 幻灯片导出为 PDF 再渲染。此备选方案由 `office.macos_fallback` 控制，无头环境可将其关闭，`doctor` 和文档也会说明一次性的自动化授权要求

### Changed

- **公网 URL 不再询问，私网 URL 永远不出本机**：公网 URL 可以直接使用远程后备服务，首次使用前会在 stderr 输出一次进程级说明，覆盖 defuddle.md、Jina、Cloudflare、FxTwitter 和 Twitter oEmbed。私网、本机、DNS 解析到非公网地址及带凭据的 URL（包括路径中的敏感令牌）始终留在本机，`MARKITAI_NO_REMOTE_FETCH=1` 会挡住所有远程路径，显式指定的远程 `-s` 策略也一样
- **`doctor` 改为检查能力是否真的可用，而不只是看包有没有装**：普通检查只有 RapidOCR 和已配置的工作流会影响退出状态；Playwright 会实际启动 Chromium 来验证，活跃模型引用的环境变量也会查一遍，你要求 `--fix` 时它只安装并复查 Chromium，不动当前项目的依赖
- **`markitai init` 默认保留已有配置**：直接按 Enter 会选择 Keep，Update 和 Overwrite 仍需明确选择
- **入门流程从便携安装脚本开始**：首页会在首次绘制前判断是 Windows 还是 macOS 或 Linux，并推荐 `setup.ps1` 或 `setup.sh`，`uv tool install markitai` 保留为手动安装方式。页面还加了一个 60 秒无 LLM 示例，中文导航、屏幕阅读器和高对比度支持也都改好了
- **显式配置 `gpt-5.6-luna` 时可识别其 Copilot 价格信息**：OpenAI 和 ChatGPT 的自动入门配置仍用已普遍开放的模型，受限预览模型要自己选

### Fixed

- **`--quiet` 模式在单项和批量任务中保持一致**：你要求输出到 stdout 的 Markdown 照常输出，错误仍写入 stderr，`--quiet --dry-run` 不显示预览，URL 批量任务部分失败时保留成功结果并退出 10，进度和成功提示则一律不显示
- **未启用任何提取方式的图片转换不再误报成功**：单张图片未指定 `--ocr` 或 `--llm` 时会退出 1 并给出处理建议，不再以成功状态结束却没有任何输出
- **重复运行安装脚本会保留用户意图**：shell 和 PowerShell 脚本发现 `~/.markitai/config.json` 已存在时会跳过 `markitai init --yes`，同时保留已有 extras；即使 Markitai 已安装，显式设置的 `MARKITAI_VERSION` 也不会被普通升级路径绕过
- **首页快速开始命令在亮色模式下保持清晰**：深色命令面板现在始终用浅色文字和透明代码背景，命令快挤不下时就切成纵向排版；站点也提供了真实的 `/favicon.ico`，不再返回 404
- **Python 3.14 的依赖解析不再固定到不兼容的 ONNX Runtime**：约束改为按平台设置，Magika 的 Windows 上限在需要的地方保留，其他平台可以用支持 Python 3.14 的 ONNX Runtime 版本

### Security

- **配置输出默认隐藏秘密**：`markitai config list` 会递归遮蔽秘密和自定义请求头，`api_base` 只保留协议、主机名和端口，显式传入 `--show-secrets` 才会显示原始值
- **URL 凭据始终留在本机，也不会出现在诊断信息中**：终端错误、进度标签、`--dry-run` 预览、控制台与文件日志，以及自动生成的输出文件名都会去掉用户信息、敏感路径令牌、查询参数和片段；主机名只要解析到任何非公网地址，就不会越过远程分发边界
- **macOS Office 自动化会隔离不受信任的文档**：备选转换打开的是只读暂存副本，宏和外部链接更新都禁用，只绑定并关闭自己打开的文档，跨进程串行访问 Office 应用，可清理的暂存文件也以私有权限保存
- **无头安装不算同意安装可选软件**：没有可用终端时，便携安装脚本只装 uv、Python 和 Markitai，别的一概不装；显式设置 `MARKITAI_INSTALL_OPTIONAL=1` 才会装可选包、浏览器二进制、系统依赖和第三方 CLI

## [0.19.0] - 2026-07-10

### Changed

- **远程提取默认不再弹出确认**（`fetch.remote_consent` 默认值 `ask` → `always`）：公网 URL 会在本地策略失败后直接按链回退到远程提取服务（defuddle.md、Jina、Cloudflare，逐个尝试，成功即停），不再打断询问；首次使用会有一条 INFO 日志说明。私有/本地 URL 无论此配置如何都不会使用远程服务，netloc 携带凭据的 URL（`user:pass@host`）现在也视同私有。可通过 `fetch.remote_consent=ask`/`never` 或 `MARKITAI_NO_REMOTE_FETCH=1` 恢复询问或禁用远程服务
- **确认提示文案重写**（针对 `remote_consent=ask`）：提示现在会说明弹出原因（本地提取未成功）、逐个尝试的机制（一次一个、成功即停），并只列出实际在链中的服务，所以 Cloudflare（使用你自己的账户凭据）仅在已配置时出现。交互式确认现在也会先暂停实时进度显示，不再撕裂界面

### Fixed

- **`--resume` 此前完全不生效**：CLI 批量入口接受该参数，但每次都会从头重新处理所有文件。现在会加载已保存的状态，跳过已完成的文件，重试失败或中断的文件，纳入本次新发现的文件，并报告 `Resuming batch: N completed, M remaining`
- **输出命名恢复为追加式方案**：`sample.pdf` → `sample.pdf.md`（而非 `sample.md`），撤销了 0.15.0 引入的"替换扩展名"命名方案。那个方案会隐藏源文件格式、破坏多重后缀文件名（如 `archive.tar.gz`），并导致同一文件在单文件模式和批量模式下的输出名不一致
- **Windows 安装一行命令 404**：站点现在会部署 `setup.ps1`（文档指向 https://markitai.dev/setup.ps1，但此前只部署了 `setup.sh`）；中文更新日志的改动现在也会触发站点重新部署
- **Prompt 尾部 REMINDER 泄漏进清理结果**：使用较小模型时（在 `gpt-5.4-mini` 上观察到），视觉清理 prompt 末尾的 `REMINDER: ...` 指令行（连同 `---` 分隔符）可能被逐字回显到 `.llm.md` 输出末尾。现改用 `<document>` 标签定界文档、全部指令置于内容之前，并新增出口防护剥离回显的 prompt 片段（对已缓存的历史结果同样生效）
- **部分 URL 转换会静默跳过图片 alt 分析**：有截图但无多源内容的 URL（如经站点提取器抓取的 X 帖子）会落入纯文本 LLM 分支、不做图片分析；URL 批量模式则完全不分析图片，`--alt`/`--desc` 在这些路径下形同虚设。现在两条路径都会分析已下载的图片（alt 文本 + `images.json`）。stdout 模式的资产链接重写也不再用文件名覆盖 LLM 生成的 alt 描述
- **批量图片分析不再因"裸负载 JSON"而失败**：小模型有时会对单图批次直接返回裸的结果对象而非 `{"images": [...]}` 包裹结构，导致 Instructor 重试耗尽后降级为逐张分析并打出 ERROR 日志。JSON 修复层现在会就地矫正这类形状，批量分析直接成功
- **弱模型可能在 LLM 清理时破坏社媒帖正文**（引用推文的 blockquote 被拍平、CJK 与英文之间被私自加空格，在 `claude-agent/haiku` 上观察到）：标记为 `social_post` 的内容现在正文原样直通，LLM 仅生成元数据；其他文档类型的处理 prompt 也补充了 blockquote 保护与 CJK 空格显式规则
- **ChatGPT 连接错误此前不可重试且信息为空**：httpx 传输层错误（连接重置/拒绝、超时）被映射为不可重试的 `ProviderError` 且消息为空，导致所有重试层被绕过。现在标记为可重试并携带底层错误文本
- **控制台日志不再撕裂实时进度显示**：日志输出改经共享的 rich stderr console 路由，日志行会打印在 StageList 转轮上方而不再留下残影帧；quiet/stdout 模式也应用与普通模式相同的第三方重试噪音过滤（此前 instructor 的原始重试错误会直接漏到控制台）
- **LLM 增强失败现在在输出中可见**：当所有 LLM 路径都失败时，兜底生成的 `.llm.md` frontmatter 会带上 `llm_enhanced: false` 标记并打出 ERROR 级日志。此前降级输出的唯一线索只有一个空的 description

### Removed

- **死代码清理**：移除了不可达的异步 enricher 注册表、未使用的异常层级、无调用方的废弃辅助函数、仅供测试的工具函数，以及约 5MB 无引用的测试固件；`markitdown` 依赖从 `[all]` 收窄为实际用到的 office 附加组件（去掉 azure/audio/pdfminer/youtube 等传递依赖）；`httpx` 和 `lxml` 现在改为直接声明

## [0.18.0] - 2026-07-09

### Changed

- **网页提取效果对齐 Defuddle**：正文主体选择和噪声内容清理算法现已高度对齐 Defuddle 的实现（评分机制、内容模式识别、内容边界检测）。基准测试语料库均分对比 Defuddle 基准：91.04 → 92.72

## [0.17.0] - 2026-07-08

### Removed

- **移除 Gemini CLI 提供商**（`gemini-cli/`）：Google 已下线它依赖的 OAuth 接入流程。请改用直接的 `GEMINI_API_KEY`，或通过 OpenRouter 接入

### Added

- 支持 `COPILOT_GITHUB_TOKEN` 认证，检测优先级高于 `GH_TOKEN`/`GITHUB_TOKEN`
- 耗时较长的转换阶段会显示已用时长提示，避免长时间运行的 LLM 调用看起来像卡死

### Fixed

- **Cloudflare 抓取现在能正确走到站点专用提取器**：由 `/markdown` 端点切换为 `/content` 端点，使经 Cloudflare 抓取的页面获得与其他策略一致的提取质量
- 减少了重复的重试/校验日志噪音

### Changed

- 本地 provider 不再为未使用的扩展思考/推理输出浪费 token
- provider 检测和 `init` 现在会建议更便宜的默认模型

## [0.16.0] - 2026-07-07

### Added

- 新增 **B 站专栏（opus）提取器**，支持 `bilibili.com/opus/<id>` 页面
- **反爬/验证码检测**：能识别 Geetest、Cloudflare、reCAPTCHA、hCaptcha 等挑战页面，不再将其当作正常内容处理

### Changed

- **X/Twitter 提取重新以 DOM 解析为主**，仅当原生提取效果不足时才回退到 FxTwitter/oEmbed 补充方案

### Fixed

- X Article 的 URL 匹配、抓取性能、frontmatter 字数统计相关问题

## [0.15.0] - 2026-07-04

一个维护版本：依赖刷新、支持 Python 3.14，并经过多轮排查，修复了批量处理、抓取/缓存、LLM provider、图片处理和配置系统中 30 多个已验证的 bug。

### Added

- 支持 Python 3.14；包元数据中新增 MIT 许可证声明
- 通过 rich-click 实现分组、更快的 `--help`
- PDF 乱码/扫描文本检测，并给出使用 `--ocr` 的建议
- 跨页面重复出现的页眉/页脚会被自动清理
- VLM 退化保护（截断视觉/OCR 输出中的重复循环内容）
- HTML 提取与脚注处理效果对齐 Defuddle（MathJax/MathML、代码块、多种站点的脚注格式）
- 统一的 `-s/--strategy` 抓取参数（旧的分策略参数保留为已弃用别名）
- **远程抓取同意机制**：未经同意不会将 URL 发送给第三方服务（`fetch.remote_consent`、`MARKITAI_NO_REMOTE_FETCH`）
- PDF 隐藏文本清理（提示注入防护）：`security.pdf_sanitize`
- 混合数字/扫描文档的按页 OCR 路由
- 针对 Defuddle 基准语料库的转换质量基准测试工具
- 通过 release-please 实现发布自动化

### Fixed

多轮 bug 排查中的主要修复：

- 针对 X 2026 年改版重写了 DOM 提取器；FxTwitter 回退路径现在能被默认抓取链路正确调用到
- 原生支持 `.eml` 邮件解析、HEIC/HEIF/AVIF 图片输入（`markitai[heif]`）、CI 质量守护门槛
- `markitai init` 现在会合并进已有配置而非直接覆盖；各 provider 的登录失败提示更清晰
- 抓取/缓存正确性修复：AUTO 策略缓存复用陈旧数据、Playwright 上下文泄漏、代理自动检测误报、空 URL 导致批量任务崩溃
- LLM/provider 正确性修复：失败不再被误报为成功、视觉分析缓存污染、Copilot 并发临时文件竞争、阻塞调用导致的事件循环停滞、重试退避期间占用并发槽位
- 图片/转换正确性修复：EXIF 方向、LA 模式透明通道、未压缩图片命名、EMF/WMF 格式误标、OCR 引擎配置漂移、临时目录泄漏
- 配置/CLI 正确性修复：配置编辑器保存前先校验、修复符号链接安全检查、`llm.concurrency` 下限校验、JSON 日志格式修复、`config set` 类型强制转换与方括号写法支持

### Changed

- 输入路径与子命令混用现在会报错，而不是静默丢弃输入路径
- 单文件/URL 输入下 `-o out.md` 会精确写入该文件
- 诊断信息改为输出到 stderr，保持 stdout 管道输出干净
- 输出命名改为"替换扩展名"方案（`sample.pdf` → `sample.md`），**已在 0.19.0 中撤销**
- `image.stdout_persist` 现在默认开启
- 转换报告（`.markitai/reports/`）默认仅在批量任务中生成

### Security

- 解除 litellm 供应链安全钉版（改为 `>=1.83.0`），上游已完成审计并对发布版本签名

## [0.14.0] - 2026-03-25

- Added: Steam News 提取器；结构化 MathML 转 LaTeX；LibreOffice 功能性检测（不只是检测程序是否存在）
- Fixed: PDF 数学公式提取回退逻辑；Steam BBCode 内容的 XSS 防护；修复了不稳定的集成测试
- Security: litellm 钉版至 `<1.82.7`（供应链安全事件应对）

## [0.13.1] - 2026-03-23

- Added: 配置编辑器重新设计：模糊搜索、可滚动列表、原地刷新 UI
- Added: 为 66 个 Pydantic 配置项新增字段说明
- Fixed: 配置编辑器中 Esc 键支持、布尔值编辑器一致性、Literal 类型取值保留

## [0.12.1] - 2026-03-22

- Added: stdout 模式下的终端内联图片显示（Kitty/iTerm2），基于内容寻址的资产存储实现
- Added: 新增中文用户旅程文档
- Fixed: 静默/stdout 模式下 LLM 错误现在可见；修复 Kitty 协议图片格式问题；`init` 不再生成重复的 provider 条目

## [0.12.0] - 2026-03-20

- Added: 原生 HTML 提取流水线：基于 resolver 的提取、frontmatter 构建器、质量档案，以及针对 GitHub Discussions、X 讨论串、YouTube 页面的结构化提取器
- Added: 新增 `--static` 与 `--kreuzberg` CLI 参数
- Changed: HTML 文件默认改为走原生 webextract 流水线
- Fixed: URL 的 stdout 回退逻辑、共享缓存/信号量的线程安全、配置原子写入

## [0.11.2] - 2026-03-14

- Fixed: Windows 下的内存检测用于任务规模调节；`~/.markitai/` 目录改为延迟创建（避免只读场景下产生副作用）；输出/日志目录默认值改为 `None`（不再硬编码路径）

## [0.11.1] - 2026-03-14

- Added: 交互式向导中新增 pure 模式选项
- Fixed: 修复 `--pure` 错误触发视觉/截图路径的问题；降低了过于激进的"内容过短"判定阈值

## [0.11.0] - 2026-03-13

- Added: **`--pure` 模式**：LLM 透明直通（仅做文本清洗，不生成 frontmatter/不做后处理），与 `--llm` 解耦
- Added: 新增 `--keep-base`，可在生成 `.llm.md` 的同时强制保留基础 `.md`
- Fixed: URL 处理流程现在与文件处理一致地遵循 `--pure`/`--llm`/`--keep-base` 参数

## [0.10.0] - 2026-03-12

- Added: 无配置文件时，自动从环境变量和已认证的 CLI 中检测可用的 LLM provider
- Changed: `-v` 改为 `--verbose`（此前是 `--version`）；`-V` 表示 `--version`
- Changed: 通过延迟导入缩短冷启动耗时（约 5s → 0.3s）
- Fixed: alt 文本语言现在跟随文档语言，不再默认使用英文

## [0.9.2] - 2026-03-11

- Fixed: Copilot/Claude 登录改为始终使用继承的标准输入输出（修复凭据存储失败问题）；错误提示更清晰，不再是难以理解的包装异常

## [0.9.1] - 2026-03-09

- Added: 新增 `markitai doctor --suggest-extras`，作为安装脚本获取推荐 extras 的唯一权威来源
- Fixed: 修复安装脚本中的登录守卫和 extras 解析 bug；provider 名称的 Rich 标记转义问题

## [0.9.0] - 2026-03-09

- Added: 支持配置全局/按域名的抓取**策略优先级**，以及 `local_only_patterns`/`inherit_no_proxy`，用于将敏感域名限制为仅使用本地策略
- Fixed: LLM 输出不再把混合语言页面内容错误地翻译成其他语言

## [0.8.1] - 2026-03-06

- Added: **新增 Defuddle 抓取策略**（免费、无需认证），并作为最高优先级选项；新增 `--defuddle` CLI 参数
- Changed: 默认策略顺序调整为优先使用 Defuddle/Jina

## [0.8.0] - 2026-03-06

- Added: 通过 markitdown/kreuzberg 新增 20 多种文件格式支持（HTML、CSV、EPUB、MSG、IPYNB、Numbers、TSV、XML、ODS、ODT、SVG、RTF、RST、ORG、TEX、EML）；新增 GIF/BMP/TIFF 图片支持
- Fixed: Claude Agent SDK 兼容性升级；i18n 测试隔离问题；kreuzberg 注册时导入阶段的日志噪音

## [0.7.0] - 2026-03-05

- Added: 新增 **ChatGPT provider**（`chatgpt/`），通过 OAuth 设备码流程接入
- Added: 新增 **Gemini CLI provider**（`gemini-cli/`），后于 0.17.0 移除
- Added: 支持 `weight: 0` 显式禁用某个模型的路由
- Fixed: 修复所有模型 weight 都为 0 时路由器除零错误

## [0.6.1] - 2026-03-05

- Fixed: Claude Agent SDK 合规性修复；认证预检查现在能识别更多基于环境变量的凭据

## [0.6.0] - 2026-03-04

- Added: **Cloudflare 集成**：URL 使用 Browser Rendering，文件使用 Workers AI `toMarkdown`
- Added: 新增 Fetch Policy 引擎，支持域名配置和 Playwright 会话持久化
- Added: 可插拔的静态 HTTP 后端（`httpx`/`curl-cffi`）
- Fixed: 修复所有视觉模型被禁用时路由器除零错误；清理了代码库中 21 个死函数

## [0.5.2] - 2026-02-07

- Fixed: SQLite 连接泄漏；Windows 路径处理 bug；OAuth 过期状态误报；清除全部 Pyright 警告

## [0.5.1] - 2026-02-07

- Added: Playwright 自动滚动以触发懒加载内容；提取前清理 DOM 噪音（导航栏/广告/Cookie 提示条）；支持 `python -m markitai` 调用方式
- Changed: `init`/交互模式/doctor 中的默认模型全面更新；缓存指纹改为基于完整内容的哈希（此前仅用较短前缀，容易冲突）

## [0.5.0] - 2026-02-06

- Added: 新增 **`markitai init`** 配置向导与**交互模式**（`-I`）；`doctor --fix` 自动安装缺失组件
- Changed: 通过延迟加载模块，CLI 启动速度提升约 3 倍；批量 UI 简化为更紧凑的进度显示
- Fixed: Windows 下 LibreOffice/FFmpeg 检测问题；修复导致挂起的 Playwright 默认等待条件

## [0.4.2] - 2026-02-03

- Changed: 调整 Playwright 等待默认值以更好支持 SPA 页面
- Fixed: X/Twitter 页面现在会等待 JS 完全渲染后再截取；缓存改为遵循配置目录，不再使用硬编码路径

## [0.4.1] - 2026-02-02

- Added: 新增 **`markitai doctor`** 诊断命令；本地 provider 支持自适应超时；Claude Agent 长系统提示词支持缓存

## [0.4.0] - 2026-01-28

- Added: 新增 **Claude Agent SDK** 与 **GitHub Copilot SDK** 本地 provider；URL 支持 HTTP 条件缓存（ETag/Last-Modified）；新增 `--quiet`/`-q` 参数
- Changed: 模块结构重大调整（`cli/`、`llm/`、`providers/`）

## [0.3.2] - 2026-01-27

- Added: 新增中文 README 和中文安装脚本

## [0.3.1] - 2026-01-27

- Added: **SPA 域名学习机制**：自动检测并记住重度依赖 JS 的站点，避免重复浪费静态抓取尝试
- Added: Windows 性能调优（线程池规模、OCR 引擎单例、更快的图片压缩）
- Fixed: 提示词泄漏防护（拆分 system/user 提示词）；抓取时的自动代理检测

## [0.3.0] - 2026-01-26

- Added: 支持**直接转换 URL**，以及 `.urls` 批量文件
- Added: 多策略抓取（`static`/`agent-browser`/`jina`/`auto`），配合 SQLite 抓取缓存和截图功能
- Added: 新增 `--no-cache-for <pattern>` 精细化跳过缓存；`cache stats -v`
- Added: 官方 VitePress 文档网站上线（中英双语）
- Added: 新增 MIT 许可证；CI/CD 工作流

## [0.2.4] - 2026-01-21

- Fixed: Office/PPTX 兼容性补丁；符号链接安全加固；LLM 空响应重试；frontmatter 字段顺序问题

## [0.2.3] - 2026-01-20

- Added: **持久化 SQLite LLM 缓存**，支持 LRU 淘汰；新增 `cache stats`/`cache clear` 命令
- Added: 视觉感知的模型路由；PDF/图片并行处理

## [0.2.2] - 2026-01-20

- Added: 新增 `constants.py` 模块统一管理硬编码常量；扩充单元测试覆盖率

## [0.2.1] - 2026-01-20

- Added: 按文件统计 LLM 用量/成本；新增类型化的用量/资产模型；跨平台 Office/LibreOffice 检测
- Changed: 文件命名冲突时的重命名方式改为 `.v2.md` 风格的自然排序

## [0.2.0] - 2026-01-19

- Added: **Monorepo 重写**：采用 uv workspace、基于 LiteLLM 的 provider 接入方式、全新的转换器/工作流架构、JSON Schema 校验的配置
- Breaking: 全新配置格式和 CLI 语法；不再支持 Python 3.13 以下版本；移除旧版 `src/markitai/` 架构

## [0.1.6] - 2026-01-14

- Fixed: 模型路由相关 bug；文档准确性修正

## [0.1.5] - 2026-01-13

- Changed: 提示词管理与清洗模块重构

## [0.1.4] - 2026-01-13

- Fixed: LLM JSON 解析边界情况；日志格式问题

## [0.1.3] - 2026-01-12

- Changed: 采用 `src` 目录布局；新增 CI 工作流

## [0.1.2] - 2026-01-12

- Added: 网络健壮性增强（重试/超时处理）；新增面向 AI 助手的文档（`CLAUDE.md`、`AGENTS.md`）

## [0.1.1] - 2026-01-11

- Changed: 架构重大重构，采用服务层模式

## [0.1.0] - 2026-01-10

- Added: 基于能力的模型路由、Provider 惰性初始化、超时并发回退、`--fast` 执行模式、按模型统计的批量处理数据

## [0.0.1] - 2026-01-08

- Added: **首次发布**：CLI（`convert`/`batch`/`config`/`provider`）、Office/PDF/HTML 转换、5 个可回退的 LLM provider、图片处理、支持断点续传的批量处理
