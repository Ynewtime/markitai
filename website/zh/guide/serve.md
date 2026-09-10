# 网页工作台

`markitai serve` 在本机打开一个网页，用来转换文件和 URL。拖入文件、粘贴链接、看进度、预览并下载结果。支持中英文，桌面和手机都能用。

一切都在本机运行。任务和历史保存在磁盘上，除了你配置的抓取策略和 LLM 提供商，不会向外发送任何内容。

![markitai 网页工作台：一张转换输入卡片，含 URL 输入框、「转换」按钮与「选项」「CLI」「上传」开关。](/workbench.zh.png){.light-only}
![markitai 网页工作台：一张转换输入卡片，含 URL 输入框、「转换」按钮与「选项」「CLI」「上传」开关。](/workbench.zh.dark.png){.dark-only}

## 启动服务

装上 `serve` extra，然后启动：

```bash
uv tool install "markitai[serve]" --force
markitai serve
```

浏览器会自动打开 `http://127.0.0.1:3600`。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | `127.0.0.1` | 绑定的网卡。要让其他设备访问，用 `0.0.0.0` |
| `--port` | `3600` | 监听端口 |
| `--no-open` | 关闭 | 启动后不打开浏览器 |
| `--no-auth` | 关闭 | 禁用访问令牌 |
| `--allowed-host <hostname>` | — | 通过域名而非 IP 访问时，额外放行的主机名（可重复） |

## 访问令牌

启动时服务会打印一个以 `#token=…` 结尾的 URL。本机请求不需要令牌；其他设备必须打开这个 URL，脚本则以 `Authorization: Bearer <token>` 发送令牌。

设 `MARKITAI_SERVE_TOKEN` 可以让令牌在重启后保持不变。`--no-auth` 完全去掉令牌：其他设备仍能上传、下载和查看历史，但 URL 目标只能是公网地址，且不能改 LLM 设置。

## 从其他设备访问

绑定全部网卡，再在手机或另一台电脑上打开打印出来的令牌 URL：

```bash
markitai serve --host 0.0.0.0
```

如果你用域名而不是 IP 访问，加上 `--allowed-host my-box.lan`。未知主机名一律拒绝，用来防止恶意网页的 DNS 重绑定攻击。

::: warning
令牌 URL 就是凭据。拿到它的人可以用你的 LLM 提供商发起转换，还能查看、下载和删除历史。只分享给你信任的设备。
:::

## 工作台功能

- **输入区**：拖入文件或文件夹，或粘贴 URL。默认界面什么都不问，所有选项收在**选项**里，分为预设、增强（LLM、OCR、图片分析）、输出（profile）和高级（抓取策略、缓存、压缩）。
- **CLI 预览**：展开选项或按下 **CLI** 按钮时，显示等价的命令行，可直接复制。
- **实时进度**：每个条目实时推送状态，任务在后台标签页完成时会发通知。
- **单条操作**：重试失败的条目，或对已完成的条目单独做 LLM 增强，不用重转其他条目。
- **预览**：渲染后的 Markdown，可对比基础版和增强版，还能打印成 PDF，页眉页脚可选。
- **下载**：单个文件、按任务打包的 zip，或整段历史打包成一个压缩包。
- **限制**：每个任务最多 50 个条目，单个上传文件最大 100 MB。

### 预设、覆盖与命令

预设和 CLI 一致：

| 预设 | LLM | alt | desc | screenshot | OCR |
|------|:---:|:---:|:----:|:----------:|:---:|
| `minimal` | – | – | – | – | – |
| `standard` | ✓ | ✓ | ✓ | – | – |
| `rich` | ✓ | ✓ | ✓ | ✓ | – |

没有预设会开 OCR，它始终单独选。选一个预设会重置这五个开关；单独改一个开关则保留其余，并显示为**自定义**。

每个选项都有悬停或轻触帮助，说明它依赖什么、数据去哪、可能花多少钱。

## LLM 设置

设置弹窗编辑的就是 `markitai config` 那套配置：发现提供商、浏览模型列表、设置权重、测试连接，且不会暴露已保存的 key。改动同时作用于网页任务和之后的 CLI 运行。

## 历史记录

完成的任务在 `~/.markitai/serve/jobs/` 下保留 7 天。在历史页面可以重新打开任务、再次下载输出、删除单条，或把全部历史打包下载。

用 [`--record-history`](/zh/guide/cli#record-history) 启动的 CLI 运行也会出现在这里，带一个 CLI 徽标，行为和其他任务一样。

## API 概览

界面建立在一套小型 REST + SSE API 之上，脚本也能直接调用：

| 端点 | 说明 |
|------|------|
| `GET /api/capabilities` | 服务版本、预设、LLM 与 extras 状态 |
| `POST /api/jobs` | 创建任务（multipart 表单：`files`、`urls` JSON 数组、`options` JSON） |
| `GET /api/jobs/{job_id}` | 任务状态与条目 |
| `GET /api/jobs/{job_id}/events` | 实时进度流（SSE） |
| `POST /api/jobs/{job_id}/items/{item_id}/retry` | 重试条目，或以 `operation: "enhance"` 做 LLM 增强 |
| `DELETE /api/jobs/{job_id}/items/{item_id}` | 从任务中移除条目 |
| `GET /api/jobs/{job_id}/items/{item_id}/result` | 条目结果；配套资源经 `GET /api/jobs/{job_id}/files/{path}` 获取 |
| `GET /api/jobs/{job_id}/archive` | 整个任务打包为 zip 下载 |
| `GET /api/history` | 列出历史条目 |
| `GET /api/history/archive` | 全部历史打包为一个 zip 下载 |
| `DELETE /api/history/{job_id}` | 删除单条历史 |
| `/api/settings/llm*` | LLM 提供商、模型与部署管理 |

来自其他网站来源的状态变更请求会被拒绝，任意网页无法借你的浏览器操作这个服务。
