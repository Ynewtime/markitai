# 网页工作台

`markitai serve` 在转换核心之上启动一个本地网页界面：上传文件或文件夹、提交 URL、查看实时进度、预览和下载结果、配置 LLM 提供商，并回看七天内的转换历史——全部在中英双语的无障碍界面中完成，手机与窄屏窗口同样可用。当你更喜欢用浏览器而非命令行、想并排对比基础输出与 LLM 增强版本，或想用可视化方式管理 LLM 提供商时，就适合用它。

一切都在本机运行：任务与历史保存在磁盘上，除了你配置的抓取策略和 LLM 提供商之外，不会向外界发送任何内容。

## 启动服务

服务依赖 `serve` 附加组件（FastAPI + uvicorn）：

```bash
uv tool install "markitai[serve]" --force
markitai serve
```

服务监听 `http://127.0.0.1:3600`，启动完成后会自动打开浏览器。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | `127.0.0.1` | 绑定的主机接口 |
| `--port` | `3600` | 监听端口 |
| `--no-open` | 关闭 | 启动后不打开浏览器 |
| `--allowed-host <hostname>` | — | 额外允许出现在 `Host`/`Origin` 头中的主机名（可重复传入） |

## 从其他设备访问

要在局域网内的其他机器或手机上使用工作台，绑定全部网卡并允许你将访问的主机名：

```bash
markitai serve --host 0.0.0.0 --allowed-host my-box.lan
```

出于安全考虑，服务只接受 `Host`/`Origin` 为 localhost、IP 字面量或经 `--allowed-host` 显式允许的主机名的请求；其他 DNS 名称一律拒绝，从而阻止恶意网页通过 DNS 重绑定攻击从你的浏览器访问 API。

::: warning
服务没有认证机制。任何能访问它的人都可以用你配置的 LLM 提供商发起转换、查看历史、读取 LLM 设置。除非你信任网络上的每一台设备，否则请保持默认的回环绑定。
:::

## 工作台功能

- **输入区**：拖入文件或文件夹，或粘贴 URL。选项与 CLI 预设一致——`minimal`、`standard`、`rich`——外加独立的 LLM 与 OCR 开关。
- **实时进度**：每个条目在转换时实时推送状态；任务在后台标签页完成时会发出通知。
- **条目操作**：就地重试失败条目，或对已完成条目单独做 LLM 增强而不重转其他条目；大量结果可快速筛选。
- **预览**：渲染后的 Markdown 预览，可对比基础版本与 LLM 增强版本；「PDF 设置」菜单可把预览打印成整洁的 A4 文档（可选自定义页眉页脚）。
- **下载**：单个输出文件、按任务打包的 zip、整段历史打包下载——还可一键复制任意任务的等效 CLI 命令。
- **限制**：每个任务最多 50 个条目，单个上传文件最大 100 MB。

## LLM 设置

设置弹窗管理与 `markitai config` 相同的配置：发现本地与 API 提供商、实时浏览模型列表、配置带权重的部署、在不暴露已保存凭据的前提下测试连接。改动同时作用于网页任务和后续的 CLI 运行。

## 历史记录

每个完成的任务会在 `~/.markitai/serve/jobs/` 下保留 **7 天**，随后自动清理。在历史页面可以重新打开任务以再次预览和下载输出、删除单个条目，或把全部历史打包下载为一个 zip。

历史条目带有来源标记。使用 [`--record-history`](/zh/guide/cli#record-history)（或 `MARKITAI_RECORD_HISTORY` 环境变量 / 配置项 `history.record`）记录的 CLI 运行会带着「CLI」徽标与浏览器创建的任务并列显示，行为完全一致——同样的 TTL、删除与打包下载——无需重启服务即可实时出现。

## API 概览

界面建立在一套小型 REST + SSE API 之上，也可以直接从脚本调用：

| 端点 | 说明 |
|------|------|
| `GET /api/capabilities` | 服务版本、可用预设、LLM 与附加组件状态 |
| `POST /api/jobs` | 创建任务（multipart 表单：`files`、`urls` JSON 数组、`options` JSON） |
| `GET /api/jobs/{job_id}` | 任务状态与条目 |
| `GET /api/jobs/{job_id}/events` | 实时进度流（SSE） |
| `POST /api/jobs/{job_id}/items/{item_id}/retry` | 重试条目，或以 `operation: "enhance"` 做 LLM 增强 |
| `DELETE /api/jobs/{job_id}/items/{item_id}` | 从任务中移除条目 |
| `GET /api/jobs/{job_id}/items/{item_id}/result` | 条目结果；配套资源经 `GET /api/jobs/{job_id}/files/{path}` 获取 |
| `GET /api/jobs/{job_id}/archive` | 整个任务打包为 zip 下载 |
| `GET /api/history` | 列出历史条目 |
| `GET /api/history/archive` | 全部历史打包为一个 zip 下载 |
| `DELETE /api/history/{job_id}` | 删除单个历史条目 |
| `/api/settings/llm*` | LLM 提供商、模型与部署管理 |

::: tip
同一套 `Host`/`Origin` 白名单同样保护 API：带跨站来源的状态变更请求会被拒绝，任意网页无法借你的浏览器驱动它。
:::
