# 输出 Profile

Profile 为特定的下游消费者塑形输出文件。它和[预设](./configuration.md#预设)是两回事：预设决定跑哪些功能，profile 决定文件长什么样。

不设 profile 时，输出就是默认形态。下面的每一项变换只在设了 profile 时才发生。

## 启用 profile

```bash
# CLI
markitai document.pdf --profile rag -o out/
```

```json
// markitai.json
{ "output": { "profile": "rag" } }
```

```python
# Python API
import markitai

out = markitai.convert("document.pdf", output_dir="out/", profile="rag")
```

## `rag` — 检索管道

默认输出把图片放在隐藏目录 `.markitai/assets/` 里，而大多数摄取器会跳过隐藏路径（LlamaIndex 的 `SimpleDirectoryReader` 默认就是）。`rag` profile 让输出对摄取器友好：

- **可见资源**：图片移到 `assets/`，Markdown 里的引用同步改写。
- **页码标记**：PDF 输出在真实分页处带 `<!-- page: N -->` 注释，文本抽取、`--ocr` 和纯截图模式都一样。
- **表格检查**：管道表格里列数和表头对不上的行会报警告。内容不会被改写。

```text
out/
├── document.pdf.md          # 引用形如 ![](assets/document.pdf-0001-10.jpg)
└── assets/
    ├── document.pdf-0001-10.jpg
    └── images.json          # 带 --llm --desc 时生成
```

页面截图（`--screenshot`）仍在 `.markitai/screenshots/` 下。它们只在 HTML 注释里被引用，不算语料的一部分。

## `obsidian` — 导入 vault

- **可见资源**：和 `rag` 一样搬到 `assets/`，输出文件夹直接贴进 vault 也不会露出隐藏目录。
- **Wikilink（可选）**：设 `output.wikilinks: true` 后，本地图片引用变成 `![[assets/x.png]]`，alt 文本保留为显示文本。
- **Frontmatter**：markitai 本来就写标准 YAML frontmatter（`title`、`source`、`tags`），Obsidian 会当作 Properties 读取。

```bash
markitai note.docx --profile obsidian -o vault/inbox/
markitai note.docx --profile obsidian --config-json '{"output":{"wikilinks":true}}' -o vault/inbox/
```

## `okf` — Open Knowledge Format

把 frontmatter 对齐到 [Open Knowledge Format](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md) 规范（v0.2）：

| markitai 字段 | OKF 字段 |
|---|---|
| — | `type: Document`（注入；OKF 唯一的必填字段） |
| `title` | `title` |
| `description` | `description` |
| `source` | `resource` |
| `tags` | `tags` |
| `markitai_processed` | `generated: {by: markitai/<version>, at: <UTC 时间戳>}` |
| 其他字段 | 保留原名 |

没有 OKF 对应项的字段（`author`、`site`、`published` 等）保留原名，规范要求消费者接受未知字段。资源目录结构不变。

```yaml
---
type: Document
title: Lorem ipsum
resource: sample.pdf
generated:
  by: markitai/1.0.0
  at: '2026-08-25T01:42:13Z'
---
```

## images.json schema（已冻结）

带 `--llm --desc` 时，每个资源目录会生成一份 `images.json`，描述分析过的图片。schema 冻结在 1.0 版本。

顶层：

| 字段 | 类型 | 说明 |
|---|---|---|
| `version` | string | 固定为 `"1.0"` |
| `created` | string | 首次写入的 ISO 8601 时间戳（合并时保留） |
| `updated` | string | 最近一次写入的 ISO 8601 时间戳 |
| `images` | array | 每张分析过的图片一条 |

`images` 里的每一条：

| 字段 | 类型 | 说明 |
|---|---|---|
| `path` | string | 图片文件的绝对路径 |
| `alt` | string | 短标题，用作 alt 文本 |
| `desc` | string | 详细描述 |
| `text` | string | 从图片里识别出的文字（可能为空） |
| `created` | string | 分析时间的 ISO 8601 时间戳 |
| `source` | string | 源文档的绝对路径 |

## Recipe：LlamaIndex 摄取

用了 `rag` profile 后没有隐藏文件，Markdown 和图片都进入语料，frontmatter 也已解析好：

```python
import markitai
from llama_index.core import SimpleDirectoryReader

out = markitai.convert("report.pdf", output_dir="corpus/", profile="rag")
print(out.frontmatter["title"])       # 可直接当元数据
print([p.name for p in out.assets])   # 图片现在在 corpus/assets/ 下

# 读取器能看到全部文件：markdown、图片、images.json
documents = SimpleDirectoryReader("corpus/").load_data()
print(len(documents))
```

## 注意

- Profile 只作用于这次写出的文件，不会回头改写早先的输出。别在同一个目录里混用带 profile 和不带 profile 的运行，需要时重新转一遍。
- Profile 只影响写出的文件。stdout 模式（不带 `-o`）不受影响。
- 批量报告和 `--resume` 状态仍在 `.markitai/` 下，不进语料。嵌套批量时每个子目录有自己的 `assets/`。
