# 输出 Profile

Profile 为特定下游消费者塑形输出。它与[预设](./configuration.md#预设)正交：预设决定运行哪些功能（LLM、OCR、截图……），profile 决定结果长什么样。

**不带 profile 时，输出与默认行为逐字节一致**——所有变换只在设置了 profile 时才执行。

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

默认输出把图片放在隐藏的 `.markitai/assets/` 目录里。多数摄取器会跳过隐藏路径（LlamaIndex 的 `SimpleDirectoryReader` 默认如此），图片会从语料中静默消失。`rag` profile 让输出对摄取器友好：

- **可见资产**：图片移到 `assets/`，markdown 中的引用同步改写。清空后的隐藏目录会被删除。
- **页标记**：PDF 输出带有 `<!-- page: N -->` 注释。它由转换器自身的 `<!-- Page number: N -->` 标记改写而来——文本抽取、`--ocr`、纯截图，每条按页切分文档的路径都会把它写在真实的页边界上，位置从不靠猜测。
- **表格校验**：LLM 清洗 prompt 增加列数一致性硬约束；写出后，行与表头列数不一致的管道表会以 warning 报告（仅检测，不改写内容）。

```text
out/
├── document.pdf.md          # 引用形如 ![](assets/document.pdf-0001-10.jpg)
└── assets/
    ├── document.pdf-0001-10.jpg
    └── images.json          # 需 --llm --desc
```

::: warning 限制
页面截图（`--screenshot`）仍在 `.markitai/screenshots/` 下——它们只被 HTML 注释引用，不属于可摄取语料。
:::

## `obsidian` — 导入 vault

- **可见资产**：与 `rag` 相同的搬迁，输出文件夹可以直接当 vault 文件夹用，不会出现隐藏目录。
- **Wikilink（可选）**：设置 `output.wikilinks: true` 后，本地图片引用变为 `![[assets/x.png]]`（alt 文本保留为显示文本：`![[assets/x.png|alt]]`）。
- **Frontmatter**：markitai 本就输出标准 YAML frontmatter（`title`、`source`、`tags` 等），Obsidian 直接作为 Properties 读取——无需改动。

```bash
markitai note.docx --profile obsidian -o vault/inbox/
markitai note.docx --profile obsidian --config-json '{"output":{"wikilinks":true}}' -o vault/inbox/
```

## `okf` — Open Knowledge Format

将 frontmatter 对齐 [Open Knowledge Format](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md) 规范（已按 v0.2 核实）：

| markitai 字段 | OKF 字段 |
|---|---|
| — | `type: Document`（注入；OKF 唯一必填字段） |
| `title` | `title`（同名） |
| `description` | `description`（同名） |
| `source` | `resource` |
| `tags` | `tags`（同名） |
| `markitai_processed` | `generated: {by: markitai/<version>, at: <UTC 时间戳>}` |
| 其余字段 | 保留现名 |

没有 OKF 对应项的字段（`author`、`site`、`published`、`canonical_url`、`fetch_strategy` 等）保留现名：规范明确要求消费者「MUST NOT reject documents with unrecognized fields」。资产布局不变。

```yaml
---
type: Document
title: Lorem ipsum
resource: sample.pdf
generated:
  by: markitai/0.24.0
  at: '2026-08-25T01:42:13Z'
---
```

## images.json schema（已冻结）

使用 `--llm --desc` 时，每个资产目录都会得到一个描述已分析图片的 `images.json`。schema **冻结在 1.0 版**；变更必须同时更新锁测试（`tests/unit/test_images_json_schema.py`）与本页。

顶层字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `version` | string | 恒为 `"1.0"` |
| `created` | string | 首次写入的 ISO 8601 时间戳（合并时保留） |
| `updated` | string | 最近写入的 ISO 8601 时间戳 |
| `images` | array | 每张已分析图片一条 |

`images` 中的每条：

| 字段 | 类型 | 说明 |
|---|---|---|
| `path` | string | 图片文件在磁盘上的绝对路径 |
| `alt` | string | 短标题（用作 alt 文本） |
| `desc` | string | 详细描述 |
| `text` | string | 从图片提取的文本（可为空） |
| `created` | string | 分析时的 ISO 8601 时间戳 |
| `source` | string | 源文档的绝对路径 |

## Recipe：LlamaIndex 摄取

`SimpleDirectoryReader` 默认跳过隐藏路径。使用 `rag` profile 后没有任何东西是隐藏的，markdown **和**图片都会进入语料，frontmatter 也已解析好：

```python
import markitai
from llama_index.core import SimpleDirectoryReader

out = markitai.convert("report.pdf", output_dir="corpus/", profile="rag")
print(out.frontmatter["title"])  # 已解析的 YAML frontmatter，可直接作为 metadata
print([p.name for p in out.assets])  # 图片现在位于 corpus/assets/

# 读取器能看到全部文件——markdown、图片、images.json
documents = SimpleDirectoryReader("corpus/").load_data()
print(len(documents))
```

## 注意

- 不要在同一输出目录混用带 profile 与不带 profile 的运行：先前引用 `.markitai/assets/` 的输出不会被追溯改写。
- Profile 作用于写出的文件；stdout 模式（不带 `-o`）不受影响。
- 批量运行的 JSON 报告与 `--resume` 状态仍在 `.markitai/` 下——它们是转换簿记，刻意排除在可摄取语料之外。嵌套批量中每个子目录都有自己的 `assets/`。
