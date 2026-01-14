# Markit 需求文档

文档转 Markdown 工具，原生支持 LLM 增强。

## 原则

1. **程序转换、LLM优化**: 即基础转换只做限定的格式转换功能，在依赖的三方库的基础上，不引入或仅引入最少的数据清洗规则。数据清洗、Markdown格式优化、排版等功能交由大模型处理
2. **不造轮子**: 无论是功能特性还是非功能特性，开发和构建时都要严格避免重新造轮子，优先基于社区优秀依赖实现
3. **测试驱动**: 所有特性都需要测试覆盖，除了支持程序自动执行的单元测试外，需要维护一个用于开发者和大模型进行测试的 tests/SKILL.md 文件，并将测试所需的相关脚本/输入放到 tests 目录，以方便大模型进行自动测试

## 接口

```bash
# 基础转换（直接打印输出结果，并默认保存到当前工作目录的 output 文件夹内）
markit document.docx

# LLM处理（支持利用 LLM 处理基础转换后的 Markdown 文本，含格式清洗、添加带 title(文章标题)|source(输入文件名)|description(全文摘要)|markit_processed(处理时间)|tags(标签如['TAG1','TAG2']) 的 YAML frontmatter）
markit document.docx --llm

# 图片分析（生成 ![alt](相对输出Markdown的图片路径) 中的 alt 文本 + 生成图片描述Markdown:<output/<如果处理的是文件夹且包含子文件夹 && 子文件夹名称>/assets/<处理文件名>.<处理文件格式>.<4位自然序号>.<图片格式>.md>）
markit document.pdf --image

# 批量转换（默认输出到当前工作目录的 output 文件夹内，多次执行采用重命名机制，不覆盖原有的文件，也可以通过 -o 指定输出目录）
markit ./docs

# 恢复中断的批处理
markit ./docs -o ./output --resume
```

### 输出结构

```bash
output/                         # 输出目录
  document.docx.md              # 基础转换后的 markdown
  document.docx.llm.md          # LLM 优化后的 markdown
  assets/
    document.docx.0001.png      # 提取的图片
    document.docx.0001.png.md   # 图片描述（使用 --image 时）
  sub_dir/                      # 输入的目录包含子文件夹时，按照相同的层级放置
```

## 功能特性

### 支持的输入格式

INPUT_TYPES = WORD | PPTX | XLSX | PDF | TXT | MARKDOWN | JPG | PNG | WEBP

### 配置管理

优先级：命令行参数 > 环境变量 > 配置文件 > 系统默认

配置文件采用 JSON 格式

### 图片压缩

基础转换阶段，默认会将输入文件中提取出来的图片做高质量压缩，可选关闭

压缩后先保存到 output/assets 目录，再进入到后续处理管道，如【图片分析】功能

### LLM 集成

支持基于 LiteLLM 的大模型相关功能，如利用 LLM 进行 Markdown 数据清洗和格式优化、图片分析等

需启用 LiteLLM 的成本追踪、负载均衡相关功能，以支持可能高达 10K 并发输入文件的 LLM 处理场景

#### 图片分析

多次进行同一任务

### 提示词管理

系统中用到的提示词全部是一级公民，需统一管理

在配置文件中，支持对所有系统提示词传入用户自定义的提示词文件路径进行覆盖，支持的提示词文件格式为 PLAINTEXT | TXT | MARKDOWN

## 非功能特性

### 可选的调试选项

`--verbose`: 终端打印详细日志

`--dry-run`: 预览执行计划，包括关键参数如输入输出目录、是否开启 LLM 处理模式、是否开启图片分析模式等

### 日志系统

默认开启日志记录，支持用户通过配置修改日志文件夹
