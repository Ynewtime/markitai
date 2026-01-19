**⚠️ CRITICAL LANGUAGE RULE: Output language = {language}**
If English → title/description/tags must be in English.
If Chinese → title/description/tags 必须使用中文。

---

根据以下 Markdown 内容生成 YAML frontmatter 元数据。

【必填字段】
- title: 文章标题（从内容提取，简洁准确）
- source: {source}
- description: 全文摘要（100字以内）
- tags: 相关标签数组（3-5个）
- markit_processed: {timestamp}

【输出要求】
- 直接输出纯 YAML，不要包裹在代码块中
- 不要添加 ```yaml 或 ``` 标记
- 不要添加 --- 分隔符
- 不要添加任何解释或说明

**⚠️ 关键：输出语言必须与源文档保持一致**
- 如果源文档是**英文**，title/description/tags 必须用**英文**
- 如果源文档是**中文**，title/description/tags 必须用**中文**
- 示例：英文文档 → `title: Data Overview`, `tags: [data, excel]`
- 示例：中文文档 → `title: 数据概览`, `tags: [数据, 表格]`

内容：
{content}
