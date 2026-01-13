Please optimize this Markdown document fragment according to the following rules.

【IMPORTANT】This is a continuation segment (not the first segment), please note:
1. **DO NOT** generate YAML Frontmatter block (`---` ... `---`)
2. Output newly discovered entities and topics at the end as an HTML comment

## Core Principle: Respect Original Content

**DO NOT modify any actual content from the original document**, including:
- **Table headers/column names**: Must keep original text, do NOT translate
- **Body text**: All paragraphs and sentences must remain unchanged
- **Data content**: Table data, list items must be preserved exactly
- **Proper nouns**: Names, locations, product names must stay in original language

## Cleanup Rules

### 1. Clean up junk content
- Remove headers, footers, watermarks
- **Chart artifact cleanup**: Remove incorrectly extracted chart data:
  - Consecutive standalone number lines (Y-axis ticks)
  - Chart axis labels (e.g., "Row 1", "Column 1", "Category A")
  - Legend text (e.g., "Series 1")
- Remove standalone page/slide numbers
- Remove OCR artifacts (garbled characters)

### 2. PowerPoint/Presentation special handling
For PPT/PPTX converted content, follow heading hierarchy:
- Each slide title → `##` (h2), e.g., `## Slide N` or `## Page N`
- First-level heading within slide → `###` (h3)
- Second-level heading within slide → `####` (h4)

### 3. Blank line normalization
- One blank line above headings
- One blank line below headings before content
- One blank line between paragraphs
- Remove more than 2 consecutive blank lines

### 4. GFM specification
- Use `-` for unordered list markers
- Use ``` with language identifier for code blocks
- **Code block content must be preserved completely**
- Properly align tables

### 5. Metadata extraction
Extract newly discovered entities and topics from this segment, output at the **end** in this format:

```
<!-- PARTIAL_METADATA: {{"entities": ["entity1", "entity2"], "topics": ["topic1"]}} -->
```

Note:
- entities are key entities (names, organizations, products, terms) - keep original
- topics are topic tags
- If no new entities or topics, leave arrays empty

---

Original Markdown fragment:
```markdown
{content}
```

Output the optimized Markdown (without YAML Frontmatter, without ```markdown markers, with PARTIAL_METADATA comment at end):
