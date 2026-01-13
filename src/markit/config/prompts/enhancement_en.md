Please optimize the following Markdown document's format according to these rules:

## Core Principle: Respect Original Content

**DO NOT modify any actual content from the original document**, including:
- **Table headers/column names**: Must keep original text, do NOT translate (e.g., keep "First Name" as is)
- **Body text**: All paragraphs and sentences must remain unchanged
- **Data content**: Table data, list items must be preserved exactly
- **Proper nouns**: Names, locations, product names must stay in original language

**Only the following may be translated/localized**:
- YAML Frontmatter values like `description`, `topics`, `domain`
- Image alt text descriptions

## Cleanup Rules

1. **Clean up junk content**:
   - Remove headers, footers, watermarks, and meaningless repeated characters
   - For PowerPoint slides: remove repetitive footer text (company names, dates, slide numbers, copyright notices) that appear on every slide
   - **Chart artifact cleanup**: Remove incorrectly extracted chart axis data:
     - Consecutive standalone number lines (e.g., "12", "10", "8"... usually Y-axis ticks)
     - Chart axis labels (e.g., "Row 1", "Column 1", "Category A")
     - Legend text (e.g., "Series 1", "Data Series 1")
     - These typically appear before/after image markers `![...]()`
   - Remove standalone page/slide numbers (e.g., "1", "Page 2", "Slide 3")
   - Remove OCR artifacts (garbled characters like "■■■", consecutive special symbols)

2. **PowerPoint/Presentation special handling**:
   For PPT/PPTX converted content, preserve slide structure:
   - Use `## Slide N` or `## Page N` as heading for each slide
   - Place each slide's content under its corresponding slide heading
   - Maintain original slide order

3. **Heading levels**: Ensure headings start from ## (h2), avoid multiple # (h1)

4. **Blank line normalization**:
   - One blank line above headings
   - One blank line below headings before content
   - One blank line between paragraphs
   - Remove more than 2 consecutive blank lines

5. **Follow GFM specification**:
   - Use `-` for unordered list markers
   - Use ``` with language identifier for code blocks
   - **Code block content must be preserved completely**
   - Properly align tables

6. **Complex table handling**:
   - Convert merged cells to multi-row representation
   - Expand nested tables to independent tables
   - Consider converting very wide tables to list format
   - **Table headers must remain in original language, do NOT translate**

7. **Knowledge Graph Metadata Extraction**:
   Extract the following metadata from the document and add to YAML Frontmatter:
   - **description**: One-sentence summary of the document's core content (max 100 characters), highlighting the theme and key information, avoiding redundant phrases like "this document"
   - **entities**: Key entities in the document (person names, organizations, products, technical terms, locations) - keep original
   - **topics**: Topic tags the document covers (3-5 most relevant topics)
   - **domain**: The domain the document belongs to (e.g., technology, business, academic, medical, legal, finance)

   **Frontmatter format requirements**:
   - Place at the very beginning of the document, wrapped with `---`
   - Do **NOT** wrap Frontmatter in Markdown code blocks (e.g., ```yaml)
   - Use double quotes for values containing special characters (colons, quotes, newlines)
   - List items on separate lines with `- ` prefix
   - Ensure valid YAML syntax parseable by standard parsers

   **Example output format**:
   ```yaml
   ---
   description: "Kubernetes container orchestration guide covering deployment, scaling, and monitoring best practices."
   entities:
     - "Kubernetes"
     - "Docker"
     - "AWS"
   topics:
     - "Containerization"
     - "Cloud Native"
     - "DevOps"
   domain: "Technology"
   ---
   ```

Original Markdown:
```markdown
{content}
```

Output the optimized Markdown (including YAML Frontmatter, without ```markdown markers):
