# Kreuzberg 评测报告 v2（充分利用 API）

> 评测日期：2026-03-05 | kreuzberg v4.4.2 | markitai v0.6.1
> 每个文件运行 3 次取中位数

---

## 一、v1 结论修正

v1 评测由于 API 利用不充分，产生了 2 个错误结论：

| 缺陷 | v1 结论 | v2 修正 |
|------|--------|--------|
| 页面标记 | ❌ 不支持 | ✅ **支持**。`PageConfig(insert_page_markers=True, marker_format="<!-- Page number: {page_num} -->")` 可自定义格式 |
| PDF 图片提取 | ❌ 不支持 | ✅ **支持**。`PdfConfig(extract_images=True)` + `ImageExtractionConfig(extract_images=True)` 可提取所有嵌入图片（含 data、page_number、dimensions） |

其余缺陷经深入验证后**确认存在**，但增加了根因分析和 workaround 方案。

---

## 二、性能对比

| 格式 | Current (ms) | Kreuzberg (ms) | 加速比 | K:images | K:tables | K:pages |
|------|-------------|----------------|--------|----------|----------|---------|
| PDF (500KB) | 362 | 67 | **5.4x** | 5 | 0 | 5 |
| Image OCR (JPG) | 1046 | 47 | **22x** | 0 | 0 | 0 |
| XLSX | 108 | 1.4 | **77x** | 0 | 1 | 0 |
| PPTX | 161 | 1.4 | **115x** | 1 | 0 | 8 |
| DOC (legacy) | 4372 | 0.6 | **7287x** | 0 | 0 | 0 |
| XLS (legacy) | 4369 | 1.3 | **3361x** | 0 | 1 | 0 |
| PPT (legacy) | 6116 | 0.3 | **20387x** | 0 | 0 | 0 |

Kreuzberg 在**所有格式**上速度领先，现代格式 5-115x，旧格式 3000-20000x。

---

## 三、缺陷逐项验证

### ✅ 已修正：API 利用不充分导致的误报

**1. 页面标记**
- v1 结论：kreuzberg 不支持页面标记 → **错误**
- 实际：`PageConfig(insert_page_markers=True, marker_format="\n\n<!-- Page number: {page_num} -->\n\n")` 完全支持
- 同时 `extract_pages=True` 返回 `result.pages` 数组，含每页 `content`、`tables`、`images`、`is_blank`
- PDF 和 PPTX 均可正确插入自定义格式的页面标记

**2. PDF 图片提取**
- v1 结论：kreuzberg 不提取嵌入图片 → **错误**
- 实际：启用 `PdfConfig(extract_images=True)` + `ImageExtractionConfig(extract_images=True)` 后，成功提取 5 张图片
- 每张图含 `data`（原始字节）、`page_number`、`width`×`height`、`format`、`colorspace`
- PPTX 也成功提取了 1 张嵌入 GIF 图片

### 📐 设计差异（非 bug）

**3. PDF 表格提取需要 OCR 模式**
- 默认 PDFium 文本提取不检测表格结构（返回 0 tables）
- 需要 `force_ocr=True` + `OcrConfig(backend="tesseract", tesseract_config=TesseractConfig(enable_table_detection=True))`
- 当前环境无 Tesseract，**未能验证 OCR 表格检测效果**
- 对比：pymupdf4llm 在非 OCR 模式下也能输出 Markdown 表格格式（通过布局分析）

### ❌ 确认存在的真实缺陷

**4. PDF Markdown 输出断词 bug**
- 状态：**确认，html-to-markdown 渲染层 bug**
- 现象：行尾换行转 Markdown 时不补空格，导致单词拼接（如 "utvarius" 应为 "ut varius"）
- 验证：6/6 个测试词在 MARKDOWN 输出中全部拼接，PLAIN 和 STRUCTURED 输出 0/6 拼接
- 根因：bug 在 html-to-markdown 转换器，不在 PDF 提取层
- **Workaround**：使用 `OutputFormat.PLAIN` 获取正确文本 + 自行格式化 Markdown

**5. PDF 链接不渲染为内联 Markdown**
- 状态：**确认**
- 现象：超链接不在 Markdown 正文中渲染为 `[text](url)` 格式
- 发现：通过 `PdfConfig(extract_annotations=True)` + `result.annotations` 可获取链接数据（含 URL、页码）
- 但链接信息**不会注入到 Markdown 内容中**，需要自行后处理

**6. DOC 格式：仅纯文本输出**
- 状态：**确认**
- OLE/CFB 原生解析仅提取纯文本
- 无标题、无表格结构、无加粗/斜体、无图片、无分页
- 超链接以 `HYPERLINK "url"` 原始标记出现（未转换为 Markdown）
- 所有配置选项（`include_document_structure`、`enable_quality_processing`、`pages`、`images`）对 DOC 格式均无效

**7. PPT 格式：仅纯文本输出**
- 状态：**确认**（PPT），**部分问题**（PPTX）
- PPT（旧格式）：无标题、无表格、无图片、无分页，同 DOC
- PPTX（现代格式）：✅ 表格提取正常、✅ 页面标记正常（8 页）、✅ 图片提取正常
- PPTX 格式问题：13 个空列表项（`- `）、标点前多余空格（52 处），为上游 cosmetic issue

---

## 四、Kreuzberg 能力全景（经验证）

| 能力 | PDF | DOCX/XLSX/PPTX | DOC/XLS/PPT |
|------|-----|----------------|-------------|
| 文本提取 | ✅ 正确（PLAIN 模式） | ✅ | ✅ |
| Markdown 输出 | ⚠️ 断词 bug | ✅ XLSX 完美 / PPTX 有 cosmetic issue | ❌ 仅纯文本 |
| 页面标记 | ✅ 自定义格式 | ✅ PPTX | ❌ |
| 分页数据 | ✅ `result.pages` | ✅ PPTX | ❌ |
| 图片提取 | ✅ 含 data/page/dimensions | ✅ PPTX | ❌ |
| 表格提取 | ⚠️ 需 OCR+Tesseract | ✅ XLSX 完美 | ❌ DOC/PPT；✅ XLS |
| 链接保留 | ⚠️ annotations 有，Markdown 无 | - | ❌ 原始标记 |
| 文档结构 | ✅ 层级检测 (K-means) | - | ❌ |
| 元数据 | ✅ 丰富（作者、日期、页数等） | ✅ | ✅（有限） |

---

## 五、修正后的综合评估

### 评分矩阵（v2）

| 维度 | 权重 | Current | Kreuzberg | 变化 |
|------|------|---------|-----------|------|
| PDF 文本质量 | 高 | ⭐⭐⭐⭐ | ⭐⭐⭐ (↑) | Markdown 断词仍存在，但 PLAIN 输出正确，有 workaround |
| PDF 图片/结构 | 高 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ (↑↑) | 图片提取、页面标记、注释均可用 |
| Office 现代格式 | 高 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ (↑) | XLSX 完美，PPTX 有页面标记和图片提取 |
| Office 旧格式 | 中 | ⭐⭐⭐⭐ | ⭐⭐ | DOC/PPT 确认仅纯文本，XLS 完美 |
| 处理速度 | 中 | ⭐⭐ | ⭐⭐⭐⭐⭐ | 全面碾压 |
| LLM 管线兼容性 | 高 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ (↑↑) | 页面标记 + 图片提取 + 分页数据均可用 |

### 结论（v2 修正）

**原先的「不建议全面替代」结论仍然成立，但可采纳范围显著扩大：**

1. **Markdown 断词 bug 是唯一阻塞性问题**（PDF 场景） — 这是 html-to-markdown 层的 bug，PLAIN 输出完全正确。可通过 PLAIN + 自行格式化绕过，或等待上游修复
2. **LLM 管线兼容性大幅提升** — 页面标记、图片提取、分页数据均已验证可用，集成改造量远小于 v1 预估
3. **DOC/PPT 仍是短板** — 但这是 OLE 格式的固有限制，对于需要高质量旧格式支持的场景仍需 LibreOffice

### 推荐的渐进式采纳路径

| 阶段 | 采纳范围 | 风险 |
|------|---------|------|
| **Phase 1** | XLSX/XLS 替换 — 输出几乎一致，速度快 77-3361x，去除 LibreOffice 依赖 | 低 |
| **Phase 2** | PPTX 替换 — 页面标记/图片/表格均可用，cosmetic issues 可接受 | 中低 |
| **Phase 3** | PDF 替换 — 等 html-to-markdown 断词 bug 修复，或用 PLAIN + 自定义格式化方案 | 中 |
| **不采纳** | DOC/PPT — 质量差距太大，继续用 LibreOffice 方案 | - |
