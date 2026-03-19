# P3-D: Webextract 增强设计（整合 Defuddle 技术）

Date: 2026-03-19

## 问题

webextract 模块选出最佳候选节点后直接转 Markdown，不做候选内部的噪声清理。导致输出包含导航、侧栏、footer、广告、作者署名等非正文内容。Defuddle 项目有成熟的噪声移除体系，可整合其核心技术。

## 设计原则

1. **移植算法，不搬代码**：用 Python/BeautifulSoup 重写，适配现有架构
2. **渐进增强**：Phase 1/2/3 递进，每 Phase 独立可用
3. **保守移除**：宁可多留也不误删正文内容
4. **零外部依赖**：不引入新的 pip 依赖

## 总体架构

### Pipeline 变更

```
HTML Input
  ↓
1. Parse & Choose Root              (现有)
2. Schema Fallback                  (现有)
3. Remove Small Images              [Phase 1]
4. Remove Hidden Elements           [Phase 1]
5. Remove by Selectors              [Phase 1]
6. Score & Remove Non-Content       [Phase 1]
7. Remove Content Patterns          [Phase 3]
8. Standardize Content              (现有 + Phase 2 增强)
9. Sanitize                         (现有)
10. HTML→Markdown                   (现有)
11. Multi-Level Adaptive Retry      [Phase 2]
```

### 新增模块结构

```
webextract/
├── removals/                    [新目录]
│   ├── __init__.py              # 公共 API: apply_removals()
│   ├── selectors.py             # Selector 移除
│   ├── hidden.py                # 隐藏元素检测
│   ├── small_images.py          # 小图片清理
│   ├── scoring.py               # 非内容块评分
│   └── content_patterns.py      # 内容模式移除 [Phase 3]
├── elements/
│   ├── headings.py              # Heading anchor 移除 [Phase 3]
│   ├── callouts.py              # Callout 标准化 [Phase 3]
│   └── (code.py, footnotes.py, images.py 已有)
├── constants.py                 # 扩展 selector 常量
├── scoring.py                   # 增强评分算法 [Phase 2]
├── standardize.py               # 增强标准化 [Phase 2]
└── pipeline.py                  # 集成入口
```

---

## Phase 1: 噪声移除 + CJK 词数

### 1.1 removals/selectors.py

**Exact selectors (177+)**：从 defuddle constants.ts 移植，按类别组织。

关键类别：
- Scripts/Styles: `noscript`, `script:not([type^="math/"])`, `style`, `meta`, `link`
- Ads: `.ad`, `[class^="ad-"]`, `[role="banner"]`, `.promo`
- Navigation: `header`, `nav`, `[role="navigation"]`, `.menu`
- Sidebar: `.sidebar`, `#sidebar`
- Footer: `footer`
- Comments: `#comments`
- Forms: `button`, `form`, `input:not([type="checkbox"])`, `textarea`

**Partial selectors (557)**：正则匹配 class/id/data-test* 属性。
预编译为单个 `re.Pattern` 提升性能。

**保护机制：**
- 包含 main content root 的元素不移除
- `<pre>` / `<code>` 内部元素不移除
- Footnote 相关元素不移除
- Math 相关元素不移除

```python
def remove_by_selectors(
    root: Tag,
    main_content: Tag,
    *,
    use_partial: bool = True,
) -> int:
    """移除匹配 selector 的非内容元素。返回移除数量。"""
```

### 1.2 removals/hidden.py

检测 inline style 中的隐藏属性：
- `display: none`
- `visibility: hidden`
- `opacity: 0`

检测 CSS framework class：
- `hidden`, `invisible`, `*:hidden`, `*:invisible`

**保护：** math 元素（`.katex`, `.MathJax`, `<math>` 等）跳过。

```python
def remove_hidden_elements(root: Tag) -> int:
    """移除隐藏元素。返回移除数量。"""
```

### 1.3 removals/small_images.py

移除小于 33×33 像素的图片（tracking pixels、icons）：
- 检查 `width`/`height` 属性
- 检查 inline style 中的尺寸
- 只在两个维度都可确定时才移除

```python
def remove_small_images(root: Tag, min_size: int = 33) -> int:
    """移除小图片和 tracking pixels。返回移除数量。"""
```

### 1.4 removals/scoring.py

对候选内部的非内容块打分，低分的移除。

**正分信号：**
- +1/word
- +1/comma（prose indicator）
- `isLikelyContent()` 白名单直接保留

**负分信号：**
- Navigation indicators (35 patterns): -10 each
- Link density > 50%: -15
- Link text ratio > 80% in <80 words: -15
- Non-content class patterns: -8 each
- Card grid: -15
- Social media profile links in <80 words: -15

**阈值：** score < 0 则移除。

```python
def score_and_remove(root: Tag) -> int:
    """评分并移除低分非内容块。返回移除数量。"""
```

### 1.5 CJK 词数计算

新增 `count_words()` 工具函数，CJK 字符逐字计数：

```python
def count_words(text: str) -> int:
    """计算词数，CJK 字符逐字计。"""
    # CJK Unified: \u4e00-\u9fff
    # Hiragana: \u3040-\u309f
    # Katakana: \u30a0-\u30ff
    # Hangul: \uac00-\ud7af
    # CJK Extensions: \u3400-\u4dbf, \u20000-\u2a6df
```

替换 pipeline 中所有 `len(text.split())` 为 `count_words(text)`。

### 1.6 constants.py 扩展

添加从 defuddle 移植的常量：
- `EXACT_SELECTORS`: list[str]（177+ CSS selectors）
- `PARTIAL_SELECTOR_PATTERNS`: list[str]（557 regex patterns）
- `PARTIAL_SELECTOR_REGEX`: re.Pattern（预编译）
- `TEST_ATTRIBUTES`: tuple（class, id, data-test, data-testid, data-test-id, data-qa, data-cy）
- `NAVIGATION_INDICATORS`: list[str]（35 patterns）
- `NON_CONTENT_CLASS_PATTERNS`: list[str]
- `CONTENT_PROTECTION_SELECTORS`: list[str]（math, code, table 等）
- `ALLOWED_EMPTY_ELEMENTS`: set[str]（31 void elements）
- `MIN_IMAGE_SIZE`: int = 33

---

## Phase 2: 评分增强 + 标准化 + 多级重试

### 2.1 scoring.py 增强

修改现有 `score_candidate()` 算法：
- Link density 改为乘法缩放：`score *= (1 - min(link_density, 0.5))`
- 增加 comma count (+1/comma)
- Card grid 检测（3+ headings + 2+ images + <20 prose/heading → 低分）
- Navigation heading 识别

### 2.2 standardize.py 增强

新增标准化步骤：
- `_convert_h1_to_h2()`: 如果有多个 H1，全部转为 H2
- `_flatten_wrapper_divs()`: 解包无意义的 wrapper div
- `_unwrap_bare_spans()`: 移除无属性的 span
- `_remove_empty_elements()`: 清理空标签（排除 void elements）
- `_remove_trailing_content()`: 清理尾部孤立 heading/hr
- `_strip_unsafe_attributes()`: 属性白名单过滤

### 2.3 多级自适应重试

```python
def _extract_with_retry(soup, url, title) -> ExtractedWebContent:
    # Level 1: 正常提取
    result = _extract_core(soup, url, title, removal_config=FULL)
    if result.word_count >= 50:
        return result

    # Level 2: 禁用 partial selectors
    result2 = _extract_core(soup_clone, url, title, removal_config=NO_PARTIAL)
    if result2.word_count > result.word_count * 2:
        result = result2
    if result.word_count >= 50:
        return result

    # Level 3: 禁用 hidden element 移除
    result3 = _extract_core(soup_clone, url, title, removal_config=NO_HIDDEN)
    if result3.word_count > result.word_count:
        result = result3
    if result.word_count >= 50:
        return result

    # Level 4: 禁用所有 removal（listing page）
    result4 = _extract_core(soup_clone, url, title, removal_config=NONE)
    return max(result, result4, key=lambda r: r.word_count)
```

**注意：** 每次重试需要从原始 HTML 重新 clone，因为 removal 是 in-place 的。

---

## Phase 3: 高级特性

### 3.1 removals/content_patterns.py

- **Read-time 检测**：`\d+\s*min(ute)?s?\s+read` → 移除
- **Author byline**：`By Author · Date` in <15 words → 移除
- **Hero header**：h1/h2 + time + <30 prose in first 300 chars → 移除
- **Trailing external links**：heading + ul/ol, all items link off-site → 移除
- **Boilerplate sentences**：`This article appeared in...` → 移除

### 3.2 elements/ 增强

- **headings.py**：移除 heading 内的 permalink anchor（#、¶、§）
- **callouts.py**：GitHub `.markdown-alert` / Bootstrap `.alert` → 标准化 blockquote
- **images.py 增强**：srcset 解析（处理含逗号的 CDN URL）、picture 元素处理
- **code.py 增强**：语言检测（13 patterns + 120+ 语言验证集）

### 3.3 metadata.py 增强

- Title 清理：fuzzy site name 匹配（处理缩写）
- DOM 级 author 检测：H1 附近的 byline 扫描
- Published date：更多来源（meta article:published_time, sailthru.date）
- Language 检测：html lang → meta content-language → og:locale

---

## 测试策略

- 每个 removal 模块独立单元测试
- Pipeline 集成测试（构造含噪声的 HTML，验证移除效果）
- 对比测试：5 个标准 URL 的提取质量
- 回归：现有测试零回归 + fixtures 真实测试

## 实施顺序

Phase 1 内部：constants → small_images → hidden → selectors → scoring → CJK → pipeline 集成
Phase 2：scoring 增强 → standardize 增强 → 多级重试
Phase 3：content_patterns → elements 增强 → metadata 增强
