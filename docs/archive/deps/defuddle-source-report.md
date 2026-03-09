# Defuddle 技术分析报告

> 仓库：https://github.com/kepano/defuddle · 版本 v0.8.0 · 作者 Steph Ango（@kepano，Obsidian CEO） · MIT

---

## 一、项目概述

Defuddle 是一个从网页中提取主要内容的开源库，名称取自 "de-fuddle"（去除混乱）。它能清理网页中的广告、导航栏、侧边栏、页脚等非核心元素，只保留文章正文。项目最初是为 Obsidian Web Clipper 服务的，目标是替代 Mozilla Readability，为 HTML-to-Markdown 转换器（如 Turndown）提供更干净、一致的输入。

项目目前已获得 3k+ Stars、101 Forks，npm 周下载量约 9,754，有 14 位贡献者参与开发。语言构成为 TypeScript 95.9%、HTML 3.2%、JavaScript 0.9%。

---

## 二、技术选型

### 2.1 核心语言与工具链

- **主语言**：TypeScript（95.9%），全量 TS 开发，类型安全
- **构建工具**：Webpack + ts-loader，多 bundle 输出
- **类型声明**：独立 tsconfig.declarations.json，分离声明文件生成与编译
- **代码规范**：ESLint（.eslintrc.json）
- **编辑器配置**：EditorConfig
- **CI/CD**：GitHub Actions，自动部署 Playground 到 GitHub Pages
- **包管理**：npm

### 2.2 运行时依赖——极简策略

- **核心 bundle（`defuddle`）**：零依赖，浏览器直接使用。直接操作浏览器原生 DOM API，非常适合体积敏感的浏览器扩展场景。
- **完整 bundle（`defuddle/full`）**：引入 mathml-to-latex 和 temml 库，用于数学公式的 MathML/LaTeX 双向转换。
- **Node.js bundle（`defuddle/node`）**：基于 JSDOM，包含完整的数学公式和 Markdown 转换能力。

### 2.3 多 Bundle 架构

三层包设计让使用者按需引入，避免依赖膨胀。这在内容提取工具中比较少见——大多数同类库只提供单一包。

---

## 三、架构与业务流程

### 3.1 核心处理管线

Defuddle 的内容处理分为七个阶段：Schema.org 提取（JSON-LD 解析）→ 站点专属检测（匹配 URL 调用 Extractor）→ 主内容区域识别 → 噪声元素移除 → HTML 标准化 → 元素精细处理 → 可选 Markdown 转换。

### 3.2 自适应重试机制

parse() 方法实现了智能重试：首次解析如果内容不足 200 词，会自动关闭模糊选择器再试一次，取更多内容的结果返回。核心逻辑（从 Go 移植版还原）：

```typescript
parse(): DefuddleResponse {
  const result = this.parseInternal();
  if (result.wordCount < 200) {
    const retryResult = this.parseInternal({ removePartialSelectors: false });
    if (retryResult.wordCount > result.wordCount) return retryResult;
  }
  return result;
}
```

### 3.3 噪声移除：双层选择器

- **精确选择器（Exact Selectors）**：匹配已知广告、社交按钮、Cookie 通知等的确定性选择器
- **模糊选择器（Partial Selectors）**：模式匹配类名/ID 中的 ad、sidebar、nav 等关键词

两者可通过 `removeExactSelectors` 和 `removePartialSelectors` 分别控制开关。

### 3.4 移动样式推断——最独特的创新

Defuddle 利用网站的移动端样式来发现可以移除的不必要元素。具体来说，它分析 CSS 媒体查询，找出在移动端被 `display: none` 的元素，推断它们是非核心内容（装饰性侧边栏、桌面端专属导航等）。这个策略在同类工具中独树一帜。

---

## 四、站点专属 Extractor 系统

### 4.1 设计理念

对于 DOM 结构复杂的网站，通用算法效果不佳。Extractor 机制为特定站点编写专门解析逻辑，绕过通用算法。

### 4.2 已支持的站点

根据 release notes 汇总，内置 Extractor 包括：

- **Twitter / X.com**：v0.7.0 加入了 X.com 文章提取
- **Reddit**：帖子和评论
- **YouTube**：视频元数据和作者信息
- **GitHub**：提取 Issues 内容，避免重复元数据
- **ChatGPT**：AI 对话内容（v0.6.2 修复）
- **Claude.ai**：AI 对话提取
- **Grok**：v0.6.2 加入
- **Hacker News**：帖子和评论线程
- **Substack**：newsletter 内容和图片

### 4.3 异步 Extractor

使用 parseAsync() 时，如果本地 HTML 无法提取内容，Defuddle 可以异步调用第三方 API 作为后备方案，这只在页面无可用内容时触发（如客户端渲染的 SPA）。

---

## 五、HTML 标准化处理

这是 Defuddle 区别于 Readability 的核心价值之一。

### 5.1 标题处理

移除与 title 重复的第一个 H1/H2；所有 H1 降级为 H2；清除标题中的锚链接。

### 5.2 代码块标准化

移除各种代码高亮库的行号和语法高亮标记，统一为 pre > code 格式，但保留语言标识（data-lang 和 class="language-*"）。

### 5.3 脚注标准化

将各种脚注实现统一为一致的格式：正文使用 sup + a 链接，脚注区域使用有序列表加回链。

### 5.4 数学公式标准化

将 MathJax 和 KaTeX 渲染的公式统一转为标准 MathML，通过 data-latex 保留原始 LaTeX 源码。

---

## 六、元数据提取

输出的 Result 对象包含丰富的元数据：title、author、description、published、domain、site、image、favicon、wordCount、parseTime、schemaOrgData、metaTags。来源涵盖 HTML meta 标签、Open Graph、Twitter Cards、Schema.org JSON-LD 以及 Extractor 自定义提取。

---

## 七、CLI 工具

v0.8.0 将 CLI 合并到了主仓库，提供完整命令行能力：

```bash
defuddle parse page.html              # 解析本地文件
defuddle parse https://example.com    # 解析 URL
defuddle parse page.html --markdown   # 输出 Markdown
defuddle parse page.html --json       # 输出含元数据的 JSON
defuddle parse page.html --property title  # 提取单个属性
```

---

## 八、亮点特性总结

### 8.1 与 Mozilla Readability 的差异化

Defuddle 相比 Readability：更宽容、移除更少不确定元素；对脚注/数学/代码块提供一致输出；利用移动端样式推断不必要元素；提取更多元数据包括 Schema.org 数据。

### 8.2 工程亮点

1. **零依赖核心包**：浏览器 bundle 无外部依赖，体积小
2. **三层 Bundle 分发**：按需加载，不同场景选不同包
3. **移动样式推断**：创新利用响应式 CSS 辅助内容识别
4. **自适应重试**：内容不足时自动放宽条件重试
5. **Extractor 可扩展架构**：主流站点定制化 + 通用算法兜底
6. **Playground 在线演示**：GitHub Pages 上的交互式调试工具
7. **Obsidian 生态联动**：Web Clipper 核心引擎，大规模用户验证

### 8.3 社区生态

- 已有 Go 语言移植版（defuddle-go），说明架构可移植性强
- 被 Obsidian Web Clipper 作为核心依赖，上游需求驱动稳定
- 约 15 个 Open Issues，社区活跃

---

## 九、潜在改进方向

1. **构建工具现代化**：Webpack + ts-loader 可迁移到 esbuild/tsup，提升构建速度
2. **Extractor 插件化**：提供注册机制，让用户自定义 Extractor 而无需 fork
3. **内容评分算法透明化**：主内容识别的算法细节文档不足
4. **Benchmark 与对比**：缺少与 Readability、Trafilatura 等的系统性对比数据

---

*报告生成日期：2026-03-06*
