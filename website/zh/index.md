---
layout: home

hero:
  name: Markitai
  text: 文件和网页，一条命令转为 Markdown
  tagline: 支持文档、图片和公网 URL，需要时再开启 LLM 增强。
  actions:
    - theme: brand
      text: 快速开始
      link: /zh/guide/getting-started
    - theme: alt
      text: GitHub
      link: https://github.com/Ynewtime/markitai

features:
  - icon: |
      <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false"><path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="M10 9H8"/><path d="M16 13H8"/><path d="M16 17H8"/></svg>
    title: 多格式支持
    details: 支持 DOCX、PPTX、XLSX、PDF、TXT、MD、图片（JPG/PNG/WebP）和 URL 转换为 Markdown。
  - icon: |
      <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false"><path d="M12 8V4H8"/><rect width="16" height="12" x="4" y="8" rx="2"/><path d="M2 14h2"/><path d="M20 14h2"/><path d="M15 13v2"/><path d="M9 13v2"/></svg>
    title: LLM 增强
    details: AI 驱动的格式清洗、元数据生成（frontmatter）和图片分析。
  - icon: |
      <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false"><path d="M16 3h5v5"/><path d="M8 3H3v5"/><path d="M12 22v-8.3a4 4 0 0 0-1.172-2.872L3 3"/><path d="m15 9 6-6"/></svg>
    title: 批量处理
    details: 并发转换，支持进度显示和断点恢复。
  - icon: |
      <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false"><path d="M15 3v4a2 2 0 0 0 2 2h4"/><path d="M12 17v-6"/><path d="M9.5 14.5 12 17l2.5-2.5"/><path d="M20 17.5a9 9 0 1 1-18 0V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2Z"/></svg>
    title: OCR 识别
    details: 使用 RapidOCR 从扫描版 PDF 和图片中提取文字。
---

<section class="home-quickstart" aria-labelledby="quickstart-title">
  <div class="home-quickstart-intro">
    <p class="eyebrow">第一次成功</p>
    <h2 id="quickstart-title">60 秒，从安装到得到 Markdown</h2>
    <p>核心转换无需 API 密钥或可选依赖。先完成一次转换，再按工作流需要添加浏览器渲染、额外格式或 LLM。</p>
    <a href="/zh/guide/getting-started">打开完整快速开始指南 <span aria-hidden="true">→</span></a>
  </div>
  <div class="home-quickstart-steps" role="list" aria-label="快速开始命令">
    <div class="home-quickstart-step" role="listitem">
      <span class="step-number" aria-hidden="true">1</span>
      <div><p>安装核心包</p><code>uv tool install markitai</code></div>
    </div>
    <div class="home-quickstart-step" role="listitem">
      <span class="step-number" aria-hidden="true">2</span>
      <div><p>创建一个简单的输入文件</p><code>echo "第一次 Markitai 转换" &gt; hello.txt</code></div>
    </div>
    <div class="home-quickstart-step output" role="listitem">
      <span class="step-number" aria-hidden="true">3</span>
      <div><p>从 stdout 得到干净的 Markdown</p><code>markitai hello.txt --pure</code></div>
    </div>
  </div>
</section>
