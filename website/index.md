---
layout: home

hero:
  name: Markitai
  text: Clean Markdown from files and URLs
  tagline: Convert documents, images, and web pages with one command. Add LLM enhancement only when you need it.
  actions:
    - theme: brand
      text: Get Started
      link: /guide/getting-started
    - theme: alt
      text: View on GitHub
      link: https://github.com/Ynewtime/markitai

features:
  - icon: |
      <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false"><path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="M10 9H8"/><path d="M16 13H8"/><path d="M16 17H8"/></svg>
    title: Multi-format Support
    details: Convert DOCX, PPTX, XLSX, PDF, TXT, MD, images (JPG/PNG/WebP) and URLs to Markdown.
  - icon: |
      <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false"><path d="M12 8V4H8"/><rect width="16" height="12" x="4" y="8" rx="2"/><path d="M2 14h2"/><path d="M20 14h2"/><path d="M15 13v2"/><path d="M9 13v2"/></svg>
    title: LLM Enhancement
    details: AI-powered format cleaning, metadata generation (frontmatter), and image analysis.
  - icon: |
      <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false"><path d="M16 3h5v5"/><path d="M8 3H3v5"/><path d="M12 22v-8.3a4 4 0 0 0-1.172-2.872L3 3"/><path d="m15 9 6-6"/></svg>
    title: Batch Processing
    details: Concurrent conversion with progress display and resume capability for interrupted jobs.
  - icon: |
      <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false"><path d="M15 3v4a2 2 0 0 0 2 2h4"/><path d="M12 17v-6"/><path d="M9.5 14.5 12 17l2.5-2.5"/><path d="M20 17.5a9 9 0 1 1-18 0V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2Z"/></svg>
    title: OCR Recognition
    details: Extract text from scanned PDFs and images using RapidOCR.
---

<section class="home-quickstart" aria-labelledby="quickstart-title">
  <div class="home-quickstart-intro">
    <p class="eyebrow">FIRST SUCCESS</p>
    <h2 id="quickstart-title">From install to Markdown in 60 seconds</h2>
    <p>The core conversion path needs no API key or optional dependency. Start small, then add browser rendering, extra formats, or an LLM when your workflow calls for them.</p>
    <a href="/guide/getting-started">Open the full getting started guide <span aria-hidden="true">→</span></a>
  </div>
  <div class="home-quickstart-steps" role="list" aria-label="Quick start commands">
    <div class="home-quickstart-step" role="listitem">
      <span class="step-number" aria-hidden="true">1</span>
      <div>
        <p class="non-windows-only">Install with the portable script for macOS or Linux</p>
        <code class="platform-install-command non-windows-only">curl -fsSL https://markitai.dev/setup.sh | sh</code>
        <p class="windows-only">Install with the portable script for Windows</p>
        <code class="platform-install-command windows-only">powershell -ExecutionPolicy ByPass -c "irm https://markitai.dev/setup.ps1 | iex"</code>
        <details class="home-install-options">
          <summary>Other platforms and manual install</summary>
          <div class="home-install-option">
            <span>macOS or Linux</span>
            <code>curl -fsSL https://markitai.dev/setup.sh | sh</code>
          </div>
          <div class="home-install-option">
            <span>Windows</span>
            <code>powershell -ExecutionPolicy ByPass -c "irm https://markitai.dev/setup.ps1 | iex"</code>
          </div>
          <div class="home-install-option">
            <span>Already have uv</span>
            <code>uv tool install markitai</code>
          </div>
        </details>
      </div>
    </div>
    <div class="home-quickstart-step" role="listitem">
      <span class="step-number" aria-hidden="true">2</span>
      <div><p>Convert a live page</p><code>markitai https://markitai.dev/guide/getting-started --pure</code></div>
    </div>
    <div class="home-quickstart-step output" role="listitem">
      <span class="step-number" aria-hidden="true">3</span>
      <div><p>Get clean Markdown on stdout</p><code># Getting Started ...</code></div>
    </div>
  </div>
</section>
