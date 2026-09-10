---
layout: home

hero:
  name: Markitai
  text: Clean Markdown from files and URLs
  tagline: Convert documents, images, and web pages with one command. LLM enhancement stays optional.
  actions:
    - theme: brand
      text: Get Started
      link: /guide/getting-started
    - theme: alt
      text: View on GitHub
      link: https://github.com/Ynewtime/markitai
---

<!-- Feature cards render via the theme's home-features-before slot (see .vitepress/theme/index.ts). -->
<section class="home-shot" aria-labelledby="shot-title">
  <p class="eyebrow">Web workspace</p>
  <h2 id="shot-title">A web workspace, when you would rather not type</h2>
  <p><code>markitai serve</code> runs the same conversion core behind a local web UI — drop files in, paste URLs, watch progress, and download results.</p>
  <picture class="home-shot-light">
    <source srcset="/workbench.webp" type="image/webp" />
    <img src="/workbench.png" alt="The markitai web workspace: a composer card with a URL field, a Convert button, and Options, CLI, and Upload toggles." width="1440" height="700" loading="lazy" />
  </picture>
  <picture class="home-shot-dark">
    <source srcset="/workbench.dark.webp" type="image/webp" />
    <img src="/workbench.dark.png" alt="The markitai web workspace: a composer card with a URL field, a Convert button, and Options, CLI, and Upload toggles." width="1440" height="700" loading="lazy" />
  </picture>
  <p class="home-shot-link"><a href="/guide/serve">Read about the web workspace <span aria-hidden="true">→</span></a></p>
</section>

<section class="home-quickstart" aria-labelledby="quickstart-title">
  <div class="home-quickstart-intro">
    <p class="eyebrow">FIRST SUCCESS</p>
    <h2 id="quickstart-title">From install to Markdown in 60 seconds</h2>
    <p>The core conversion path needs no API key or optional dependency. Start small, then add browser rendering, extra formats, or an LLM when your workflow calls for them.</p>
    <a href="/guide/getting-started">Open the full getting started guide <span aria-hidden="true">→</span></a>
    <p class="home-install-options-note">Choosing between converters? <a href="/guide/comparison">See how markitai compares</a> with markitdown, docling and anydoc.</p>
  </div>
  <div class="home-quickstart-steps" role="list" aria-label="Quick start commands">
    <div class="home-quickstart-step" role="listitem">
      <span class="step-number" aria-hidden="true">1</span>
      <div>
        <p class="non-windows-only">Install with the portable script for macOS or Linux</p>
        <code class="platform-install-command non-windows-only">curl -fsSL https://markitai.dev/setup.sh | sh</code>
        <p class="windows-only">Install with the portable script for Windows</p>
        <code class="platform-install-command windows-only">powershell -ExecutionPolicy ByPass -c "irm https://markitai.dev/setup.ps1 | iex"</code>
        <noscript>
          <p>On Windows (PowerShell):</p>
          <code>powershell -ExecutionPolicy ByPass -c "irm https://markitai.dev/setup.ps1 | iex"</code>
        </noscript>
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
      <div><p>Convert a live page</p><code>markitai https://github.com/Ynewtime/markitai --pure</code></div>
    </div>
    <div class="home-quickstart-step output" role="listitem">
      <span class="step-number" aria-hidden="true">3</span>
      <div>
        <p>Get clean Markdown on stdout</p>
        <code># Markitai<br />Opinionated Markdown converter with native LLM enhancement support.<br /><br />- **Multi-format**: DOCX, PPTX, XLSX, PDF, EPUB,<br />&nbsp;&nbsp;EML, TXT, MD, images, and URLs &#8594; clean Markdown<br />- **LLM enhancement**: format cleaning, frontmatter<br />&nbsp;&nbsp;metadata, and vision analysis<br />- **Batch processing**: concurrent conversion with<br />&nbsp;&nbsp;progress display and --resume<br />&#8230;</code>
      </div>
    </div>
  </div>
</section>
