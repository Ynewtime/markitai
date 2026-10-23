---
layout: home

hero:
  name: Markitai
  text: 文件和网页，一条命令转成 Markdown
  tagline: 文档、图片和网页链接都能转。LLM 增强按需开启，不开也完全够用。
  actions:
    - theme: brand
      text: 快速开始
      link: /zh/guide/getting-started
    - theme: alt
      text: GitHub
      link: https://github.com/Ynewtime/markitai
---

<!-- 功能卡片由主题的 home-features-before 插槽渲染（见 .vitepress/theme/index.ts）。 -->
<section class="home-shot" aria-labelledby="shot-title">
  <p class="eyebrow">网页工作台</p>
  <h2 id="shot-title">不想敲命令时，有个工作台</h2>
  <p><code>markitai serve</code> 把同一套转换内核放进本地网页界面。拖入文件、粘贴 URL、看进度，对照原件逐条检查结果，再全部下载。</p>
  <picture class="home-shot-light">
    <source srcset="../images/workbench.zh.webp" type="image/webp" />
    <img src="../images/workbench.zh.png" alt="markitai 网页工作台：一个已转换的单页 PDF 在窗口中打开，窗口居中于变暗的转换任务列表上方，位于「对照」标签。左侧的原件页面和右侧的 Markdown 同时框出同一张图：文件和网页、markitai、Markdown。" width="1600" height="1000" loading="lazy" />
  </picture>
  <picture class="home-shot-dark">
    <source srcset="../images/workbench.zh.dark.webp" type="image/webp" />
    <img src="../images/workbench.zh.dark.png" alt="markitai 网页工作台：一个已转换的单页 PDF 在窗口中打开，窗口居中于变暗的转换任务列表上方，位于「对照」标签。左侧的原件页面和右侧的 Markdown 同时框出同一张图：文件和网页、markitai、Markdown。" width="1600" height="1000" loading="lazy" />
  </picture>
  <p class="home-shot-link"><a href="/zh/guide/serve">了解网页工作台 <span aria-hidden="true">→</span></a></p>
</section>

<section class="home-quickstart" aria-labelledby="quickstart-title">
  <div class="home-quickstart-intro">
    <p class="eyebrow">快速上手</p>
    <h2 id="quickstart-title">60 秒，转成 Markdown</h2>
    <p>无需 API 密钥。</p>
    <a href="/zh/guide/getting-started">快速开始 <span aria-hidden="true">→</span></a>
  </div>
  <div class="home-quickstart-steps" role="list" aria-label="快速开始命令">
    <div class="home-quickstart-step" role="listitem">
      <span class="step-number" aria-hidden="true">1</span>
      <div>
        <p class="non-windows-only">安装（macOS / Linux）</p>
        <code class="platform-install-command non-windows-only">curl -fsSL https://markitai.dev/setup.sh | sh</code>
        <p class="windows-only">安装（Windows）</p>
        <code class="platform-install-command windows-only">powershell -ExecutionPolicy ByPass -c "irm https://markitai.dev/setup.ps1 | iex"</code>
        <noscript>
          <p>Windows（PowerShell）：</p>
          <code>powershell -ExecutionPolicy ByPass -c "irm https://markitai.dev/setup.ps1 | iex"</code>
        </noscript>
        <details class="home-install-options">
          <summary>其他平台和手动安装方式</summary>
          <div class="home-install-option">
            <span>macOS 或 Linux</span>
            <code>curl -fsSL https://markitai.dev/setup.sh | sh</code>
          </div>
          <div class="home-install-option">
            <span>Windows</span>
            <code>powershell -ExecutionPolicy ByPass -c "irm https://markitai.dev/setup.ps1 | iex"</code>
          </div>
          <div class="home-install-option">
            <span>通过 Python</span>
            <code>uv tool install markitai</code>
          </div>
          <div class="home-install-option">
            <span>发布包</span>
            <a href="https://github.com/Ynewtime/markitai/releases">GitHub Releases</a>
          </div>
        </details>
      </div>
    </div>
    <div class="home-quickstart-step" role="listitem">
      <span class="step-number" aria-hidden="true">2</span>
      <div><p>转换网页</p><code>markitai https://github.com/Ynewtime/markitai --pure</code></div>
    </div>
    <div class="home-quickstart-step output" role="listitem">
      <span class="step-number" aria-hidden="true">3</span>
      <div>
        <p>得到 Markdown</p>
        <code># Markitai</code>
      </div>
    </div>
  </div>
</section>
