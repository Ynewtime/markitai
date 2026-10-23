import { defineConfig } from 'vitepress'
import { mermaid } from './mermaid.ts'

const SITE_URL = 'https://markitai.dev'
const SITE_NAME = 'Markitai'
const OG_IMAGE = `${SITE_URL}/og.jpg`
const OG_IMAGE_ALT =
  'Markitai — clean Markdown from files and URLs'

/**
 * Per-page descriptions. VitePress publishes `frontmatter.description` as the
 * meta description; without an entry here a page inherits the one site-wide
 * description, which makes it indistinguishable from every other page in
 * search results and social previews. A page missing from this map also gets
 * no hreflang alternates, so add both locales together.
 */
const PAGE_DESCRIPTIONS: Record<string, string> = {
  '/': 'Convert documents, images and web pages to clean Markdown with one command. LLM enhancement stays optional.',
  // pagePath() strips the index suffix, so the locale root is '/zh'.
  '/zh': '文档、图片和网页，一条命令转成干净的 Markdown。LLM 增强按需开启，不开也完全够用。',
  '/guide/getting-started':
    'Install markitai (setup script, release archive, pip or uv), run your first conversion, and see the output structure and supported formats.',
  '/zh/guide/getting-started':
    '安装 markitai（一键脚本、发布包、pip 或 uv），完成第一次转换，并了解输出结构与支持的格式。',
  '/guide/upgrading':
    'Upgrading from the Python markitai 1.x to markitai 2.0: how the setup script replaces it, and what changed in the CLI, output, configuration and Python API.',
  '/zh/guide/upgrading':
    '从 Python 版 markitai 1.x 升级到 markitai 2.0：安装脚本如何替换旧版，以及命令行、输出、配置与 Python API 的变化。',
  '/guide/comparison':
    'How markitai compares with markitdown, docling and anydoc — and which job each tool optimizes for.',
  '/zh/guide/comparison':
    'markitai 与 markitdown、docling、anydoc 的对比，以及每个工具各自优化的目标。',
  '/guide/serve':
    'Run the local markitai web workspace: upload files or URLs, watch live progress, preview results, and revisit seven days of history.',
  '/zh/guide/serve':
    '运行本地 markitai 网页工作台：上传文件或 URL、查看实时进度、预览结果，并回看七天内的历史记录。',
  '/guide/serve-reference':
    "Web workspace reference: access tokens and user tokens, users' and shared models, LLM key and address rules, Edit config, and every API route and error code.",
  '/zh/guide/serve-reference':
    '网页工作台参考：访问令牌与用户令牌、用户与共享模型、LLM key 与地址规则、编辑配置，以及全部 API 路由和错误代码。',
  '/guide/configuration':
    'Every markitai setting: configuration file and environment variables, LLM providers, image, screenshot, OCR, cache and security options.',
  '/zh/guide/configuration':
    'markitai 的全部设置：配置文件与环境变量、LLM 提供商、图片、截图、OCR、缓存与安全选项。',
  '/guide/cli':
    'Complete markitai command reference: conversion, output, concurrency, cache, URL, server and agent commands with every flag.',
  '/zh/guide/cli':
    '完整的 markitai 命令参考：转换、输出、并发、缓存、URL、服务与 Agent 命令，以及全部参数。',
  '/guide/output-profiles':
    'Shape markitai output for a downstream consumer with rag, obsidian and okf profiles, including the frozen images.json schema.',
  '/zh/guide/output-profiles':
    '用 rag、obsidian 与 okf 输出 Profile 为下游消费者塑形 markitai 输出，含冻结的 images.json schema。',
  '/guide/python-api':
    'Use markitai as a library: convert() and aconvert() with typed results, LLM enhancement and the same configuration as the CLI, plus the Node.js, Go and C bindings.',
  '/zh/guide/python-api':
    '把 markitai 当库使用：convert() 与 aconvert() 返回类型化结果，支持 LLM 增强并复用 CLI 的配置层级；另有 Node.js、Go 与 C 绑定。',
  '/guide/mcp':
    'Run the built-in markitai MCP server so AI agents can convert documents and URLs to Markdown over stdio.',
  '/zh/guide/mcp':
    '运行内置的 markitai MCP 服务器，让 AI Agent 通过 stdio 把文档与 URL 转换为 Markdown。',
  '/guide/performance':
    'What keeps a markitai conversion fast: one native program, local-first fetching, separate concurrency limits, caches and OCR threads, and how to diagnose a slow run.',
  '/zh/guide/performance':
    'markitai 如何让转换更快：单一原生程序、本地优先的抓取、分开的并发上限、缓存与 OCR 线程，以及慢速转换的排查方法。',
  '/guide/fetch-policy':
    'How markitai fetches URLs: static and local-browser strategies first, Defuddle, Jina and Cloudflare only once you allow them, domain profiles and remote-fetch consent.',
  '/zh/guide/fetch-policy':
    'markitai 如何抓取 URL：先用 static 与本机浏览器，经你同意后才用 Defuddle、Jina 与 Cloudflare，以及域名配置与远程抓取同意。',
  '/guide/subscriptions':
    'Use a Claude Code or GitHub Copilot subscription with markitai: install the official CLI, sign in, and fix what the web workbench reports.',
  '/zh/guide/subscriptions':
    '在 markitai 中使用 Claude Code 或 GitHub Copilot 订阅：安装官方命令行、登录，并处理网页工作台给出的提示。',
  '/guide/subscription-chatgpt':
    'Use a ChatGPT subscription with markitai: install the Codex CLI and sign in through it.',
  '/zh/guide/subscription-chatgpt':
    '在 markitai 中使用 ChatGPT 订阅：安装 Codex 命令行并通过它登录。',
  '/guide/pricing':
    'How markitai prices a conversion, what an unknown price means, how to get a cost for a model, and the per-document spending limit.',
  '/zh/guide/pricing':
    'markitai 如何为转换计费、价格未知意味着什么、怎样让某个模型显示费用，以及每个文档的费用上限。',
  '/changelog':
    'Release history for markitai — every added, changed, removed and fixed item, newest first.',
  '/zh/changelog':
    'markitai 的发布历史——新增、变更、移除与修复的每一条记录，最新在前。',
}

/** '/guide/cli' -> '/zh/guide/cli' and back. The root locale has no prefix. */
function alternatePath(path: string): string {
  if (path === '/') return '/zh'
  if (path === '/zh') return '/'
  return path.startsWith('/zh/') ? path.slice(3) : `/zh${path}`
}

/** Page path -> absolute URL. `'/'` and `'/zh'` are locale roots. */
function canonicalUrl(path: string): string {
  if (path === '/') return `${SITE_URL}/`
  if (path === '/zh') return `${SITE_URL}/zh/`
  return `${SITE_URL}${path}`
}

/** VitePress relativePath ('guide/cli.md') -> page path ('/guide/cli'). */
function pagePath(relativePath: string): string {
  const clean = relativePath.replace(/\.md$/, '')
  if (clean === '' || clean === 'index') return '/'
  const stripped = clean.replace(/\/index$/, '')
  return stripped === '' ? '/' : `/${stripped}`
}

export default defineConfig({
  title: 'Markitai',
  description: 'Opinionated Markdown converter with native LLM enhancement support',

  lastUpdated: true,
  cleanUrls: true,
  // Not pages: the release notes' sources (scripts/changelog.mjs publishes
  // their released sections as /changelog and /zh/changelog) and shared
  // snippets (`<!--@include: …/snippets/setup.md-->`).
  srcExclude: ['changelog/**', 'snippets/**'],

  // ```mermaid fences become SVG at build time (see mermaid.ts).
  markdown: {
    config(md) {
      md.use(mermaid)
    },
  },

  sitemap: {
    hostname: SITE_URL,
  },

  head: [
    ['link', { rel: 'icon', type: 'image/svg+xml', href: '/logo.svg' }],
    ['link', { rel: 'icon', type: 'image/x-icon', href: '/favicon.ico' }],
    ['link', { rel: 'alternate', type: 'text/plain', href: '/llms.txt' }],
    [
      'script',
      {},
      `(function () {
        try {
          var nav = navigator
          var platform = (nav.userAgentData && nav.userAgentData.platform) || nav.platform || nav.userAgent || ''
          if (/windows|win32|win64|wow64/i.test(platform)) {
            document.documentElement.classList.add('platform-windows')
          }
        } catch (_) {}
      })()`,
    ],
  ],

  transformPageData(pageData) {
    const path = pagePath(pageData.relativePath)
    const isZh = path === '/zh' || path.startsWith('/zh/')
    const url = canonicalUrl(path)
    const translated = alternatePath(path)
    const altUrl = canonicalUrl(translated)
    const title = pageData.title || SITE_NAME
    const description = PAGE_DESCRIPTIONS[path] ?? PAGE_DESCRIPTIONS['/']

    // VitePress derives pageData.description from frontmatter.description or
    // the first <meta name="description"> in frontmatter.head; the home layout
    // strips frontmatter, so seed it through the head array instead.
    pageData.frontmatter.head ??= []
    pageData.frontmatter.head.push(
      ['meta', { name: 'description', content: description }],
      ['meta', { property: 'og:type', content: 'website' }],
      ['meta', { property: 'og:site_name', content: SITE_NAME }],
      ['meta', { property: 'og:title', content: title }],
      ['meta', { property: 'og:description', content: description }],
      ['meta', { property: 'og:url', content: url }],
      ['meta', { property: 'og:image', content: OG_IMAGE }],
      ['meta', { property: 'og:image:width', content: '1200' }],
      ['meta', { property: 'og:image:height', content: '630' }],
      ['meta', { property: 'og:image:alt', content: OG_IMAGE_ALT }],
      ['meta', { property: 'og:locale', content: isZh ? 'zh_CN' : 'en' }],
      ['meta', { name: 'twitter:card', content: 'summary_large_image' }],
      ['meta', { name: 'twitter:title', content: title }],
      ['meta', { name: 'twitter:description', content: description }],
      ['meta', { name: 'twitter:image', content: OG_IMAGE }],
      ['link', { rel: 'canonical', href: url }],
    )

    // Alternates only exist for pages that ship in both locales; pointing
    // hreflang at a page that does not exist is worse than omitting it.
    if (PAGE_DESCRIPTIONS[translated] !== undefined) {
      pageData.frontmatter.head.push(
        [
          'meta',
          { property: 'og:locale:alternate', content: isZh ? 'en' : 'zh_CN' },
        ],
        ['link', { rel: 'alternate', hreflang: 'en', href: isZh ? altUrl : url }],
        [
          'link',
          { rel: 'alternate', hreflang: 'zh-CN', href: isZh ? url : altUrl },
        ],
        // x-default points at the default locale's copy, i.e. English.
        [
          'link',
          { rel: 'alternate', hreflang: 'x-default', href: isZh ? altUrl : url },
        ],
      )
    }

    // The hook runs after VitePress derives pageData.description, so return it
    // explicitly; the head entry above stops a duplicate meta tag.
    return { description }
  },

  locales: {
    root: {
      label: 'English',
      lang: 'en',
    },
    zh: {
      label: '简体中文',
      lang: 'zh-CN',
      description: '开箱即用的 Markdown 转换器，原生支持 LLM 增强',
      themeConfig: {
        nav: [
          { text: '指南', link: '/zh/guide/getting-started' },
          { text: '更新日志', link: '/zh/changelog' },
        ],
        sidebar: {
          '/zh/guide/': [
            {
              text: '开始',
              items: [
                { text: '快速开始', link: '/zh/guide/getting-started' },
                { text: '从 1.x 升级', link: '/zh/guide/upgrading' },
                { text: '为什么选 Markitai', link: '/zh/guide/comparison' },
              ],
            },
            {
              text: '指南',
              items: [
                { text: '网页工作台', link: '/zh/guide/serve' },
                { text: 'Claude 与 Copilot 订阅', link: '/zh/guide/subscriptions' },
                { text: 'ChatGPT 订阅', link: '/zh/guide/subscription-chatgpt' },
                { text: '价格与费用显示', link: '/zh/guide/pricing' },
                { text: '抓取策略', link: '/zh/guide/fetch-policy' },
                { text: '转换性能', link: '/zh/guide/performance' },
                { text: '输出 Profile', link: '/zh/guide/output-profiles' },
              ],
            },
            {
              text: '参考',
              items: [
                {
                  text: '配置说明',
                  link: '/zh/guide/configuration',
                  collapsed: true,
                  items: [
                    { text: '配置文件与环境变量', link: '/zh/guide/configuration#配置优先级' },
                    { text: 'LLM 提供商', link: '/zh/guide/configuration#llm-配置' },
                    { text: 'URL 抓取与隐私', link: '/zh/guide/configuration#fetch-policy-domain-profiles' },
                    { text: '缓存与输出', link: '/zh/guide/configuration#缓存配置' },
                    { text: '安全设置', link: '/zh/guide/configuration#安全配置' },
                  ],
                },
                { text: 'CLI 命令', link: '/zh/guide/cli' },
                { text: '工作台参考', link: '/zh/guide/serve-reference' },
              ],
            },
            {
              text: '集成',
              items: [
                { text: 'MCP 服务器', link: '/zh/guide/mcp' },
                { text: 'Python 与其他语言', link: '/zh/guide/python-api' },
              ],
            },
          ],
        },
        outline: {
          label: '本页目录',
        },
        lastUpdated: {
          text: '更新于',
        },
        docFooter: {
          prev: '上一页',
          next: '下一页',
        },
        darkModeSwitchLabel: '外观',
        lightModeSwitchTitle: '切换到浅色主题',
        darkModeSwitchTitle: '切换到深色主题',
        sidebarMenuLabel: '菜单',
        returnToTopLabel: '返回顶部',
        langMenuLabel: '切换语言',
        skipToContentLabel: '跳到正文',
        notFound: {
          title: '页面不存在',
          quote: '你访问的页面可能已移动或删除。',
          linkLabel: '返回首页',
          linkText: '返回首页',
        },
        footer: {
          copyright: 'Copyright © 2026-present',
        },
        editLink: {
          pattern: 'https://github.com/Ynewtime/markitai/edit/main/website/:path',
          text: '在 GitHub 上编辑此页',
        },
      },
    },
  },

  themeConfig: {
    logo: '/logo.svg',

    search: {
      provider: 'local',
      options: {
        locales: {
          zh: {
            translations: {
              button: {
                buttonText: '搜索文档',
                buttonAriaLabel: '搜索文档',
              },
              modal: {
                displayDetails: '显示详细列表',
                noResultsText: '无法找到相关结果',
                resetButtonTitle: '清除查询条件',
                backButtonTitle: '返回搜索结果',
                footer: {
                  selectText: '选择',
                  selectKeyAriaLabel: '回车键',
                  navigateText: '切换',
                  navigateUpKeyAriaLabel: '向上键',
                  navigateDownKeyAriaLabel: '向下键',
                  closeText: '关闭',
                  closeKeyAriaLabel: 'Esc 键',
                },
              },
            },
          },
        },
      },
    },

    nav: [
      { text: 'Guide', link: '/guide/getting-started' },
      { text: 'Changelog', link: '/changelog' },
    ],

    sidebar: {
      '/guide/': [
        {
          text: 'Start',
          items: [
            { text: 'Getting Started', link: '/guide/getting-started' },
            { text: 'Upgrading from 1.x', link: '/guide/upgrading' },
            { text: 'Why Markitai', link: '/guide/comparison' },
          ],
        },
        {
          text: 'Guide',
          items: [
            { text: 'Web Workspace', link: '/guide/serve' },
            { text: 'Claude and Copilot', link: '/guide/subscriptions' },
            { text: 'ChatGPT', link: '/guide/subscription-chatgpt' },
            { text: 'Prices and Cost', link: '/guide/pricing' },
            { text: 'Fetch Policy', link: '/guide/fetch-policy' },
            { text: 'Performance', link: '/guide/performance' },
            { text: 'Output Profiles', link: '/guide/output-profiles' },
          ],
        },
        {
          text: 'Reference',
          items: [
            {
              text: 'Configuration',
              link: '/guide/configuration',
              collapsed: true,
              items: [
                { text: 'Files & environment', link: '/guide/configuration#configuration-priority' },
                { text: 'LLM providers', link: '/guide/configuration#llm-configuration' },
                { text: 'URL fetching & privacy', link: '/guide/configuration#fetch-policy-domain-profiles' },
                { text: 'Cache & output', link: '/guide/configuration#cache-configuration' },
                { text: 'Security', link: '/guide/configuration#security-configuration' },
              ],
            },
            { text: 'CLI Reference', link: '/guide/cli' },
            { text: 'Workspace Reference', link: '/guide/serve-reference' },
          ],
        },
        {
          text: 'Integrations',
          items: [
            { text: 'MCP Server', link: '/guide/mcp' },
            { text: 'Python and other languages', link: '/guide/python-api' },
          ],
        },
      ],
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/Ynewtime/markitai' },
    ],

    footer: {
      copyright: 'Copyright © 2026-present',
    },

    editLink: {
      pattern: 'https://github.com/Ynewtime/markitai/edit/main/website/:path',
      text: 'Edit this page on GitHub',
    },
  },
})
