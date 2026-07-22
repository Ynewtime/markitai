import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'Markitai',
  description: 'Opinionated Markdown converter with native LLM support',

  lastUpdated: true,
  cleanUrls: true,

  head: [
    ['link', { rel: 'icon', type: 'image/svg+xml', href: '/logo.svg' }],
    ['link', { rel: 'icon', type: 'image/x-icon', href: '/favicon.ico' }],
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
              text: '指南',
              items: [
                { text: '快速开始', link: '/zh/guide/getting-started' },
                { text: '网页工作台', link: '/zh/guide/serve' },
                {
                  text: '配置说明',
                  link: '/zh/guide/configuration',
                  collapsed: true,
                  items: [
                    { text: '配置文件与环境变量', link: '/zh/guide/configuration#配置优先级' },
                    { text: 'LLM 提供商', link: '/zh/guide/configuration#llm-配置' },
                    { text: 'URL 抓取与隐私', link: '/zh/guide/configuration#url-抓取配置' },
                    { text: '缓存与输出', link: '/zh/guide/configuration#缓存配置' },
                    { text: '安全设置', link: '/zh/guide/configuration#安全配置' },
                  ],
                },
                { text: 'CLI 命令', link: '/zh/guide/cli' },
                { text: '抓取策略', link: '/zh/guide/fetch-policy' },
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
          text: 'Guide',
          items: [
            { text: 'Getting Started', link: '/guide/getting-started' },
            { text: 'Web Workspace', link: '/guide/serve' },
            {
              text: 'Configuration',
              link: '/guide/configuration',
              collapsed: true,
              items: [
                { text: 'Files & environment', link: '/guide/configuration#configuration-priority' },
                { text: 'LLM providers', link: '/guide/configuration#llm-configuration' },
                { text: 'URL fetching & privacy', link: '/guide/configuration#url-fetch-configuration' },
                { text: 'Cache & output', link: '/guide/configuration#cache-configuration' },
                { text: 'Security', link: '/guide/configuration#security-configuration' },
              ],
            },
            { text: 'CLI Reference', link: '/guide/cli' },
            { text: 'Fetch Policy', link: '/guide/fetch-policy' },
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
