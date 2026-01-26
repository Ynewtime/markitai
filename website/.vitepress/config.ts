import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'Markitai',
  description: 'Opinionated Markdown converter with native LLM support',

  lastUpdated: true,
  cleanUrls: true,

  head: [
    ['link', { rel: 'icon', type: 'image/svg+xml', href: '/logo.svg' }],
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
          { text: '更新日志', link: '/changelog' },
        ],
        sidebar: {
          '/zh/guide/': [
            {
              text: '指南',
              items: [
                { text: '快速开始', link: '/zh/guide/getting-started' },
                { text: '配置说明', link: '/zh/guide/configuration' },
                { text: 'CLI 命令', link: '/zh/guide/cli' },
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
                noResultsText: '无法找到相关结果',
                resetButtonTitle: '清除查询条件',
                footer: {
                  selectText: '选择',
                  navigateText: '切换',
                  closeText: '关闭',
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
            { text: 'Configuration', link: '/guide/configuration' },
            { text: 'CLI Reference', link: '/guide/cli' },
          ],
        },
      ],
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/Ynewtime/markitai' },
    ],

    footer: {
      message: 'Released under the MIT License.',
      copyright: 'Copyright © 2026-present',
    },

    editLink: {
      pattern: 'https://github.com/Ynewtime/markitai/edit/main/website/:path',
      text: 'Edit this page on GitHub',
    },
  },
})
