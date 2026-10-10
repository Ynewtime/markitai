// Runs before first paint as a classic script (the CSP allows only same-origin
// scripts, so this cannot be inline): apply a stored theme, the sidebar's
// collapsed state and the resolved language, so a remembered choice never
// flashes the other appearance.
(function () {
  var root = document.documentElement, theme = null, lang = null, rail = null;
  try {
    theme = localStorage.getItem('markitai.theme');
    lang = localStorage.getItem('markitai.lang');
    rail = localStorage.getItem('markitai.sidebar');
  } catch (error) { /* Defaults apply. */ }
  if (theme === 'light' || theme === 'dark') root.setAttribute('data-theme', theme);
  if (rail === 'collapsed') root.setAttribute('data-rail', '');
  if (lang !== 'en' && lang !== 'zh' && lang !== 'ja') {
    var first = String(navigator.language || '').toLowerCase();
    lang = first.indexOf('zh') === 0 ? 'zh' : first.indexOf('ja') === 0 ? 'ja' : 'en';
  }
  root.lang = lang === 'zh' ? 'zh-CN' : lang;
  // Fetch the language pack alongside the application rather than after it.
  if (lang !== 'en') {
    var preload = document.createElement('link');
    preload.rel = 'modulepreload';
    preload.href = '/ui/i18n-' + lang + '.js';
    document.head.appendChild(preload);
  }
  if (lang === 'zh') document.title = '文档与网页转 Markdown · Markitai';
  if (lang === 'ja') document.title = '文書とウェブページを Markdown に · Markitai';
}());
