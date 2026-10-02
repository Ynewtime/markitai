// Runs before first paint as a classic script (the CSP allows only same-origin
// scripts, so this cannot be inline): apply a stored theme and the resolved
// language, so an explicit choice never flashes the other appearance.
(function () {
  var root = document.documentElement, theme = null, lang = null;
  try {
    theme = localStorage.getItem('markitai.theme');
    lang = localStorage.getItem('markitai.lang');
  } catch (error) { /* Defaults apply. */ }
  if (theme === 'light' || theme === 'dark') root.setAttribute('data-theme', theme);
  if (lang !== 'en' && lang !== 'zh') lang = String(navigator.language || '').toLowerCase().indexOf('zh') === 0 ? 'zh' : 'en';
  root.lang = lang === 'zh' ? 'zh-CN' : 'en';
  if (lang === 'zh') document.title = '文档与网页转 Markdown · Markitai';
}());
