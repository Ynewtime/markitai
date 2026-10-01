// Runs before first paint (classic script, CSP 'self'): apply a stored theme and
// language so an explicit choice does not flash the other appearance.
(function () {
  var root = document.documentElement, theme = null, lang = null;
  try { theme = localStorage.getItem('markitai.theme'); lang = localStorage.getItem('markitai.lang'); } catch (error) { /* Defaults apply. */ }
  if (theme === 'light' || theme === 'dark') root.setAttribute('data-theme', theme);
  if (lang !== 'en' && lang !== 'zh') lang = String(navigator.language || '').toLowerCase().indexOf('zh') === 0 ? 'zh' : 'en';
  root.lang = lang === 'zh' ? 'zh-CN' : 'en';
  if (lang !== 'zh') return;
  // The page's static text is English. Keep it hidden (style.css) until i18n.js
  // has applied Chinese; if the modules never run, show it after three seconds.
  root.setAttribute('data-i18n-pending', '');
  document.title = 'Markitai · 让文档更好用';
  setTimeout(function () { root.removeAttribute('data-i18n-pending'); }, 3000);
}());
