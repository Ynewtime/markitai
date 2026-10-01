// Runs before first paint (classic script, CSP 'self'): apply a stored theme and
// language so an explicit choice does not flash the other appearance.
(function () {
  var root = document.documentElement, theme = null, lang = null;
  try { theme = localStorage.getItem('markitai.theme'); lang = localStorage.getItem('markitai.lang'); } catch (error) { /* Defaults apply. */ }
  if (theme === 'light' || theme === 'dark') root.setAttribute('data-theme', theme);
  if (lang !== 'en' && lang !== 'zh') lang = String(navigator.language || '').toLowerCase().indexOf('zh') === 0 ? 'zh' : 'en';
  root.lang = lang === 'zh' ? 'zh-CN' : 'en';
}());
