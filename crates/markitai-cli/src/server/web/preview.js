import {marked} from './marked.js';
import DOMPurify from './purify.js';
import {artifactPath, authenticatedURL, fileURL} from './api.js';
import {previewBody} from './result-tools.js';
import {t} from './i18n.js';

export function preview(markdown, job, documentPath, artifacts, target) {
  const allowed = new Set(artifacts.map(item => item.relpath));
  const fragment = DOMPurify.sanitize(marked.parse(previewBody(markdown), {async: false, gfm: true}), {
    RETURN_DOM_FRAGMENT: true, USE_PROFILES: {html: true},
    FORBID_TAGS: ['style','script','iframe','object','embed','form','input','button','textarea','select','video','audio','source','picture','link','meta','base'],
    FORBID_ATTR: ['style','srcset','poster','ping','target','id','name'],
    ALLOW_DATA_ATTR: false, ALLOW_ARIA_ATTR: false,
  });
  for (const node of fragment.querySelectorAll('img')) {
    const path = artifactPath(node.getAttribute('src') || '', documentPath, allowed);
    if (path && /\.(png|jpe?g|gif|webp|avif)$/i.test(path)) {
      node.src = authenticatedURL(fileURL(job, path)); node.loading = 'lazy'; node.referrerPolicy = 'no-referrer';
    } else {
      const placeholder = document.createElement('span'); placeholder.className = 'blocked-image';
      placeholder.textContent = t('imagePlaceholder', {alt: node.getAttribute('alt') || t('imageUnavailable')});
      node.replaceWith(placeholder);
    }
  }
  for (const node of fragment.querySelectorAll('a')) {
    const href = node.getAttribute('href') || '';
    const path = artifactPath(href, documentPath, allowed);
    if (path) { node.href = authenticatedURL(fileURL(job, path)); node.setAttribute('download', ''); }
    else {
      try {
        const url = new URL(href);
        if (!['https:', 'http:', 'mailto:'].includes(url.protocol) || url.username || url.password) throw new Error();
        node.href = url.href;
      } catch { node.removeAttribute('href'); }
    }
    node.target = '_blank'; node.rel = 'noopener noreferrer'; node.referrerPolicy = 'no-referrer';
  }
  target.replaceChildren(fragment);
}
