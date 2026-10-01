import test from 'node:test';
import assert from 'node:assert/strict';
import {markdownPair, compareLines, renderComparison, waitForPrintImages, printPreview, previewBody} from './result-tools.js';
import {api} from './api.js';
import {useLocale} from './i18n.js';

const restored = (result, side) => result.rows.filter(row => row.kind !== (side === 'base' ? 'add' : 'remove')).map(row => row.text).join('');

test('pairing uses complete artifact identity, including dotted stems, and rejects ambiguity', () => {
  const items = paths => paths.map(relpath => ({relpath}));
  assert.deepEqual(markdownPair(items(['folder/report.v2.md','folder/report.v2.llm.md','assets/pic.png'])), {base:'folder/report.v2.md',enhanced:'folder/report.v2.llm.md'});
  assert.deepEqual(markdownPair(items(['x.llm.md','x.llm.llm.md'])), {base:'x.llm.md',enhanced:'x.llm.llm.md'});
  for (const value of [['a.md','other.llm.md'],['a.md'],['a.md','a.llm.md','b.md','b.llm.md'],['a.md','a.llm.md','a.llm.llm.md']]) assert.equal(markdownPair(items(value)),null);
});

test('diff retains Unicode, blank lines, CRLF, terminal newline and hostile literal text', () => {
  const before='# 文档🚀\r\n\r\n<script>alert(1)</script>\nkeep  \nlast';
  const after='# 文档🚀\r\n\r\n<img src=x onerror=alert(1)>\nkeep  \nlast\n';
  const result=compareLines(before,after);
  assert.equal(result.kind,'diff'); assert.equal(restored(result,'base'),before); assert.equal(restored(result,'enhanced'),after);
  assert.equal(result.added,2);assert.equal(result.removed,2);
  assert.ok(result.rows.some(row=>row.kind==='same'&&row.text==='keep  \n'));
});

test('ordered insertions/deletions/repeated lines reconstruct both complete documents', () => {
  for(const [base,enhanced] of [['',''],['','\n'],['gone\n',''],['a\na\nb\n','a\nb\na\n'],['a\nb\n','x\na\nb\ny\n'],['\n\n','\n'],['same','same']]) {
    const result=compareLines(base,enhanced);assert.equal(restored(result,'base'),base);assert.equal(restored(result,'enhanced'),enhanced);
  }
  assert.deepEqual(compareLines('same\n','same\n').rows,[{kind:'same',text:'same\n'}]);
});

test('comparison refuses oversized work without showing a partial diff', () => {
  const limits={characters:200,lines:10,cells:4};
  assert.deepEqual(compareLines('a'.repeat(201),'',limits),{kind:'too-large'});
  assert.deepEqual(compareLines('a\n'.repeat(11),'',limits),{kind:'too-large'});
  assert.deepEqual(compareLines('a\nb\nc\n','d\ne\nf\n',limits),{kind:'too-large'});
  // Equal edges do not consume the middle-work budget.
  assert.equal(compareLines('same\na\ntail\n','same\nb\ntail\n',limits).kind,'diff');
});

class Node {
  constructor(tag,doc){this.tag=tag;this.ownerDocument=doc;this.children=[];this.attributes=new Map();this.textContent='';this.className='';this.parent=null;}
  set innerHTML(_value){throw new Error('HTML insertion is forbidden in a comparison');}
  setAttribute(name,value){this.attributes.set(name,value);}
  removeAttribute(name){this.attributes.delete(name);}
  append(...nodes){for(const node of nodes){node.parent=this;this.children.push(node);}}
  replaceChildren(...nodes){this.children=[];this.append(...nodes);}
  remove(){if(this.parent)this.parent.children=this.parent.children.filter(child=>child!==this);this.parent=null;}
  querySelectorAll(tag){return this.children.flatMap(child=>[...(child.tag===tag?[child]:[]),...child.querySelectorAll(tag)]);}
  cloneNode(deep){const clone=new Node(this.tag,this.ownerDocument);clone.attributes=new Map(this.attributes);clone.textContent=this.textContent;clone.className=this.className;Object.assign(clone,{complete:this.complete,naturalWidth:this.naturalWidth,decode:this.decode});if(deep)for(const child of this.children)clone.append(child.cloneNode(true));return clone;}
}
function environment(){
  const listeners=new Map(), classes=new Set();
  const doc={title:'Original application',createElement(tag){return new Node(tag,this);}};
  doc.body=new Node('body',doc);doc.body.classList={add:value=>classes.add(value),remove:value=>classes.delete(value),contains:value=>classes.has(value)};
  let prints=0;
  const win={addEventListener:(event,fn)=>listeners.set(event,fn),removeEventListener:(event,fn)=>{if(listeners.get(event)===fn)listeners.delete(event);},print(){prints++;},dispatch(event){listeners.get(event)?.();}};
  return {doc,win,listeners,classes,get prints(){return prints;}};
}
const tick=()=>new Promise(resolve=>setImmediate(resolve));

test('comparison renderer uses text nodes for executable-looking source',()=>{
  const {doc}=environment(),target=new Node('div',doc);
  renderComparison(compareLines('<script>OLD</script>\n','<img onerror=evil()>\n'),target);
  const code=target.querySelectorAll('code');assert.deepEqual(code.map(row=>row.textContent),['<script>OLD</script>','<img onerror=evil()>']);
  assert.equal(target.querySelectorAll('script').length,0);assert.equal(target.querySelectorAll('img').length,0);
});

test('printing clones only preview, removes link credentials, waits for images and restores state',async()=>{
  const env=environment(),preview=new Node('div',env.doc);preview.setAttribute('id','rendered');preview.setAttribute('hidden','');
  const link=new Node('a',env.doc);link.setAttribute('href','http://service/api/file?token=secret');link.textContent='download';preview.append(link);
  const image=new Node('img',env.doc);image.complete=false;image.naturalWidth=20;let release;image.decode=()=>new Promise(resolve=>{release=resolve;});preview.append(image);
  const session=printPreview({preview,title:'Report\n<title>',doc:env.doc,win:env.win});
  assert.equal(env.prints,0);assert.equal(env.doc.title,'Original application');assert.equal(env.doc.body.children.length,1);
  const clone=env.doc.body.children[0].children[0];assert.equal(clone.attributes.has('hidden'),false);assert.equal(clone.attributes.has('id'),false);assert.equal(clone.querySelectorAll('a')[0].attributes.has('href'),false);assert.equal(link.attributes.get('href'),'http://service/api/file?token=secret');
  release();await tick();assert.equal(env.prints,1);assert.equal(env.doc.title,'Report<title>');assert.ok(env.classes.has('printing-result'));
  env.win.dispatch('afterprint');await session.done;
  assert.equal(env.doc.title,'Original application');assert.equal(env.doc.body.children.length,0);assert.equal(env.classes.size,0);assert.equal(env.listeners.size,0);
});

test('broken/stalled images abort printing explicitly and release temporary state',async()=>{
  for(const kind of ['broken','stalled']){
    const env=environment(),preview=new Node('div',env.doc),image=new Node('img',env.doc);
    image.complete=kind==='broken';image.naturalWidth=0;image.decode=()=>new Promise(()=>{});preview.append(image);
    const session=printPreview({preview,title:'not printed',doc:env.doc,win:env.win,timeoutMs:5});
    await assert.rejects(session.done,/image|Images/);assert.equal(env.prints,0);assert.equal(env.doc.body.children.length,0);assert.equal(env.doc.title,'Original application');
  }
});

test('selection change, explicit cancel and thrown print never leave page in print mode',async()=>{
  for(const mode of ['stale','cancel','throws']){
    const env=environment(),preview=new Node('div',env.doc);
    if(mode==='throws')env.win.print=()=>{throw new Error('dialog blocked');};
    const session=printPreview({preview,title:'Document',doc:env.doc,win:env.win,isCurrent:()=>mode!=='stale'});
    if(mode==='cancel')session.cancel();
    await assert.rejects(session.done,mode==='throws'?/dialog blocked/:{name:'AbortError'});
    assert.equal(env.doc.title,'Original application');assert.equal(env.doc.body.children.length,0);assert.equal(env.classes.size,0);
  }
});

test('image readiness handles an already-aborted selection without waiting',async()=>{
  const controller=new AbortController();controller.abort();
  await assert.rejects(waitForPrintImages([],{signal:controller.signal}),{name:'AbortError'});
});

test('comparison file reads enforce declared and streamed byte limits on the authenticated API',async()=>{
  const oldFetch=globalThis.fetch, oldLocation=globalThis.location;globalThis.location={origin:'http://127.0.0.1:8080'};
  const calls=[];
  try{
    globalThis.fetch=async(url,options)=>{calls.push({url,options});return new Response('abcd',{headers:{'content-length':'4'}});};
    await assert.rejects(api('/api/jobs/test/files/base.md',{text:true,maxTextBytes:3}),/limit/);
    globalThis.fetch=async()=>new Response(new ReadableStream({start(controller){controller.enqueue(new Uint8Array([97,98]));controller.enqueue(new Uint8Array([99,100]));controller.close();}}));
    await assert.rejects(api('/api/jobs/test/files/base.md',{text:true,maxTextBytes:3}),/limit/);
    globalThis.fetch=async()=>new Response(new ReadableStream({start(controller){controller.enqueue(new Uint8Array([0xe6]));controller.enqueue(new Uint8Array([0x96,0x87]));controller.close();}}));
    assert.equal(await api('/api/jobs/test/files/base.md',{text:true,maxTextBytes:3}),'文');
    assert.equal(calls[0].options.redirect,'error');assert.equal(calls[0].options.cache,'no-store');
  }finally{globalThis.fetch=oldFetch;globalThis.location=oldLocation;}
});

test('rendered body omits only a leading YAML frontmatter block, as the reference preview does', () => {
  assert.equal(previewBody('---\ntitle: A\ntags:\n- x\n---\n\n# A\n\nBody\n'), '\n\n# A\n\nBody\n');
  assert.equal(previewBody('---\ntitle: only\n---'), '');
  for (const text of ['# No frontmatter\n\n---\nnot: yaml\n---\n', ' ---\nx: 1\n---\n', '---\nunterminated\n', '']) assert.equal(previewBody(text), text);
  assert.equal(previewBody('---\na: 1\n---\n---\nb: 2\n---\n'), '\n---\nb: 2\n---\n');
});

test('an image whose load fails reports the explicit print reason, not the decoder wording',async()=>{
  const broken={complete:false,naturalWidth:0,decode:async()=>{throw new DOMException('The source image cannot be decoded.','EncodingError');}};
  await assert.rejects(waitForPrintImages([broken]),/preview image could not load/);
  const ready={complete:false,naturalWidth:0,decode:async()=>{ready.naturalWidth=3;}};
  await waitForPrintImages([ready]);
  assert.equal(ready.loading,'eager');
});

test('comparison and print messages follow the selected interface language',async()=>{
  try{
    useLocale('zh');
    const {doc}=environment(),target=new Node('div',doc);
    renderComparison(compareLines('a\nb\n','a\nc\nd\n'),target);
    assert.equal(target.children[0].textContent,'基础版 → 增强版 · 新增 2 行，删除 1 行');
    assert.equal(target.children[1].attributes.get('aria-label'),'从基础版到增强版的改动');
    renderComparison({kind:'too-large'},target);
    assert.match(target.children[0].textContent,/超出浏览器限制/);
    await assert.rejects(waitForPrintImages([{complete:true,naturalWidth:0}]),/预览图片无法加载/);
  }finally{useLocale('en');}
  const {doc}=environment(),target=new Node('div',doc);
  renderComparison(compareLines('a\n','b\n'),target);
  assert.equal(target.children[0].textContent,'Base → Enhanced · 1 added, 1 removed lines');
});
