import test from 'node:test';
import assert from 'node:assert/strict';
import {bootstrapToken,authenticatedURL,serviceURL,artifactPath,api,NetworkError,parseUrls,mergeFiles,formatSize,copyText} from './api.js';

const origin='http://127.0.0.1:9876';
globalThis.location={origin};
const storage=()=>{const map=new Map();return{getItem:key=>map.get(key),setItem:(key,value)=>map.set(key,value),map};};
test('fragment token wins, both URL credentials disappear, and job selection survives',()=>{
  const saved=storage();let clean;
  bootstrapToken({href:origin+'/?token=query-secret&job=abc#token=fragment-secret'}, {replaceState:(_a,_b,url)=>clean=url},saved);
  assert.equal(clean,'/?job=abc');
  assert.equal(saved.map.get('markitai.service-token'),'fragment-secret');
  assert.equal(new URL(authenticatedURL('/api/jobs/abc/events')).searchParams.get('token'),'fragment-secret');
});
test('blocked storage still permits memory-only access and strips token first',()=>{
  let clean;Object.defineProperty(globalThis,'sessionStorage',{configurable:true,get(){throw new Error('blocked');}});
  bootstrapToken({href:origin+'/#token=memory-only'},{replaceState:(_a,_b,url)=>clean=url});
  assert.equal(clean,'/');assert.equal(new URL(authenticatedURL('/api/history')).searchParams.get('token'),'memory-only');
  delete globalThis.sessionStorage;
});
test('tokens cannot be added to foreign, credentialed or non-API URLs',()=>{
  for(const path of ['https://evil.test/api/jobs','//evil.test/api/jobs','javascript:alert(1)','http://user:pass@127.0.0.1:9876/api/jobs','/ui/app.js'])assert.throws(()=>authenticatedURL(path));
  assert.equal(serviceURL('/api/jobs?x=1',origin).origin,origin);
});
test('artifact references require exact manifest membership and resolve encoded filenames once',()=>{
  const allowed=new Set(['assets/a b%?.png','base.md','assets/same.png']);
  assert.equal(artifactPath('assets/a%20b%25%3F.png?download=1#view','base.md',allowed),'assets/a b%?.png');
  assert.equal(artifactPath('../assets/same.png','folder/note.md',allowed),'assets/same.png');
  for(const path of ['https://evil.test/assets/same.png','//evil.test/a','/assets/same.png','javascript:alert(1)','assets/%ZZ','assets/no.png','..\\assets\\same.png'])assert.equal(artifactPath(path,'base.md',allowed),null);
});
test('authenticated fetch rejects redirects and never dispatches an external request',async()=>{
  const previous=globalThis.fetch;const calls=[];globalThis.fetch=async(url,options)=>{calls.push({url,options});return new Response(JSON.stringify({ok:true}),{status:200,headers:{'content-type':'application/json'}});};
  try{assert.deepEqual(await api('/api/capabilities'),{ok:true});assert.equal(calls.length,1);assert.equal(calls[0].options.redirect,'error');assert.equal(calls[0].options.cache,'no-store');assert.equal(calls[0].options.headers.get('Authorization'),'Bearer memory-only');await assert.rejects(api('https://evil.test/api/jobs'));assert.equal(calls.length,1);}finally{globalThis.fetch=previous;}
});
test('URL lines accept bare domains, report the first unusable line and skip blanks',()=>{
  assert.deepEqual(parseUrls(' https://a.test/x \n\nexample.com/page?q=1\nhttp://127.0.0.1:8080/y\n'),{urls:['https://a.test/x','https://example.com/page?q=1','http://127.0.0.1:8080/y'],invalid:null});
  assert.deepEqual(parseUrls('https://ok.test\nnot a url\nftp://x.test').invalid,{value:'not a url',reason:'badUrl'});
  assert.deepEqual(parseUrls('ftp://files.test/a').invalid,{value:'ftp://files.test/a',reason:'badScheme'});
  assert.deepEqual(parseUrls('mailto:someone@example.test').invalid,{value:'mailto:someone@example.test',reason:'badScheme'});
  assert.deepEqual(parseUrls('   \n'),{urls:[],invalid:null});
});
test('the same chosen file is listed once and sizes stay readable',()=>{
  const file=(name,size,lastModified=1)=>({name,size,lastModified});
  const first=mergeFiles([],[file('a.pdf',10),file('b.pdf',20),file('a.pdf',10)]);
  assert.deepEqual(first.files.map(f=>f.name),['a.pdf','b.pdf']);assert.equal(first.duplicates,1);
  const second=mergeFiles(first.files,[file('a.pdf',10),file('a.pdf',10,2),file('a.pdf',11)]);
  assert.equal(second.files.length,4);assert.equal(second.duplicates,1);
  assert.equal(formatSize(0),'0 KiB');assert.equal(formatSize(1500),'2 KiB');assert.equal(formatSize(101*1024*1024),'101.0 MiB');
});
test('copy falls back to the selection command when the clipboard API is missing or refused',async()=>{
  const written=[];await copyText('direct',{clipboard:{writeText:async text=>written.push(text)}});assert.deepEqual(written,['direct']);
  const fake=result=>{const doc={commands:[],body:{append(node){doc.appended=node;}},createElement(){return {value:'',setAttribute(){},select(){doc.selected=this.value;},remove(){doc.removed=true;}};},execCommand(name){doc.commands.push(name);if(result instanceof Error)throw result;return result;}};return doc;};
  const doc=fake(true);await copyText('fallback',{clipboard:{writeText:async()=>{throw new Error('denied');}},doc});
  assert.equal(doc.selected,'fallback');assert.deepEqual(doc.commands,['copy']);assert.equal(doc.removed,true);
  const blocked=fake(false);await assert.rejects(copyText('x',{clipboard:undefined,doc:blocked}),/Copying is blocked/);assert.equal(blocked.removed,true);
  const thrown=fake(new Error('no'));await assert.rejects(copyText('x',{clipboard:undefined,doc:thrown}),/Copying is blocked/);
});
test('an unreachable service becomes an explicit network error, while aborts and HTTP errors keep their meaning',async()=>{
  const previous=globalThis.fetch;
  try{
    globalThis.fetch=async()=>{throw new TypeError('Failed to fetch');};
    await assert.rejects(api('/api/capabilities'),error=>error instanceof NetworkError&&/Cannot reach the Markitai service/.test(error.message));
    globalThis.fetch=async()=>{throw new DOMException('The user aborted a request.','AbortError');};
    await assert.rejects(api('/api/jobs',{method:'POST'}),{name:'AbortError'});
    globalThis.fetch=async()=>new Response(JSON.stringify({detail:'file exceeds upload limit',code:'payload_too_large'}),{status:413});
    await assert.rejects(api('/api/jobs',{method:'POST'}),{message:'file exceeds upload limit',status:413});
    globalThis.fetch=async()=>new Response('',{status:502});
    await assert.rejects(api('/api/jobs'),{message:'Request failed (502)',status:502});
  }finally{globalThis.fetch=previous;}
});
