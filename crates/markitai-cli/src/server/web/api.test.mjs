import test from 'node:test';
import assert from 'node:assert/strict';
import {bootstrapToken,authenticatedURL,serviceURL,artifactPath,api,ApiError,NetworkError,errorDetail,upload,uploadState,throttle,parseUrls,mergeFiles,formatSize,copyText} from './api.js';
import {useLocale} from './i18n.js';

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
    globalThis.fetch=async()=>new Response(JSON.stringify({detail:'file exceeds upload limit',code:'payload_too_large',reason:'file_too_large'}),{status:413});
    await assert.rejects(api('/api/jobs',{method:'POST'}),{message:'A file is larger than the 100 MiB upload limit.',detail:'file exceeds upload limit',reason:'file_too_large',status:413});
    globalThis.fetch=async()=>new Response('',{status:502});
    await assert.rejects(api('/api/jobs'),{message:'Request failed (502)',detail:'',status:502});
  }finally{globalThis.fetch=previous;}
});
test('service errors are localized by reason, then status code, and keep the original wording',()=>{
  try{
    for(const [locale,expected] of [['en','This job no longer exists. It may have been deleted.'],['zh','这个任务已不存在，可能已被删除。']]){
      useLocale(locale);
      const error=new ApiError(404,{detail:'job not found',code:'not_found',reason:'job_not_found'});
      assert.equal(error.message,expected);assert.equal(error.detail,'job not found');assert.equal(error.reason,'job_not_found');
      assert.equal(errorDetail(error),'job not found');
    }
    useLocale('zh');
    // An unknown reason falls back to the status-derived code; structured details give their own code.
    assert.equal(new ApiError(413,{detail:'too big',code:'payload_too_large',reason:'something_new'}).message,'本次上传超出服务单次请求可接收的大小。');
    const stale=new ApiError(409,{detail:{code:'stale_revision',current_revision:'r2'},code:'conflict'});
    assert.equal(stale.reason,'stale_revision');assert.equal(stale.message,'设置已在别处被修改。');assert.equal(stale.detail,'stale_revision');
    // Text the page cannot classify stays as the service wrote it, with nothing repeated as a detail.
    const plain=new ApiError(418,{detail:'teapot'});assert.equal(plain.message,'teapot');assert.equal(errorDetail(plain),'');
    assert.equal(new ApiError(500,null).message,'请求失败（500）');
  }finally{useLocale('en');}
});
test('only a revision conflict keeps a settings draft for review; other 409s stay plain errors',async()=>{
  const {conflicted}=await import('./settings.js');
  for(const reason of ['stale_revision','config_changed',null])assert.equal(conflicted({status:409,reason}),true,String(reason));
  for(const reason of ['settings_read_only','ambiguous_legacy_model_name'])assert.equal(conflicted({status:409,reason}),false,reason);
  assert.equal(conflicted({status:422,reason:'stale_revision'}),false);
});
class FakeRequest{
  static last=null;
  constructor(){this.headers={};this.upload={};this.aborted=false;FakeRequest.last=this;}
  open(method,url){this.method=method;this.url=url;}
  setRequestHeader(name,value){this.headers[name]=value;}
  send(body){this.body=body;}
  abort(){this.aborted=true;this.onabort?.();}
  progress(loaded,total){this.upload.onprogress?.({loaded,total,lengthComputable:total>0});}
  respond(status,value,responseURL=this.url){this.status=status;this.responseText=value===undefined?'':JSON.stringify(value);this.responseURL=responseURL;this.onload?.();}
}
test('uploads report real byte progress, keep the token and resolve with the created job',async()=>{
  const events=[];const previous=globalThis.window;globalThis.window={dispatchEvent:event=>events.push(event.type)};
  globalThis.CustomEvent??=class extends Event{};
  try{
    const seen=[];const pending=upload('/api/jobs','BODY',{Request:FakeRequest,onProgress:(loaded,total)=>seen.push([loaded,total])});
    const request=FakeRequest.last;
    assert.equal(request.method,'POST');assert.equal(request.url,origin+'/api/jobs');assert.equal(request.body,'BODY');
    assert.equal(request.headers.Authorization,'Bearer memory-only');
    request.progress(10,40);request.progress(40,40);request.upload.onload?.({lengthComputable:true,total:40});
    request.respond(201,{job_id:'abc',items:[]});
    assert.deepEqual(await pending,{job_id:'abc',items:[]});
    assert.deepEqual(seen,[[10,40],[40,40],[40,40]]);
    assert.ok(events.includes('markitai:online'));
    const unknown=[];upload('/api/jobs','x',{Request:FakeRequest,onProgress:(loaded,total)=>unknown.push([loaded,total])});
    FakeRequest.last.progress(5,0);assert.deepEqual(unknown,[[5,0]]);
  }finally{globalThis.window=previous;}
});
test('an upload can be cancelled, and service, network and redirect failures keep their meaning',async()=>{
  const events=[];const previous=globalThis.window;globalThis.window={dispatchEvent:event=>events.push(event.type)};
  globalThis.CustomEvent??=class extends Event{};
  try{
    const controller=new AbortController();
    const cancelled=upload('/api/jobs','x',{Request:FakeRequest,signal:controller.signal});
    const request=FakeRequest.last;controller.abort();
    await assert.rejects(cancelled,{name:'AbortError'});assert.equal(request.aborted,true);
    const early=new AbortController();early.abort();FakeRequest.last=null;
    await assert.rejects(upload('/api/jobs','x',{Request:FakeRequest,signal:early.signal}),{name:'AbortError'});assert.equal(FakeRequest.last,null);
    const refused=upload('/api/jobs','x',{Request:FakeRequest});FakeRequest.last.respond(413,{detail:'file exceeds upload limit',code:'payload_too_large',reason:'file_too_large'});
    await assert.rejects(refused,error=>error instanceof ApiError&&error.status===413&&error.reason==='file_too_large'&&error.detail==='file exceeds upload limit');
    const offline=upload('/api/jobs','x',{Request:FakeRequest});FakeRequest.last.onerror();
    await assert.rejects(offline,error=>error instanceof NetworkError);assert.ok(events.includes('markitai:offline'));
    const unauthorized=upload('/api/jobs','x',{Request:FakeRequest});FakeRequest.last.respond(401,{detail:'authentication required',code:'unauthorized',reason:'token_required'});
    await assert.rejects(unauthorized,{status:401});assert.ok(events.includes('markitai:unauthorized'));
    const redirected=upload('/api/jobs','x',{Request:FakeRequest});FakeRequest.last.respond(201,{job_id:'elsewhere'},'https://evil.test/api/jobs');
    await assert.rejects(redirected,error=>!(error instanceof ApiError)&&error.message==='Request failed');
  }finally{globalThis.window=previous;}
});
test('upload progress text gives a whole percentage and only reaches 100 when every byte is sent',()=>{
  try{
    useLocale('en');
    assert.deepEqual(uploadState(0,0),{percent:null,done:false,text:'Starting upload…'});
    assert.deepEqual(uploadState(2048,0),{percent:null,done:false,text:'Uploading · 2 KiB sent'});
    assert.deepEqual(uploadState(1,3*1024*1024),{percent:0,done:false,text:'Uploading 0% · 1 KiB of 3.0 MiB'});
    assert.equal(uploadState(3*1024*1024-1,3*1024*1024).percent,99);
    assert.deepEqual(uploadState(3*1024*1024,3*1024*1024),{percent:100,done:true,text:'Upload complete · the service is saving the files and creating the job…'});
    useLocale('zh');
    assert.equal(uploadState(1536*1024,3*1024*1024).text,'正在上传 50% · 1.5 MiB / 3.0 MiB');
  }finally{useLocale('en');}
});
test('progress redraws are throttled, keep the latest value and never drop completion',async()=>{
  let clock=0;const drawn=[];const report=throttle((loaded,total)=>drawn.push(loaded),20,()=>clock);
  report(1,100);report(2,100);report(3,100);
  assert.deepEqual(drawn,[1]);
  clock=25;await new Promise(resolve=>setTimeout(resolve,40));
  assert.deepEqual(drawn,[1,3]);
  report(4,100);report(100,100);
  assert.deepEqual(drawn,[1,3,100]);
  clock=30;report(50,100);report.cancel();
  await new Promise(resolve=>setTimeout(resolve,40));
  assert.deepEqual(drawn,[1,3,100]);
});
