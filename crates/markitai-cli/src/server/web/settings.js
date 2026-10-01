import {api,element,button,errorText,errorDetail,detailNode} from './api.js';
import {t,serviceNote,discoveryStatus,providerName} from './i18n.js';
const $=id=>document.getElementById(id), base='/api/settings/llm';
let hooks, settings=null, providers=[], selected=null, draftRevision=null, discovered=[], edit=null, busy=false;
let loadGeneration=0;
function label(node,key){node.dataset.i18n=key;node.textContent=t(key);}
// A translated service message keeps the service's own wording folded beside it.
function status(text,bad=false,detail=''){$('settings-message').replaceChildren(text);if(detail&&detail!==text)$('settings-message').append(detailNode(detail));$('settings-message').classList.toggle('item-error',bad);}
// Probe details are fixed provider-runtime phrases; known ones are translated.
function probeStatus(result){const note=serviceNote(result.detail);status(note.text,!result.ok,note.detail);}
// Only a revision conflict keeps the draft for review; other 409s (read-only settings, ambiguous names) are plain errors.
export const conflicted=error=>error.status===409&&(!error.reason||['stale_revision','config_changed'].includes(error.reason));
function formBusy(value){busy=value;for(const node of $('connection-form').querySelectorAll('button'))node.disabled=value;}
async function guarded(action){if(busy)return;formBusy(true);try{await action();}catch(error){if(conflicted(error)){await conflict(error);}else{status(errorText(error),true,errorDetail(error));}}finally{formBusy(false);}}
async function conflict(error){
  const previous=draftRevision;await loadSettings();draftRevision=previous;
  $('conflict-message').hidden=false;
  $('conflict-message').replaceChildren(element('strong','',t('conflictTitle')),element('p','',t('conflictText')));
  $('conflict-message').append(button(t('conflictAccept'),()=>{draftRevision=settings.revision;$('conflict-message').hidden=true;status(t('conflictKept'));}));
  status(errorText(error),true,errorDetail(error));
}
function credentials(){const fields={};for(const [prefix,key]of[['key','api_key'],['base','api_base']]){const mode=$(prefix+'-mode').value;if(mode==='clear')fields[key]=null;if(mode==='replace'){const value=$('provider-'+prefix).value.trim();if(!value)throw new Error(t(key==='api_key'?'enterKey':'enterBase'));fields[key]=value;}}return fields;}
function currentModels(){const entries=[...$('discovered-models').querySelectorAll('input:checked')].map(node=>node.value);entries.push(...$('manual-models').value.split(/\r?\n/).map(line=>line.trim()).filter(Boolean));return [...new Set(entries)];}
function modelPayload(model){const value={model_name:$('routing-group').value.trim(),model,weight:Number($('model-weight').value),...credentials()};if(!value.model_name||!Number.isInteger(value.weight)||value.weight<0)throw new Error(t('routingRequired'));if(selected){if(selected.provider_id.startsWith('legacy:'))value.credential_deployment_id=selected.deployment_id||selected.provider_id.slice(7);else value.credential_provider_id=selected.provider_id;}else value.provider=$('provider-type').value;return value;}
function clearSecrets(){for(const key of ['key','base']){$('provider-'+key).value='';$(key+'-mode').value='keep';$('provider-'+key).disabled=true;}}
function reset(){selected=null;draftRevision=settings?.revision||null;discovered=[];clearSecrets();$('provider-type').disabled=false;$('provider-type').value='openai';label($('connection-title'),'addConnectionTitle');label($('connection-hint'),'connectionHint');$('manual-models').value='';$('discovered-models').replaceChildren();$('discovery-status').textContent='';$('routing-group').value='default';$('model-weight').value='1';$('save-connection').hidden=true;$('conflict-message').hidden=true;status('');}
// The editor sits below the lists on narrow screens; bring it into view after choosing what to edit.
function revealEditor(focus){const box=$('connection-form').getBoundingClientRect();if(box.top<0||box.top>innerHeight*0.5)$('connection-form').scrollIntoView({behavior:'smooth',block:'start'});focus?.focus({preventScroll:true});}
function setUp(provider,state){reset();$('provider-type').value=provider;if(state!=='needs_credentials'){revealEditor($('discover-models'));return;}$('key-mode').value='replace';$('provider-key').disabled=false;revealEditor($('provider-key'));}
async function editConnection(provider){
  await guarded(async()=>{
    const values=await api(`${base}/providers/${encodeURIComponent(provider.provider_id)}/credentials`);
    selected=provider;draftRevision=settings.revision;clearSecrets();
    $('provider-type').value=provider.provider;$('provider-type').disabled=true;
    $('provider-key').value=values.api_key||'';$('provider-base').value=values.api_base||'';
    $('provider-base').placeholder=values.api_base_placeholder||'https://api.example.com/v1';
    delete $('connection-title').dataset.i18n;$('connection-title').textContent=providerName(provider);
    label($('connection-hint'),'editHint');
    $('save-connection').hidden=false;$('conflict-message').hidden=true;$('manual-models').value='';$('discovered-models').replaceChildren();$('discovery-status').textContent='';status(t('connectionLoaded'));
  });
  if(selected===provider)revealEditor($('key-mode'));
}
const KINDS={common:'kindCommon',environment:'kindEnvironment',configured:'kindConfigured'};
const STATES={ready:'provReady',needs_credentials:'provNeedsCredentials',unknown:'provUnknown',disabled:'provDisabled'};
const ORIGINS={explicit:'originExplicit',environment:'originEnvironment',project:'originProject',user:'originUser',default:'originDefault'};
const word=(table,value)=>table[value]?t(table[value]):value;
export function renderSettings(){
  if(!settings)return;
  // The conversion runtime's own answer (also used for Enhance) wins over the saved-list estimate.
  const runtime=hooks?.routable?.(),routable=typeof runtime==='boolean'?runtime:settings.routable;
  $('settings-context').textContent=t('settingsContext',{origin:word(ORIGINS,settings.config_origin),path:settings.config_path,state:t(routable?'modelsAvailable':'noRoutableModel')});
  $('provider-list').replaceChildren();
  if(!providers.length)$('provider-list').append(element('p','empty compact',t('noConnections')));
  for(const provider of providers){
    const row=element('article','connection-row');const info=element('div');
    const meta=[word(KINDS,provider.kind),word(STATES,provider.status)];if(provider.model_count!==undefined)meta.push(t('modelCount',{count:provider.model_count}));
    info.append(element('strong','',providerName(provider)),element('small',`muted state-${provider.status}`,meta.join(' · ')));if(provider.api_base)info.append(element('small','mono muted',provider.api_base));
    const actions=element('div','row');
    if(provider.provider_id){actions.append(button(t('actEdit'),()=>editConnection(provider)),button(t('actDelete'),()=>guarded(async()=>{if(!await hooks.confirmDelete(t('deleteConnectionTitle'),t('deleteConnectionText',{label:providerName(provider)})))return;await api(`${base}/providers/${encodeURIComponent(provider.provider_id)}?expected_revision=${encodeURIComponent(settings.revision)}`,{method:'DELETE'});if(selected?.provider_id===provider.provider_id)reset();await saved(t('connectionDeleted'));}),'quiet small delete'));}
    else if(['openai','anthropic','gemini','deepseek','openrouter','azure','ollama','custom'].includes(provider.provider)){actions.append(button(t('actSetUp'),()=>setUp(provider.provider,provider.status)));}
    row.append(info,actions);$('provider-list').append(row);
  }
  $('deployment-list').replaceChildren();
  if(!settings?.deployments?.length&&!settings?.detected?.length)$('deployment-list').append(element('p','empty compact',t('noModels')));
  for(const deployment of [...(settings?.deployments||[]),...(settings?.detected||[])]){
    const row=element('article','connection-row');const info=element('div');info.append(element('strong','filename',deployment.model),element('small','muted',t('deploymentMeta',{group:deployment.routing_group,weight:deployment.weight,state:t(deployment.persisted?'deploymentSaved':'deploymentDetected')})));const actions=element('div','row');
    if(deployment.persisted===false){actions.append(button(t('actUseModel'),()=>{reset();$('provider-type').value=deployment.model.split('/')[0];$('manual-models').value=deployment.model;revealEditor($('manual-models'));}),button(t('actTest'),()=>guarded(async()=>{status(t('testingConnection'));probeStatus(await api(base+'/test',{method:'POST',body:{deployment_id:deployment.deployment_id}}));})));row.append(info,actions);$('deployment-list').append(row);continue;}
    actions.append(button(t('actTest'),()=>guarded(async()=>{status(t('testingConnection'));probeStatus(await api(base+'/test',{method:'POST',body:{deployment_id:deployment.deployment_id}}));})),button(t('actEdit'),()=>{edit={...deployment,revision:settings.revision};$('edit-model').value=deployment.model;$('edit-group').value=deployment.routing_group;$('edit-weight').value=deployment.weight;$('edit-message').textContent='';$('deployment-dialog').showModal();}),button(t('actDelete'),()=>guarded(async()=>{if(!await hooks.confirmDelete(t('deleteModelTitle'),t('deleteModelText',{model:deployment.model})))return;await api(`${base}/deployments/${encodeURIComponent(deployment.deployment_id)}?expected_revision=${encodeURIComponent(settings.revision)}`,{method:'DELETE'});await saved(t('modelDeleted'));}),'quiet small delete'));
    row.append(info,actions);$('deployment-list').append(row);
  }
}
export async function loadSettings(){
  const expected=++loadGeneration;
  const [view,cardData]=await Promise.all([api(base),api(base+'/providers')]);
  if(expected!==loadGeneration)return;settings=view;providers=cardData.providers||[];if(draftRevision===null)draftRevision=view.revision;renderSettings();
}
async function saved(message){await loadSettings();draftRevision=settings.revision;await hooks.onSaved();renderSettings();status(message);$('conflict-message').hidden=true;}
function discoveryText(result){
  const parts=[discoveryStatus(result.status),t('discoveryModels',{count:discovered.length})];
  if(result.cached)parts.push(t('discoveryCached'));if(result.stale)parts.push(t('discoveryStale'));if(result.authoritative===false)parts.push(t('discoveryVerify'));if(result.detail)parts.push(serviceNote(result.detail).text);
  if(!discovered.length)parts.push(t('discoveryManual'));
  return parts.filter(Boolean).join(' · ');
}
export function initSettings(value){
  hooks=value;
  $('settings-refresh').addEventListener('click',()=>guarded(loadSettings));
  $('connection-new').addEventListener('click',()=>{reset();revealEditor($('provider-type'));});$('connection-reset').addEventListener('click',reset);
  for(const prefix of ['key','base'])$(prefix+'-mode').addEventListener('change',()=>{$('provider-'+prefix).disabled=$(prefix+'-mode').value!=='replace';if(!$('provider-'+prefix).disabled)$('provider-'+prefix).focus();});
  $('discover-models').addEventListener('click',()=>guarded(async()=>{
    const body={provider:$('provider-type').value,...credentials()};if(selected)body.provider_id=selected.provider_id;
    $('discovery-status').textContent=t('discovering');
    const result=await api(base+'/model-discovery',{method:'POST',body});discovered=result.models||[];
    $('discovery-status').textContent=discoveryText(result);if(result.detail)$('discovery-status').title=result.detail;else $('discovery-status').removeAttribute('title');
    $('discovered-models').replaceChildren();
    if(discovered.length){$('discovered-models').append(button(t('selectAll'),()=>{for(const input of $('discovered-models').querySelectorAll('input'))input.checked=true;}),button(t('clearSelection'),()=>{for(const input of $('discovered-models').querySelectorAll('input'))input.checked=false;}));}
    for(const model of discovered){const choice=element('label','model-choice'),input=element('input');input.type='checkbox';input.value=model.model;choice.append(input,element('span','',model.label||model.model));if(model.supports_vision)choice.append(element('small','tag',t('visionTag')));$('discovered-models').append(choice);}
  }));
  $('connection-form').addEventListener('submit',event=>{event.preventDefault();guarded(async()=>{
    if(!settings)throw new Error(t('loadFirst'));const models=currentModels();if(!models.length||models.length>50)throw new Error(t('modelRange'));
    const deployments=models.map(modelPayload);
    await api(base+'/deployments/batch',{method:'POST',body:{expected_revision:draftRevision,deployments}});
    reset();await saved(t('modelsAdded',{count:models.length}));
  });});
  $('save-connection').addEventListener('click',()=>guarded(async()=>{
    if(!selected)return;const fields=credentials();if(!Object.keys(fields).length)throw new Error(t('chooseCredentialMode'));
    await api(`${base}/providers/${encodeURIComponent(selected.provider_id)}`,{method:'PATCH',body:{expected_revision:draftRevision,...fields}});
    reset();await saved(t('connectionSaved'));
  }));
  $('test-draft').addEventListener('click',()=>guarded(async()=>{
    const model=currentModels()[0];if(!model)throw new Error(t('pickModel'));
    // Explicit edit loaded the saved reference, not its environment-expanded secret.
    const body={model,...credentials()};
    if(selected){if(!('api_key'in body)&&$('provider-key').value)body.api_key=$('provider-key').value;if(!('api_base'in body)&&$('provider-base').value)body.api_base=$('provider-base').value;}
    status(t('testingModel'));probeStatus(await api(base+'/test',{method:'POST',body}));
  }));
  $('edit-cancel').addEventListener('click',()=>$('deployment-dialog').close());
  $('deployment-form').addEventListener('submit',async event=>{
    event.preventDefault();const weight=Number($('edit-weight').value);if(!Number.isInteger(weight)||weight<0)return;
    try{await api(`${base}/deployments/${encodeURIComponent(edit.deployment_id)}`,{method:'PATCH',body:{expected_revision:edit.revision,model:$('edit-model').value.trim(),model_name:$('edit-group').value.trim(),weight}});$('deployment-dialog').close();await saved(t('modelUpdated'));}
    catch(error){if(conflicted(error)){await loadSettings();$('edit-message').replaceChildren(element('p','',t('editConflict')),button(t('conflictAccept'),()=>{edit.revision=settings.revision;$('edit-message').textContent=t('editKept');}));}else{$('edit-message').replaceChildren(errorText(error));if(errorDetail(error))$('edit-message').append(detailNode(errorDetail(error)));}}
  });
}
