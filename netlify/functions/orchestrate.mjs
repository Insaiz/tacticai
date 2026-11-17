const cors={'Access-Control-Allow-Origin':'*','Access-Control-Allow-Headers':'Content-Type','Access-Control-Allow-Methods':'POST,OPTIONS'};
const json=(body,status=200)=>({statusCode:status,headers:{...cors,'Content-Type':'application/json'},body:JSON.stringify(body)});
function makePlan(request,context){
 const q=request.toLowerCase();let intent='general_analysis',ops=['summarize_match_context'];
 if(/pérdid|perdid|salida/.test(q)){intent='build_up_turnovers';ops=['find_turnovers_in_build_up'];}
 else if(/presión|pressing/.test(q)){intent='pressing';ops=['find_pressing_sequences'];}
 else if(/transición|contra/.test(q)){intent='transition';ops=['find_transition_sequences'];}
 else if(/tiro|ocasión|finaliz/.test(q)){intent='chance_creation';ops=['find_shots_and_chances'];}
 else if(/lateral|extremo|mediocentro|central|delantero|jugador/.test(q)){intent='role_diagnosis';ops=['inspect_role_sequences'];}
 const video=/vídeo|video|clip|secuencia|localiza|ocurr|pérdid|presión|transición|tiro|ocasión|lateral|extremo|jugador/.test(q);
 return{request,intent,required_sources:video?['manual_data','video_events','clips']:['manual_data','diagnosis'],operations:ops,evidence_requirements:video?['timestamped_video_evidence','statistical_context']:['statistical_context'],execution:{needs_video:video,video_query:video?request:null}};
}
function localResponse(plan,c){
 const d=c.match||{};const p=(c.problems||[])[0];let answer='He revisado el contexto disponible. ';
 if(p)answer+='La prioridad actual es '+p.title+'. Antes de tratarla como causa confirmada, conviene contrastarla con evidencia temporal del vídeo.';
 else answer+='Todavía no hay un diagnóstico suficiente. Puedo cruzar los datos manuales con las secuencias del vídeo.';
 return{answer,diagnosis:p?{title:p.title,status:'hypothesis',evidence:[p.evidence],cause:p.cause||null,confidence:'medium'}:null,solution:p?.solution||'Buscar primero el patrón que explique el problema.',training_task:null,evidence_to_check:plan.evidence_requirements,follow_up:plan.execution.needs_video?'Puedo localizar las secuencias relevantes del vídeo.':'Añade vídeo si quieres validar el patrón temporal.',analysis_plan:{...plan,status:'planned'}};
}
function compact(c){return{match:c.match||null,problems:(c.problems||[]).slice(0,20),video:c.video||null,clips:(c.clips||[]).slice(0,30),evidence_ledger:(c.evidence_ledger||[]).slice(0,50),mission:c.mission||null};}
async function llm(key,model,instructions,payload){
 const r=await fetch('https://api.openai.com/v1/responses',{method:'POST',headers:{Authorization:'Bearer '+key,'Content-Type':'application/json'},body:JSON.stringify({model,instructions,input:[{role:'user',content:JSON.stringify(payload)}],temperature:0.15})});
 const j=await r.json();if(!r.ok)throw new Error(j.error?.message||('OpenAI '+r.status));return j.output_text||'';
}
function parse(t){try{return JSON.parse(t)}catch(e){const m=String(t||'').match(/\{[\s\S]*\}/);if(m)try{return JSON.parse(m[0])}catch(_){}return null}}
exports.handler=async(event)=>{
 if(event.httpMethod==='OPTIONS')return{statusCode:204,headers:cors,body:''};
 if(event.httpMethod!=='POST')return json({error:'Método no permitido'},405);
 let b;try{b=JSON.parse(event.body||'{}')}catch(e){return json({error:'JSON inválido'},400)}
 const request=String(b.request||'').trim();if(!request)return json({error:'Falta la petición del entrenador'},400);
 const context=compact(b.context||{}),plan=makePlan(request,context),key=process.env.OPENAI_API_KEY;
 if(!key)return json({mode:'local',plan,response:localResponse(plan,context)});
 const instructions='Eres TacticAI, asistente táctico para entrenadores. Usa SOLO el contexto. Distingue HECHO, EVIDENCIA, HIPÓTESIS y RECOMENDACIÓN. Nunca inventes eventos, jugadores, métricas ni clips. Los eventos candidate no son hechos confirmados. Si falta evidencia, dilo. Devuelve JSON con plan, answer, diagnosis, solution, training_task, evidence_to_check, follow_up y analysis_plan.';
 try{const raw=await llm(key,process.env.TACTICAI_OPENAI_MODEL||'gpt-4.1-mini',instructions,{request,context,plan});const parsed=parse(raw);if(parsed)return json({mode:'llm',plan:parsed.plan||plan,response:parsed});return json({mode:'llm',plan,response:{answer:raw}})}catch(e){return json({mode:'local',plan,response:localResponse(plan,context),provider_error:e.message})}
};