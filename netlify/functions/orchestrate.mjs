const cors={'Access-Control-Allow-Origin':'*','Access-Control-Allow-Headers':'Content-Type','Access-Control-Allow-Methods':'POST,OPTIONS'};
const json=(body,status=200)=>({statusCode:status,headers:{...cors,'Content-Type':'application/json'},body:JSON.stringify(body)});

function makePlan(request,context){
 const q=request.toLowerCase();
 let intent='general_analysis',ops=['summarize_match_context'];
 if(/pérdid|perdid|salida/.test(q)){intent='build_up_turnovers';ops=['find_turnovers_in_build_up'];}
 else if(/presión|pressing/.test(q)){intent='pressing';ops=['find_pressing_sequences'];}
 else if(/transición|contra/.test(q)){intent='transition';ops=['find_transition_sequences'];}
 else if(/tiro|ocasión|finaliz/.test(q)){intent='chance_creation';ops=['find_shots_and_chances'];}
 else if(/lateral|extremo|mediocentro|central|delantero|jugador/.test(q)){intent='role_diagnosis';ops=['inspect_role_sequences'];}
 const video=/vídeo|video|clip|secuencia|localiza|ocurr|pérdid|presión|transición|tiro|ocasión|lateral|extremo|jugador/.test(q);
 return{request,intent,required_sources:video?['manual_data','video_events','clips']:['manual_data','diagnosis'],operations:ops,evidence_requirements:video?['timestamped_video_evidence','statistical_context']:['statistical_context'],execution:{needs_video:video,video_query:video?request:null}};
}

function videoEvidence(c){
 const v=c.video||{},s=v.sequence||{};
 return {
   possession_episodes:(s.possession_episodes||[]).slice(0,80),
   turnover_candidates:(s.turnover_candidates||[]).slice(0,80),
   structural_context:(s.turnover_candidates||[]).map(x=>({timestamp:x.timestamp,clip_id:x.clip_id,context:x.structural_context})).filter(x=>x.context).slice(0,80),
   transition_candidates:(s.transition_candidates||[]).slice(0,80),
   defensive_transition_candidates:(s.defensive_transition_candidates||[]).slice(0,80),
   visual_facts:(s.visual_facts||[]).slice(0,80),
   clips:(c.clips||v.clips||[]).slice(0,40)
 };
}

function tacticalSynthesis(plan,c){
 const v=videoEvidence(c),m=c.match||c.manual_data||{};
 const turnovers=v.turnover_candidates||[], transitions=v.transition_candidates||[], clips=v.clips||[];
 const facts=[],evidence=[],hypotheses=[];
 if(m.score)facts.push({type:'fact',label:'Resultado',value:String(m.score),source:'manual_data'});
 if(m.formation)facts.push({type:'fact',label:'Estructura inicial',value:String(m.formation),source:'manual_data'});
 if(Number.isFinite(Number(m.losses)))facts.push({type:'fact',label:'Pérdidas registradas',value:Number(m.losses),source:'manual_data'});
 if(turnovers.length){
   evidence.push({type:'evidence',label:'Cambios de posesión candidatos',value:turnovers.length,source:'video.sequence.turnover_candidates',clip_refs:turnovers.map(x=>x.clip_id||x.id).filter(Boolean).slice(0,10)});
   const byZone={};for(const t of turnovers){const z=t.to_zone||t.from_zone||'unknown';byZone[z]=(byZone[z]||0)+1;}
   const top=Object.entries(byZone).sort((a,b)=>b[1]-a[1])[0];
   if(top)hypotheses.push({type:'hypothesis',label:'Concentración de cambios de posesión',value:top[0],count:top[1],confidence:'medium'});
   const geom=turnovers.map(t=>t.post_turnover_geometry).filter(Boolean);
   const structural=turnovers.map(t=>t.structural_context).filter(Boolean);
   if(structural.length){
     const support=structural.map(x=>x.nearest_support_distance_px).filter(Number.isFinite);
     const weak=structural.map(x=>x.weak_side_candidate).filter(Boolean);
     const avg=a=>a.length?Math.round(a.reduce((s,x)=>s+x,0)/a.length):null;
     evidence.push({type:'evidence',label:'Estructura espacial tras pérdida',value:{
       samples:structural.length,
       avg_nearest_support_distance_px:avg(support),
       weak_side_candidates:[...new Set(weak)]
     },source:'video.sequence.turnover_candidates.structural_context',clip_refs:turnovers.map(x=>x.clip_id).filter(Boolean).slice(0,10)});
     if(support.length&&avg(support)>220)
       hypotheses.push({type:'hypothesis',label:'Posible distancia excesiva de apoyo',value:'el jugador del equipo que pierde más cercano al balón aparece relativamente alejado en la ventana analizada',confidence:'medium'});
     if(weak.length)
       hypotheses.push({type:'hypothesis',label:'Posible lado débil expuesto',value:'la distribución espacial presenta una asimetría lateral candidata hacia '+[...new Set(weak)].join(', '),confidence:'low-medium'});
   }
   if(geom.length){
     const nearOwn=geom.filter(g=>g.nearest_losing_team_px!=null).map(g=>g.nearest_losing_team_px);
     const nearOpp=geom.filter(g=>g.nearest_recovering_team_px!=null).map(g=>g.nearest_recovering_team_px);
     const avg=a=>a.length?Math.round(a.reduce((s,x)=>s+x,0)/a.length):null;
     evidence.push({type:'evidence',label:'Geometría tras cambio de posesión',value:{samples:geom.length,avg_losing_team_distance_px:avg(nearOwn),avg_recovering_team_distance_px:avg(nearOpp)},source:'video.sequence.turnover_candidates.post_turnover_geometry',clip_refs:turnovers.map(x=>x.clip_id).filter(Boolean).slice(0,10)});
     if(nearOwn.length&&nearOpp.length&&avg(nearOwn)>avg(nearOpp)+35)
       hypotheses.push({type:'hypothesis',label:'Posible reacción defensiva tardía',value:'el equipo que pierde queda más alejado del balón que el equipo que lo recupera en la ventana posterior',confidence:'medium'});
   }
 }
 if(transitions.length)evidence.push({type:'evidence',label:'Transiciones candidatas',value:transitions.length,source:'video.sequence.transition_candidates',clip_refs:transitions.map(x=>x.clip_id||x.id).filter(Boolean).slice(0,10)});
 const askedRole=/lateral|extremo|mediocentro|central|delantero|jugador/.test(plan.request.toLowerCase());
 let diagnosis=null,solution=null,task=null;
 if(turnovers.length && transitions.length){
   const late= hypotheses.some(h=>h.label==='Posible reacción defensiva tardía');
   const supportGap= hypotheses.some(h=>h.label==='Posible distancia excesiva de apoyo');
   const weakSide= hypotheses.some(h=>h.label==='Posible lado débil expuesto');
   diagnosis={title:late?'Patrón candidato de pérdida + transición con posible reacción tardía':(supportGap||weakSide?'Patrón candidato de pérdida con posible problema estructural':'Existe un patrón candidato de pérdida seguido de transición'),status:'hypothesis',confidence:late?'medium':'medium',facts,evidence,hypotheses};
   solution=late
     ?'La evidencia espacial apunta a comprobar primero la reacción colectiva tras pérdida: quién salta al balón, quién protege dentro y qué distancia queda entre el jugador que pierde y el balón. No permite culpar a un lateral concreto.'
     :(supportGap||weakSide
       ?'Prioriza comprobar la estructura colectiva: distancia del apoyo más cercano, protección del carril central y equilibrio del lado débil. La geometría es una hipótesis en espacio de cámara, no una posición táctica confirmada.'
       :'Prioriza verificar qué ocurre en los 3–5 segundos posteriores a las pérdidas: distancia al balón, ocupación de carriles, protección del espacio central y velocidad de reacción.');
   task={title:'Pérdida + transición defensiva',objective:'mejorar la reacción tras pérdida',format:'oleada 5v4+comodines o juego de posición con transición inmediata',coaching_points:['reacción del jugador más cercano','cerrar línea de pase interior','proteger profundidad antes de saltar','reorganización del lado débil'],evidence_basis:'video.sequence.turnover_candidates + post_turnover_geometry'};
 } else if(turnovers.length){
   diagnosis={title:'Hay pérdidas candidatas detectadas, pero no basta para atribuir la causa',status:'hypothesis',confidence:'low-medium',facts,evidence,hypotheses};
   solution='Revisar las pérdidas por zona y la secuencia inmediatamente anterior y posterior antes de señalar a un jugador concreto.';
   task={title:'Pérdida y reacción',objective:'mejorar continuidad tras pérdida',format:'posesión con transición de 5–8 segundos',coaching_points:['distancia entre líneas','jugador más cercano al balón','cierre interior']};
 } else if(askedRole){
   diagnosis={title:'No hay evidencia de vídeo suficiente para confirmar un problema individual',status:'insufficient_evidence',confidence:'low',facts,evidence,hypotheses};
   solution='No atribuir el problema al jugador todavía. Comparar sus acciones con la estructura colectiva y localizar secuencias concretas.';
   task=null;
 } else {
   diagnosis={title:'Diagnóstico pendiente de evidencia temporal',status:'insufficient_evidence',confidence:'low',facts,evidence,hypotheses};
   solution='Cruzar los datos del partido con secuencias temporales antes de convertir una observación en diagnóstico.';
   task=null;
 }
 return {answer:diagnosis.title+'. '+solution,diagnosis,solution,training_task:task,evidence_to_check:plan.evidence_requirements,follow_up:'Usa las secuencias candidatas como punto de partida; los eventos de vídeo actuales no son hechos confirmados.',analysis_plan:{...plan,status:'completed',evidence_summary:{turnover_candidates:turnovers.length,transition_candidates:transitions.length,clips:clips.length}},facts,evidence,hypotheses,clip_refs:[...new Set(clips.map(x=>x.id).filter(Boolean))].slice(0,20)};
}

function localResponse(plan,c){return tacticalSynthesis(plan,c)}
function compact(c){
 const x={match:c.match||c.manual_data||null,problems:(c.problems||[]).slice(0,20),video:c.video||null,clips:(c.clips||[]).slice(0,40),evidence_ledger:(c.evidence_ledger||[]).slice(0,50),mission:c.mission||null};
 return x;
}
async function llm(key,model,instructions,payload){
 const r=await fetch('https://api.openai.com/v1/responses',{method:'POST',headers:{Authorization:'Bearer '+key,'Content-Type':'application/json'},body:JSON.stringify({model,instructions,input:[{role:'user',content:JSON.stringify(payload)}],temperature:0.15})});
 const j=await r.json();if(!r.ok)throw new Error(j.error?.message||('OpenAI '+r.status));return j.output_text||'';
}
function parse(t){try{return JSON.parse(t)}catch(e){const m=String(t||'').match(/\{[\s\S]*\}/);if(m)try{return JSON.parse(m[0])}catch(_){}return null}}

export const handler=async(event)=>{
 if(event.httpMethod==='OPTIONS')return{statusCode:204,headers:cors,body:''};
 if(event.httpMethod!=='POST')return json({error:'Método no permitido'},405);
 let b;try{b=JSON.parse(event.body||'{}')}catch(e){return json({error:'JSON inválido'},400)}
 const request=String(b.request||'').trim();if(!request)return json({error:'Falta la petición del entrenador'},400);
 const context=compact(b.context||{}),plan=makePlan(request,context),local=tacticalSynthesis(plan,context),key=process.env.OPENAI_API_KEY;
 if(!key)return json({mode:'local',plan,response:local});
 const instructions='Eres TacticAI, asistente táctico para entrenadores. Usa SOLO el contexto proporcionado. Distingue HECHO, EVIDENCIA, HIPÓTESIS y RECOMENDACIÓN. Nunca inventes eventos, jugadores, métricas ni clips. Los eventos candidate son candidatos, no hechos confirmados. Si la evidencia no permite confirmar una causa, dilo. Devuelve JSON válido con: answer, diagnosis{title,status,confidence,facts,evidence,hypotheses}, solution, training_task, evidence_to_check, follow_up, analysis_plan, facts, evidence, hypotheses, clip_refs. Cada clip_ref debe existir en el contexto. No conviertas geometría, proximidad o cambio de equipo en un pase, tiro, pérdida o acción técnica confirmada.';
 try{
   const raw=await llm(key,process.env.TACTICAI_OPENAI_MODEL||'gpt-4.1-mini',instructions,{request,context,plan,local_baseline:local});
   const parsed=parse(raw);
   if(parsed)return json({mode:'llm',plan:parsed.plan||plan,response:{...local,...parsed,plan:parsed.plan||plan}});
   return json({mode:'llm',plan,response:{...local,answer:raw}});
 }catch(e){return json({mode:'local',plan,response:{...local,provider_error:e.message}})}
};