const cors={'Access-Control-Allow-Origin':'*','Access-Control-Allow-Headers':'Content-Type','Access-Control-Allow-Methods':'GET,POST,OPTIONS'};
const out=(body,status=200)=>({statusCode:status,headers:{...cors,'Content-Type':'application/json'},body:JSON.stringify(body)});
export const handler=async(event)=>{
 if(event.httpMethod==='OPTIONS')return{statusCode:204,headers:cors,body:''};
 const base=process.env.TACTICAI_WORKER_URL;
 if(!base)return out({error:'TACTICAI_WORKER_URL no configurado'},500);
 try{
   let path=event.path.replace(/^.*\/video/,'')||'/';
   if(path==='/')path='';
   const url=base+path+(event.rawQuery?'?'+event.rawQuery:'');
   const incomingHeaders=event.headers||{};
   const contentType=incomingHeaders['content-type']||incomingHeaders['Content-Type'];
   const headers={};
   if(contentType)headers['Content-Type']=contentType;
   if(process.env.TACTICAI_WORKER_TOKEN)headers['x-tacticai-token']=process.env.TACTICAI_WORKER_TOKEN;
   const body=event.body ? Buffer.from(event.body,event.isBase64Encoded?'base64':'utf8') : undefined;
   if(body)headers['Content-Length']=String(body.length);
   const r=await fetch(url,{method:event.httpMethod,headers,body});
   const responseText=await r.text();
   return {
     statusCode:r.status,
     headers:{...cors,'Content-Type':r.headers.get('content-type')||'application/json'},
     body:responseText||JSON.stringify({error:'Worker returned an empty response',status:r.status})
   };
 }catch(e){return out({error:e.message},502)}
};