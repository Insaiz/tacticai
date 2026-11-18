from __future__ import annotations
import os,json,uuid,shutil,subprocess,math
from pathlib import Path
from typing import Any,Dict
from fastapi import FastAPI,UploadFile,File,Form,HTTPException,Request
from fastapi.middleware.cors import CORSMiddleware
from football_engine import build_analysis

BASE=Path(os.getenv('TACTICAI_DATA','/data'));BASE.mkdir(parents=True,exist_ok=True)
TOKEN=os.getenv('TACTICAI_WORKER_TOKEN','')
app=FastAPI(title='TacticAI Worker',version='1.8')
app.add_middleware(CORSMiddleware,allow_origins=os.getenv('TACTICAI_CORS_ORIGINS','*').split(','),allow_methods=['*'],allow_headers=['*'])

def auth(request:Request):
    if TOKEN and request.headers.get('x-tacticai-token')!=TOKEN: raise HTTPException(401,'Invalid worker token')
def job_dir(j): return BASE/j
def readj(j): 
    p=job_dir(j)/'job.json'
    return json.loads(p.read_text()) if p.exists() else None
def writej(j,x): (job_dir(j)/'job.json').write_text(json.dumps(x,ensure_ascii=False,indent=2))

@app.get('/health')
def health(): return {'ok':True,'service':'tacticai-worker','version':'1.8'}

@app.post('/projects')
def create_project(payload:Dict[str,Any]):
    j=str(uuid.uuid4());job_dir(j).mkdir(parents=True,exist_ok=True)
    ctx={'manual_data':payload.get('data',payload),'problems':payload.get('problems',[]),'chat':[],'clips':[],'evidence_ledger':[]}
    writej(j,{'id':j,'status':'created','project_id':j,'context':ctx})
    (job_dir(j)/'project_context.json').write_text(json.dumps(ctx,ensure_ascii=False,indent=2))
    return {'project':{'id':j},'context':ctx}

@app.get('/projects/{j}/context')
def get_context(j:str,request:Request):
    auth(request);x=readj(j)
    if not x: raise HTTPException(404,'Project not found')
    return x.get('context',{})

@app.put('/projects/{j}/context')
def update_context(j:str,payload:Dict[str,Any],request:Request):
    auth(request);x=readj(j)
    if not x: raise HTTPException(404,'Project not found')
    c=x.get('context',{});c.update(payload);x['context']=c;writej(j,x);(job_dir(j)/'project_context.json').write_text(json.dumps(c,ensure_ascii=False,indent=2));return {'context':c}

@app.post('/jobs/upload')
async def upload(request:Request,file:UploadFile=File(...),project_id:str=Form(''),team:str=Form(''),opponent:str=Form(''),sample_fps:float=Form(2)):
    j=project_id or str(uuid.uuid4())
    try:
        auth(request)
        job_dir(j).mkdir(parents=True,exist_ok=True)
        ext=Path(file.filename or 'video.mp4').suffix or '.mp4'
        src=job_dir(j)/('source'+ext)
        try:
            with src.open('wb') as out:
                shutil.copyfileobj(file.file,out)
        except Exception as e:
            msg=f'file_save_error: {type(e).__name__}: {e}'
            writej(j,{'id':j,'project_id':j,'status':'error','message':msg})
            raise HTTPException(500,msg)
        old=readj(j) or {'id':j,'project_id':j,'context':{'manual_data':{},'problems':[],'chat':[],'clips':[],'evidence_ledger':[]}}
        old.update({'status':'queued','message':'Vídeo recibido','progress':0,'source':src.name,'sample_fps':sample_fps,'team':team,'opponent':opponent})
        old['context']['video']={'source':src.name,'team':team,'opponent':opponent}
        writej(j,old)
        import asyncio
        try:
            asyncio.create_task(process(j,src,sample_fps,team,opponent))
        except Exception as e:
            msg=f'queue_error: {type(e).__name__}: {e}'
            old['status']='error';old['message']=msg;writej(j,old)
            raise HTTPException(500,msg)
        return {'id':j,'project_id':j,'status':'queued'}
    except HTTPException:
        raise
    except Exception as e:
        msg=f'upload_error: {type(e).__name__}: {e}'
        try:
            x=readj(j) or {'id':j,'project_id':j}
            x['status']='error';x['message']=msg;writej(j,x)
        except Exception:
            pass
        raise HTTPException(500,msg)

async def process(j,src,fps,team,opp):
    x=readj(j);x['status']='processing';x['progress']=5;writej(j,x)
    try:
        result=build_analysis(str(src),fps,team,opp)
        x=readj(j);x['status']='completed';x['progress']=100;x['message']='Análisis completado';x['result']=result
        c=x.get('context',{});c['video']=result;c['clips']=result.get('clips',[]);c['evidence_ledger']=result.get('evidence_ledger',[]);x['context']=c
        writej(j,x);(job_dir(j)/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
        (job_dir(j)/'project_context.json').write_text(json.dumps(c,ensure_ascii=False,indent=2))
    except Exception as e:
        x=readj(j);x['status']='error';x['message']=str(e);writej(j,x)

@app.get('/jobs/{j}')
def job(j:str,request:Request):
    auth(request);x=readj(j)
    if not x: raise HTTPException(404,'Job not found')
    r={k:x.get(k) for k in ['id','project_id','status','message','progress','team','opponent']};r['metrics']=x.get('result',{}).get('metrics',{});r['duration']=x.get('result',{}).get('duration',0);return r

@app.get('/jobs/{j}/context')
def job_context(j:str,request:Request):
    auth(request);x=readj(j)
    if not x: raise HTTPException(404,'Job not found')
    return x.get('context',x.get('result',{}))

@app.get('/jobs/{j}/clips')
def clips(j:str,request:Request):
    auth(request);x=readj(j)
    if not x: raise HTTPException(404,'Job not found')
    return {'clips':x.get('context',{}).get('clips',[])}

@app.post('/jobs/{j}/query')
def query(j:str,payload:Dict[str,Any],request:Request):
    auth(request);x=readj(j)
    if not x: raise HTTPException(404,'Job not found')
    q=str(payload.get('query','')).lower();events=x.get('result',{}).get('events',[])
    matches=[e for e in events if any(k in str(e).lower() for k in q.split() if len(k)>3)][:30]
    clips=[{'id':str(uuid.uuid4()),'title':e.get('type','candidate'),'type':e.get('type','candidate'),'timestamp':e.get('timestamp',0),'start':max(0,e.get('start',0)-payload.get('pre_seconds',5)),'end':e.get('end',e.get('timestamp',0)+payload.get('post_seconds',10)),'status':e.get('status','candidate'),'reason':e.get('reason','Evidencia temporal candidata')} for e in matches]
    c=x.get('context',{});c['clips']=c.get('clips',[])+clips;c['evidence_ledger']=c.get('evidence_ledger',[])+clips;x['context']=c;writej(j,x);return {'clips':clips,'evidence':clips,'query':payload.get('query')}

@app.post('/projects/{j}/calibration')
def calibration(j:str,payload:Dict[str,Any],request:Request):
    auth(request);x=readj(j)
    if not x: raise HTTPException(404,'Project not found')
    c=x.get('context',{});c['pitch_calibration']=payload.get('calibration');x['context']=c;writej(j,x);return {'context':c}

@app.get('/library')
def library(request:Request):
    auth(request);items=[]
    for p in BASE.iterdir():
        if (p/'job.json').exists():
            x=readj(p.name);r=x.get('result',{});items.append({'id':p.name,'status':x.get('status'),'team':x.get('team'),'opponent':x.get('opponent'),'duration_seconds':r.get('duration',0),'clips':len(x.get('context',{}).get('clips',[])),'has_video':bool(x.get('source'))})
    return {'items':items}

@app.get('/jobs/{j}/artifacts')
def artifacts(j:str,request:Request):
    auth(request);p=job_dir(j)
    if not p.exists(): raise HTTPException(404,'Job not found')
    return {'artifacts':[{'name':f.name,'size_bytes':f.stat().st_size,'download_url':'/jobs/'+j+'/artifacts/'+f.name} for f in p.iterdir() if f.is_file()]}

@app.get('/jobs/{j}/artifacts/{name}')
def artifact(j:str,name:str,request:Request):
    auth(request);p=job_dir(j)/name
    if not p.exists() or not p.is_file(): raise HTTPException(404,'Artifact not found')
    from fastapi.responses import FileResponse;return FileResponse(p)

@app.post('/jobs/{j}/archive')
def archive(j:str,request:Request):
    auth(request);p=job_dir(j)
    if not p.exists(): raise HTTPException(404,'Job not found')
    out=shutil.make_archive(str(BASE/(j+'_project')),'zip',root_dir=p);return {'file':'/jobs/'+j+'/artifacts/'+Path(out).name}
