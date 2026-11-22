from __future__ import annotations
import os,json,uuid,shutil,subprocess,math
import cv2
from pathlib import Path
from typing import Any,Dict
from fastapi import FastAPI,UploadFile,File,Form,HTTPException,Request
from fastapi.middleware.cors import CORSMiddleware
from football_engine import build_analysis

BASE=Path(os.getenv('TACTICAI_DATA','/data'));BASE.mkdir(parents=True,exist_ok=True)
# Persist source video per job so evidence clips can be rendered after analysis.
# Completed jobs keep their source until the project is archived/deleted.
import tempfile
TOKEN=os.getenv('TACTICAI_WORKER_TOKEN','')
app=FastAPI(title='TacticAI Worker',version='1.8')
app.add_middleware(CORSMiddleware,allow_origins=os.getenv('TACTICAI_CORS_ORIGINS','*').split(','),allow_methods=['*'],allow_headers=['*'])

def auth(request:Request):
    if TOKEN and request.headers.get('x-tacticai-token')!=TOKEN: raise HTTPException(401,'Invalid worker token')
def job_dir(j): return BASE/j
def readj(j):
    p=job_dir(j)/'job.json'
    if not p.exists(): return None
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        # A concurrent status poll can briefly observe a file while it is being replaced.
        return None

def writej(j,x):
    p=job_dir(j)/'job.json'
    tmp=p.with_name('job.json.tmp')
    tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
    tmp.replace(p)
def persisted_source(j):
    x=readj(j) or {}
    name=Path(str(x.get('source',''))).name
    if not name:return None
    p=job_dir(j)/name
    return p if p.exists() else None

def render_evidence_clips(j:str,src:Path,result:dict,max_clips:int=3):
    clips_dir=job_dir(j)/'clips';clips_dir.mkdir(parents=True,exist_ok=True)
    rendered=[]
    windows=result.get('sequence',{}).get('windows',[])
    by_ts=sorted(windows,key=lambda x:float(x.get('timestamp',0)))
    overlay=os.getenv('TACTICAI_RENDER_OVERLAY','0')!='0'

    def nearest_window(ts):
        if not by_ts:return None
        return min(by_ts,key=lambda x:abs(float(x.get('timestamp',0))-ts))

    clips_to_render=result.get('clips',[])[:max_clips]
    total_render=len(clips_to_render)
    for idx,clip in enumerate(clips_to_render,1):
        try:
            x=readj(j) or {}
            x['progress']=92+round(7*idx/max(1,total_render))
            x['message']=f'Generando clip {idx}/{total_render}'
            writej(j,x)
        except Exception:
            pass
        print(f'TacticAI render: clip {idx}/{total_render}',flush=True)
        start=max(0.0,float(clip.get('start',0)))
        end=max(start+0.5,float(clip.get('end',start+1)))
        out=clips_dir/f"clip-{idx}.mp4"
        duration=min(end-start,8.0)
        ok=False
        if overlay:
            cap=cv2.VideoCapture(str(src))
            fps=cap.get(cv2.CAP_PROP_FPS) or 25.0
            w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            if w>0 and h>0:
                tmp=clips_dir/f".overlay-{idx}.mp4"
                writer=cv2.VideoWriter(str(tmp),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
                cap.set(cv2.CAP_PROP_POS_MSEC,start*1000)
                frames=int(max(1,duration*fps))
                for _ in range(frames):
                    ret,frame=cap.read()
                    if not ret:break
                    ts=cap.get(cv2.CAP_PROP_POS_MSEC)/1000.0
                    win=nearest_window(ts)
                    if win:
                        for p in win.get('detections',[]):
                            x,y,ww,hh=[int(float(p.get(k,0))) for k in ('x','y','w','h')]
                            cls=p.get('class_id')
                            if cls==1:
                                team=p.get('team','unknown')
                                if team=='team_a': c=(40,180,255)
                                elif team=='team_b': c=(255,160,40)
                                else: c=(180,180,180)
                                cv2.rectangle(frame,(x-ww//2,y-hh//2),(x+ww//2,y+hh//2),c,2)
                                cv2.putText(frame,f"P{p.get('id','?')} {team}",(x-ww//2,max(15,y-hh//2-4)),cv2.FONT_HERSHEY_SIMPLEX,0.42,c,1,cv2.LINE_AA)
                            elif cls==0:
                                cv2.circle(frame,(x,y),max(5,int(min(w,h)*0.009)),(255,255,255),-1)
                                cv2.circle(frame,(x,y),max(7,int(min(w,h)*0.012)),(30,30,30),2)
                        evs=[e for e in result.get('events',[]) if e.get('clip_id')==clip.get('id')]
                        ev=evs[0] if evs else None
                        sc=ev.get('structural_context',{}) if ev else {}
                        cv2.rectangle(frame,(10,10),(min(w-10,620),130), (20,20,20), -1)
                        cv2.putText(frame,f"TacticAI · {clip.get('type','candidate')}",(24,36),cv2.FONT_HERSHEY_SIMPLEX,0.62,(255,255,255),2,cv2.LINE_AA)
                        cv2.putText(frame,f"t={ts:.1f}s  status={clip.get('status','candidate')}",(24,62),cv2.FONT_HERSHEY_SIMPLEX,0.48,(220,220,220),1,cv2.LINE_AA)
                        if sc:
                            txt=f"support={sc.get('nearest_support_distance_px','—')}px  central={sc.get('central_corridor_count_avg','—')}"
                            cv2.putText(frame,txt,(24,88),cv2.FONT_HERSHEY_SIMPLEX,0.45,(220,220,220),1,cv2.LINE_AA)
                            cv2.putText(frame,f"weak_side={sc.get('weak_side_candidate') or 'none'}  camera-space",(24,112),cv2.FONT_HERSHEY_SIMPLEX,0.45,(220,220,220),1,cv2.LINE_AA)
                    writer.write(frame)
                writer.release();cap.release()
                print(f'TacticAI render: overlay frames finished for clip {idx}/{total_render}',flush=True)
                if tmp.exists() and tmp.stat().st_size>0:
                    ff=['ffmpeg','-y','-hide_banner','-loglevel','error','-i',str(tmp),'-c:v','libx264','-preset','veryfast','-crf','23','-movflags','+faststart',str(out)]
                    p=subprocess.run(ff,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=60)
                    try:tmp.unlink()
                    except Exception:pass
                    ok=p.returncode==0 and out.exists() and out.stat().st_size>0
                print(f'TacticAI render: ffmpeg clip {idx}/{total_render} rc={p.returncode}',flush=True)
        if not ok:
            cmd=['ffmpeg','-y','-hide_banner','-loglevel','error','-ss',f'{start:.3f}','-i',str(src),
                 '-t',f'{duration:.3f}','-map','0:v:0?','-map','0:a:0?','-c:v','libx264','-preset','veryfast',
                 '-crf','23','-c:a','aac','-movflags','+faststart',str(out)]
            try:
                p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=60)
                ok=p.returncode==0 and out.exists() and out.stat().st_size>0
            except Exception:
                ok=False
        if ok:
            x=dict(clip)
            x['file']=f'clips/{out.name}'
            x['download_url']=f'/jobs/{j}/artifacts/clips/{out.name}'
            x['playable']=True
            x['visualization']='TacticAI overlay' if overlay else 'raw clip'
            rendered.append(x)
    print(f'TacticAI render: completed {len(rendered)}/{total_render} clips',flush=True)
    return rendered

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


@app.post('/jobs/upload/init')
def upload_init(request:Request, project_id:str=Form(''), team:str=Form(''), opponent:str=Form(''), sample_fps:float=Form(2), filename:str=Form('video.mp4'), total_chunks:int=Form(...)):
    auth(request)
    j=project_id or str(uuid.uuid4())
    job_dir(j).mkdir(parents=True,exist_ok=True)
    ext=Path(filename or 'video.mp4').suffix or '.mp4'
    src=Path(tempfile.gettempdir())/('tacticai-'+j+ext)
    try:
        if src.exists(): src.unlink()
    except Exception: pass
    old=readj(j) or {'id':j,'project_id':j,'context':{'manual_data':{},'problems':[],'chat':[],'clips':[],'evidence_ledger':[]}}
    old.update({'status':'uploading','message':'Preparando subida por bloques','progress':0,'source':src.name,'sample_fps':sample_fps,'team':team,'opponent':opponent,'upload_total_chunks':total_chunks,'upload_received_chunks':0})
    old['context']['video']={'source':src.name,'team':team,'opponent':opponent}
    writej(j,old)
    src.touch()
    return {'id':j,'project_id':j,'status':'uploading','total_chunks':total_chunks}

@app.post('/jobs/{j}/upload/chunk')
async def upload_chunk(j:str,request:Request,chunk:UploadFile=File(...),index:int=Form(...),total_chunks:int=Form(...)):
    auth(request)
    x=readj(j)
    if not x: raise HTTPException(404,'Job not found')
    src=Path(tempfile.gettempdir())/x.get('source','')
    if not src.name: raise HTTPException(400,'Upload source not initialized')
    if total_chunks != int(x.get('upload_total_chunks',total_chunks)): raise HTTPException(400,'Chunk total mismatch')
    try:
        data=await chunk.read()
        mode='ab' if index>0 else 'wb'
        with src.open(mode) as out: out.write(data)
        x['upload_received_chunks']=max(int(x.get('upload_received_chunks',0)),index+1)
        x['progress']=min(95,round(95*x['upload_received_chunks']/total_chunks))
        x['message']=f'Bloque {index+1}/{total_chunks} recibido'
        writej(j,x)
        return {'ok':True,'id':j,'chunk':index,'received_chunks':x['upload_received_chunks'],'total_chunks':total_chunks}
    except Exception as e:
        raise HTTPException(500,f'chunk_save_error: {type(e).__name__}: {e}')

@app.post('/jobs/{j}/upload/finalize')
async def upload_finalize(j:str,request:Request):
    auth(request)
    x=readj(j)
    if not x: raise HTTPException(404,'Job not found')
    total=int(x.get('upload_total_chunks',0)); received=int(x.get('upload_received_chunks',0))
    if total<=0 or received<total: raise HTTPException(409,f'Upload incomplete: {received}/{total} chunks')
    src=Path(tempfile.gettempdir())/x.get('source','')
    if not src.exists() or src.stat().st_size<=0: raise HTTPException(400,'Uploaded video is empty or missing')
    x['status']='queued';x['message']='Vídeo recibido';x['progress']=0
    writej(j,x)
    import asyncio
    try:
        asyncio.create_task(process(j,src,float(x.get('sample_fps',2)),x.get('team',''),x.get('opponent','')))
    except Exception as e:
        x['status']='error';x['message']=f'queue_error: {type(e).__name__}: {e}';writej(j,x);raise HTTPException(500,x['message'])
    return {'id':j,'project_id':j,'status':'queued'}

@app.post('/jobs/upload')
async def upload(request:Request,file:UploadFile=File(...),project_id:str=Form(''),team:str=Form(''),opponent:str=Form(''),sample_fps:float=Form(2)):
    j=project_id or str(uuid.uuid4())
    try:
        auth(request)
        job_dir(j).mkdir(parents=True,exist_ok=True)
        ext=Path(file.filename or 'video.mp4').suffix or '.mp4'
        src=Path(tempfile.gettempdir())/('tacticai-'+j+ext)
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
    def report_analysis_progress(percent, sampled, total):
        try:
            y=readj(j)
            if y and y.get('status')=='processing':
                y['progress']=round(min(90, max(5, percent)))
                y['message']=f'Analizando vídeo · {sampled}/{total} muestras'
                writej(j,y)
        except Exception:
            pass

    x=readj(j);x['status']='processing';x['progress']=5;writej(j,x)
    try:
        # build_analysis is CPU/blocking work. Keep it off FastAPI's event loop
        # /jobs/{id} can continue responding while the video is analyzed.
        import asyncio
        result=await asyncio.to_thread(build_analysis,str(src),fps,team,opp,report_analysis_progress)
        x=readj(j)
        if x:
            x['progress']=92
            x['message']='Generando clips y overlay visual'
            writej(j,x)
        # Do not block completion on video rendering. The tactical analysis is the
        # primary artifact; clips can be rendered separately/on demand.
        x=readj(j)
        if x:
            x['progress']=98
            x['message']='Análisis táctico listo; finalizando resultados'
            writej(j,x)
        for clip in result.get('clips',[]):
            clip.setdefault('playable',False)
            clip.setdefault('visualization','pending_render')
        x=readj(j);x['status']='completed';x['progress']=100;x['message']='Análisis completado';x['result']=result
        c=x.get('context',{});c['video']=result;c['clips']=result.get('clips',[]);c['evidence_ledger']=result.get('evidence_ledger',[]);x['context']=c
        writej(j,x)
        if os.getenv('TACTICAI_PERSIST_SOURCE','1')!='0':
            persisted=job_dir(j)/src.name
            shutil.copy2(src,persisted)
            x=readj(j) or {}
            x['source']=persisted.name
            x['source_persisted']=True
            writej(j,x)
        try: src.unlink()
        except Exception: pass
        (job_dir(j)/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
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

@app.post('/jobs/{j}/render')
async def render_job(j:str,request:Request):
    auth(request)
    x=readj(j)
    if not x: raise HTTPException(404,'Job not found')
    src=persisted_source(j)
    if not src: raise HTTPException(404,'Source video unavailable for rendering')
    result=x.get('result',{})
    if not result.get('clips'): return {'clips':[],'message':'No evidence clips available'}
    import asyncio
    x['status']='rendering';x['progress']=92;x['message']='Preparando clips de evidencia';writej(j,x)
    try:
        rendered=await asyncio.to_thread(render_evidence_clips,j,src,result,int(os.getenv('TACTICAI_MAX_RENDER_CLIPS','3')))
        by_id={str(c.get('id')):c for c in rendered if c.get('id') is not None}
        merged=[]
        for clip in result.get('clips',[]):
            rc=by_id.get(str(clip.get('id')))
            merged.append(rc if rc else clip)
        result['clips']=merged
        x=readj(j) or x
        x['result']=result
        ctx=x.get('context',{})
        ctx['video']=result
        ctx['clips']=merged
        ctx['evidence_ledger']=result.get('evidence_ledger',ctx.get('evidence_ledger',[]))
        x['context']=ctx
        x['status']='completed';x['progress']=100;x['message']='Clips de evidencia listos'
        writej(j,x)
        (job_dir(j)/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
        (job_dir(j)/'project_context.json').write_text(json.dumps(ctx,ensure_ascii=False,indent=2))
        return {'clips':merged,'status':'completed'}
    except Exception as e:
        x=readj(j) or x
        x['status']='completed';x['progress']=100;x['message']=f'Análisis listo; error al renderizar clips: {e}'
        writej(j,x)
        raise HTTPException(500,str(e))

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
            x=readj(p.name);r=x.get('result',{});items.append({'id':p.name,'status':x.get('status'),'team':x.get('team'),'opponent':x.get('opponent'),'duration_seconds':r.get('duration',0),'clips':len(x.get('context',{}).get('clips',[])),'has_video':bool(persisted_source(p.name))})
    return {'items':items}

@app.get('/jobs/{j}/artifacts')
def artifacts(j:str,request:Request):
    auth(request);p=job_dir(j)
    if not p.exists(): raise HTTPException(404,'Job not found')
    return {'artifacts':[{'name':f.name,'size_bytes':f.stat().st_size,'download_url':'/jobs/'+j+'/artifacts/'+f.name} for f in p.iterdir() if f.is_file()]}

@app.get('/jobs/{j}/artifacts/{name:path}')
def artifact(j:str,name:str,request:Request):
    auth(request)
    base=job_dir(j).resolve()
    p=(base/name).resolve()
    if base not in p.parents or not p.is_file(): raise HTTPException(404,'Artifact not found')
    from fastapi.responses import FileResponse;return FileResponse(p)

@app.post('/jobs/{j}/archive')
def archive(j:str,request:Request):
    auth(request);p=job_dir(j)
    if not p.exists(): raise HTTPException(404,'Job not found')
    out=shutil.make_archive(str(BASE/(j+'_project')),'zip',root_dir=p);return {'file':'/jobs/'+j+'/artifacts/'+Path(out).name}
