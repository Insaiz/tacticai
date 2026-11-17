from __future__ import annotations
import os
try:
 from ultralytics import YOLO
except Exception: YOLO=None
class FootballDetector:
 def __init__(self):
  self.model=None;path=os.getenv('TACTICAI_YOLO_MODEL','')
  if path and YOLO:self.model=YOLO(path)
 def detect(self,frame):
  if not self.model:return []
  r=self.model.predict(frame,conf=float(os.getenv('TACTICAI_DETECTOR_CONF','0.15')),imgsz=int(os.getenv('TACTICAI_DETECTOR_IMGSZ','1280')),verbose=False)[0];out=[]
  for b in r.boxes:
   cls=int(b.cls[0]);conf=float(b.conf[0]);x1,y1,x2,y2=map(float,b.xyxy[0]);out.append({'class_id':cls,'confidence':conf,'x':(x1+x2)/2,'y':(y1+y2)/2,'w':x2-x1,'h':y2-y1})
  return out
