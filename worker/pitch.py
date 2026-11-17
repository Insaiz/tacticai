from __future__ import annotations
import numpy as np

def homography_from_4pts(src,dst):
 A=[];b=[]
 for (x,y),(u,v) in zip(src,dst):
  A += [[x,y,1,0,0,0,-u*x,-u*y],[0,0,0,x,y,1,-v*x,-v*y]];b += [u,v]
 h=np.linalg.solve(np.asarray(A,float),np.asarray(b,float));return np.array([[h[0],h[1],h[2]],[h[3],h[4],h[5]],[h[6],h[7],1.]])
def project_point(H,x,y):
 p=H@np.array([x,y,1.]);return None if abs(p[2])<1e-9 else (float(p[0]/p[2]),float(p[1]/p[2]))
def pitch_zone(x,y,length=105,width=68):
 return ('defensive_third' if x<length/3 else 'middle_third' if x<2*length/3 else 'attacking_third')+'_'+('left' if y<width/3 else 'central' if y<2*width/3 else 'right')