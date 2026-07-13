from math import dist
def check(frame,local_points,world_points,extrusion,normal=(0,0,1),tolerance=1e-6):
 errors=tuple(dist(frame.local_to_world(p),w) for p,w in zip(local_points,world_points))
 dot=sum(a*b for a,b in zip(extrusion,normal))
 issues=[]
 if len(local_points)!=len(world_points):issues.append("point_count")
 if any(x>tolerance for x in errors):issues.append("frame_drift")
 if dot<=0:issues.append("reversed_extrusion")
 return {"errors":errors,"issues":tuple(issues)}
