def test_edit_modules():
 from dataengine.edit_triplets import build
 from quality.directional_edit_alignment import alignment,rank
 from quality.edit_locality import locality
 from agent.iterative_edit_policy import IterativeEditPolicy
 assert build("x",[1],[2],"a","b").edits==1
 assert build("x",[1],[1],"a","a") is None
 assert alignment((0,0),(1,0),(0,0),(1,0))==1
 assert rank(({"id":"b","valid":True,"score":.2},{"id":"a","valid":False,"score":1}))[0]["id"]=="b"
 assert locality({"a"},{"a","b"},{"a","b"})["collateral"]==("b",)
 cur={"alignment":.5,"digest":"a","valid":True}; bad={"alignment":.4,"digest":"b","valid":True}
 assert IterativeEditPolicy().choose(cur,bad,())[1]=="rollback"
def test_spatial_modules():
 from ingest.sketch_frame_tokens import SketchFrame,quantize
 from quality.spatial_sequence_accuracy import angle_error,score
 from quality.frame_coherence import check
 from bench.spatial_challenge_set import fixtures,stratify
 from math import pi
 f=SketchFrame((1,2,3),pi/2); p=(2,3); w=f.local_to_world(p)
 q=f.world_to_local(w); assert max(abs(a-b) for a,b in zip(p,q))<1e-9
 assert quantize((0,),3)[0]==(1,) and angle_error(0,2*pi)<1e-7
 assert score([{"op":"x"}],[{"op":"x"}])["command"]==1
 assert not check(f,(p,),(w,),(0,0,1))["issues"]
 fs=fixtures(); assert stratify([(fs[0],1)])["orientation"]==1
