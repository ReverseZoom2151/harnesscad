def test_smoke():
 from procedural.cad_patterns import linear,grid,radial,pipe
 from quality.grammar_compression import compression
 from dataengine.prior_cache import PriorCache,cache_key
 from bench.render_distribution import summarize
 assert len(linear(3,2))==3 and len(grid(2,2,(1,1)))==4 and len(radial(4,1))==4
 assert len(pipe(((0,0,0),(1,0,0))))==1
 assert compression([1,2],{1})["reuse"]==2
 c=PriorCache(); k=cache_key(prompt="x"); c.put(k,1); assert c.get(k)[0]==1
 assert summarize([{"id":"a","quality":1,"prompt":1,"feature":0}],lambda a,b:abs(a-b))["count"]==1

def test_camera_and_scene():
 from quality.camera_pruning import CameraSample,prune
 from procedural.lazy_scene import expand
 a,r=prune([CameraSample("a",0,0,0)],1,1,1); assert len(a)==1 and not r
 out,b,c=expand([1],lambda x:True,lambda x:() if x>2 else (x+1,),lambda x:str(x))
 assert out==(3,) and c["visited"]==3

def test_grammar_consistency_stages():
 from procedural.shape_grammar import Production,derive
 from quality.multiview_consistency import consistency
 from quality.render_stages import assess
 assert derive("S",[Production("S",("T",))],{"T"})[0]
 assert consistency({"a":0,"b":1},lambda a,b:abs(a-b))["mean"]==1
 assert assess(0,1,lambda a,b:1,lambda a,b:.1,.2)["accepted"]
