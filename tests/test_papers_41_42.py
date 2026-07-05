def test_p41():
 from cisp.explicit_context import Context
 c=Context();h=c.bind("x","face");s=c.snapshot();c.require(h,"face");c.rollback(s)
 try:c.require(h,"face");assert False
 except ValueError:pass
 from reliability.code_error import normalize
 assert normalize(TypeError(),"x").category=="type"
 from rag.cad_api_knowledge import API,validate,chunks
 assert not validate((API("x","x()","face",("x",)),)) and chunks((API("x","x()","face",()),))
 from bench.correction_trajectory import score
 assert score(({"valid":False},{"valid":True}))["recovered"]
def test_p42():
 from reconstruction.primitive_relations import Primitive,infer,project
 a=Primitive("a",(1,0,0));b=Primitive("b",(1,0,0));assert infer(a,b)=="parallel"
 assert project(a,b,"parallel")[1].axis==a.axis
 from reconstruction.primitive_intersections import assemble
 ps=(a,b,Primitive("c",(0,1,0)));r=assemble(ps,{("a","b"),("b","c"),("a","c")},lambda a,b:(a.id,b.id),lambda a,b,c:(0,0,0))
 assert len(r["edges"])==3
 from reconstruction.primitive_stitch import stitch
 assert stitch(4,lambda x:x/2,abs)["residual"]<1e-5
 from quality.view_coverage import audit
 assert audit({"a","b"},({"id":"v","visible":{"a"},"potential":{"b"}},))["recommendation"]=="v"
 from bench.primitive_reconstruction_metrics import metrics
 assert metrics((((1,0,0),(1,0,0)),),(),(True,),0,1)["normal_consistency"]==1
