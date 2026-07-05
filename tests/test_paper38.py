def test_fsa_and_pooling():
 from grammar_fsa import run,State
 seq=("line","curve_end","loop_end","face_end","sketch_end","add","pad")
 assert run(seq)[0] is State.PAD
 assert run(("curve_end",))[0] is State.DEAD
 from quality.primitive_pooling import spans
 s,r=spans(seq);assert r["exact_coverage"] and not r["overflow"]
def test_metrics_splits():
 from bench.tokenizer_frontier import evaluate,frontier
 a=evaluate("a",(1,2),(1,),(1,2));b=evaluate("b",(1,2),(1,2),(1,0))
 assert a in frontier((a,b))
 from bench.tokenizer_split_audit import audit
 r=audit({1},{1,2});assert r["nested"] and r["has_heldout_exposure"]
