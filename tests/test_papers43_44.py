from adapters.rhino_contract import HostCapabilities, HostResult, HostScript, validate_script
from agent.host_feedback import HostProposal, confirm, execute, preview, refine
from agent.intent_resolution import resolve_intent
from bench.evolution_dynamics import evolution_dynamics, lineage_stats
from bench.nl_cad_casebook import evaluate_case, paper_casebook
from bench.operator_profile import operator_profile
from dataengine.template_collapse import identifier_leakage, template_collapse
from datagen.cube_rotations import apply_rotation, cube_rotations, inverse_rotation, rewrite_calls
from datagen.evolution import GeneratorRecord, sample_parents, termination, validate_lineage
from datagen.evolution_validation import canonical_seven_views, validate_candidate
from datagen.parameter_qd import fill_archive
from datagen.trace_slice import slice_trace, verify_slice

def test_evolution_lineage_sampling_and_budget():
    records=(GeneratorRecord("a","a","","",""),GeneratorRecord("b","b","","","",("a",)))
    assert not validate_lineage(records)
    assert sample_parents(records,1,seed=4)==sample_parents(records,1,seed=4)
    assert termination(({"novelty_ratio":0},)*3,budget=9)=="novelty-saturation"
    assert lineage_stats(records)["max_depth"]==1

def test_quality_diversity_archive_enforces_geometry_and_novelty():
    def evaluate(params):
        x=params[0]
        return {"valid":True,"solid_count":1,"watertight":True,
                "bounds":(-40,-30,-30,40,30,30),"descriptor":(x,)}
    report=fill_archive(lambda i:(i%2,),evaluate,target=3,budget=4,epsilon=.1)
    assert len(report["entries"])==2 and report["termination"]=="budget"
    assert any(x["reason"]=="not-novel" for x in report["attempts"])

def test_trace_slice_is_flat_canonical_and_equivalence_gated():
    source=slice_trace((("width",4),),(
        {"kind":"log","statement":"print('x')"},
        {"kind":"extrude","statement":"shape = cq.Workplane('XY').box(width, 2, 3)",
         "output":"shape"},
    ))
    assert "print" not in source and source.endswith("result = shape\n")
    assert verify_slice(source,lambda s:"shape",
                        lambda value:{"equivalent":value=="shape"})["accepted"]

def test_cube_has_exactly_24_proper_reversible_rotations_and_rewrites_globals():
    rotations=cube_rotations()
    assert len(rotations)==24
    vector=(1,2,3); matrix=rotations[7]
    assert apply_rotation(inverse_rotation(matrix),apply_rotation(matrix,vector))==vector
    calls=({"kind":"line","args":{"point":vector}},
           {"kind":"translate_global","args":{"vector":vector}})
    rewritten=rewrite_calls(calls,matrix)
    assert rewritten[0]==calls[0] and rewritten[1]["args"]["vector"]!=vector

def test_template_collapse_and_operator_profile():
    records=({"family":"a","code":"x=box(1)","operations":("box","extrude"),"face_count":6},
             {"family":"a","code":"y=box(2)","operations":("box",),"face_count":5})
    report=template_collapse(records)
    assert report["families"][0]["concentration"]==1
    assert "x" in identifier_leakage(records)
    profile=operator_profile(records,reference={"box":.5})
    assert profile["operation_rates"]["box"]==1 and profile["operation_delta"]["box"]==.5

def test_evolution_validation_stops_at_first_failed_stage():
    ok=lambda value:{"accepted":True,"output":value}
    admission=validate_candidate("candidate",execute=ok,integrity=ok,render=ok,
                                 semantic=lambda value:{"accepted":False,"reason":"mismatch"})
    assert admission.stage=="semantic" and admission.repair_packet["reason"]=="mismatch"
    assert len(canonical_seven_views())==7
    dynamics=evolution_dynamics(({"proposed":10,"invalid":1,"novel":8,"accepted":7},
                                 {"proposed":10,"invalid":3,"novel":4,"accepted":3}))
    assert dynamics["diminishing"]

def test_intent_resolution_requires_seed_for_random_and_routes_analysis():
    unresolved=resolve_intent("Create a 100 mm box at a random edge and union it")
    assert unresolved.needs_clarification
    intent=resolve_intent("Create a 100 mm box at a random edge, union and bake",seed=7)
    assert not intent.needs_clarification and "random-choice-seed:7" in intent.assumptions
    case=paper_casebook()[0]
    assert evaluate_case(intent,required_operations=case["operations"],
                         required_capabilities=case["capabilities"])["intent_coverage"]==1

def test_rhino_contract_and_confirmation_lifecycle_are_host_neutral():
    script=HostScript("s","rhinoscript","Box()",("box",),True)
    assert not validate_script(script,HostCapabilities(frozenset({"box"})))
    class Host:
        def preview(self,script): return {"safe":True}
        def execute(self,script): return HostResult(True,"done","rollback")
    proposal=HostProposal("p",script,"make box")
    staged=preview(proposal,Host())
    done=execute(confirm(staged),Host())
    assert done.status=="executed"
    revised=refine(done,script)
    assert revised.status=="proposed" and revised.lineage==("p",)
