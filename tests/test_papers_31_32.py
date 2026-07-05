def test_all():
 from quality.spcc_structure import Component,collapse,expand
 c=Component((1,)); x=(c,)*4; assert expand(collapse(x))==x
 from quality.cad_complexity import classify
 assert classify(components=1,loops=1,curves=1,type_diversity=1,feature_depth=1)["level"]>=1
 from dataengine.hierarchical_cad_annotation import validate
 assert not validate({"a"},{"a":"x"},{"a"})
 from bench.cad_domain_shift import audit
 assert audit([{"a"}],[{"b"}])["unseen"]==("b",)
 from bench.prefix_completion import cuts,auc
 assert len(cuts(range(10)))==4 and auc(((0,0),(1,1)))==.5
 from bench.sketch_sequence_metrics import metrics
 assert metrics({"s":(1,)},{"s":(1,)})["f1"]==1
