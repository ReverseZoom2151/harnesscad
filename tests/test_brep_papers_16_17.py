import unittest
from reconstruction.point_labels import *
from ingest.scan_brep_labels import *
from quality.scan_label_audit import audit
from reconstruction.failure_audit import classify,Failure
from reconstruction.structured_brep import Node,pad_children,unique_children,validate_tree
from reconstruction.brep_merge import GeometryNode,cluster
from reconstruction.geometry_stitch import average_vertices,align_edge,consistency
from bench.generative_brep_metrics import ratios,coverage_mmd,jsd

class Tests(unittest.TestCase):
 def test_labels(self):
  p=lambda i,x,c,l:PointPrediction(i,(x,0,0),c,l)
  self.assertEqual(suppress([p("b",0,.5,"b"),p("a",0,.5,"b")],1)[0].id,"a")
  self.assertEqual(len(boundary_first([p("b",0,1,"b")],[p("j",0,1,"j")],.1)[1]),1)
  self.assertEqual(precision_recall_f1([1],[1])["f1"],1)
 def test_chain(self):
  c=ChainComplex(frozenset({"f"}),{"b":("f","f","f")},frozenset({"j"}))
  self.assertIn("junction_without_boundary:p",validate([ScanLabel("p",junction_id="j")],c))
 def test_audit_failure(self):
  self.assertEqual(audit([(0,0,0),(2,0,0)],["b","f"]).occupied_bins,2)
  self.assertIn(Failure.NON_WATERTIGHT,classify(watertight=False))
 def test_tree(self):
  v1,v2=Node("1","vertex",()),Node("2","vertex",())
  e=Node("e","edge",(),(v1,v2))
  self.assertFalse(validate_tree(e)); self.assertEqual(len(pad_children((e,),3,1)),3)
  self.assertEqual(len(unique_children((e,e))),1)
 def test_merge(self):
  n=lambda i,x:GeometryNode(i,(x,)*6,((x,0,0),))
  self.assertEqual(len(cluster((n("a",0),n("b",.01)))),1)
 def test_stitch(self):
  self.assertEqual(average_vertices(((0,0,0),(2,0,0))),(1,0,0))
  self.assertEqual(align_edge(((1,0,0),(0,0,0)),(0,0,0),(2,0,0))[-1],(2,0,0))
  self.assertEqual(consistency([((0,0,0),)],lambda p:1),(1,1))
 def test_metrics(self):
  self.assertEqual(ratios([1,1,2],[1])["novel"],1/3)
  self.assertEqual(coverage_mmd([0,2],[0,1],lambda a,b:abs(a-b))["mmd"],.5)
  self.assertAlmostEqual(jsd({"a":1},{"b":1}),1)
if __name__=="__main__":unittest.main()
