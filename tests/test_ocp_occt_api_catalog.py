import os
import unittest

from harnesscad.io.backends.occt_catalog import (
    ClassEntry, MethodEntry, OcctApiCatalog, build_catalog_from_headers,
    module_of, parse_ocp_config,
)
from harnesscad.io.formats.cpp_header import parse_header


HEADER = """\
#ifndef _guard_
class gp_Pnt
{
public:
  gp_Pnt();
  gp_Pnt(const Standard_Real x, const Standard_Real y, const Standard_Real z);
  void SetCoord(const Standard_Real x, const Standard_Real y, const Standard_Real z);
  Standard_Real Distance(const gp_Pnt& other) const;
  Standard_EXPORT void DumpJson(Standard_OStream& s, Standard_Integer depth = -1) const;
  static gp_Pnt Origin();
private:
  void Hidden();
};

class BRepBuilderAPI_MakeShape
{
public:
  const TopoDS_Shape& Shape();
  Standard_Boolean IsDone() const;
};

class BRepPrimAPI_MakeBox : public BRepBuilderAPI_MakeShape
{
public:
  BRepPrimAPI_MakeBox(const Standard_Real dx, const Standard_Real dy, const Standard_Real dz);
  const TopoDS_Solid& Solid();

  enum Mode
  {
    Mode_A,
    Mode_B
  };
};

enum TopAbs_ShapeEnum
{
  TopAbs_COMPOUND,
  TopAbs_FACE
};
"""

OCP_TOML = """\
name = "OCP"
input_folder = "./opencascade"  # source headers
output_folder = "./OCP"

pats = ["{}_*.hxx","{}.hxx"]

exclude_namespaces = [
"std",
"opencascade",
]

modules = [
"gp",
"TopAbs",
"BRepPrimAPI",
]

[Standard]
exclude_classes = ["Standard_Transient"]
"""


def _catalog():
    catalog = OcctApiCatalog()
    catalog.add_header(parse_header(HEADER, path="test.hxx"))
    return catalog


class TestModuleOf(unittest.TestCase):
    def test_module_extraction(self):
        self.assertEqual(module_of("gp_Pnt"), "gp")
        self.assertEqual(module_of("BRepPrimAPI_MakeBox"), "BRepPrimAPI")
        self.assertEqual(module_of("TopoDS"), "TopoDS")
        self.assertEqual(module_of("OCP::gp_Pnt"), "gp")


class TestOcpConfig(unittest.TestCase):
    def test_parse(self):
        cfg = parse_ocp_config(OCP_TOML)
        self.assertEqual(cfg.name, "OCP")
        self.assertEqual(cfg.input_folder, "./opencascade")
        self.assertEqual(cfg.modules, ("gp", "TopAbs", "BRepPrimAPI"))
        self.assertEqual(cfg.exclude_namespaces, ("std", "opencascade"))
        self.assertTrue(cfg.includes_module("gp"))
        self.assertFalse(cfg.includes_module("AIS"))

    def test_table_sections_ignored(self):
        # exclude_classes lives under [Standard], not at the root
        self.assertEqual(parse_ocp_config(OCP_TOML).exclude_classes, ())

    def test_comment_stripping_does_not_eat_strings(self):
        cfg = parse_ocp_config('name = "a#b" # trailing\n')
        self.assertEqual(cfg.name, "a#b")


class TestCatalogBuild(unittest.TestCase):
    def setUp(self):
        self.cat = _catalog()

    def test_classes_and_modules(self):
        self.assertEqual(len(self.cat), 4)
        self.assertIn("gp_Pnt", self.cat)
        self.assertEqual(
            self.cat.modules(), ("BRepBuilderAPI", "BRepPrimAPI", "TopAbs", "gp")
        )
        self.assertEqual(self.cat.classes_in("gp"), ("gp_Pnt",))

    def test_private_methods_excluded(self):
        self.assertFalse(self.cat.has_method("gp_Pnt", "Hidden"))
        self.assertTrue(self.cat.has_method("gp_Pnt", "Distance"))

    def test_overload_arity_merge(self):
        ctor = self.cat.get("gp_Pnt").constructor()
        self.assertEqual((ctor.min_args, ctor.max_args), (0, 3))
        self.assertEqual(ctor.overloads, 2)
        dump = self.cat.get("gp_Pnt").methods["DumpJson"]
        self.assertEqual((dump.min_args, dump.max_args), (1, 2))
        self.assertTrue(self.cat.get("gp_Pnt").methods["Origin"].is_static)

    def test_enums_captured(self):
        self.assertEqual(self.cat.get("BRepPrimAPI_MakeBox").enums["Mode"],
                         ("Mode_A", "Mode_B"))
        self.assertEqual(
            self.cat.get("TopAbs_ShapeEnum").enums["TopAbs_ShapeEnum"],
            ("TopAbs_COMPOUND", "TopAbs_FACE"),
        )

    def test_summary(self):
        summary = self.cat.summary()
        self.assertEqual(summary["gp"]["classes"], 1)
        self.assertEqual(summary["gp"]["methods"], 5)
        self.assertEqual(list(summary), sorted(summary))

    def test_filter_modules(self):
        sub = self.cat.filter_modules(["gp"])
        self.assertEqual(len(sub), 1)
        self.assertIn("gp_Pnt", sub)


class TestCallChecking(unittest.TestCase):
    def setUp(self):
        self.cat = _catalog()

    def test_valid_call(self):
        check = self.cat.check_call("gp_Pnt", "SetCoord", 3)
        self.assertTrue(check.ok)
        self.assertEqual(check.kind, "ok")
        self.assertEqual(check.expected, (3, 3))

    def test_unknown_class_suggests(self):
        check = self.cat.check_call("gp_Pntt", "SetCoord", 3)
        self.assertFalse(check.ok)
        self.assertEqual(check.kind, "unknown_class")
        self.assertEqual(check.suggestions, ("gp_Pnt",))

    def test_unknown_method_suggests(self):
        check = self.cat.check_call("gp_Pnt", "Distence", 1)
        self.assertEqual(check.kind, "unknown_method")
        self.assertEqual(check.suggestions, ("Distance",))

    def test_bad_arity(self):
        check = self.cat.check_call("gp_Pnt", "SetCoord", 5)
        self.assertEqual(check.kind, "bad_arity")
        self.assertEqual(check.expected, (3, 3))
        self.assertIn("takes 3..3 args, got 5", check.reason)

    def test_inherited_method_resolution(self):
        self.assertTrue(self.cat.check_call("BRepPrimAPI_MakeBox", "Shape", 0).ok)
        self.assertTrue(self.cat.check_call("BRepPrimAPI_MakeBox", "IsDone", 0).ok)
        self.assertIsNone(self.cat.resolve_method("BRepPrimAPI_MakeBox", "Nope"))

    def test_construction_check(self):
        self.assertTrue(self.cat.check_construction("BRepPrimAPI_MakeBox", 3).ok)
        self.assertFalse(self.cat.check_construction("BRepPrimAPI_MakeBox", 1).ok)
        self.assertTrue(self.cat.check_construction("gp_Pnt", 0).ok)

    def test_bool_protocol(self):
        self.assertTrue(bool(self.cat.check_call("gp_Pnt", "Distance", 1)))
        self.assertFalse(bool(self.cat.check_call("gp_Pnt", "Distance", 2)))


class TestSerialisation(unittest.TestCase):
    def test_json_round_trip_is_stable(self):
        cat = _catalog()
        text = cat.to_json()
        back = OcctApiCatalog.from_json(text)
        self.assertEqual(back.to_dict(), cat.to_dict())
        self.assertEqual(back.to_json(), text)
        self.assertTrue(back.check_call("gp_Pnt", "SetCoord", 3).ok)

    def test_add_class_merges(self):
        cat = OcctApiCatalog()
        cat.add_class(ClassEntry(name="A", module="A",
                                 methods={"f": MethodEntry("f", 0, 0)}))
        cat.add_class(ClassEntry(name="A", module="A", bases=("B",),
                                 methods={"g": MethodEntry("g", 1, 1)}))
        self.assertEqual(cat.get("A").method_names(), ("f", "g"))
        self.assertEqual(cat.get("A").bases, ("B",))


OCP_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "resources", "cadbible", "OCP-master", "OCP-master",
)


class TestRealOcctHeaders(unittest.TestCase):
    def test_catalog_from_vendored_headers(self):
        headers = os.path.join(OCP_ROOT, "opencascade")
        if not os.path.isdir(headers):
            self.skipTest("OCCT headers not vendored")
        paths = [
            os.path.join(headers, name)
            for name in ("gp_Pnt.hxx", "BRepPrimAPI_MakeBox.hxx", "TopAbs_ShapeEnum.hxx")
            if os.path.exists(os.path.join(headers, name))
        ]
        cat = build_catalog_from_headers(headers, paths=paths)
        self.assertIn("gp_Pnt", cat)
        self.assertTrue(cat.check_call("gp_Pnt", "Distance", 1).ok)
        self.assertEqual(cat.check_call("gp_Pnt", "SetCoord", 9).kind, "bad_arity")
        box = cat.get("BRepPrimAPI_MakeBox")
        self.assertIsNotNone(box)
        self.assertEqual(box.module, "BRepPrimAPI")
        self.assertIn("Shell", box.method_names())

    def test_real_ocp_toml(self):
        path = os.path.join(OCP_ROOT, "ocp.toml")
        if not os.path.exists(path):
            self.skipTest("ocp.toml not vendored")
        with open(path, "r", encoding="utf-8") as fh:
            cfg = parse_ocp_config(fh.read())
        self.assertEqual(cfg.name, "OCP")
        self.assertEqual(cfg.input_folder, "./opencascade")
        for module in ("gp", "TopoDS", "BRepPrimAPI", "BRepAlgoAPI"):
            self.assertIn(module, cfg.modules)
        self.assertIn("std", cfg.exclude_namespaces)


if __name__ == "__main__":
    unittest.main()
