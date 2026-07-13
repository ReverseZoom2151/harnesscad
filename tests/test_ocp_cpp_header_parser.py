import os
import unittest

from harnesscad.io.formats.ocp_cpp_header_parser import (
    ParseError, parse_header, parse_header_file, parse_param, split_params,
    split_top_level, strip_comments, strip_preprocessor,
)


HEADER = """\
// Copyright (c) 1991 Matra Datavision
/* block
   comment */
#ifndef _gp_Pnt_HeaderFile
#define _gp_Pnt_HeaderFile

#include <gp_XYZ.hxx>

class gp_Ax1;
class gp_Vec;

typedef double Standard_Real;

//! Defines a 3D cartesian point.
class gp_Pnt
{
public:
  DEFINE_STANDARD_ALLOC

  //! Creates a point with zero coordinates.
  gp_Pnt() {}

  gp_Pnt(const gp_XYZ& theCoord)
      : coord(theCoord)
  {
  }

  gp_Pnt(const Standard_Real theXp, const Standard_Real theYp, const Standard_Real theZp);

  void SetCoord(const Standard_Integer theIndex, const Standard_Real theXi);

  void SetCoord(const Standard_Real theXp, const Standard_Real theYp, const Standard_Real theZp);

  Standard_Real X() const { return coord.X(); }

  Standard_EXPORT Standard_Boolean IsEqual(const gp_Pnt& theOther,
                                           const Standard_Real theLinearTolerance) const;

  Standard_EXPORT virtual void DumpJson(Standard_OStream& theOStream, Standard_Integer theDepth = -1) const;

  static gp_Pnt Origin();

  gp_Pnt& operator=(const gp_Pnt& theOther);

  virtual Standard_Real Area() const = 0;

  enum Kind
  {
    Kind_Simple,
    Kind_Complex = 4
  };

  ~gp_Pnt();

protected:
  void Internal(int a);

private:
  gp_XYZ coord;
};

struct Bare : public gp_Pnt, private Other
{
  int value(int a, int b = 2);
};

enum TopAbs_ShapeEnum
{
  TopAbs_COMPOUND,
  TopAbs_SOLID
};

Standard_EXPORT Standard_Real Free(const gp_Pnt& p);
"""


class TestNormalisation(unittest.TestCase):
    def test_strip_comments_preserves_lines(self):
        src = "int a; // x\n/* two\nlines */\nint b;\n"
        out = strip_comments(src)
        self.assertNotIn("//", out)
        self.assertNotIn("two", out)
        self.assertEqual(src.count("\n"), out.count("\n"))

    def test_strip_comments_keeps_string_literals(self):
        self.assertIn('"a//b"', strip_comments('const char* s = "a//b"; // gone'))

    def test_strip_preprocessor_handles_continuations(self):
        src = "#define M(x) \\\n  do_it(x)\nint a;\n"
        out = strip_preprocessor(src)
        self.assertNotIn("do_it", out)
        self.assertIn("int a;", out)

    def test_split_top_level(self):
        items = split_top_level("int a; class X { int b; }; void f() { g(); }")
        self.assertEqual(items[0], ("int a", None))
        self.assertEqual(items[1][0], "class X")
        self.assertEqual(items[2][0], "void f()")

    def test_unbalanced_brace_raises(self):
        with self.assertRaises(ParseError):
            split_top_level("class X { int a;")


class TestParams(unittest.TestCase):
    def test_parse_param_type_and_name(self):
        p = parse_param("const gp_Pnt& theP")
        self.assertEqual(p.type, "const gp_Pnt&")
        self.assertEqual(p.name, "theP")
        self.assertFalse(p.has_default)
        self.assertTrue(p.is_const_ref)

    def test_parse_param_default(self):
        p = parse_param("Standard_Integer theDepth = -1")
        self.assertEqual(p.name, "theDepth")
        self.assertEqual(p.default, "-1")
        self.assertTrue(p.has_default)

    def test_split_params_respects_templates_and_calls(self):
        ps = split_params("NCollection_List<int> a, const gp_Pnt& b = gp_Pnt(0, 0, 0)")
        self.assertEqual(len(ps), 2)
        self.assertEqual(ps[0].type, "NCollection_List<int>")
        self.assertEqual(ps[1].default, "gp_Pnt(0, 0, 0)")

    def test_split_params_void_and_empty(self):
        self.assertEqual(split_params(""), ())
        self.assertEqual(split_params("void"), ())


class TestParseHeader(unittest.TestCase):
    def setUp(self):
        self.h = parse_header(HEADER, path="gp_Pnt.hxx")
        self.cls = self.h.find("gp_Pnt")

    def test_forward_decls_and_typedefs(self):
        self.assertEqual(self.h.forward_decls, ["gp_Ax1", "gp_Vec"])
        self.assertEqual([t.name for t in self.h.typedefs], ["Standard_Real"])
        self.assertEqual(self.h.typedefs[0].target, "double")

    def test_classes_found(self):
        self.assertEqual([c.name for c in self.h.classes], ["gp_Pnt", "Bare"])
        self.assertEqual(self.cls.kind, "class")
        self.assertEqual(self.h.class_map()["Bare"].bases, ("gp_Pnt", "Other"))

    def test_constructors_and_arity(self):
        ctors = self.cls.constructors()
        self.assertEqual([c.max_args for c in ctors], [0, 1, 3])
        self.assertTrue(all(c.is_constructor for c in ctors))

    def test_overloads_and_defaults(self):
        setc = self.cls.overloads("SetCoord")
        self.assertEqual(len(setc), 2)
        self.assertEqual({m.arity_range for m in setc}, {(2, 2), (3, 3)})
        dump = self.cls.overloads("DumpJson")[0]
        self.assertEqual(dump.arity_range, (1, 2))
        self.assertTrue(dump.accepts(1))
        self.assertTrue(dump.accepts(2))
        self.assertFalse(dump.accepts(3))
        self.assertTrue(dump.is_virtual)
        self.assertTrue(dump.is_exported)
        self.assertTrue(dump.is_const)

    def test_inline_body_method_kept(self):
        x = self.cls.overloads("X")[0]
        self.assertEqual(x.return_type, "Standard_Real")
        self.assertTrue(x.is_const)

    def test_static_pure_operator_destructor(self):
        self.assertTrue(self.cls.overloads("Origin")[0].is_static)
        area = self.cls.overloads("Area")[0]
        self.assertTrue(area.is_pure)
        self.assertTrue(area.is_virtual)
        self.assertTrue(self.cls.overloads("operator=")[0].is_operator)
        self.assertTrue(self.cls.overloads("~gp_Pnt")[0].is_destructor)

    def test_access_tracking(self):
        by_name = {m.name: m.access for m in self.cls.methods}
        self.assertEqual(by_name["SetCoord"], "public")
        self.assertEqual(by_name["Internal"], "protected")
        self.assertNotIn("DEFINE_STANDARD_ALLOC", by_name)

    def test_struct_default_access_and_default_arg(self):
        bare = self.h.find("Bare")
        self.assertEqual(bare.kind, "struct")
        value = bare.overloads("value")[0]
        self.assertEqual(value.access, "public")
        self.assertEqual(value.arity_range, (1, 2))

    def test_enums(self):
        self.assertEqual(
            self.cls.enums[0].values, ("Kind_Simple", "Kind_Complex")
        )
        self.assertEqual(self.h.enums[0].name, "TopAbs_ShapeEnum")
        self.assertEqual(
            self.h.enums[0].values, ("TopAbs_COMPOUND", "TopAbs_SOLID")
        )

    def test_free_function(self):
        self.assertEqual([f.name for f in self.h.functions], ["Free"])
        self.assertEqual(self.h.functions[0].return_type, "Standard_Real")

    def test_signature_and_determinism(self):
        again = parse_header(HEADER, path="gp_Pnt.hxx")
        self.assertEqual(
            [m.signature() for m in again.find("gp_Pnt").methods],
            [m.signature() for m in self.cls.methods],
        )
        self.assertEqual(
            self.cls.overloads("IsEqual")[0].signature(),
            "Standard_Boolean IsEqual(const gp_Pnt& theOther, "
            "const Standard_Real theLinearTolerance) const",
        )


OCCT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "resources", "cadbible", "OCP-master", "OCP-master", "opencascade",
)


class TestRealOcctHeader(unittest.TestCase):
    def test_gp_pnt_from_disk(self):
        path = os.path.join(OCCT_DIR, "gp_Pnt.hxx")
        if not os.path.exists(path):
            self.skipTest("OCCT headers not vendored")
        cls = parse_header_file(path).find("gp_Pnt")
        self.assertIsNotNone(cls)
        for name in ("SetCoord", "X", "Y", "Z", "Distance", "Transformed"):
            self.assertIn(name, cls.method_names())
        self.assertEqual(
            sorted(c.arity_range for c in cls.constructors()),
            [(0, 0), (1, 1), (3, 3)],
        )


if __name__ == "__main__":
    unittest.main()
