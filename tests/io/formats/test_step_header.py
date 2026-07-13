"""Tests for formats.step_header_model."""

import unittest

from harnesscad.io.formats.step import parse
from harnesscad.io.formats.step_header import HeaderError, StepHeader, parse_header

# From the ABC dataset example used in ruststep's header.rs test.
_STEP = """ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('a model'),'2;1');
FILE_NAME('/tmp/5ae2.step','2018-04-27T08:23:47',('Alice','Bob'),(''),'preproc 1.0','sys 2.0','none');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));
ENDSEC;
DATA;
#1=CARTESIAN_POINT('',(0.,0.,0.));
ENDSEC;
END-ISO-10303-21;
"""


class HeaderModelTest(unittest.TestCase):
    def setUp(self):
        self.step = parse(_STEP)
        self.header = parse_header(self.step)

    def test_returns_step_header(self):
        self.assertIsInstance(self.header, StepHeader)

    def test_file_description(self):
        fd = self.header.file_description
        self.assertEqual(fd.description, ("a model",))
        self.assertEqual(fd.implementation_level, "2;1")

    def test_file_name_fields(self):
        fn = self.header.file_name
        self.assertEqual(fn.name, "/tmp/5ae2.step")
        self.assertEqual(fn.time_stamp, "2018-04-27T08:23:47")
        self.assertEqual(fn.author, ("Alice", "Bob"))
        self.assertEqual(fn.organization, ("",))
        self.assertEqual(fn.preprocessor_version, "preproc 1.0")
        self.assertEqual(fn.originating_system, "sys 2.0")
        self.assertEqual(fn.authorization, "none")

    def test_schema_name(self):
        self.assertEqual(
            self.header.file_schema.schema_identifiers,
            ("AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }",))
        self.assertTrue(self.header.schema_name.startswith("AUTOMOTIVE_DESIGN"))

    def test_semicolon_inside_string_preserved(self):
        # implementation_level '2;1' contains a semicolon; the parser must not
        # split on it. This guards the full parse->header path.
        self.assertEqual(self.header.file_description.implementation_level,
                         "2;1")


class HeaderErrorTest(unittest.TestCase):
    def test_missing_record_raises(self):
        step = parse(
            "ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION((''),'');\nENDSEC;\n"
            "DATA;\nENDSEC;\nEND-ISO-10303-21;\n")
        with self.assertRaises(HeaderError):
            parse_header(step)

    def test_wrong_arity_raises(self):
        step = parse(
            "ISO-10303-21;\nHEADER;\n"
            "FILE_DESCRIPTION((''),'','extra');\n"
            "FILE_NAME('','',(''),(''),'','','');\n"
            "FILE_SCHEMA((''));\nENDSEC;\n"
            "DATA;\nENDSEC;\nEND-ISO-10303-21;\n")
        with self.assertRaises(HeaderError):
            parse_header(step)

    def test_extra_header_records_collected(self):
        step = parse(
            "ISO-10303-21;\nHEADER;\n"
            "FILE_DESCRIPTION((''),'');\n"
            "FILE_NAME('','',(''),(''),'','','');\n"
            "FILE_SCHEMA((''));\n"
            "FILE_POPULATION('X',(#1));\nENDSEC;\n"
            "DATA;\nENDSEC;\nEND-ISO-10303-21;\n")
        header = parse_header(step)
        self.assertEqual(len(header.extra), 1)
        self.assertEqual(header.extra[0].keyword, "FILE_POPULATION")


if __name__ == "__main__":
    unittest.main()
