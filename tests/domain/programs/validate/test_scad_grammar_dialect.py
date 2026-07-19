"""The scad_grammar dialect gate: refuse RapCAD instead of mis-diagnosing it.

This module implements OpenSCAD only (transliterated from RapCAD's
``doc/openscad.bnf``). RapCAD's own, larger grammar (``doc/rapcad.bnf``) was
never ported, so ``.rcad`` input must be refused by name rather than reported as
a heap of invented syntax errors.

Every snippet here is written from scratch; no RapCAD (GPL-3) file content is
reproduced or vendored.
"""

import pytest

from harnesscad.domain.programs.validate.scad_grammar import (
    RAPCAD_MARKERS,
    Rule,
    detect_rapcad,
    main,
    validate,
)

PM = chr(0x00B1)


RAPCAD_ONLY = [
    ("return", "function f(x) { return x * 2; }"),
    ("const", "const width = 10;"),
    ("param", "param depth = 4;"),
    ("namespace ::", "x = lib::helper(1);"),
    ("append ~=", "xs ~= [1, 2];"),
    ("exponent ^", "x = 2 ^ 8;"),
    ("cross **", "n = a ** b;"),
    ("componentwise .*", "v = a .* b;"),
    ("componentwise ./", "v = a ./ b;"),
    ("tolerance operator", "d = 10 " + PM + " 0.1;"),
    ("interval literal", "d = 10[0.1, 0.2];"),
    ("doc comment", "/** @param r radius */\nmodule ring(r) { }"),
]

NOT_RAPCAD = [
    ("relative include path", "include <./lib/util.scad>\ncube(1);"),
    ("relative use path", "use <../shared/gears.scad>\ncube(1);"),
    ("markers inside a string", 'echo("return const param :: ^ ~= .*");'),
    ("markers in a line comment", "// return const ^ ::\ncube(1);"),
    ("markers in a block comment", "/* param ~= .* */\ncube(1);"),
    ("identifiers containing markers", "constant = 1;\nreturns = 2;"),
    ("ordinary indexing", "x = v[0];"),
    ("plain openscad", "translate([1, 0, 0]) cube([1, 2, 3], center = true);"),
]


@pytest.mark.parametrize("label,source", RAPCAD_ONLY,
                         ids=[lbl for lbl, _ in RAPCAD_ONLY])
def test_rapcad_only_constructs_are_refused_not_mis_diagnosed(label, source):
    result = validate(source)
    assert not result.ok
    # exactly one diagnostic, and it names the dialect -- not a pile of
    # invented statement-level syntax errors
    assert len(result.diagnostics) == 1
    diag = result.diagnostics[0]
    assert diag.rule == Rule.DIALECT.value
    assert "RapCAD" in diag.found
    assert "rapcad_language" in diag.expected


@pytest.mark.parametrize("label,source", NOT_RAPCAD,
                         ids=[lbl for lbl, _ in NOT_RAPCAD])
def test_lookalikes_do_not_trip_the_dialect_gate(label, source):
    assert detect_rapcad(source) is None
    result = validate(source)
    assert not any(d.rule == Rule.DIALECT.value for d in result.diagnostics)


def test_rcad_extension_alone_is_enough():
    # text that is perfectly good OpenSCAD, but the file says it is RapCAD
    assert validate("cube(1);").ok
    refused = validate("cube(1);", path="parts/bracket.rcad")
    assert not refused.ok
    assert refused.diagnostics[0].rule == Rule.DIALECT.value


def test_extension_matching_is_case_insensitive_and_suffix_anchored():
    assert validate("cube(1);", path="A.RCAD").diagnostics[0].rule == \
        Rule.DIALECT.value
    # ".rcad" in the middle of a name is not the extension
    assert validate("cube(1);", path="rcad_examples/a.scad").ok


def test_path_defaults_to_none_so_the_old_call_still_works():
    assert validate("cube(1);").ok


def test_refusal_reports_the_earliest_marker_position():
    source = "cube(1);\nx = 2 ^ 8;\ny = a ** b;\n"
    diag = detect_rapcad(source)
    assert diag is not None
    assert diag.line == 2
    assert "^" in diag.found


def test_detect_rapcad_returns_none_for_clean_openscad():
    assert detect_rapcad("module a(n = 3) { for (i = [0 : n]) cube(i); }") is None


def test_marker_table_is_non_empty_and_described():
    assert RAPCAD_MARKERS
    for text, what in RAPCAD_MARKERS:
        assert text
        assert "RapCAD" in what


def test_unterminated_string_or_comment_does_not_hang_the_scanner():
    assert detect_rapcad('echo("unterminated') is None
    assert detect_rapcad("/* unterminated const") is None
    # a doc comment is still caught even when unterminated
    assert detect_rapcad("/** unterminated") is not None


def test_selfcheck_exits_zero():
    assert main(["--selfcheck"]) == 0


def test_cli_refuses_an_rcad_file(tmp_path, capsys):
    target = tmp_path / "bracket.rcad"
    target.write_text("cube(1);\n", encoding="utf-8")
    assert main([str(target)]) == 1
    assert "RapCAD" in capsys.readouterr().out
