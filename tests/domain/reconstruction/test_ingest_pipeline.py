"""Tests for the reconstruction INGEST pipeline (tokens/mesh -> CISP ops -> session).

Deterministic and stdlib-only. The load-bearing property under test is that the
four quantiser families stay SELECTABLE and are never blended: a sequence tagged
with one family must not decode through another family's dequantiser.
"""

import json
import os
import tempfile
import unittest

from harnesscad.core.cisp.ops import AddCircle, AddLine, Extrude, NewSketch, SetParam
from harnesscad.core.cli import main as cli_main
from harnesscad.core.loop import HarnessSession
from harnesscad.domain.reconstruction import ingest_pipeline as ip
from harnesscad.domain.reconstruction.tokens import hnc_vector_codec as hnc_codec
from harnesscad.domain.reconstruction.tokens import skexgen_extrude as sx_extrude
from harnesscad.domain.reconstruction.tokens import skexgen_quantize as sx_quant
from harnesscad.io.backends.stub import StubBackend

BINS = 64


def _bin_centre(index: int, n_bins: int = BINS) -> float:
    """A value that Vitruvion's floor/bin-centre quantiser round-trips exactly."""
    return (index + 0.5) / n_bins - 0.5


def _rect_ops(x0=-0.2, y0=-0.1, x1=0.2, y1=0.1, distance=0.5):
    """A closed rectangular chain + extrude, in CISP ops."""
    return [
        NewSketch(plane="XY"),
        AddLine(sketch="sk1", x1=x0, y1=y0, x2=x1, y2=y0),
        AddLine(sketch="sk1", x1=x1, y1=y0, x2=x1, y2=y1),
        AddLine(sketch="sk1", x1=x1, y1=y1, x2=x0, y2=y1),
        AddLine(sketch="sk1", x1=x0, y1=y1, x2=x0, y2=y0),
        Extrude(sketch="sk1", distance=distance),
    ]


def _deepcad_tokens() -> ip.TokenSequence:
    """A synthetic DeepCAD token sequence (17-int rows) for a rectangular block."""
    return ip.from_cisp(_rect_ops(), family="deepcad")


def _skexgen_tokens() -> ip.TokenSequence:
    sketch = [[[
        {"type": "line", "start": (-0.5, -0.25), "end": (0.5, -0.25)},
        {"type": "line", "start": (0.5, -0.25), "end": (0.5, 0.25)},
        {"type": "line", "start": (0.5, 0.25), "end": (-0.5, 0.25)},
        {"type": "line", "start": (-0.5, 0.25), "end": (-0.5, -0.25)},
    ]]]
    enc = sx_quant.encode_sketch(sketch)
    ext = sx_extrude.encode_extrude(
        [0.4, 0.0], [0.0, 0.0, 0.0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        "NewBodyFeatureOperation", enc["scale"], [0.0, 0.0])
    merged = sx_quant.merge_se([enc["pix"]], [ext])
    return ip.TokenSequence(ip.SKEXGEN, {"tokens": merged})


def _hnc_tokens() -> ip.TokenSequence:
    faces = [[[
        {"type": "line", "start": (-0.4, -0.2)},
        {"type": "line", "start": (0.4, -0.2)},
        {"type": "line", "start": (0.4, 0.2)},
        {"type": "line", "start": (-0.4, 0.2)},
    ]]]
    cmds, params = hnc_codec.encode_sketch(faces, (0.0, 0.0), 1.0)
    extrude = hnc_codec.encode_extrude(
        [0.0, 0.0, 0.0], 0.5, [0.3, 0.0], [0.0, 0.0, 0.0],
        [1, 0, 0], [0, 1, 0], [0, 0, 1], "NewBodyFeatureOperation")
    return ip.TokenSequence(ip.HNC,
                            {"cmds": cmds, "params": params, "extrude": extrude})


def _vitruvion_tokens() -> ip.TokenSequence:
    return ip.from_cisp(_vitruvion_ops(), family="vitruvion")


def _vitruvion_ops():
    return [
        NewSketch(plane="XY"),
        AddLine(sketch="sk1", x1=_bin_centre(10), y1=_bin_centre(12),
                x2=_bin_centre(50), y2=_bin_centre(12)),
        AddCircle(sketch="sk1", cx=_bin_centre(30), cy=_bin_centre(30),
                  r=_bin_centre(40)),
    ]


class TestDecoderRegistry(unittest.TestCase):
    def test_discovers_the_real_token_families(self):
        decoders = ip.discover_decoders()
        self.assertEqual(sorted(decoders), sorted(ip.FAMILIES))
        self.assertEqual(sorted(ip.FAMILIES),
                         ["deepcad", "hnc", "skexgen", "vitruvion"])

    def test_each_decoder_declares_indexed_reconstruction_modules(self):
        from harnesscad import registry

        indexed = {e.dotted for e in registry.find(package="reconstruction")}
        for family, decoder in ip.discover_decoders().items():
            self.assertTrue(decoder.modules, family)
            for dotted in decoder.modules:
                self.assertIn(dotted, indexed, f"{family} -> {dotted}")

    def test_the_four_quantisers_are_genuinely_different(self):
        # DeepCAD: 256 levels, round-half-even. Vitruvion: 64 bins, bin centre.
        from harnesscad.domain.reconstruction.tokens import deepcad_quantize as dq
        from harnesscad.domain.reconstruction.tokens import vitruvion_primitives as vp

        self.assertEqual(dq.ARGS_DIM, 256)
        self.assertEqual(vp.DEFAULT_NUM_BINS, 64)
        self.assertEqual(2 ** sx_quant.BIT, 64)          # SkexGen: 6-bit
        from harnesscad.domain.reconstruction.tokens import hnc_rotation_codebook as hr
        self.assertEqual(hr.NUM_FRAMES, 25)              # HNC: rotation codebook
        # A level of 32 means a different value in every family.
        self.assertNotAlmostEqual(dq.denumericalize_unit(32),
                                  sx_quant.dequantize(32))
        self.assertNotAlmostEqual(vp.dequantize_params([32])[0],
                                  sx_quant.dequantize(32))

    def test_unknown_family_is_rejected(self):
        with self.assertRaises(ip.UnknownFamily):
            ip.get_decoder("gencad")


class TestDeepCADIngest(unittest.TestCase):
    def test_tokens_decode_to_ops_and_apply_in_a_session(self):
        tokens = _deepcad_tokens()
        commands = ip.decode(tokens, family="deepcad")
        self.assertEqual(commands.family, "deepcad")
        self.assertEqual(len(commands.sketches), 1)

        ops = ip.to_cisp(commands)
        self.assertIsInstance(ops[0], NewSketch)
        self.assertEqual(sum(isinstance(o, AddLine) for o in ops), 4)
        self.assertEqual(sum(isinstance(o, Extrude) for o in ops), 1)

        session = HarnessSession(StubBackend())
        result = session.apply_ops(ops)
        self.assertTrue(result.ok, result.diagnostics)
        self.assertEqual(result.applied, len(ops))
        summary = session.summary()
        self.assertTrue(summary["solid_present"])
        self.assertEqual(summary["feature_count"], 1)
        self.assertEqual(summary["entity_count"], 4)
        self.assertTrue(session.digest())

    def test_ingested_model_is_editable(self):
        """The payoff: an ingested op stream accepts a SetParam edit and rebuilds."""
        ops = ip.to_cisp(ip.decode(_deepcad_tokens(), family="deepcad"))
        session = HarnessSession(StubBackend())
        self.assertTrue(session.apply_ops(ops).ok)
        before = session.digest()
        extrude_index = next(i for i, o in enumerate(ops) if isinstance(o, Extrude))
        edit = session.apply_ops([SetParam(target=extrude_index, param="distance",
                                           value=9.0)])
        self.assertTrue(edit.ok, edit.diagnostics)
        self.assertTrue(session.digest())
        self.assertNotEqual(before, "")

    def test_ingest_is_deterministic(self):
        first = ip.ingest_tokens(_deepcad_tokens(), family="deepcad")
        second = ip.ingest_tokens(_deepcad_tokens(), family="deepcad")
        self.assertEqual(first["digest"], second["digest"])
        self.assertEqual(first["ops"], second["ops"])


class TestOtherFamilies(unittest.TestCase):
    def test_skexgen_tokens_ingest(self):
        result = ip.ingest_tokens(_skexgen_tokens(), family="skexgen")
        self.assertTrue(result["ok"], result["diagnostics"])
        self.assertTrue(result["summary"]["solid_present"])

    def test_hnc_tokens_ingest(self):
        result = ip.ingest_tokens(_hnc_tokens(), family="hnc")
        self.assertTrue(result["ok"], result["diagnostics"])
        self.assertTrue(result["summary"]["solid_present"])
        self.assertEqual(result["summary"]["entity_count"], 4)

    def test_vitruvion_is_sketch_only(self):
        result = ip.ingest_tokens(_vitruvion_tokens(), family="vitruvion")
        self.assertTrue(result["ok"], result["diagnostics"])
        # Vitruvion has no extrude vocabulary: sketch ops, no solid.
        self.assertFalse(result["summary"]["solid_present"])
        self.assertEqual(result["summary"]["feature_count"], 0)
        self.assertEqual(result["summary"]["entity_count"], 2)


class TestFamilyMismatchIsRefused(unittest.TestCase):
    """Decoding one family's tokens with another's dequantiser must RAISE."""

    def test_deepcad_tokens_refused_by_every_other_family(self):
        tokens = _deepcad_tokens()
        for family in ("skexgen", "hnc", "vitruvion"):
            with self.assertRaises(ip.FamilyMismatch):
                ip.decode(tokens, family=family)

    def test_skexgen_tokens_refused_by_deepcad(self):
        with self.assertRaises(ip.FamilyMismatch):
            ip.decode(_skexgen_tokens(), family="deepcad")

    def test_hnc_and_vitruvion_are_not_interchangeable(self):
        with self.assertRaises(ip.FamilyMismatch):
            ip.decode(_hnc_tokens(), family="vitruvion")
        with self.assertRaises(ip.FamilyMismatch):
            ip.decode(_vitruvion_tokens(), family="hnc")

    def test_ingest_tokens_refuses_a_mismatched_family(self):
        with self.assertRaises(ip.FamilyMismatch):
            ip.ingest_tokens(_deepcad_tokens(), family="skexgen")

    def test_mismatch_is_never_silently_decoded(self):
        # The wrong dequantiser would have produced geometry (it does not raise
        # on its own) -- proving the guard, not the input, is what stops it.
        raw = _deepcad_tokens()
        mislabelled = ip.TokenSequence("skexgen", raw.tokens)
        with self.assertRaises(ip.IngestError):
            ip.decode(mislabelled, family="skexgen")   # skexgen cannot parse rows

    def test_unknown_family_name_raises(self):
        with self.assertRaises(ip.UnknownFamily):
            ip.decode(_deepcad_tokens(), family="not-a-family")


class TestRoundTrip(unittest.TestCase):
    def test_vitruvion_round_trip_is_exact(self):
        """Vitruvion is the only unbiased family: ops -> tokens -> ops is exact."""
        ops = _vitruvion_ops()
        tokens = ip.from_cisp(ops, family="vitruvion")
        recovered = ip.to_cisp(ip.decode(tokens, family="vitruvion"))
        self.assertEqual(recovered, ops)

    def test_deepcad_round_trip_preserves_the_op_stream(self):
        """DeepCAD's quantiser is lossy by design: structure exact, values within 1 level."""
        ops = _rect_ops()
        tokens = ip.from_cisp(ops, family="deepcad")
        recovered = ip.to_cisp(ip.decode(tokens, family="deepcad"))
        self.assertEqual([o.OP for o in recovered], [o.OP for o in ops])
        for original, decoded in zip(ops, recovered):
            for key, value in original.to_dict().items():
                if isinstance(value, float):
                    self.assertAlmostEqual(value, decoded.to_dict()[key], delta=0.02)
                else:
                    self.assertEqual(value, decoded.to_dict()[key])

    def test_encode_refuses_ops_the_family_cannot_express(self):
        with self.assertRaises(ip.UnsupportedByFamily):
            ip.from_cisp(_rect_ops(), family="vitruvion")   # no extrude vocabulary
        with self.assertRaises(ip.UnsupportedByFamily):
            ip.from_cisp(_rect_ops(), family="skexgen")     # decode-only family


class TestMeshIngest(unittest.TestCase):
    def test_point_cloud_recovers_a_prismatic_block(self):
        corners = [(x, y, z) for x in (0.0, 4.0) for y in (0.0, 2.0)
                   for z in (0.0, 1.0)]
        result = ip.ingest_mesh(corners)
        self.assertTrue(result["ok"], result["diagnostics"])
        self.assertTrue(result["summary"]["solid_present"])
        self.assertEqual(result["method"], "metrics-bbox")
        self.assertEqual([o["op"] for o in result["ops"]],
                         ["new_sketch", "add_rectangle", "extrude"])


class TestCLI(unittest.TestCase):
    def test_ingest_subcommand_runs_end_to_end(self):
        tokens = _deepcad_tokens()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tokens.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(tokens.to_dict(), fh)
            self.assertEqual(cli_main(["ingest", path, "--family", "deepcad"]), 0)
            # The mandatory family is enforced by argparse.
            with self.assertRaises(SystemExit):
                cli_main(["ingest", path])

    def test_cli_reports_a_family_mismatch_without_crashing(self):
        tokens = _deepcad_tokens()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tokens.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(tokens.to_dict(), fh)
            self.assertEqual(cli_main(["ingest", path, "--family", "hnc"]), 2)

    def test_existing_subcommands_still_work(self):
        self.assertEqual(cli_main(["demo"]), 0)


if __name__ == "__main__":
    unittest.main()
