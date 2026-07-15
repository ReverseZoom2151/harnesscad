"""Tests for the Anvilate provenance evidence bundle."""

import unittest

from harnesscad.domain.standards import evidence_bundle as eb


def _spec():
    return {
        "name": "motor_mount_bracket",
        "material": "AA-6061-T6",
        "interfaces": ["NEMA23", "6204"],
        "dimensions": [{"tag": "pilot_bore", "fit": "H7"}],
        "geometric_tolerances": [{"feature": "motor_face", "characteristic": "flatness"}],
    }


class CollectTest(unittest.TestCase):
    def test_material_cited(self):
        bundle = eb.collect_provenance(_spec())
        mats = [r for r in bundle.records if r.kind == "material"]
        self.assertEqual(len(mats), 1)
        self.assertEqual(mats[0].ref, "AA-6061-T6")

    def test_components_cited(self):
        bundle = eb.collect_provenance(_spec())
        comps = [r for r in bundle.records if r.kind == "component"]
        self.assertEqual({c.ref for c in comps}, {"NEMA23", "6204"})

    def test_general_tolerance_always_present(self):
        bundle = eb.collect_provenance({"name": "bare"})
        self.assertIn("general_tolerance", bundle.kinds())

    def test_fit_cited(self):
        bundle = eb.collect_provenance(_spec())
        fits = [r for r in bundle.records if r.kind == "fit"]
        self.assertEqual(len(fits), 1)
        self.assertIn("ISO 286", fits[0].sources[0])

    def test_gdt_cited(self):
        bundle = eb.collect_provenance(_spec())
        gdt = [r for r in bundle.records if r.kind == "gdt"]
        self.assertEqual(len(gdt), 1)
        self.assertEqual(gdt[0].sources[0], "ISO 1101")

    def test_every_record_has_a_source(self):
        bundle = eb.collect_provenance(_spec())
        for r in bundle.records:
            self.assertTrue(r.sources, f"{r.ref} has no citation")


class MissingCitationTest(unittest.TestCase):
    def test_unknown_material_unresolved(self):
        bundle = eb.collect_provenance({"material": "UNOBTAINIUM"})
        self.assertFalse(bundle.is_fully_cited())
        self.assertIn("material:UNOBTAINIUM", bundle.missing_citations())

    def test_unknown_component_unresolved(self):
        bundle = eb.collect_provenance({"interfaces": ["NEMA23", "ZZZ99"]})
        self.assertIn("component:ZZZ99", bundle.missing_citations())

    def test_fully_cited_spec(self):
        bundle = eb.collect_provenance(_spec())
        self.assertTrue(bundle.is_fully_cited())


class DigestTest(unittest.TestCase):
    def test_deterministic(self):
        d1 = eb.collect_provenance(_spec()).digest()
        d2 = eb.collect_provenance(_spec()).digest()
        self.assertEqual(d1, d2)

    def test_order_independent(self):
        s1 = _spec()
        s2 = dict(s1)
        s2["interfaces"] = ["6204", "NEMA23"]  # reversed order
        self.assertEqual(
            eb.collect_provenance(s1).digest(),
            eb.collect_provenance(s2).digest(),
        )

    def test_changes_with_content(self):
        base = eb.collect_provenance(_spec()).digest()
        other = _spec()
        other["material"] = "SS-304"
        self.assertNotEqual(base, eb.collect_provenance(other).digest())

    def test_digest_is_hex_sha256(self):
        d = eb.collect_provenance(_spec()).digest()
        self.assertEqual(len(d), 64)
        int(d, 16)  # parses as hex


if __name__ == "__main__":
    unittest.main()
