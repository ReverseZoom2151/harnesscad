"""Tests for the Wu & Palmer taxonomy similarity metric."""
import unittest

from harnesscad.agents.exploration.llmdesopt_wup_similarity import (
    Taxonomy,
    wup_similarity,
    wup_distance,
    rank_by_similarity,
)


def build_animal_taxonomy():
    # root -> entity -> animal -> {reptile->snake, amphibian->frog}
    #                          -> vehicle -> car
    t = Taxonomy("entity")
    t.add("animal", "entity")
    t.add("vehicle", "entity")
    t.add("reptile", "animal")
    t.add("amphibian", "animal")
    t.add("snake", "reptile")
    t.add("frog", "amphibian")
    t.add("car", "vehicle")
    return t


class StructureTests(unittest.TestCase):
    def test_depth_root_is_one(self):
        t = build_animal_taxonomy()
        self.assertEqual(t.depth("entity"), 1)

    def test_ancestors_reach_root(self):
        t = build_animal_taxonomy()
        self.assertEqual(t.ancestors("snake"),
                         ["snake", "reptile", "animal", "entity"])

    def test_lcs(self):
        t = build_animal_taxonomy()
        self.assertEqual(t.lcs("snake", "frog"), "animal")
        self.assertEqual(t.lcs("snake", "car"), "entity")

    def test_add_unknown_parent_raises(self):
        t = build_animal_taxonomy()
        with self.assertRaises(KeyError):
            t.add("x", "nonexistent")

    def test_duplicate_raises(self):
        t = build_animal_taxonomy()
        with self.assertRaises(ValueError):
            t.add("snake", "reptile")


class SimilarityTests(unittest.TestCase):
    def test_self_similarity_is_one(self):
        t = build_animal_taxonomy()
        self.assertAlmostEqual(wup_similarity(t, "car", "car"), 1.0)

    def test_snake_frog_close_to_car_equal(self):
        # Paper's observation (Fig. 8): snake & frog have near-identical WUP
        # w.r.t. car despite different geometry.  Here they are exactly equal
        # because both sit two levels under 'animal' at the same depth.
        t = build_animal_taxonomy()
        s = wup_similarity(t, "car", "snake")
        f = wup_similarity(t, "car", "frog")
        self.assertAlmostEqual(s, f)

    def test_known_value(self):
        t = build_animal_taxonomy()
        # LCS(snake, car) = entity depth 1; depth snake 4, depth car 3.
        self.assertAlmostEqual(wup_similarity(t, "snake", "car"),
                               2.0 * 1 / (4 + 3))

    def test_distance_is_complement(self):
        t = build_animal_taxonomy()
        self.assertAlmostEqual(wup_distance(t, "snake", "car"),
                               1.0 - wup_similarity(t, "snake", "car"))

    def test_closer_pair_higher_similarity(self):
        t = build_animal_taxonomy()
        # snake vs frog (share 'animal') should be more similar than snake vs car
        self.assertGreater(wup_similarity(t, "snake", "frog"),
                           wup_similarity(t, "snake", "car"))


class RankingTests(unittest.TestCase):
    def test_rank_by_similarity_orders_and_ties_break(self):
        t = build_animal_taxonomy()
        ranked = rank_by_similarity(t, "car", ["snake", "frog", "car"])
        self.assertEqual(ranked[0][0], "car")   # self first
        # snake and frog tie -> alphabetical
        self.assertEqual([w for w, _ in ranked[1:]], ["frog", "snake"])


if __name__ == "__main__":
    unittest.main()
