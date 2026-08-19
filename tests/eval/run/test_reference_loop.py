"""The reference-policy loop actually closes, measures, discriminates, and ranks.

These tests drive the REAL CADGymEnv over the kernel-free frep backend with the
scripted reference policy -- no model, no API. The single most important assertion
is that a DELIBERATELY WRONG op stream SCORES WORSE than the reference: that is the
proof the environment discriminates, which is the whole value proposition.
"""

import json
import unittest

from harnesscad.agents.agent.tool_trajectory import ToolTrajectory
from harnesscad.eval.leaderboard.hardcorpus_board import Standing, ranking
from harnesscad.eval.run import reference_loop as R

# A small, fast subset for the structural tests; the full set is exercised by the
# selfcheck test below.
SUBSET = ["dev_plate_60x40x10", "dev_disc_d40_h12", "dev_spacer_d40_bore14"]


class TestReferenceLoop(unittest.TestCase):
    def test_reference_episode_closes_and_passes_contract(self):
        task = R.tasks(["dev_plate_60x40x10"])[0]
        env = R._new_env()
        res = R.run_episode(env, task, R.reference_policy)
        self.assertTrue(res.completed)                     # done=True through env
        self.assertTrue(res.contract_pass)                 # measured oracle solved
        self.assertTrue(res.built)
        self.assertEqual(res.n_actions, len(task.reference))
        # measured volume is within a hair of the closed form.
        self.assertLess(res.volume_rel_error, 0.01)

    def test_trajectory_has_obs_action_reward_structure(self):
        task = R.tasks(["dev_plate_60x40x10"])[0]
        env = R._new_env()
        res = R.run_episode(env, task, R.reference_policy)
        traj = res.trajectory
        self.assertIsInstance(traj, ToolTrajectory)
        self.assertEqual(len(traj), res.n_actions)
        self.assertTrue(traj.completed)
        for step in traj.steps:
            # action: a typed CISP tool call.
            self.assertTrue(step.call.name)
            # think: scripted reasoning is present.
            self.assertTrue(step.think)
            # reward + obs: the tool_response carries the env verdict.
            payload = json.loads(step.result.description)
            self.assertIn("reward", payload)
            self.assertIn("digest", payload)
            self.assertTrue(step.result.success)

    def test_wrong_policy_is_caught_and_scores_worse(self):
        """THE proof the environment discriminates rather than rubber-stamps."""
        task = R.tasks(["dev_plate_60x40x10"])[0]
        env_ref = R._new_env()
        ref = R.run_episode(env_ref, task, R.reference_policy)
        env_wrong = R._new_env()
        wrong = R.run_episode(env_wrong, task, R.wrong_policy)

        # The env still built a valid solid and paid a POSITIVE reward for the
        # wrong plan -- a plain valid-solid reward cannot tell them apart.
        self.assertTrue(wrong.built)
        self.assertGreater(wrong.total_reward, 0.0)
        # ...but the measured oracle caught it.
        self.assertTrue(ref.contract_pass)
        self.assertFalse(wrong.contract_pass)
        # ...and the episode score is strictly worse.
        self.assertLess(wrong.score, ref.score)
        # ...because the volume is far off the closed form.
        self.assertGreater(wrong.volume_rel_error, 0.5)

    def test_determinism(self):
        task = R.tasks(["dev_disc_d40_h12"])[0]
        a = R.run_episode(R._new_env(), task, R.reference_policy)
        b = R.run_episode(R._new_env(), task, R.reference_policy)
        self.assertEqual(a.final_digest, b.final_digest)
        self.assertEqual(a.measured_volume, b.measured_volume)
        self.assertEqual(a.score, b.score)

    def test_evaluate_emits_table_with_pass_rate(self):
        results, table = R.evaluate(R.tasks(SUBSET), R.reference_policy)
        self.assertEqual(table["n_tasks"], len(SUBSET))
        self.assertEqual(table["episodes_closed"], len(SUBSET))
        self.assertEqual(table["contract_pass"], len(SUBSET))
        self.assertEqual(table["pass_rate"], 1.0)
        self.assertEqual(len(table["rows"]), len(SUBSET))
        for row in table["rows"]:
            self.assertTrue(row["completed"])
            self.assertTrue(row["contract_pass"])
            self.assertIn("measured_volume", row)
            self.assertIn("expected_volume", row)

    def test_leaderboard_row_is_produced_and_ranks(self):
        ref_results, _ = R.evaluate(R.tasks(SUBSET), R.reference_policy,
                                    model="reference-policy")
        wrong_results, _ = R.evaluate(R.tasks(SUBSET), R.wrong_policy,
                                      model="wrong-policy")
        board, ranked = R.leaderboard([
            ("reference-policy", ref_results),
            ("wrong-policy", wrong_results),
        ])
        self.assertTrue(all(isinstance(s, Standing) for s in ranked))
        self.assertEqual(ranked[0].name, "reference-policy")
        self.assertEqual(ranked[0].oracle_rate, 1.0)
        self.assertEqual(ranked[1].oracle_rate, 0.0)
        # the wrong policy fools the field's weak (valid-solid) metric, not the oracle.
        self.assertEqual(ranked[1].field_fooled, len(SUBSET))
        # ranking() is the real leaderboard ranker.
        self.assertEqual(
            [s.name for s in ranking(list(board.standings))],
            ["reference-policy", "wrong-policy"])

    def test_selfcheck_passes_and_writes_artifacts(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            saved = R.ARTIFACT_DIR
            R.ARTIFACT_DIR = tmp
            try:
                rc = R.selfcheck(write=True, verbose=False)
            finally:
                R.ARTIFACT_DIR = saved
            self.assertEqual(rc, 0)
            import os
            traj = os.path.join(tmp, "reference_trajectory.json")
            table = os.path.join(tmp, "eval_table.json")
            self.assertTrue(os.path.exists(traj))
            self.assertTrue(os.path.exists(table))
            with open(table, encoding="ascii") as fh:
                tdata = json.load(fh)
            self.assertEqual(tdata["pass_rate"], 1.0)
            self.assertEqual(tdata["model"], "reference-policy")
            self.assertEqual(
                tdata["leaderboard"]["ranking"][0]["name"], "reference-policy")
            with open(traj, encoding="ascii") as fh:
                jdata = json.load(fh)
            self.assertTrue(jdata["trajectory"]["completed"])
            self.assertTrue(jdata["outcome"]["contract_pass"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
