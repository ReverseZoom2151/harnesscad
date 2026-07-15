"""The observation template is a contract: termination and prediction are required."""

import unittest

from harnesscad.agents.cua.prompts import (
    CAD_SYSTEM, OBSERVATION_TEMPLATE, Observation, ObservationError,
    parse_observation, parse_termination, predicted_outcome, prediction_held,
    render_observation_prompt,
)

_GOOD = (
    "The objective is: build a 10mm cube\n"
    "On the screen, I see: the FreeCAD window, a Part Design workbench, an empty "
    "document tree, and the Pad toolbar\n"
    "This means the objective is: not complete\n"
    "The next step is to click the New Sketch button in order to open the plane "
    "selection dialog"
)

_DONE = (
    "The objective is: build a 10mm cube\n"
    "On the screen, I see: a solid cube in the 3D view and Pad001 in the tree\n"
    "This means the objective is: complete"
)


class TestTemplate(unittest.TestCase):
    def test_render_restates_objective(self):
        p = render_observation_prompt("draw a plate")
        self.assertTrue(p.startswith("OBJECTIVE: draw a plate"))
        self.assertIn(OBSERVATION_TEMPLATE, p)

    def test_cad_system_bakes_in_the_traps(self):
        self.assertIn("37.5", CAD_SYSTEM)
        self.assertIn("NEVER save", CAD_SYSTEM)


class TestTermination(unittest.TestCase):
    def test_not_complete_beats_complete_substring(self):
        self.assertFalse(parse_termination("the objective is: not complete"))

    def test_complete(self):
        self.assertTrue(parse_termination("this means the objective is: complete"))

    def test_missing_termination_raises(self):
        with self.assertRaises(ObservationError):
            parse_termination("the objective is unclear")


class TestPrediction(unittest.TestCase):
    def test_extracted(self):
        self.assertEqual(
            predicted_outcome("click X in order to open the dialog."),
            "open the dialog")

    def test_absent(self):
        self.assertIsNone(predicted_outcome("click the button"))


class TestParseObservation(unittest.TestCase):
    def test_full_parse(self):
        obs = parse_observation(_GOOD)
        self.assertEqual(obs.objective, "build a 10mm cube")
        self.assertFalse(obs.complete)
        self.assertEqual(obs.prediction, "open the plane selection dialog")

    def test_complete_needs_no_next_step(self):
        obs = parse_observation(_DONE)
        self.assertTrue(obs.complete)
        self.assertEqual(obs.next_step, "")

    def test_empty_enumeration_raises(self):
        txt = ("The objective is: x\nOn the screen, I see:\n"
               "This means the objective is: not complete\n"
               "The next step is to click y in order to do z")
        with self.assertRaises(ObservationError):
            parse_observation(txt)

    def test_incomplete_without_prediction_raises(self):
        txt = ("The objective is: x\nOn the screen, I see: a window\n"
               "This means the objective is: not complete\n"
               "The next step is to click the button")
        with self.assertRaises(ObservationError):
            parse_observation(txt)

    def test_missing_objective_raises(self):
        with self.assertRaises(ObservationError):
            parse_observation("On the screen, I see: things\n"
                              "This means the objective is: complete")


class TestPredictionHeld(unittest.TestCase):
    def test_prediction_confirmed_next_step(self):
        prev = parse_observation(_GOOD)
        nxt = "a plane selection dialog is now open with XY, XZ, YZ options"
        self.assertTrue(prediction_held(prev, nxt))

    def test_prediction_not_confirmed(self):
        prev = parse_observation(_GOOD)
        self.assertFalse(prediction_held(prev, "nothing changed, same toolbar"))

    def test_none_when_no_prediction(self):
        done = parse_observation(_DONE)
        self.assertIsNone(prediction_held(done, "whatever"))


if __name__ == "__main__":
    unittest.main()
