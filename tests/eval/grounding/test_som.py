"""Tests for the Set-of-Marks numbered element list (model-free grounding adaptor)."""

import unittest

from harnesscad.eval.grounding.som import (
    BBox, Mark, SOM_TOOL_DESCRIPTION, SetOfMarks,
)


class _Elem:
    """A minimal stand-in for harnesscad.io.cua.uia.Element."""

    def __init__(self, rect, name="", control_type="ButtonControl", enabled=True):
        self.rect = rect
        self.name = name
        self.control_type = control_type
        self.enabled = enabled


class TestBBox(unittest.TestCase):
    def test_center_and_area(self):
        b = BBox(0, 0, 10, 4)
        self.assertEqual(b.center, (5, 2))
        self.assertEqual(b.area, 40)

    def test_from_rect(self):
        self.assertEqual(BBox.from_rect((1, 2, 3, 4)), BBox(1, 2, 3, 4))

    def test_degenerate_rejected(self):
        with self.assertRaises(ValueError):
            BBox(10, 0, 5, 5)

    def test_to_dict(self):
        self.assertEqual(BBox(1, 2, 3, 4).to_dict(),
                         {"x1": 1, "y1": 2, "x2": 3, "y2": 4})


class TestMark(unittest.TestCase):
    def test_center_and_dict(self):
        m = Mark(id=3, bbox=BBox(0, 0, 8, 8), label="Pad", kind="ButtonControl")
        self.assertEqual(m.center, (4, 4))
        d = m.to_dict()
        self.assertEqual(d["id"], 3)
        self.assertEqual(d["center"], [4, 4])


class TestFromBoxes(unittest.TestCase):
    def test_numbering_is_reading_order_independent_of_input_order(self):
        # Same three boxes, two different input orders -> same id/label mapping.
        boxes = [
            {"rect": (0, 100, 10, 110), "label": "bottom"},
            {"rect": (0, 0, 10, 10), "label": "top"},
            {"rect": (50, 0, 60, 10), "label": "top-right"},
        ]
        a = SetOfMarks.from_boxes(boxes)
        b = SetOfMarks.from_boxes(list(reversed(boxes)))
        self.assertEqual([m.label for m in a.marks], [m.label for m in b.marks])
        # top-to-bottom then left-to-right.
        self.assertEqual([m.label for m in a.marks],
                         ["top", "top-right", "bottom"])

    def test_ids_start_at_one_and_are_contiguous(self):
        boxes = [{"rect": (0, 0, 5, 5)}, {"rect": (0, 10, 5, 15)}]
        som = SetOfMarks.from_boxes(boxes)
        self.assertEqual([m.id for m in som.marks], [1, 2])

    def test_accepts_bbox_and_dict_forms(self):
        boxes = [
            {"bbox": BBox(0, 0, 4, 4), "label": "a"},
            {"bbox": {"x1": 0, "y1": 10, "x2": 4, "y2": 14}, "label": "b"},
        ]
        som = SetOfMarks.from_boxes(boxes)
        self.assertEqual(len(som), 2)


class TestFromElements(unittest.TestCase):
    def test_clickable_only_filters_noninteractive_and_zero_area(self):
        elems = [
            _Elem((0, 0, 10, 10), "Pad", "ButtonControl"),
            _Elem((0, 20, 10, 30), "label", "TextControl"),        # not interactable
            _Elem((0, 40, 0, 40), "zero", "ButtonControl"),         # zero area
            _Elem((0, 60, 10, 70), "disabled", "ButtonControl", enabled=False),
        ]
        som = SetOfMarks.from_elements(elems)
        self.assertEqual([m.label for m in som.marks], ["Pad"])
        self.assertEqual(som.marks[0].source, "a11y")

    def test_without_clickable_only_keeps_all_sized(self):
        elems = [_Elem((0, 0, 10, 10), "Pad", "ButtonControl"),
                 _Elem((0, 20, 10, 30), "label", "TextControl")]
        som = SetOfMarks.from_elements(elems, clickable_only=False)
        self.assertEqual(len(som), 2)


class TestLookups(unittest.TestCase):
    def setUp(self):
        self.som = SetOfMarks.from_boxes([
            {"rect": (0, 0, 10, 10), "label": "Pad", "kind": "ButtonControl"},
            {"rect": (0, 20, 20, 40), "label": "Length field", "kind": "EditControl"},
        ])

    def test_id2xy_maps_id_to_center(self):
        mapping = self.som.id2xy()
        self.assertEqual(mapping[1], (5, 5))
        self.assertEqual(mapping[2], (10, 30))

    def test_center_of_and_get(self):
        self.assertEqual(self.som.center_of(1), (5, 5))
        self.assertIsNone(self.som.center_of(99))
        self.assertEqual(self.som.get(2).label, "Length field")

    def test_find_substring_and_exact(self):
        self.assertEqual(self.som.find("length").id, 2)
        self.assertIsNone(self.som.find("Length", exact=True))
        self.assertEqual(self.som.find("Pad", exact=True).id, 1)

    def test_element_list_omits_pixels(self):
        lst = self.som.element_list()
        self.assertEqual(set(lst[0]), {"id", "label", "kind"})

    def test_to_dict_carries_description_and_id2xy(self):
        d = self.som.to_dict()
        self.assertEqual(d["description"], SOM_TOOL_DESCRIPTION)
        self.assertEqual(d["id2xy"]["1"], [5, 5])


if __name__ == "__main__":
    unittest.main()
