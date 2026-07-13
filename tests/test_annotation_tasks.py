import unittest

from harnesscad.data.dataengine.annotation.tasks import decompose_annotation


class AnnotationTaskTests(unittest.TestCase):
    def test_decomposes_expert_and_nonexpert_tasks(self):
        tasks = decompose_annotation(
            "part-1",
            {"thumbnail", "geometry", "material", "op_stream", "requirements"},
        )
        self.assertEqual(tasks[0].task_id, "part-1:01:family")
        self.assertTrue(any(task.requires_expert for task in tasks))
        self.assertTrue(any(not task.requires_expert for task in tasks))
        self.assertEqual(len({task.task_id for task in tasks}), len(tasks))

    def test_omits_tasks_without_evidence(self):
        self.assertEqual(decompose_annotation("p", set()), [])
