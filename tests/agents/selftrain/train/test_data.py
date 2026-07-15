"""Data formatting is where a fine-tune silently goes wrong: if the training text
does not match the inference prompt, the model optimises a distribution it is never
asked for. These tests assert the message shape and the completion-only masking
seam. They SKIP cleanly when the training stack (transformers/datasets) is absent,
because the core suite must run on a CPU-only box."""

from __future__ import annotations

import unittest

from harnesscad.agents.selftrain import train
from harnesscad.agents.selftrain.train import data as data_mod


class TestMessages(unittest.TestCase):

    def test_inference_messages_are_system_plus_user(self):
        msgs = data_mod.messages_for("a 60x40x5 plate")
        self.assertEqual([m["role"] for m in msgs], ["system", "user"])
        # The system turn must carry the op schema so train and test share it.
        self.assertIn("CISP", msgs[0]["content"])
        self.assertIn("60x40x5 plate", msgs[1]["content"])


@unittest.skipIf(train.MISSING, "training stack absent: " + train.why_unavailable())
class TestTokenisedShape(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        # A small, cached Qwen tokenizer shares the chat template with the 7B.
        for name in ("Qwen/Qwen2.5-Coder-0.5B-Instruct",
                     "Qwen/Qwen2.5-Coder-7B-Instruct"):
            try:
                cls.tok = AutoTokenizer.from_pretrained(name)
                return
            except Exception:  # noqa: BLE001
                continue
        raise unittest.SkipTest("no Qwen tokenizer available offline")

    def test_response_template_ids_nonempty(self):
        ids = data_mod._response_template_ids(self.tok)
        self.assertTrue(ids)
        # The template header must decode to the assistant turn marker.
        decoded = self.tok.decode(ids)
        self.assertIn("assistant", decoded)

    def test_kto_dataset_columns(self):
        import os
        path = "assets/selftrain/kto.jsonl"
        if not os.path.exists(path):
            self.skipTest("kto.jsonl not present")
        ds, stats = data_mod.kto_dataset(path, self.tok)
        self.assertEqual(set(ds.column_names), {"prompt", "completion", "label"})
        self.assertEqual(stats.records, len(ds))
        # Every label is a bool and the desirable count matches the manifest split.
        self.assertTrue(all(isinstance(x, bool) for x in ds["label"]))
        self.assertEqual(stats.desirable + stats.undesirable, stats.records)


if __name__ == "__main__":
    unittest.main()
