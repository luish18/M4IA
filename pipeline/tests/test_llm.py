import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "pipeline") not in sys.path:
    sys.path.insert(0, str(ROOT / "pipeline"))

import llm  # noqa: E402


class TaskEncodingTests(unittest.TestCase):

    def test_answer_is_reversed_and_terminated(self):
        seq = llm.encode(7, 48)
        self.assertEqual(len(seq), llm.SEQ_LEN)
        self.assertEqual(seq[:llm.PROMPT_LEN], [0, 7, llm.PLUS, 4, 8, llm.EQUALS])
        self.assertEqual(seq[llm.PROMPT_LEN:], [5, 5, 0, llm.END])
        self.assertEqual(llm.decode_answer(seq[llm.PROMPT_LEN:]), 55)

    def test_malformed_answers_decode_to_none(self):
        self.assertIsNone(llm.decode_answer([5, 5, 0, 0]))
        self.assertIsNone(llm.decode_answer([5, llm.PLUS, 0, llm.END]))
        self.assertIsNone(llm.decode_answer([5, 5, llm.END]))


class Fp32TypingTests(unittest.TestCase):
    """Deeploy types inputs from sample values; narrower types disable clusters."""

    def _rejects(self, values):
        with self.assertRaises(SystemExit):
            llm.check_fp32_typing("t", np.asarray(values, dtype = np.float32))

    def test_integer_valued_sample_is_rejected(self):
        self._rejects([0.0, -1e9, 0.0])

    def test_bfloat16_exact_sample_is_rejected(self):
        # The original sel sample: 0, 1 and 0.5 all fit bfloat16.
        self._rejects([0.0, 1.0, 0.5, 0.5])

    def test_decode_inputs_type_as_fp32(self):
        state = llm.DecodeState(np.zeros((llm.VOCAB, llm.D_MODEL), np.float32) + 0.1,
                                np.zeros((llm.CTX, llm.D_MODEL), np.float32))
        feed = state.feed(3, 4)
        llm.check_fp32_typing("mask", feed["mask"])
        llm.check_fp32_typing("sel", feed["sel"])
        self.assertEqual(feed["sel"][0, 4], 1.0)
        self.assertTrue(np.all(feed["sel"][0, :4] == 0.0))
        self.assertTrue(np.all(feed["mask"][0, :5] == 0.0))


class ModelTests(unittest.TestCase):

    def test_backward_matches_central_differences(self):
        llm.gradcheck(np.random.default_rng(1))

    def test_decode_graph_matches_full_sequence_model(self):
        import onnxruntime as ort

        rng = np.random.default_rng(2)
        params = llm.init_params(rng)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "network.onnx"
            llm.export_onnx(params, path)
            sess = ort.InferenceSession(str(path), providers = ["CPUExecutionProvider"])
        tok = params["tok"].astype(np.float32)
        pos = params["pos"].astype(np.float32)
        prompt = llm.encode(31, 69)[:llm.PROMPT_LEN]
        answer, logits, _ = llm.ort_decode(sess, tok, pos, prompt)
        full, _ = llm.forward(params, np.array([prompt + answer[:-1]]))
        self.assertLess(np.abs(full[0] - logits).max(), 1e-3)

        # The codegen sample has to be self-consistent: running the graph on the
        # captured inputs must reproduce the captured outputs. It was not once,
        # because the cache inputs were views later steps wrote into.
        _, _, (feed, out) = llm.ort_decode(sess, tok, pos, prompt, capture = llm.PROMPT_LEN)
        again = sess.run(None, feed)
        for got, want in zip(again, out):
            np.testing.assert_allclose(got, want, rtol = 0, atol = 1e-6)


if __name__ == "__main__":
    unittest.main()
