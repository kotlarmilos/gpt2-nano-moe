from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from gpt2_nano_moe.train import _corpus_batch, _load_corpus


class TrainingDataTests(unittest.TestCase):
    def test_uint16_corpus_is_split_and_sampled_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokens.bin"
            np.arange(40, dtype="<u2").tofile(path)
            train, validation = _load_corpus(
                path,
                vocab_size=64,
                context_length=8,
                validation_fraction=0.25,
            )
            first = _corpus_batch(
                train,
                batch_size=2,
                context_length=8,
                generator=torch.Generator().manual_seed(3),
                device=torch.device("cpu"),
            )
            second = _corpus_batch(
                train,
                batch_size=2,
                context_length=8,
                generator=torch.Generator().manual_seed(3),
                device=torch.device("cpu"),
            )

        self.assertEqual(train.size, 30)
        self.assertEqual(validation.size, 10)
        torch.testing.assert_close(first, second)

    def test_out_of_vocabulary_corpus_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokens.bin"
            np.arange(20, dtype="<u2").tofile(path)
            with self.assertRaisesRegex(ValueError, "vocabulary"):
                _load_corpus(
                    path,
                    vocab_size=10,
                    context_length=4,
                    validation_fraction=0.25,
                )


if __name__ == "__main__":
    unittest.main()
