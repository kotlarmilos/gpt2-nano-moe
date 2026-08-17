from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from gpt2_nano_moe.train import _corpus_batch, _load_corpus


def _write_shard(path: Path, tokens: np.ndarray) -> None:
    header = np.zeros(256, dtype="<i4")
    header[0] = 20250320
    header[1] = 1
    header[2] = tokens.size
    path.write_bytes(header.tobytes() + tokens.astype("<u2").tobytes())


class TrainingDataTests(unittest.TestCase):
    def test_uint16_corpus_is_split_and_sampled_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokens.bin"
            _write_shard(path, np.arange(40, dtype="<u2"))
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
            _write_shard(path, np.arange(20, dtype="<u2"))
            with self.assertRaisesRegex(ValueError, "vocabulary"):
                _load_corpus(
                    path,
                    vocab_size=10,
                    context_length=4,
                    validation_fraction=0.25,
                )


if __name__ == "__main__":
    unittest.main()
