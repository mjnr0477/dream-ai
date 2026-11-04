"""Regression tests for DREAM next-token training-window boundaries."""
import unittest
from unittest.mock import patch

import torch

from train import batches


class TrainingWindowBoundaryTests(unittest.TestCase):
    def test_sampler_includes_final_valid_next_token_window(self):
        # For 10 tokens and a block size of 3, start index 6 is valid:
        # x=[6, 7, 8] and y=[7, 8, 9]. The upper bound must therefore be 7.
        data = torch.arange(10)
        with patch("train.torch.randint", return_value=torch.tensor([6])) as randint:
            x, y = batches(data, batch_size=1, block_size=3, device="cpu")

        randint.assert_called_once_with(7, (1,))
        self.assertEqual(x.tolist(), [[6, 7, 8]])
        self.assertEqual(y.tolist(), [[7, 8, 9]])

    def test_accepts_exactly_one_valid_window(self):
        data = torch.arange(4)
        with patch("train.torch.randint", return_value=torch.tensor([0])) as randint:
            x, y = batches(data, batch_size=1, block_size=3, device="cpu")

        randint.assert_called_once_with(1, (1,))
        self.assertEqual(x.tolist(), [[0, 1, 2]])
        self.assertEqual(y.tolist(), [[1, 2, 3]])


if __name__ == "__main__":
    unittest.main()
