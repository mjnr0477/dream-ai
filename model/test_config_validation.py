import unittest

from dream_transformer import DreamConfig


class DreamConfigBooleanValidationTests(unittest.TestCase):
    def test_boolean_dimensions_are_rejected(self):
        for field in ("vocab_size", "block_size", "d_model", "n_heads", "n_layers"):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    DreamConfig(**{field: True})

    def test_boolean_dropout_is_rejected(self):
        for value in (True, False):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    DreamConfig(dropout=value)


if __name__ == "__main__":
    unittest.main()
