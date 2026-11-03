from dream_transformer import DreamConfig


def test_boolean_dimensions_are_rejected():
    for field in ("block_size", "d_model", "n_heads", "n_layers"):
        try:
            DreamConfig(**{field: True})
        except ValueError:
            continue
        raise AssertionError(field)


def test_boolean_dropout_is_rejected():
    for value in (True, False):
        try:
            DreamConfig(dropout=value)
        except ValueError:
            continue
        raise AssertionError(value)
