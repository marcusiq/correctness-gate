import pytest

from correctness_gate.items import Item, fingerprint, normalize_gold


def test_letter_labels():
    assert normalize_gold(["A", "B", "C", "D"], "B") == 1


def test_numeric_labels():
    assert normalize_gold(["1", "2", "3", "4"], "2") == 1


def test_unknown_label_raises():
    with pytest.raises(ValueError):
        normalize_gold(["A", "B"], "3")


def _item(q="q", gold=0):
    return Item(qid="x", question=q, choices=["a", "b"], gold=gold)


def test_fingerprint_changes_with_content():
    assert fingerprint([_item()]) != fingerprint([_item(gold=1)])


def test_fingerprint_is_order_sensitive():
    a, b = _item(q="one"), _item(q="two")
    assert fingerprint([a, b]) != fingerprint([b, a])