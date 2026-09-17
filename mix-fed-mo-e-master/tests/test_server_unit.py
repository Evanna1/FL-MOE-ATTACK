from __future__ import annotations

from torch.utils.data import DataLoader

from mixfedmoe_fl.server import _make_eval_loader


def test_make_eval_loader_returns_none_for_missing_optional_dataset() -> None:
    assert _make_eval_loader(None, batch_size=2, collator=lambda rows: rows) is None  # type: ignore[arg-type]


def test_make_eval_loader_builds_loader_for_present_dataset() -> None:
    loader = _make_eval_loader(
        dataset=[{"value": 1}, {"value": 2}],
        batch_size=2,
        collator=lambda rows: rows,  # type: ignore[arg-type]
    )
    assert isinstance(loader, DataLoader)
    assert next(iter(loader)) == [{"value": 1}, {"value": 2}]
