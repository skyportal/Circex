"""Tests for fine-tuning dataset construction."""

from __future__ import annotations

import json

from circex.schema import CircularExtraction, Event, ExtractionMeta, Redshift
from circex.train import to_example, vidushi_examples, write_jsonl
from circex.train.dataset import _has_signal


def _meta() -> ExtractionMeta:
    return ExtractionMeta(extractor="gold")


def test_to_example_is_chat_format_with_target_json() -> None:
    ex = CircularExtraction(
        circular_id=1,
        event=Event(event_name="GRB 010101A"),
        redshift=Redshift(redshift=0.5),
        extraction_meta=_meta(),
    )
    example = to_example("We observed GRB 010101A, z = 0.5.", ex)
    assert example["messages"][0]["role"] == "user"
    assert "GRB 010101A" in example["messages"][0]["content"]
    completion = json.loads(example["messages"][1]["content"])
    assert completion["event"]["event_name"] == "GRB 010101A"
    assert completion["redshift"]["redshift"] == 0.5
    # bookkeeping keys are not training targets
    assert "circular_id" not in completion and "provenance" not in completion


def test_has_signal_filters_empty_labels() -> None:
    empty = CircularExtraction(circular_id=9, extraction_meta=_meta())
    assert not _has_signal(empty)
    populated = CircularExtraction(
        circular_id=9, event=Event(event_name="GRB 010101A"), extraction_meta=_meta()
    )
    assert _has_signal(populated)


def test_write_jsonl_deterministic_split(tmp_path) -> None:
    examples = [
        {"messages": [{"role": "user", "content": str(i)}, {"role": "assistant", "content": "{}"}]}
        for i in range(10)
    ]
    n_train, n_val = write_jsonl(examples, tmp_path, val_every=5)
    assert (n_train, n_val) == (8, 2)  # i=0,5 -> val
    assert (tmp_path / "train.jsonl").exists() and (tmp_path / "val.jsonl").exists()
    assert sum(1 for _ in (tmp_path / "train.jsonl").open()) == 8


def test_vidushi_examples_from_rows() -> None:
    from circex.data.swift_gold import SwiftEvaluationRow

    row = SwiftEvaluationRow(
        circular_id=1,
        text="GRB 091018: VLT observations yield z = 0.971.",
        circular_date="2009",
        actual_redshift=0.971,
        actual_grb_number="091018",
        actual_telescope="VLT",
        actual_redshift_type=None,
        predicted_redshift=None,
        predicted_grb_number=None,
        predicted_telescope=None,
        predicted_redshift_type=None,
    )
    examples = list(vidushi_examples([row]))
    assert len(examples) == 1
    completion = json.loads(examples[0]["messages"][1]["content"])
    assert completion["redshift"]["redshift"] == 0.971


def test_vidushi_examples_skips_empty_text() -> None:
    from circex.data.swift_gold import SwiftEvaluationRow

    row = SwiftEvaluationRow(
        circular_id=2,
        text="   ",
        circular_date="2009",
        actual_redshift=0.5,
        actual_grb_number="000000",
        actual_telescope=None,
        actual_redshift_type=None,
        predicted_redshift=None,
        predicted_grb_number=None,
        predicted_telescope=None,
        predicted_redshift_type=None,
    )
    assert list(vidushi_examples([row])) == []


def test_label_dir_examples_reads_labels_and_bodies(tmp_path) -> None:
    from circex.schema import Event, PhotometryExt
    from circex.train import label_dir_examples

    ex = CircularExtraction(
        circular_id=5,
        event=Event(event_name="GRB 010101A"),
        photometry=[PhotometryExt(filter="r", mag=19.0, obs_mjd=60000.0)],
        extraction_meta=_meta(),
    )
    (tmp_path / "000005.label.json").write_text(ex.model_dump_json())
    examples = list(
        label_dir_examples(tmp_path, lambda cid: "GRB 010101A r = 19." if cid == 5 else None)
    )
    assert len(examples) == 1
    completion = json.loads(examples[0]["messages"][1]["content"])
    assert completion["photometry"][0]["filter"] == "r"
