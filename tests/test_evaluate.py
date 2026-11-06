"""Evaluation handles a split with no pairs in its long-protein subsets."""

import gzip
import json
import sys
from pathlib import Path

import numpy as np

from d2ppi.evaluate import main

DATA = Path(__file__).parent / "data"


def test_evaluate_skips_empty_subsets(tmp_path, monkeypatch, capsys):
    with np.load(DATA / "bernett_example_650m.npz", allow_pickle=False) as fixture:
        pids = fixture["pids"].tolist()
    pairs = list(zip(pids[::2], pids[1::2], strict=True))
    benchmark = tmp_path / "bernett_human"
    benchmark.mkdir()
    tsv = "id1\tid2\tlabel\n" + "".join(f"{a}\t{b}\t{i % 2}\n" for i, (a, b) in enumerate(pairs))
    (benchmark / "test.tsv").write_text(tsv)
    out = tmp_path / "results"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "d2ppi.evaluate",
            "--benchmark",
            "bernett",
            "--data",
            str(tmp_path),
            "--emb",
            str(DATA / "bernett_example_650m.npz"),
            "--out",
            str(out),
            "--device",
            "cpu",
            "--save-probs",
        ],
    )

    main()

    result = json.loads((out / "bernett_human_esm2_650m_test.json").read_text())
    assert set(result["metrics"]) == {"test"}
    assert result["metrics"]["test"]["n"] == len(pairs)
    stderr = capsys.readouterr().err
    assert "skipping test_long: no pairs" in stderr
    assert "skipping both_long: no pairs" in stderr
    with gzip.open(out / "bernett_human_esm2_650m_test_probs.tsv.gz", "rt") as f:
        assert len(f.read().splitlines()) == len(pairs) + 1
