"""Embedding tables: shards merged with --merge equal the table of one unsharded run."""

import json

import numpy as np

from d2ppi.data import Table
from d2ppi.embed import merge_tables, write_table

LENGTHS = {"P3": 40, "P1": 25, "P4": 90, "P2": 60}
DOMAINS = {"P1": [[(1, 25)]], "P2": [[(1, 30)], [(31, 60)]], "P3": [[(1, 40)]], "P4": [[(1, 20), (50, 90)], [(21, 49)]]}


def test_merge_tables(tmp_path):
    rng = np.random.default_rng(0)
    seqs = {p: "M" * n for p, n in LENGTHS.items()}
    res = {p: (rng.normal(size=(len(d), 8)), rng.normal(size=8)) for p, d in DOMAINS.items()}

    def write(name, pids, shard):
        path = str(tmp_path / name)
        write_table(path, seqs, {p: res[p] for p in pids}, DOMAINS, {"encoder": "test", "shard": shard})
        return path

    full = write("full.npz", list(seqs), None)
    out = str(tmp_path / "merged" / "table")  # a new folder, and a name without .npz: written exactly there
    merge_tables(out, [write("s0.npz", ["P4", "P1"], "0/2"), write("s1.npz", ["P2", "P3"], "1/2")])

    with np.load(full) as a, np.load(out) as b:
        assert a.files == b.files
        for k in a.files:
            np.testing.assert_array_equal(a[k], b[k], err_msg=k)
        assert json.loads(str(b["meta"])) == {"encoder": "test", "shard": None}
    table = Table(out, pids={"P4", "P2"})
    assert table.pids == ["P2", "P4"] and table.dom.shape == (4, 8) and table.length.tolist() == [60, 90]
