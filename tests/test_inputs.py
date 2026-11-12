"""Pair files for d2ppi.predict: named columns, the first two columns, and files without a header line."""

import pytest

from d2ppi.data import read_id_pairs

KNOWN = {"P1", "P2", "P3"}
PAIRS = [("P1", "P2"), ("P3", "P1"), ("P2", "P2")]


def write(tmp_path, name, lines):
    path = tmp_path / name
    path.write_text("".join(line + "\n" for line in lines))
    return str(path)


@pytest.mark.parametrize(
    "name,lines",
    [
        ("named.tsv", ["label\tid2\tid1", "1\tP2\tP1", "0\tP1\tP3", "1\tP2\tP2"]),
        ("header.tsv", ["protein_a\tprotein_b\tlabel", "P1\tP2\t1", "P3\tP1\t0", "P2\tP2\t1"]),
        ("header.csv", ["a,b", "P1,P2", "P3,P1", "P2,P2"]),
        ("no_header.tsv", ["P1\tP2", "P3\tP1", "", "P2\tP2"]),
        ("no_header.csv", ["P1,P2", "P3,P1", "P2,P2"]),
    ],
)
def test_read_id_pairs(tmp_path, name, lines):
    assert read_id_pairs(write(tmp_path, name, lines), KNOWN) == PAIRS


def test_first_line_self_pair_is_kept(tmp_path):
    assert read_id_pairs(write(tmp_path, "self.tsv", ["P2\tP2", "P1\tP3"]), KNOWN) == [("P2", "P2"), ("P1", "P3")]


@pytest.mark.parametrize("lines", [[], ["P1", "P2"], ["P1 P2", "P3 P1"], ["a\tb", "P1\tP2", "P3"]])
def test_needs_two_columns(tmp_path, lines):
    with pytest.raises(SystemExit, match="two protein ids"):
        read_id_pairs(write(tmp_path, "bad.tsv", lines), KNOWN)
