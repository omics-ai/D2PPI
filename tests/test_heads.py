"""Released heads: files intact, loading, symmetry, and exact scores on a small fixture from the Bernett test set."""

import hashlib
import os

import numpy as np
import pytest
import torch

from d2ppi.data import Scorer, Table
from d2ppi.heads import BENCHMARKS, CKPT_DIR, ENCODERS, load_head

HERE = os.path.dirname(os.path.abspath(__file__))
HEADS = [(b, e) for b in BENCHMARKS for e in ENCODERS]


def test_checksums():
    with open(os.path.join(CKPT_DIR, "SHA256SUMS")) as f:
        for line in f:
            digest, name = line.split()
            with open(os.path.join(CKPT_DIR, name), "rb") as g:
                assert hashlib.sha256(g.read()).hexdigest() == digest, name


@pytest.mark.parametrize("bench,enc", HEADS)
def test_symmetric(bench, enc):
    torch.manual_seed(0)
    model, mean, std, meta = load_head(bench, enc)
    assert meta["benchmark"] == bench and meta["encoder"] == enc
    D = ENCODERS[enc]["dim"]
    d1, d2 = torch.randn(4, 5, D), torch.randn(4, 3, D)
    m1 = torch.tensor([[1, 1, 1, 0, 0]] * 4, dtype=torch.bool)
    m2 = torch.tensor([[1, 1, 0]] * 4, dtype=torch.bool)
    g1, g2 = torch.randn(4, D), torch.randn(4, D)
    with torch.no_grad():
        ab, ba = model(d1, m1, g1, d2, m2, g2), model(d2, m2, g2, d1, m1, g1)
    assert torch.allclose(ab, ba, atol=1e-5)


def test_bernett_example_scores():
    """the ESM-2-650M Bernett head on 4 test pairs (embeddings copied from the Zenodo file) = the paper's run."""
    torch.backends.cuda.matmul.allow_tf32 = False
    table = Table(os.path.join(HERE, "data", "bernett_example_650m.npz"))
    expected = {
        "O43638\tP55265": 0.836088061,
        "P61313\tQ8N8A6": 0.872331798,
        "Q9NUL7\tQ8IVS2": 0.97747314,
        "Q5BKZ1\tO95711": 0.0716691092,
    }
    pairs = [k.split("\t") for k in expected]
    model, mean, std, _ = load_head("bernett", "esm2_650m")
    p = Scorer(table, mean, std, "cpu").score(model, [a for a, _ in pairs], [b for _, b in pairs])
    np.testing.assert_allclose(p, list(expected.values()), atol=1e-5)
