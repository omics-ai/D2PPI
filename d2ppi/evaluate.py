"""Score a benchmark split with a released head and report its metrics.

    python -m d2ppi.evaluate --benchmark bernett --encoder esm2_650m
    python -m d2ppi.evaluate --benchmark lfxs --encoder esm2_15b --emb data/embeddings/d2ppi_xspecies/test_esm2_15b.npz

ESM-2-650M reads the embeddings shipped with the benchmark (data/benchmark/<benchmark>/<split>.npz), so it needs
no language model and runs in minutes. ESM-2-15B needs a table built with d2ppi.embed first (see its --help).

Reported sets: Bernett test, test_long (pairs with len(seq1) + len(seq2) > 2,193) and both_long (both proteins
> 1,022 aa); cross-species: each species and the mean over species. Thresholded metrics use 0.5.
"""

import argparse
import gzip
import json
import os
import sys

import numpy as np
import torch

from d2ppi.data import Scorer, Table, read_pairs
from d2ppi.heads import BENCHMARKS, ENCODERS, benchmark_name, load_head
from d2ppi.metrics import all_metrics

SHOW = ("aupr", "auroc", "f1", "mcc", "precision", "recall", "accuracy")
LONG_PAIR = 2193  # Bernett test_long: combined length above the longest training pair of train_cut_0.8
LONG_PROTEIN = 1022  # Bernett both_long: both proteins longer than the ESM-2 window


def subsets(bench, cols, length):
    """name -> boolean mask over the rows of the split."""
    n = len(cols["label"])
    sets = {}
    if "species" in cols:
        sp = np.array(cols["species"])
        for s in dict.fromkeys(cols["species"]):
            sets[s] = sp == s
    else:
        sets["test" if bench == "bernett_human" else "all"] = np.ones(n, bool)
        if bench == "bernett_human" and length is not None:
            la, lb = length
            sets["test_long"] = la + lb > LONG_PAIR
            sets["both_long"] = (la > LONG_PROTEIN) & (lb > LONG_PROTEIN)
    return sets


def main():
    ap = argparse.ArgumentParser(description="Evaluate a released D2PPI head on a benchmark split")
    ap.add_argument("--benchmark", required=True, help="bernett | dscript | lfxs (or bernett_human, ...)")
    ap.add_argument("--encoder", default="esm2_650m", choices=list(ENCODERS))
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--data", default="data/benchmark", help="folder of the downloaded benchmarks")
    ap.add_argument("--emb", help="embedding table (default for esm2_650m: the benchmark's own <split>.npz)")
    ap.add_argument("--out", default="results", help="folder for <benchmark>_<encoder>_<split>.json")
    ap.add_argument("--save-probs", action="store_true", help="also write the per-pair probabilities (.tsv.gz)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=1024)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # IEEE fp32, as in training

    bench = benchmark_name(a.benchmark)
    tsv = os.path.join(a.data, bench, f"{a.split}.tsv")
    if not os.path.exists(tsv):
        raise SystemExit(f"{tsv} not found: download the benchmark first (python -m d2ppi.download)")
    emb = a.emb or (os.path.join(a.data, bench, f"{a.split}.npz") if a.encoder == "esm2_650m" else None)
    if emb is None:
        raise SystemExit(
            f"--emb is required for {a.encoder}: the benchmark ships ESM-2-650M embeddings only. Build the table "
            f"with: python -m d2ppi.embed --pairs {tsv} --domains-from {tsv[:-4]}.npz --encoder {a.encoder} "
            f"--out data/embeddings/{bench}/{a.split}_{a.encoder}.npz"
        )

    cols = read_pairs(tsv, ["id1", "id2", "label", "species"])
    y = np.array(cols["label"], dtype=int)
    print(f"[evaluate] {bench} {a.split}: {len(y):,} pairs ({y.sum():,} positive); head {a.encoder}; table {emb}",
          file=sys.stderr, flush=True)  # fmt: skip
    table = Table(emb, pids=set(cols["id1"]) | set(cols["id2"]))
    model, mean, std, meta = load_head(bench, a.encoder, a.device)
    scorer = Scorer(table, mean, std, a.device)
    length = None
    if table.length is not None:
        ln = np.asarray(table.length)
        length = (ln[[table.index[x] for x in cols["id1"]]], ln[[table.index[x] for x in cols["id2"]]])
    sets = subsets(bench, cols, length)
    # score each species on its own, so the batches (and so the last float bits) match the paper's runs
    groups = list(sets.values()) if "species" in cols else [np.ones(len(y), bool)]
    p = np.zeros(len(y), np.float32)
    for m in groups:
        ix = np.flatnonzero(m)
        p[ix] = scorer.score(model, [cols["id1"][i] for i in ix], [cols["id2"][i] for i in ix], a.batch)
    res = {}
    for n, m in sets.items():
        if not m.any():
            print(f"[evaluate] skipping {n}: no pairs in this subset", file=sys.stderr)
            continue
        res[n] = all_metrics(y[m], p[m])
    if "species" in cols:
        sp = list(res)
        res["mean"] = {k: float(np.mean([res[s][k] for s in sp])) for k in SHOW}

    head = f"{'set':<12}" + "".join(f"{k:>10}" for k in SHOW) + f"{'pairs':>11}"
    print(f"\n{BENCHMARKS[bench]['title']} · {a.encoder} head · {a.split}\n{head}")
    for n, m in res.items():
        print(f"{n:<12}" + "".join(f"{m[k]:>10.4f}" for k in SHOW) + (f"{m['n']:>11,}" if "n" in m else ""))

    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, f"{bench}_{a.encoder}_{a.split}.json")
    info = {"benchmark": bench, "encoder": a.encoder, "split": a.split, "table": emb, "head": meta.get("name")}
    with open(out, "w") as f:
        json.dump({**info, "metrics": res}, f, indent=1)
    print(f"\n[evaluate] wrote {out}", file=sys.stderr)
    if a.save_probs:
        pp = out[:-5] + "_probs.tsv.gz"
        with gzip.open(pp, "wt") as f:
            extra = "species" in cols
            f.write("id1\tid2\tlabel" + ("\tspecies" if extra else "") + "\tprob\n")
            for k in range(len(y)):
                sp = f"\t{cols['species'][k]}" if extra else ""
                f.write(f"{cols['id1'][k]}\t{cols['id2'][k]}\t{y[k]}{sp}\t{p[k]:.9g}\n")
        print(f"[evaluate] wrote {pp}", file=sys.stderr)


if __name__ == "__main__":
    main()
