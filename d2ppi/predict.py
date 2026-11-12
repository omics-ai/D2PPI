"""Score your own protein pairs from their sequences.

    # one pair
    python -m d2ppi.predict --seq-a MKTAYIAKQR... --seq-b MSEQNNTEMT...

    # many pairs: a FASTA of the proteins and a TSV/CSV of id pairs (columns id1/id2, or else the first two)
    python -m d2ppi.predict --fasta proteins.fasta --pairs pairs.tsv --out scores.tsv

Each protein is cut into domains (RecCut on the ESM-2-150M contact map, d2ppi.segment), the domains are embedded
by the chosen encoder (d2ppi.embed), and every requested head scores the pair. By default all three heads report,
one column each:

  p_bernett_human     trained on human pairs of the Bernett benchmark, 50 % positives
  p_dscript_xspecies  trained on human STRING pairs (D-SCRIPT benchmark), 1 positive : 10 negatives
  p_d2ppi_xspecies    trained on human STRING v12.5 pairs (LF-XS benchmark), 1 positive : 10 negatives

The probabilities reflect those training priors: the same pair scores higher under the Bernett head. Compare
scores within one head; do not mix heads. Each head segments proteins with the window its training domains were
made with (500 aa for Bernett, 1,022 aa for the cross-species heads), so a protein longer than 500 aa can have
different domains under the two.
"""

import argparse
import os
import sys

import torch

from d2ppi.data import Scorer, Table, read_fasta, read_id_pairs, write_domain_file
from d2ppi.embed import Encoder, embed_proteins
from d2ppi.heads import BENCHMARKS, ENCODERS, benchmark_name, load_head
from d2ppi.segment import ContactPredictor, segment

AMINO = set("ACDEFGHIKLMNPQRSTVWYXBZUO")


def clean_sequence(pid, s):
    s = "".join(s.split()).upper().rstrip("*")
    bad = sorted(set(s) - AMINO)
    if bad:
        raise SystemExit(f"{pid}: sequence contains non-amino-acid characters {bad}")
    if not s:
        raise SystemExit(f"{pid}: empty sequence")
    return s


def load_inputs(a):
    if a.seq_a or a.seq_b:
        if not (a.seq_a and a.seq_b):
            raise SystemExit("give both --seq-a and --seq-b")
        seqs = {a.id_a: a.seq_a, a.id_b: a.seq_b}
        if a.id_a == a.id_b:
            raise SystemExit("--id-a and --id-b must differ")
        pairs = [(a.id_a, a.id_b)]
    elif a.fasta:
        seqs = read_fasta(a.fasta)
        if a.pairs:
            pairs = read_id_pairs(a.pairs, seqs)
        elif len(seqs) == 2:
            pairs = [tuple(seqs)]
        else:
            raise SystemExit(f"{a.fasta} holds {len(seqs)} sequences: give --pairs (or a FASTA of exactly two)")
        missing = sorted({p for pr in pairs for p in pr} - set(seqs))
        if missing:
            raise SystemExit(f"{len(missing)} ids of the pairs are not in {a.fasta}, e.g. {missing[:3]}")
    else:
        raise SystemExit("give --seq-a/--seq-b, or --fasta (+ --pairs)")
    used = {p for pr in pairs for p in pr}
    return {p: clean_sequence(p, s) for p, s in seqs.items() if p in used}, pairs


def segment_all(seqs, windows, device):
    """{window: {pid: domains}} for every window. A protein no longer than a window is not windowed, so it is
    segmented once, under the largest window, and shared with the smaller ones."""
    predictor = ContactPredictor(device=device)  # ESM-2-150M, loaded once for every window
    out = {}
    for w in sorted(windows, reverse=True):
        d = segment({p: s for p, s in seqs.items() if not out or len(s) > w}, maxlen=w, predictor=predictor)
        out[w] = {p: d[p] if p in d else out[max(out)][p] for p in seqs}
    del predictor
    return out


def main():
    ap = argparse.ArgumentParser(description="Score protein pairs with D2PPI")
    ap.add_argument("--seq-a", help="sequence of protein A")
    ap.add_argument("--seq-b", help="sequence of protein B")
    ap.add_argument("--id-a", default="A")
    ap.add_argument("--id-b", default="B")
    ap.add_argument("--fasta", help="FASTA of the proteins")
    ap.add_argument("--pairs", help="TSV/CSV of protein-id pairs (columns id1, id2, or the first two; the header "
                    "line is optional)")  # fmt: skip
    ap.add_argument("--encoder", default="esm2_650m", choices=list(ENCODERS),
                    help="esm2_650m (default; ~9 GB GPU) or esm2_15b (~40 GB GPU)")  # fmt: skip
    ap.add_argument("--heads", nargs="+", default=["all"], help="all (default), or any of bernett dscript lfxs")
    ap.add_argument("--out", help="output TSV (default: print to stdout)")
    ap.add_argument("--domains-out", help="also write each protein's domains (TSV) here; heads that segment with "
                    "different windows get one file each, <name>.window<W>.tsv")  # fmt: skip
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    if a.device == "cpu" and a.encoder == "esm2_15b":
        print("[predict] warning: ESM-2-15B on the CPU needs ~60 GB of RAM and is very slow", file=sys.stderr)

    heads = list(BENCHMARKS) if a.heads == ["all"] else list(dict.fromkeys(benchmark_name(h) for h in a.heads))
    seqs, pairs = load_inputs(a)
    print(f"[predict] {len(pairs)} pair(s), {len(seqs)} protein(s); encoder {a.encoder}; heads {', '.join(heads)}",
          file=sys.stderr, flush=True)  # fmt: skip

    # 1. domains, once per segmentation window the heads need
    windows = {BENCHMARKS[h]["segment_maxlen"] for h in heads}
    doms = segment_all(seqs, windows, a.device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. embeddings: one per distinct (protein, domain set)
    keys, todo = {}, {}
    for w in windows:
        for p, d in doms[w].items():
            k = (p, tuple(tuple(r) for r in d))
            keys[(w, p)] = todo.setdefault(k, f"{p}\t{len(todo)}")
    enc = Encoder(a.encoder, a.device)
    res, _ = embed_proteins(enc, [(name, seqs[k[0]], [list(r) for r in k[1]]) for k, name in todo.items()])
    del enc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3. scores
    cols = {}
    for h in heads:
        w = BENCHMARKS[h]["segment_maxlen"]
        table = Table.from_embeddings({p: res[keys[(w, p)]] for p in seqs})
        model, mean, std, _meta = load_head(h, a.encoder, a.device)
        cols[h] = Scorer(table, mean, std, a.device).score(model, [x for x, _ in pairs], [y for _, y in pairs])

    if a.domains_out:
        base, ext = os.path.splitext(a.domains_out)
        for w in sorted(windows):  # heads with different segmentation windows: one file per window
            path = a.domains_out if len(windows) == 1 else f"{base}.window{w}{ext or '.tsv'}"
            write_domain_file(path, doms[w])
            print(f"[predict] wrote domains to {path}", file=sys.stderr)

    lines = ["id_a\tid_b\tlen_a\tlen_b\t" + "\t".join(f"p_{h}" for h in heads)]
    for k, (x, y) in enumerate(pairs):
        lines.append(f"{x}\t{y}\t{len(seqs[x])}\t{len(seqs[y])}\t" + "\t".join(f"{cols[h][k]:.4f}" for h in heads))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[predict] wrote {a.out}", file=sys.stderr)
    if not a.out or len(pairs) <= 5:  # a readable summary for small runs
        for k, (x, y) in enumerate(pairs[:5]):
            print(f"\n{x} ({len(seqs[x])} aa) x {y} ({len(seqs[y])} aa)")
            for h in heads:
                print(f"  P(interact) {h:<17} {cols[h][k]:.4f}")
        if not a.out:
            print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
