"""Domain embeddings from a frozen ESM-2 encoder (the recipe of the paper and of the Zenodo embedding files).

* A protein of <= 1,022 residues is encoded once as a whole. A domain's embedding is the mean of the last-layer
  residue embeddings over the domain's residues; the whole-protein embedding is the mean over the residues of all
  its domains.
* A longer protein is encoded once per domain, in a 1,022-residue window centred on the domain's span and clamped
  to the sequence ends (windows that coincide are encoded once); the domain embedding averages the domain's own
  residues in that window. The whole-protein embedding is the domain-length-weighted mean of the domain embeddings.

Command line: build an embedding table for the proteins of one or more pair tables, with domains from a Zenodo
embedding file (benchmark proteins) or from a domain file (d2ppi.segment):

    python -m d2ppi.embed --pairs data/benchmark/bernett_human/test.tsv \
        --domains-from data/benchmark/bernett_human/test.npz --encoder esm2_15b \
        --out data/embeddings/bernett_human/test_esm2_15b.npz

Long jobs can be split over GPUs with --shard K/N (one process per GPU) and joined with --merge.
"""

import argparse
import json
import os
import sys
import time
import types

import numpy as np
import torch

from d2ppi.data import clean_domains, domains_from_table, format_ranges, read_domain_file, read_pairs
from d2ppi.heads import ENCODERS

CTX = 1022  # ESM-2 residue window (1,024 tokens minus BOS/EOS)


def _linear_fp32(self, x):
    """nn.Linear.forward for fp32act: fp16-stored weight, fp32 arithmetic."""
    return torch.nn.functional.linear(x, self.weight.float(), None if self.bias is None else self.bias.float())


class Encoder:
    """A Hugging Face ESM-2 model; residues(seqs) -> (B, maxlen + 2, D) last-layer states with BOS at index 0."""

    def __init__(self, encoder="esm2_650m", device="cuda", dtype=None):
        from transformers import AutoTokenizer, EsmModel
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()  # the checkpoints' unused LM head would be reported as UNEXPECTED

        info = ENCODERS.get(encoder, {"hf": encoder, "dtype": "fp32"})
        self.name, self.device = info["hf"], device
        self.dtype = dtype or info["dtype"]
        on_gpu = str(device).startswith("cuda")
        half = self.dtype == "fp32act" and on_gpu
        # half: load straight into fp16 (the 15B fp32 checkpoint would need 60 GB of host RAM). Older transformers
        # call the dtype argument torch_dtype and may lack SDPA attention for ESM; try the variants in turn.
        variants = [{"dtype": torch.float16}, {"torch_dtype": torch.float16}] if half else [{}]
        variants = [{"attn_implementation": "sdpa", **v} for v in variants] + variants
        for i, kw in enumerate(variants):
            try:
                m = EsmModel.from_pretrained(self.name, add_pooling_layer=False, **kw)
                break
            except (TypeError, ValueError):
                if i == len(variants) - 1:
                    raise
        self.tok = AutoTokenizer.from_pretrained(self.name)
        self.model = (m.half() if half else m.float()).to(device).eval()
        self.hidden_size = self.model.config.hidden_size
        if half:  # fp32 activations: fp32 embeddings, LayerNorms and Linear arithmetic, un-rounded RoPE
            self.model.embeddings.float()
            for mod in self.model.modules():
                if isinstance(mod, torch.nn.LayerNorm):
                    mod.float()
                elif isinstance(mod, torch.nn.Linear):
                    mod.forward = types.MethodType(_linear_fp32, mod)
                if hasattr(mod, "inv_freq") and hasattr(mod, "compute_default_rope_parameters"):
                    mod.inv_freq = mod.compute_default_rope_parameters(mod.config)[0].to(device)

    @torch.no_grad()
    def residues(self, seqs):
        enc = self.tok(list(seqs), return_tensors="pt", padding=True, add_special_tokens=True)
        enc = {k: enc[k].to(self.device) for k in ("input_ids", "attention_mask")}
        return self.model(**enc).last_hidden_state.float().cpu()


def pool(emb, ranges):
    """mean of residue embeddings over 1-based inclusive ranges."""
    rows = torch.cat([emb[s - 1 : e] for s, e in ranges], 0)
    return rows.mean(0) if rows.shape[0] > 0 else emb.mean(0)


@torch.no_grad()
def batched(enc, seqs, budget, on_item):
    """length-sorted batches under a token budget; on_item(index, residue embeddings without BOS/EOS)."""
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    i = 0
    while i < len(order):
        batch, longest, j = [], 0, i
        while j < len(order):
            n = len(seqs[order[j]]) + 2
            if batch and max(longest, n) * (len(batch) + 1) > budget:
                break
            batch.append(order[j])
            longest = max(longest, n)
            j += 1
        hs = enc.residues([seqs[k] for k in batch])
        for bi, k in enumerate(batch):
            on_item(k, hs[bi, 1 : 1 + len(seqs[k])])
        i = j


def window_start(ranges, L, ctx=CTX):
    """0-based start of the ctx-residue window centred on a domain's span (clamped to the sequence)."""
    lo, hi = min(s for s, _ in ranges), max(e for _, e in ranges)
    if hi - lo + 1 > ctx and not any(
        min(e, lo + (hi - lo) // 2 + ctx // 2) >= max(s, lo + (hi - lo) // 2 - ctx // 2 + 1) for s, e in ranges
    ):  # a discontinuous domain wider than the window whose centre falls in a gap: centre on its longest segment
        lo, hi = max(ranges, key=lambda r: r[1] - r[0])
    st = int(round((lo + hi) / 2.0 - ctx / 2.0))
    return max(0, min(st, L - ctx))


@torch.no_grad()
def embed_proteins(enc, proteins, budget=16384, log=True):
    """proteins: [(pid, sequence, domains)] with domains as lists of (start, end) ranges (see clean_domains).

    -> {pid: (domain embeddings (k, D) float32, whole-protein embedding (D,) float32)}
    """
    t0 = time.time()
    prots = [(p, s, clean_domains(s, d)) for p, s, d in proteins]
    short = [x for x in prots if len(x[1]) <= CTX]
    long_ = [x for x in prots if len(x[1]) > CTX]
    res = {}

    def on_short(k, emb):
        pid, _seq, doms = short[k]
        dom = torch.stack([pool(emb, r) for r in doms], 0)
        glob = pool(emb, [(s, e) for r in doms for s, e in r])
        res[pid] = (dom.numpy(), glob.numpy())

    batched(enc, [x[1] for x in short], budget, on_short)
    if log and short:
        print(f"[embed] {len(short)} proteins <= {CTX} aa ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)

    # long proteins: one window per distinct domain-centred offset
    wins, owners, key2w = [], [], {}
    for pi, (_pid, seq, doms) in enumerate(long_):
        for di, ranges in enumerate(doms):
            st = window_start(ranges, len(seq))
            if (pi, st) not in key2w:
                key2w[(pi, st)] = len(wins)
                wins.append(seq[st : st + CTX])
                owners.append([])
            owners[key2w[(pi, st)]].append((pi, di, st))
    vecs = [[None] * len(x[2]) for x in long_]

    def on_win(k, emb):
        for pi, di, st in owners[k]:
            rr = [(max(s - st, 1), min(e - st, CTX)) for s, e in long_[pi][2][di]]
            vecs[pi][di] = pool(emb, [(s, e) for s, e in rr if e >= s])

    batched(enc, wins, budget, on_win)
    for pi, (pid, _seq, doms) in enumerate(long_):
        dom = torch.stack(vecs[pi], 0)
        w = torch.tensor([sum(e - s + 1 for s, e in r) for r in doms], dtype=torch.float32).unsqueeze(1)
        res[pid] = (dom.numpy(), ((dom * w).sum(0) / w.sum()).numpy())
    if log and long_:
        print(f"[embed] {len(long_)} proteins > {CTX} aa, {len(wins)} windows ({time.time() - t0:.0f}s)",
              file=sys.stderr, flush=True)  # fmt: skip
    return res, {p: d for p, _s, d in prots}


def write_table(path, seqs, res, doms, meta):
    """write an embedding table (format: d2ppi.data) for the proteins of `res`, sorted by id."""
    pids = sorted(res)
    off = np.concatenate([[0], np.cumsum([res[p][0].shape[0] for p in pids])]).astype(np.int64)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(
        tmp,
        pids=np.array(pids, dtype=str),
        length=np.array([len(seqs[p]) for p in pids], dtype=np.int32),
        dom_packed=np.concatenate([res[p][0] for p in pids], 0).astype(np.float32),
        dom_off=off,
        dom_range=np.array([format_ranges(r) for p in pids for r in doms[p]], dtype=str),
        glob=np.stack([res[p][1] for p in pids], 0).astype(np.float32),
        meta=np.array(json.dumps(meta)),
    )
    os.replace(tmp, path)


def merge_tables(out, parts):
    zs = [np.load(p, allow_pickle=False) for p in parts]
    pids = np.concatenate([z["pids"] for z in zs])
    if len(set(pids.tolist())) != len(pids):
        raise SystemExit("the shards overlap")
    order = np.argsort(pids, kind="stable")
    doms = [np.split(z["dom_packed"], z["dom_off"][1:-1]) for z in zs]
    rngs = [np.split(z["dom_range"], z["dom_off"][1:-1]) for z in zs]
    doms, rngs = [d for x in doms for d in x], [r for x in rngs for r in x]
    off = np.concatenate([[0], np.cumsum([len(doms[i]) for i in order])]).astype(np.int64)
    np.savez(
        out,
        pids=pids[order],
        length=np.concatenate([z["length"] for z in zs])[order],
        dom_packed=np.concatenate([doms[i] for i in order], 0),
        dom_off=off,
        dom_range=np.concatenate([rngs[i] for i in order]),
        glob=np.concatenate([z["glob"] for z in zs])[order],
        meta=zs[0]["meta"],
    )
    print(f"[embed] merged {len(parts)} shards -> {out} ({len(pids)} proteins)", file=sys.stderr)


def read_sequences(pair_files):
    """{pid: sequence} from pair tables with id1, id2, seq1, seq2 columns."""
    seqs = {}
    for path in pair_files:
        c = read_pairs(path, ["id1", "id2", "seq1", "seq2"])
        for a, b in (("id1", "seq1"), ("id2", "seq2")):
            for p, s in zip(c[a], c[b], strict=True):
                if seqs.setdefault(p, s) != s:
                    raise SystemExit(f"{path}: protein {p} appears with two different sequences")
    return seqs


def main():
    ap = argparse.ArgumentParser(description="Build an embedding table for the proteins of pair tables")
    ap.add_argument("--pairs", nargs="+", help="pair tables (TSV with id1 id2 seq1 seq2)")
    ap.add_argument("--domains-from", help="embedding table whose dom_range gives the domains (Zenodo .npz)")
    ap.add_argument("--domains", help="domain TSV pid, domain, ranges (d2ppi.predict --domains-out), instead of "
                    "--domains-from")  # fmt: skip
    ap.add_argument("--encoder", default="esm2_650m", choices=list(ENCODERS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--token-budget", type=int, default=16384, help="residues per batch (lower it if out of memory)")
    ap.add_argument("--shard", metavar="K/N", help="embed only shard K of N (0-based); join the shards with --merge")
    ap.add_argument("--merge", nargs="+", metavar="SHARD", help="join shard tables into --out and exit")
    a = ap.parse_args()
    if a.merge:
        return merge_tables(a.out, a.merge)
    if not a.pairs or not (a.domains_from or a.domains):
        ap.error("--pairs and one of --domains-from / --domains are required")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    if str(a.device).startswith("cuda"):
        torch.set_num_threads(1)  # CPU work is only pooling; one thread keeps its summation order fixed

    seqs = read_sequences(a.pairs)
    if a.domains_from:
        doms = domains_from_table(a.domains_from)
    else:
        doms = read_domain_file(a.domains)
    pids = sorted(seqs)
    if a.shard:
        k, n = map(int, a.shard.split("/"))
        by_len = sorted(pids, key=lambda p: (len(seqs[p]), p))
        pids = sorted(by_len[k::n])
    missing = [p for p in pids if p not in doms]
    if missing:
        print(f"[embed] {len(missing)} proteins have no domains: each is embedded as one whole-protein domain",
              file=sys.stderr)  # fmt: skip
    print(f"[embed] {len(pids)} proteins, encoder {a.encoder} on {a.device}", file=sys.stderr, flush=True)
    enc = Encoder(a.encoder, a.device)
    res, used = embed_proteins(enc, [(p, seqs[p], doms.get(p)) for p in pids], a.token_budget)
    meta = {"encoder": enc.name, "precision": enc.dtype, "recipe": "d2ppi.embed (domain-context embeddings)",
            "domains": a.domains_from or a.domains, "shard": a.shard}  # fmt: skip
    write_table(a.out, seqs, res, used, meta)
    print(f"[embed] wrote {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
