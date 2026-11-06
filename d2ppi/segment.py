"""Protein sequences -> RecCut domains, as in the domain files the D2PPI models were trained on.

Pipeline (DCTdomain's, Iovino, Tang & Ye 2024; RecCut is ported in :mod:`d2ppi.reccut`):

1. ESM-2-150M (``facebook/esm2_t30_150M_UR50D``) predicts the residue contact map, in fp32. A protein longer than
   ``maxlen`` is cut into ``maxlen``-residue windows overlapping by 200 residues (DCTdomain's
   ``Embedding.embed_seq``); each window is predicted alone and the window maps are placed on the diagonal of the
   full map, the overlapping blocks averaged.
2. RecCut segments the top 2.6 * L contacts.
3. Segments that run past the sequence end are dropped (never seen in the benchmark files), at most 86 domains are
   kept, and a protein RecCut does not cut is one whole-length domain (:func:`d2ppi.data.clean_domains`).

The window decides the domains of long proteins, so use the one the head was trained with
(``BENCHMARKS[<benchmark>]["segment_maxlen"]`` in :mod:`d2ppi.heads`): 500 for Bernett (DCTdomain's default),
1,022 for the cross-species heads.

Example::

    from d2ppi.segment import segment
    doms = segment({"P1": "MKT..."}, maxlen=500)   # {"P1": [[(1, 84)], [(85, 118)], ...]}

or from the command line::

    python -m d2ppi.segment --fasta proteins.fasta --benchmark bernett --out domains.tsv

RecCut runs on a pool of CPU processes (``workers``); call :func:`segment` under ``if __name__ == "__main__":`` in
scripts, as for any :mod:`multiprocessing` code.
"""

import argparse
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from d2ppi.data import clean_domains, read_fasta, write_domain_file
from d2ppi.heads import BENCHMARKS, benchmark_name
from d2ppi.reccut import reccut_edges, top_contacts

CONTACT_MODEL = "facebook/esm2_t30_150M_UR50D"
OVERLAP = 200  # DCTdomain's window overlap


# ------- contact maps
class ContactPredictor:
    """ESM-2-150M contact maps (Hugging Face transformers), fp32 with TF32 off. ``device`` defaults to CUDA."""

    def __init__(self, device=None):
        from transformers import AutoTokenizer, EsmForMaskedLM

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        self.tok = AutoTokenizer.from_pretrained(CONTACT_MODEL)
        model = EsmForMaskedLM.from_pretrained(CONTACT_MODEL, attn_implementation="eager")
        self.model = model.float().eval().to(self.device)

    @torch.no_grad()
    def __call__(self, seqs):
        """One padded batch of sequences -> list of (L, L) float32 contact maps."""
        enc = self.tok(list(seqs), return_tensors="pt", padding=True)
        ct = self.model.predict_contacts(enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device))
        ct = ct.float().cpu().numpy()
        return [ct[k, : len(s), : len(s)].copy() for k, s in enumerate(seqs)]

    def windowed(self, seq, maxlen):
        """DCTdomain ``Embedding.embed_seq`` for one sequence: one pass if ``len(seq) <= maxlen``, else overlapping
        ``maxlen`` windows (each predicted alone) merged by ``combine_contacts``."""
        if len(seq) <= maxlen:
            return self([seq])[0]
        step = maxlen - OVERLAP
        starts = [i for i in range(0, len(seq), step) if len(seq[i : i + maxlen]) > OVERLAP]
        # combine_contacts grows the map window by window; filling one preallocated map is the same arithmetic
        # (each overlapping block is (previous + new) / 2 in float32) without the repeated copies
        ct, n1 = np.zeros((len(seq), len(seq)), dtype=np.float32), 0
        for k, st in enumerate(starts):
            c = self([seq[st : st + maxlen]])[0]
            ct[st : st + c.shape[0], st : st + c.shape[0]] += c
            if k:
                ct[st:n1, st:n1] /= 2
            n1 = st + c.shape[0]
        assert n1 == len(seq), (n1, len(seq))
        return ct


def length_batches(lengths, budget, max_batch):
    """Length-sorted batches of indices under a B * L^2 budget (the contact head is quadratic in L)."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    cur = []
    for i in order:
        L = lengths[i]
        if cur and ((len(cur) + 1) * L * L > budget or len(cur) >= max_batch):
            yield cur
            cur = []
        cur.append(i)
    if cur:
        yield cur


def contact_maps(seqs, maxlen, predictor, batched=True, budget=1_000_000, max_batch=16):
    """Yield ``(pid, contact_map)`` for a dict of sequences. Proteins up to ``maxlen`` residues are predicted in
    padded, length-sorted batches when ``batched`` (one at a time otherwise); longer ones window by window."""
    pids = list(seqs)
    short = [p for p in pids if len(seqs[p]) <= maxlen]
    long_ = [p for p in pids if len(seqs[p]) > maxlen]
    if batched:
        for b in length_batches([len(seqs[p]) for p in short], budget, max_batch):
            batch = [short[i] for i in b]
            yield from zip(batch, predictor([seqs[p] for p in batch]), strict=True)
    else:
        for p in short:
            yield p, predictor([seqs[p]])[0]
    for p in long_:
        yield p, predictor.windowed(seqs[p], maxlen)


# ------- segmentation
def final_domains(domains, seq):
    """RecCut's domains -> the embedder's: segments past the sequence end dropped, then clean_domains."""
    return clean_domains(seq, [[(a, b) for a, b in d if max(a, b) <= len(seq)] for d in domains])


def segment(seqs, maxlen, device=None, predictor=None, batched=True, budget=1_000_000, max_batch=16, workers=None,
            verbose=False):  # fmt: skip
    """Sequences -> the domains D2PPI embeds.

    Args:
        seqs: ``{pid: sequence}``.
        maxlen: window for long proteins (the head's ``segment_maxlen``).
        device: for the :class:`ContactPredictor` (ignored when ``predictor`` is given).
        batched, budget, max_batch: GPU batching of proteins up to ``maxlen`` residues.
        workers: RecCut processes (default: ``min(8, cpu_count)`` for 16 or more proteins, else 0 = RecCut runs in
            this process).

    Returns:
        ``{pid: [[(start, end), ...], ...]}`` in the input order: one list of 1-based inclusive segments per domain.
    """
    seqs = {str(k): str(v).strip().upper() for k, v in seqs.items()}
    predictor = predictor or ContactPredictor(device)
    if workers is None:
        workers = min(8, os.cpu_count() or 1) if len(seqs) >= 16 else 0
    pool = ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) if workers > 0 else None
    out, t0 = {}, time.time()
    try:
        for n, (pid, ct) in enumerate(contact_maps(seqs, maxlen, predictor, batched, budget, max_batch), 1):
            job = (len(seqs[pid]), *top_contacts(ct))
            out[pid] = pool.submit(reccut_edges, *job) if pool else reccut_edges(*job)
            if verbose and n % 500 == 0:
                print(f"[segment] contacts {n}/{len(seqs)} ({time.time() - t0:.0f} s)", file=sys.stderr, flush=True)
        out = {p: final_domains(out[p].result() if pool else out[p], seqs[p]) for p in seqs}
    finally:
        if pool:
            pool.shutdown()
    if verbose:
        print(f"[segment] {len(seqs)} proteins in {time.time() - t0:.0f} s", file=sys.stderr, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description="RecCut domains for protein sequences (ESM-2-150M contacts)")
    ap.add_argument("--fasta", required=True, help="FASTA of the proteins")
    ap.add_argument("--out", required=True, help="domain TSV (pid, domain, ranges): the domains D2PPI embeds")
    ap.add_argument("--benchmark", default="bernett_human", help="whose window to use: bernett | dscript | lfxs")
    ap.add_argument("--maxlen", type=int, default=None, help="window for long proteins (default: --benchmark's)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    maxlen = a.maxlen or BENCHMARKS[benchmark_name(a.benchmark)]["segment_maxlen"]
    doms = segment(read_fasta(a.fasta), maxlen, device=a.device, workers=a.workers, verbose=True)
    write_domain_file(a.out, doms)
    nd = sum(len(d) for d in doms.values())
    print(f"[segment] wrote {a.out}: {len(doms)} proteins, {nd} domains (maxlen {maxlen})", file=sys.stderr)


if __name__ == "__main__":
    main()
