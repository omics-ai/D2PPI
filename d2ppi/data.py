"""Embedding tables, pair tables and batched scoring.

An embedding table is a NumPy .npz (the format of the Zenodo benchmark files, also written by d2ppi.embed):

    pids        (N,)        str      protein ids
    length      (N,)        int32    sequence lengths
    dom_packed  (D, F)      float32  domain embeddings of all proteins, concatenated protein by protein
    dom_off     (N + 1,)    int64    protein i's domains are dom_packed[dom_off[i]:dom_off[i + 1]]
    dom_range   (D,)        str      residues of each domain, 1-based inclusive, e.g. "87-137" or "140-171;190-232"
    glob        (N, F)      float32  whole-protein embeddings
    meta        ()          str      JSON: encoder and recipe

Embeddings are stored raw; every head standardises them with its own training statistics (heads.load_head).
"""

import csv
import gzip
import os

import numpy as np
import torch

MAX_DOMAINS = 86  # domains per protein the heads were trained with

csv.field_size_limit(2**31 - 1)


class Table:
    """An embedding table in memory (fp32, CPU)."""

    def __init__(self, path, pids=None):
        """`pids`: keep only these proteins (saves memory when a table holds far more proteins than needed)."""
        z = np.load(path, allow_pickle=False)
        all_pids = [str(p) for p in z["pids"]]
        off = z["dom_off"].astype(np.int64)
        if pids is None:
            keep = np.arange(len(all_pids))
        else:
            idx = {p: i for i, p in enumerate(all_pids)}
            missing = [p for p in pids if p not in idx]
            if missing:
                raise SystemExit(f"{len(missing)} proteins are not in {path}, e.g. {missing[:3]}")
            keep = np.array(sorted({idx[p] for p in pids}), dtype=np.int64)
        cnt = off[1:] - off[:-1]
        dom = z["dom_packed"]
        if pids is not None:
            rows = np.concatenate([np.arange(off[i], off[i + 1]) for i in keep]) if len(keep) else np.zeros(0, int)
            dom = dom[rows]
        self.pids = [all_pids[i] for i in keep]
        self.index = {p: i for i, p in enumerate(self.pids)}
        self.dom_off = np.concatenate([[0], np.cumsum(cnt[keep])]).astype(np.int64)
        self.dom = torch.from_numpy(np.asarray(dom, dtype=np.float32))
        self.glob = torch.from_numpy(np.asarray(z["glob"][keep], dtype=np.float32))
        self.length = np.asarray(z["length"][keep]) if "length" in z.files else None
        self.dim = self.dom.shape[1]

    def __len__(self):
        return len(self.pids)

    @classmethod
    def from_embeddings(cls, res, lengths=None):
        """a table from {pid: (domain embeddings (k, F), whole-protein embedding (F,))} (d2ppi.embed output)."""
        t = cls.__new__(cls)
        t.pids = list(res)
        t.index = {p: i for i, p in enumerate(t.pids)}
        t.dom_off = np.concatenate([[0], np.cumsum([res[p][0].shape[0] for p in t.pids])]).astype(np.int64)
        t.dom = torch.from_numpy(np.concatenate([res[p][0] for p in t.pids], 0).astype(np.float32))
        t.glob = torch.from_numpy(np.stack([res[p][1] for p in t.pids], 0).astype(np.float32))
        t.length = None if lengths is None else np.array([lengths[p] for p in t.pids])
        t.dim = t.dom.shape[1]
        return t


class Scorer:
    """A table standardised with one head's statistics and kept on `device`; scores pairs in fixed-size batches."""

    def __init__(self, table, mean, std, device, max_domains=MAX_DOMAINS):
        if mean.shape[0] != table.dim:
            raise SystemExit(
                f"the head expects {mean.shape[0]}-d embeddings but the table holds {table.dim}-d ones: "
                "embed the proteins with the head's encoder"
            )
        self.device = device
        self.dom = ((table.dom - mean) / std).to(device)
        self.glob = ((table.glob - mean) / std).to(device)
        off = torch.from_numpy(table.dom_off)
        self.start = off[:-1].to(device)
        self.count = (off[1:] - off[:-1]).clamp(max=max_domains).to(device)
        self.index = table.index

    def side(self, idx):
        """protein indices (B,) -> zero-padded domains (B, T, F), mask (B, T), whole-protein vectors (B, F)."""
        cnt = self.count[idx]
        n = int(cnt.max())
        pos = torch.arange(n, device=self.device)
        mask = pos[None, :] < cnt[:, None]
        dom = torch.zeros(len(idx), n, self.dom.shape[1], device=self.device)
        dom[mask] = self.dom[(self.start[idx][:, None] + pos[None, :])[mask]]
        return dom, mask, self.glob[idx]

    @torch.no_grad()
    def score(self, model, ids_a, ids_b, batch=1024):
        """P(interact) for the pairs (ids_a[k], ids_b[k]), as a float32 numpy array."""
        i1 = torch.tensor([self.index[p] for p in ids_a], dtype=torch.long)
        i2 = torch.tensor([self.index[p] for p in ids_b], dtype=torch.long)
        model.eval()
        out = []
        for s in range(0, len(i1), batch):
            a, b = i1[s : s + batch].to(self.device), i2[s : s + batch].to(self.device)
            logit = model(*self.side(a), *self.side(b))
            out.append(torch.sigmoid(logit.float()).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, np.float32)


def read_pairs(path, columns=None):
    """Read a TSV/CSV pair table (optionally .gz) -> dict column -> list of str.

    `columns`: keep only these (missing optional ones are skipped). The delimiter is taken from the header line.
    """
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", newline="") as f:
        header = f.readline()
        delim = "\t" if "\t" in header else ","
        f.seek(0)
        reader = csv.reader(f, delimiter=delim)
        names = next(reader)
        want = [c for c in (columns or names) if c in names]
        pos = [names.index(c) for c in want]
        cols = {c: [] for c in want}
        for row in reader:
            if not row:
                continue
            for c, p in zip(want, pos, strict=True):
                cols[c].append(row[p])
    return cols


def parse_ranges(s):
    """'12-88' or '140-171;190-232' -> [(12, 88)] / [(140, 171), (190, 232)] (1-based inclusive)."""
    out = []
    for part in s.replace(",", ";").split(";"):
        part = part.strip()
        if part:
            a, b = part.split("-")
            out.append((int(a), int(b)))
    return out


def format_ranges(ranges):
    return ";".join(f"{a}-{b}" for a, b in ranges)


def clean_domains(seq, domains):
    """the domains the heads were trained with: at most 86, clipped to the sequence; whole protein if none."""
    out = []
    for ranges in (domains or [])[:MAX_DOMAINS]:
        rr = [(max(1, s), min(len(seq), e)) for s, e in ranges]
        rr = [(s, e) for s, e in rr if e >= s]
        if rr:
            out.append(rr)
    return out or [[(1, len(seq))]]


def write_domain_file(path, doms):
    """domains as a TSV: pid, domain (1, 2, ...), ranges (1-based inclusive, ';' between segments)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write("pid\tdomain\tranges\n")
        for p, dd in doms.items():
            for k, r in enumerate(dd, 1):
                f.write(f"{p}\t{k}\t{format_ranges(r)}\n")


def read_domain_file(path):
    """{pid: [ranges per domain]} from write_domain_file's TSV (rows in domain order)."""
    c = read_pairs(path, ["pid", "ranges"])
    out = {}
    for p, r in zip(c["pid"], c["ranges"], strict=True):
        out.setdefault(p, []).append(parse_ranges(r))
    return out


def read_fasta(path):
    """{id: sequence} (id = first word of the header)."""
    seqs, pid, buf = {}, None, []
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if pid is not None:
                    seqs[pid] = "".join(buf)
                pid, buf = line[1:].split()[0], []
                if pid in seqs:
                    raise SystemExit(f"{path}: duplicate id {pid}")
            elif line:
                buf.append(line)
    if pid is not None:
        seqs[pid] = "".join(buf)
    return seqs


def domains_from_table(path):
    """{pid: [ranges per domain]} from an embedding table's dom_range (e.g. a Zenodo benchmark file)."""
    z = np.load(path, allow_pickle=False)
    pids, off, rng = [str(p) for p in z["pids"]], z["dom_off"], z["dom_range"]
    return {p: [parse_ranges(str(r)) for r in rng[off[i] : off[i + 1]]] for i, p in enumerate(pids)}
