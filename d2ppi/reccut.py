"""RecCut: protein domain segmentation from a predicted residue contact map.

Attribution notice
------------------
RecCut was written by Haixu Tang (Indiana University) as part of DCTdomain

    B. Iovino, H. Tang & Y. Ye (2024). Protein domain embeddings for fast and accurate similarity search.
    Genome Research 34:1434-1444.  https://github.com/mgtools/DCTdomain

and is licensed under the Academic Free License 3.0 (see ``LICENSES/DCTdomain.txt``). This file is a derivative
work: the upstream C++ program ``src/RecCut.cpp`` at commit 17c1a423a337e808e2734071154495cfa56af295 was ported to
Python/NumPy and modified by the D2PPI authors. The modifications are mechanical: the contact graph is held as a
sparse edge list instead of a dense matrix, the O(L^2) scans are vectorised with cumulative sums, and the
contact-file round trip is replaced by :func:`top_contacts`. The output is meant to be identical to the C++
program's for every input.

Algorithm
---------
The graph has one vertex per residue. Edge weights are ``int(100 * p + 0.5)`` for the top ``alpha * L`` predicted
contacts with ``|i - j| >= 5`` (``p`` rounded to 6 decimals, as in DCTdomain's contact file) and 100 between
residues at sequence distance 1-3. RecCut recursively applies the best of two cuts, scored by the normalised cut
``cut * total / internal_1 / internal_2``: a single cut (one boundary) or a double cut (a middle segment against the
two termini joined). A cut is applied when its score is below 0.08 (single) or 0.07 (double) and both sides keep at
least 22 residues. The leaves are the domains; a domain can be discontinuous (several segments).

Faithfulness notes
------------------
* All bookkeeping (including the original's quirks in how segment boundaries are carried into sub-problems) is
  transliterated from ``SplitDomain`` / ``SplitDomain_2cuts``. In one branch the original reads ``cutsite[-1]``
  (one element before the vector). With glibc this reads the upper half of the heap chunk header, which is 0 for
  any vector below 4 GiB; the port uses 0, which is also the value the surrounding logic expects.
* Scores are evaluated in IEEE double precision in the same operation order as the C++ code, and ties go to the
  first candidate in the original loop order, so the chosen cut is the same, not just equally good.
"""

import numpy as np

ALPHA = 2.6  # contacts kept: the top ALPHA * L (DCTdomain's threshold; "alpha" of FUpred)
CUT1 = 0.08  # single-cut score threshold (cuts_threshold)
CUT2 = 0.07  # double-cut score threshold (cuts_2_threshold)
MIN_SIZE = 22  # smallest domain (Min_Size)
MIN_TERMINAL = 10  # residues kept off each terminus by a double cut (Min_Terminal)
SEQ_SEP = 5  # contacts are taken with |i - j| >= SEQ_SEP
ADJ = 3  # residues at sequence distance 1..ADJ are joined with weight 100


def top_contacts(contacts, alpha=ALPHA, chunk=1 << 24):
    """Select the contacts RecCut reads, exactly as DCTdomain writes them to its contact (CE) file.

    Candidates are the pairs i < j with j - i >= 5, ordered by probability (descending); ties keep the (i, j)
    row-major order (a stable sort). The top ``int(alpha * L)`` are kept, and each probability is rounded to the
    6 decimals of the file.

    Args:
        contacts: (L, L) predicted contact probabilities (float32, as produced by ESM-2).
        alpha: contacts kept per residue.
        chunk: matrix entries scanned at a time (bounds memory for very long proteins).

    Returns:
        (i, j, p): int64, int64, float64 arrays, 0-based, in file order.
    """
    c = np.asarray(contacts)
    L = c.shape[0]
    npairs = (L - SEQ_SEP) * (L - SEQ_SEP + 1) // 2 if L > SEQ_SEP else 0
    tot = min(int(alpha * L), npairs)
    empty = np.zeros(0, dtype=np.int64)
    if tot <= 0:
        return empty, empty, np.zeros(0)
    rows = max(1, chunk // max(L, 1))

    def blocks():  # (flat row-major index, value) of the candidate pairs, rows r0..r1
        for r0 in range(0, L - SEQ_SEP, rows):
            r1 = min(r0 + rows, L - SEQ_SEP)
            ii, jj = np.nonzero(np.arange(L)[None, :] >= np.arange(r0, r1)[:, None] + SEQ_SEP)
            ii += r0
            yield ii.astype(np.int64) * L + jj, c[ii, jj]

    # pass 1: the tot-th largest value (with multiplicity)
    pool = np.zeros(0, dtype=c.dtype)
    for _, v in blocks():
        pool = np.concatenate([pool, v])
        if pool.size > tot:
            pool = np.partition(pool, pool.size - tot)[pool.size - tot :]
    thr = pool.min()
    # pass 2: everything above the threshold, then ties at the threshold in row-major order
    sel_idx, sel_val = [], []
    for f, v in blocks():
        keep = v >= thr
        sel_idx.append(f[keep])
        sel_val.append(v[keep])
    f = np.concatenate(sel_idx)
    v = np.concatenate(sel_val)
    order = np.lexsort((f, -v.astype(np.float64)))[:tot]  # value descending, then row-major
    f, v = f[order], v[order]
    # the CE file stores '%.6f' text; RecCut parses it back as a double
    p = np.array([float(f"{x:.6f}") for x in v.tolist()], dtype=np.float64)
    return f // L, f % L, p


def contact_graph(L, ci, cj, p):
    """RecCut's ``readGraph``: weighted edge list (u < v, int64 weights) of the residue graph."""
    w = (p * 100 + 0.5).astype(np.int64)  # C++ (int)(value * 100 + 0.5): truncation, value >= 0
    u = np.minimum(ci, cj).astype(np.int64)
    v = np.maximum(ci, cj).astype(np.int64)
    au = np.concatenate([np.arange(L - d, dtype=np.int64) for d in range(1, ADJ + 1)])
    av = np.concatenate([np.arange(d, L, dtype=np.int64) for d in range(1, ADJ + 1)])
    # adjacency is written after the contacts in the original and never overlaps them (|i - j| >= 5 > ADJ)
    keep = w > 0
    u = np.concatenate([u[keep], au])
    v = np.concatenate([v[keep], av])
    w = np.concatenate([w[keep], np.full(au.size, 100, dtype=np.int64)])
    return u, v, w


def _induced(V, u, v, w, order):
    """Edges of the subgraph on vertices ``order`` (relabelled 0..len(order)-1 in that order)."""
    pos = np.full(V, -1, dtype=np.int64)
    pos[order] = np.arange(len(order), dtype=np.int64)
    pu, pv = pos[u], pos[v]
    keep = (pu >= 0) & (pv >= 0)
    pu, pv = pu[keep], pv[keep]
    return np.minimum(pu, pv), np.maximum(pu, pv), w[keep]


def _best_cuts(V, u, v, w):
    """Scores of the best single and double cut of a graph (the two scans of ``recursiveMaxCut``)."""
    total = int(w.sum())
    pre = np.bincount(v, weights=w, minlength=V).astype(np.int64)  # sum_{j<i} G[i][j]
    post = np.bincount(u, weights=w, minlength=V).astype(np.int64)  # sum_{j>i} G[i][j]
    # single cut after vertex i (the C++ scan over i = 1 .. V-3); cutv_i = weight between {0..i} and {i+1..}
    cutv = post[0] + np.concatenate([[0], np.cumsum(post[1:] - pre[1:])])
    N1 = np.cumsum(pre)  # N1[0] = 0
    N2 = total - np.concatenate([[0], np.cumsum(post[1:])])  # N2[0] = total (as in the original)
    ii = np.arange(1, V - 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        ave = cutv[ii].astype(np.float64) * float(total) / N1[ii].astype(np.float64) / N2[ii].astype(np.float64)
    mincut, cuts = 2.0, None
    ok = ave < mincut  # NaN never wins, as in the C++ comparison
    if ok.any():
        k = int(np.argmin(np.where(ok, ave, np.inf)))
        mincut, cuts = float(ave[k]), int(ii[k]) + 1

    # double cut: segment [i, j] against the rest, i in [10, V-10), j in [i+21, V-10)
    mincut2, cuts1, cuts2 = 2.0, None, None
    order = np.argsort(u, kind="stable")
    su, sv, sw = u[order], v[order], w[order]
    bound = np.searchsorted(su, np.arange(V + 1))  # edges with u == a: bound[a]:bound[a+1]
    A = np.cumsum(np.bincount(v, weights=w, minlength=V).astype(np.int64))  # A[j] = weight of edges with v <= j
    col = np.zeros(V, dtype=np.int64)  # col[k] = weight of edges (a, k) with a < i
    np.add.at(col, sv[: bound[MIN_TERMINAL]], sw[: bound[MIN_TERMINAL]])
    jmax = V - MIN_TERMINAL
    for i in range(MIN_TERMINAL, V - MIN_TERMINAL):
        if i > MIN_TERMINAL:
            a, b = bound[i - 1], bound[i]
            if b > a:
                np.add.at(col, sv[a:b], sw[a:b])
        j0 = i + MIN_SIZE - 1
        if j0 >= jmax:
            break
        NS2 = A[j0:jmax] - np.cumsum(col[:jmax])[j0:]  # internal weight of [i, j], j = j0 .. jmax-1
        cv = N1[j0:jmax] + N2[i] - N1[i - 1] - N2[j0 + 1 : jmax + 1] - NS2 * 2
        NS1 = total - cv - NS2
        valid = (NS1 > 0) & (NS2 > 0)
        if not valid.any():
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            ave2 = cv.astype(np.float64) * float(total) / NS1.astype(np.float64) / NS2.astype(np.float64)
        ave2 = np.where(valid, ave2, np.inf)
        k = int(np.argmin(ave2))
        if ave2[k] < mincut2:
            mincut2, cuts1, cuts2 = float(ave2[k]), i, j0 + k
    return mincut, cuts, mincut2, cuts1, cuts2


def _at(vec, k):
    """``vec[k]`` with the C++ out-of-bounds read at k == -1 (0 under glibc, see the module notes)."""
    if k == -1:
        return 0
    return vec[k]


def _split_domain(cuts, domain, cutsite, length):
    """Transliteration of ``SplitDomain`` (single cut at ``cuts``)."""
    d1, d2, cs1, cs2 = [], [], [], []
    c1 = 0
    while c1 < len(cutsite):
        if cutsite[c1] < cuts:
            d1.append(domain[c1])
            cs1.append(cutsite[c1])
            c1 += 1
        else:
            break
    len1 = cuts if c1 == 0 else cuts - cutsite[c1 - 1]
    d1.append((domain[c1][0], domain[c1][0] + len1 - 1))
    if len(cutsite) == 0 or c1 == len(cutsite):
        len2 = length - cuts
    else:
        len2 = cutsite[c1] - cuts
    if len2 > 0:
        d2.append((domain[c1][1] - len2 + 1, domain[c1][1]))
        if c1 < len(cutsite):
            cs2.append(len2)
    c1 += 1
    while c1 < len(domain):
        d2.append(domain[c1])
        if c1 < len(cutsite) and len1 > 0:
            cs2.append(len1)
        c1 += 1
    return d1, d2, cs1, cs2


def _split_domain_2cuts(cuts1, cuts2, domain, cutsite, length):
    """Transliteration of ``SplitDomain_2cuts`` (middle segment [cuts1, cuts2) cut out)."""
    d1, d2, cs1, cs2 = [], [], [], []
    ln = length - cuts2
    c1, c2 = -1, 0
    for i in range(len(cutsite) - 1, -1, -1):
        if cutsite[i] < cuts2:
            c2 = i + 1
        if cutsite[i] <= cuts1:
            c1 = i
            break
    len2 = cuts2 if c2 == 0 else cuts2 - cutsite[c2 - 1]
    d1.append((domain[c2][0] + len2, domain[c2][1]))
    for i in range(c2 + 1, len(cutsite) + 1):
        d1.append(domain[i])
    for i in range(c2, len(cutsite)):
        cs1.append(cutsite[i] - cuts2)
    cs1.append(ln)
    for i in range(0, c1 + 1):
        d1.append(domain[i])
        if cutsite[i] != cuts1:
            cs1.append(cutsite[i] + ln)
    len1 = cuts1 - cutsite[c1] if c1 >= 0 else cuts1
    c1 += 1
    if len1 > 0:
        d1.append((domain[c1][0], domain[c1][0] + len1 - 1))
    if c2 > c1 and _at(cutsite, c1 - 1) + len1 < cutsite[c1]:
        if c1 >= 0:
            d2.append((domain[c1][0] + len1, domain[c1][1]))
    for i in range(c1 + 1, c2):
        d2.append(domain[i])
    if c2 > c1 and len2 > 0:
        d2.append((domain[c2][0], domain[c2][0] + len2 - 1))
    elif c1 == c2:
        d2.append((domain[c2][0] + len1, domain[c2][0] + len2 - 1))
    for i in range(c1, c2):
        if cutsite[i] - cuts1 > 0:
            cs2.append(cutsite[i] - cuts1)
    return d1, d2, cs1, cs2


def _recursive_cut(V, u, v, w, cutsite, domain):
    """``recursiveMaxCut`` for one domain (list of 0-based segments) -> list of domains."""
    if V < MIN_SIZE:
        raise RuntimeError("RecCut: sub-problem below the minimum domain size")  # unreachable, as in the C++
    mincut, cuts, mincut2, cuts1, cuts2 = _best_cuts(V, u, v, w)
    if mincut - CUT1 <= mincut2 - CUT2:
        if mincut > CUT1 or cuts < MIN_SIZE or V - cuts < MIN_SIZE:
            return [domain]
        first = np.arange(cuts)
        second = np.arange(cuts, V)
        d1, d2, cs1, cs2 = _split_domain(cuts, domain, cutsite, V)
    else:
        length = cuts2 - cuts1
        if mincut2 > CUT2 or length < MIN_SIZE or V - length < MIN_SIZE:
            return [domain]
        first = np.concatenate([np.arange(cuts2, V), np.arange(cuts1)])
        second = np.arange(cuts1, cuts2)
        d1, d2, cs1, cs2 = _split_domain_2cuts(cuts1, cuts2, domain, cutsite, V)
    out = _recursive_cut(len(first), *_induced(V, u, v, w, first), cs1, d1)
    out += _recursive_cut(len(second), *_induced(V, u, v, w, second), cs2, d2)
    return out


def reccut_edges(L, ci, cj, p):
    """RecCut on selected contacts (0-based ``ci``, ``cj``, probabilities ``p`` as in the CE file).

    Returns RecCut's domains in its output order: a list of domains, each a list of 1-based inclusive
    ``(start, end)`` segments.
    """
    if L < MIN_SIZE:
        return [[(1, L)]]
    u, v, w = contact_graph(L, ci, cj, p)
    doms = _recursive_cut(L, u, v, w, [], [(0, L - 1)])
    return [[(a + 1, b + 1) for a, b in d] for d in doms]


def reccut(contacts, alpha=ALPHA):
    """Domains of one protein from its (L, L) contact map: DCTdomain's ``writece`` + ``RecCut``.

    Returns a list of domains, each a list of 1-based inclusive ``(start, end)`` segments, in RecCut's order.
    """
    L = np.asarray(contacts).shape[0]
    return reccut_edges(L, *top_contacts(contacts, alpha))


def format_domains(domains):
    """RecCut's stdout field: '1-70,180-261;262-425;' (',' within a domain, ';' after each)."""
    return "".join(",".join(f"{a}-{b}" for a, b in d) + ";" for d in domains)


def ce_text(pid, seq, ci, cj, p):
    """DCTdomain's CE contact file (the C++ program's input) for selected contacts."""
    con = ",".join(f"{a} {b} {x:.6f}" for a, b, x in zip(ci.tolist(), cj.tolist(), p.tolist(), strict=True))
    return f"INF   {pid} {len(seq)}\nSEQ   {seq}\nSS    {'C' * len(seq)}\nCON   {con}\n"
