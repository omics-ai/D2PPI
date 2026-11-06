# D2PPI: Learning Context-Aware Domain Representations for Protein–Protein Interaction Prediction

D2PPI predicts whether two proteins interact from their sequences. Each protein is represented as a **set of
domains**:

1. RecCut cuts the protein into domains on the contact map predicted by ESM-2-150M.
2. A frozen protein language model embeds every domain in its full-sequence context.
3. A small symmetric head (1.4 M parameters for ESM-2-650M) scores the pair. It combines pooled protein-level
   features with a multiple-instance readout over all domain pairs.

This repository holds the inference code, the trained heads for two encoders (ESM-2-650M and ESM-2-15B) on three
benchmarks, a script that downloads the benchmarks from Zenodo, and a script that scores your own sequence pairs.
The domain segmentation ships with it, so DCTdomain does not need to be installed.

## Contents

- [1. Install](#1-install)
- [2. Score your own protein pairs](#2-score-your-own-protein-pairs)
- [3. Repository layout](#3-repository-layout)
- [4. Licence and citation](#4-licence-and-citation)

## 1. Install

```bash
git clone https://github.com/omics-ai/D2PPI && cd D2PPI
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The six trained heads are included in the checkout (about 54 MiB total). Each `.pt` checkpoint contains the
model weights, normalization statistics and training metadata, so no JSON or NPZ sidecars are needed.
The much larger ESM-2 encoders are downloaded separately on first use. To install D2PPI as a package, run
`pip install .`; the installed package also includes the heads and statistics.

You need Python ≥ 3.10. The encoders are downloaded from Hugging Face the first time they are used
(ESM-2-150M 0.6 GB, ESM-2-650M 2.6 GB, ESM-2-15B 60 GB). Approximate GPU memory:

| step                                    | GPU memory |
| --------------------------------------- | ---------: |
| segmentation (ESM-2-150M contact maps)  |     ~12 GB |
| ESM-2-650M embeddings (fp32)            |       9 GB |
| ESM-2-15B embeddings (fp32 activations) |     ~40 GB |
| scoring with a head                     |     < 4 GB |

ESM-2-650M also runs on a CPU for a handful of proteins (`--device cpu`).

## 2. Score your own protein pairs

One pair, given as sequences:

```bash
python -m d2ppi.predict --seq-a MSTNPKPQRKTKRNTNRRPQDVKFPGG... --seq-b MKVLAAGIVGLLLAPAAQA...
```

Many pairs: a FASTA file of the proteins plus a TSV or CSV of id pairs (columns `id1`/`id2`, or else the first two
columns):

```bash
python -m d2ppi.predict --fasta examples/proteins.fasta --pairs examples/pairs.tsv --out examples/scores.tsv
```

Its output:

```
id_a          id_b          len_a  len_b  p_bernett_human  p_dscript_xspecies  p_d2ppi_xspecies
P20783        P23560        257    247    0.7678           0.9949              0.9762
P56556        Q16718        128    116    0.9833           0.9998              0.9976
511145.b4372  511145.b4259  137    147    0.6997           0.9989              0.8601
3702.Q8W575   3702.P43255   429    197    0.8692           1.0000              0.8989
```

Every head calls all four pairs interacting (p > 0.5). With `--encoder esm2_15b` the scores range from 0.67 to 1.00.
For each pair, the head of its own test set (Bernett for the human pairs, LF-XS for the others) agrees with the
paper run within 1e-6 under either encoder. The run takes about 15 s on a GPU, plus the one-time model downloads.

Things to know when you read the scores:

- **The probabilities follow each head's training prior.** The Bernett head saw 50 % positives and the
  cross-species heads 1 positive per 10 negatives, so the Bernett head's scores run higher. The three heads were
  also trained on different data and can disagree strongly on a pair.
- **Which head to trust.**
  - For human pairs between proteins unlike those in public PPI databases, use the Bernett head.
  - For other species, use a cross-species head. Both were trained on human pairs only. LF-XS is the harder,
    leakage-free test of that transfer, and its absolute scores are far lower.
- **Encoder:** `--encoder esm2_650m` (default) or `--encoder esm2_15b`. The 15B heads rank pairs somewhat better,
  especially across species, but need a ~40 GB GPU.

## 3. Repository layout

```
d2ppi/
  predict.py    score your own pairs from sequences
  download.py   fetch the benchmarks from Zenodo
  evaluate.py   score a benchmark split with a released head
  embed.py      domain embeddings (ESM-2-650M / ESM-2-15B), embedding tables
  segment.py    ESM-2-150M contact maps -> RecCut domains
  reccut.py     RecCut (NumPy port of DCTdomain's RecCut.cpp)
  model.py      the D2PPI head
  heads.py      registry of the released heads; loading
  data.py       embedding tables, pair tables, batched scoring
  metrics.py    AUPR, AUROC and thresholded metrics
checkpoints/    the six released heads (+ SHA256SUMS)
examples/       a small FASTA and pair list
tests/          pytest checks (checkpoint integrity, symmetry, released-head scores and evaluation)
LICENSES/       DCTdomain's licence (the NumPy RecCut port keeps its attribution in reccut.py)
```

After installing the requirements, run the tests with `pip install pytest && python -m pytest -q`.

## 4. Licence and citation

The code is released under the MIT licence (`LICENSE`). Third-party pieces:

- `d2ppi/reccut.py` is a port of RecCut from DCTdomain (Academic Free License 3.0). See
  `LICENSES/DCTdomain.txt` and the attribution in `d2ppi/reccut.py`.
- ESM-2 (Meta AI) is MIT-licensed and downloaded from Hugging Face at run time.
- The benchmarks on Zenodo are CC BY 4.0, derived from Bernett et al., D-SCRIPT and STRING.

If you use D2PPI, please cite:

```bibtex
@inproceedings{xiao2026d2ppi,
  title     = {Learning Context-Aware Domain Representations for Protein-Protein Interaction Prediction},
  author    = {Xiao, Qingyang and Ye, Yuzhen and Tang, Haixu},
  year      = {2026},
  note      = {Code: https://github.com/omics-ai/D2PPI; data: https://doi.org/10.5281/zenodo.23874759}
}
```
