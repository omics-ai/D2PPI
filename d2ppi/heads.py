"""The released heads: which benchmark and encoder each was trained on, and how to load it.

checkpoints/<benchmark>/<encoder>.pt bundles the D2PPI state dict, training mean / std for standardisation,
and metadata with provenance, the training recipe and test metrics. No sidecar files are needed.
"""

import os

import torch

from d2ppi.model import D2PPI

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
if not os.path.isdir(CKPT_DIR):
    CKPT_DIR = os.path.join(ROOT, "checkpoints")

ENCODERS = {
    # name: Hugging Face id, width, inference precision (see d2ppi.embed.Encoder)
    "esm2_650m": {"hf": "facebook/esm2_t33_650M_UR50D", "dim": 1280, "dtype": "fp32"},
    "esm2_15b": {"hf": "facebook/esm2_t48_15B_UR50D", "dim": 5120, "dtype": "fp32act"},
}

BENCHMARKS = {
    # segment_maxlen: the DCTdomain window with which the training proteins' RecCut domains were made; a protein
    # longer than it gets its contact map from overlapping windows (d2ppi.segment). Applied to new proteins too.
    "bernett_human": {
        "aliases": ("bernett",),
        "title": "Bernett (human, leakage-free)",
        "train_positive_rate": 0.5,
        "segment_maxlen": 500,
    },
    "dscript_xspecies": {
        "aliases": ("dscript", "d-script"),
        "title": "D-SCRIPT cross-species (trained on human)",
        "train_positive_rate": 1 / 11,
        "segment_maxlen": 1022,
    },
    "d2ppi_xspecies": {
        "aliases": ("lfxs", "lf-xs"),
        "title": "LF-XS leakage-free cross-species (trained on human)",
        "train_positive_rate": 1 / 11,
        "segment_maxlen": 1022,
    },
}


def benchmark_name(name):
    """canonical benchmark name from a name or alias (bernett, dscript, lfxs, ...)."""
    key = name.lower()
    for b, info in BENCHMARKS.items():
        if key == b or key in info["aliases"]:
            return b
    choices = ", ".join(BENCHMARKS)
    raise SystemExit(f"unknown benchmark {name!r}; choose from {choices} (aliases bernett, dscript, lfxs)")


def head_path(benchmark, encoder):
    return os.path.join(CKPT_DIR, benchmark, encoder + ".pt")


def load_head(benchmark, encoder, device="cpu"):
    """-> (model in eval mode on `device`, mean (D,) float32 tensor, std (D,) float32 tensor, meta dict)."""
    benchmark = benchmark_name(benchmark)
    if encoder not in ENCODERS:
        raise SystemExit(f"unknown encoder {encoder!r}; choose from {', '.join(ENCODERS)}")
    pt = head_path(benchmark, encoder)
    if not os.path.exists(pt):
        raise SystemExit(f"checkpoint {pt} not found")
    checkpoint = torch.load(pt, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise SystemExit(f"unsupported checkpoint format in {pt}: use a bundled D2PPI checkpoint")
    model = D2PPI(in_dim=ENCODERS[encoder]["dim"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()  # eval mode: dropout off, spectral norm uses the stored estimates
    mean = checkpoint["mean"].float()
    std = checkpoint["std"].float()
    return model, mean, std, checkpoint["meta"]
