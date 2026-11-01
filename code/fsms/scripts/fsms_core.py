"""
FSMS core - shared feature loading and per-image max-similarity.

Mirrors `fsms/metric.py` exactly in its maths, with one addition:
that module returns only the aggregate scalar (it computes the per-image max
similarities and discards them). Every validation in this folder needs
the per-image vector - to detect planted copies, to count near-duplicates, to plot
distributions - so `max_sim_to_real()` returns it and `fsms_from_max_sims()` does the
final aggregation.

Deliberately self-contained (same rationale as the classifier scripts in classifiers/metric_tournament): validation
must not be able to perturb the code path that produced the published numbers.

    FSMS = 1 - mean(max cosine similarity to nearest real training image)
    Higher = safer (further from any real training image).
"""
from pathlib import Path

import pandas as pd
import torch

BATCH_SIZE = 2048


# ===================================================================
# LOADING
# ===================================================================

def load_features(path: Path):
    """
    Load a feature asset in either supported layout and return (names, matrix).

      * raw dict     {filename: tensor[D]}          - as written by extract_dino_features.py
      * packed dict  {'names': [...], 'matrix': T}  - as written by pack_features.py

    The packed layout is preferred: 275k separate tensor objects carry ~1.5 GB of
    Python overhead on top of the data, which matters on a laptop.
    """
    print(f"  Loading features: {path}")
    obj = torch.load(path, map_location='cpu', weights_only=False)

    if isinstance(obj, dict) and 'names' in obj and 'matrix' in obj:
        names, matrix = list(obj['names']), obj['matrix'].float()
    elif isinstance(obj, dict):
        names = list(obj.keys())
        matrix = torch.stack([obj[k] for k in names]).float()
    else:
        raise ValueError(f"Unrecognised feature file layout: {path}")

    print(f"  {len(names)} entries, dim {matrix.shape[1]}")
    return names, matrix


def build_safe_key_map(names) -> dict:
    """
    {safe_filename -> original key}.

    Feature keys use forward slashes ('edm2/ckpt_25k_cfg_1.0/class_0/000649.png');
    curated CSVs store safe_filename with '/' replaced by '_'. Identical to
    fsms/metric.py:30.
    """
    return {k.replace('/', '_').replace('\\', '_'): k for k in names}


def rows_for_csv(csv_path: Path, names, matrix):
    """
    Resolve one curated CSV to its feature rows.

    Returns (submatrix [n, D], list of resolved keys). Warns on unresolved images
    rather than failing, matching the behaviour of the production FSMS script.
    """
    df = pd.read_csv(csv_path)
    index = {k: i for i, k in enumerate(names)}

    if 'safe_filename' in df.columns:
        safe_to_orig = build_safe_key_map(names)
        wanted = [safe_to_orig.get(fn, fn) for fn in df['safe_filename']]
    elif 'source_path' in df.columns:
        wanted = ['/'.join(Path(p).parts[-2:]) for p in df['source_path']]
    else:
        raise ValueError(f"CSV has no 'safe_filename' or 'source_path': {csv_path}")

    idx = [index[k] for k in wanted if k in index]
    missing = len(wanted) - len(idx)
    if missing:
        print(f"    [WARNING] {missing}/{len(wanted)} images not found in feature set.")
    if not idx:
        raise ValueError(f"No features resolved for {csv_path}")

    keys = [names[i] for i in idx]
    return matrix[idx], keys


# ===================================================================
# CORE COMPUTATION
# ===================================================================

def _normalise(m: torch.Tensor) -> torch.Tensor:
    return m / (m.norm(dim=1, keepdim=True) + 1e-8)


def max_sim_to_real(query: torch.Tensor,
                    real: torch.Tensor,
                    exclude_self: bool = False,
                    batch_size: int = BATCH_SIZE) -> torch.Tensor:
    """
    Per-image maximum cosine similarity from `query` to any row of `real`.

    exclude_self: mask the diagonal. Only meaningful when query IS real (same rows in
    the same order) - gives "nearest *other* real image", i.e. the domain's internal
    density rather than the trivial self-match of 1.0.
    """
    qn, rn = _normalise(query), _normalise(real)

    out = []
    for i in range(0, len(qn), batch_size):
        sims = qn[i:i + batch_size] @ rn.T
        if exclude_self:
            for j in range(sims.shape[0]):
                sims[j, i + j] = -1.0
        out.append(sims.max(dim=1).values)
    return torch.cat(out)


def fsms_from_max_sims(max_sims: torch.Tensor) -> float:
    """FSMS = 1 - mean(max similarity). Higher = safer."""
    return 1.0 - max_sims.mean().item()


def near_duplicate_rate(max_sims: torch.Tensor, threshold: float) -> float:
    """Fraction of images whose nearest real neighbour exceeds `threshold`."""
    return (max_sims >= threshold).float().mean().item()


def describe(max_sims: torch.Tensor, label: str, thresholds=(0.90, 0.95, 0.99)):
    """One-line summary plus near-duplicate rates. Returns a dict for tabulation."""
    row = {
        'set': label,
        'n': len(max_sims),
        'fsms': fsms_from_max_sims(max_sims),
        'mean_max_sim': max_sims.mean().item(),
        'p50': max_sims.median().item(),
        'p99': max_sims.quantile(0.99).item(),
        'max': max_sims.max().item(),
    }
    for t in thresholds:
        row[f'pct_ge_{t}'] = 100.0 * near_duplicate_rate(max_sims, t)

    print(f"  {label:32s} FSMS={row['fsms']:.4f}  mean_sim={row['mean_max_sim']:.4f}  "
          f"max={row['max']:.4f}  >=0.95: {row['pct_ge_0.95']:.2f}%")
    return row
