#!/usr/bin/env python3
"""APEX-style Pareto-frontier analysis over the candidate paths produced
by the clustered NAMOA*-dr stage.

Given a list of candidate paths each with a 3-D cost vector
(construction, environmental, geometry), this module:
  * filters to the ε-Pareto frontier,
  * computes ideal / nadir points,
  * produces a ranked list with normalised distance-to-ideal,
  * selects the single "best" compromise path.

The scoring mirrors the A*pex (pranav-dev) convention:
solutions are Pareto-filtered with ε-dominance, then ranked by the
Euclidean distance of their normalised objective vector to the ideal
corner.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


Vec = Tuple[float, float, float]


@dataclass
class ApexRanking:
    pareto: List[dict]          # Pareto-optimal subset (with extra fields)
    dominated: List[dict]       # Candidates dropped by Pareto filter
    ideal: Vec
    nadir: Vec
    best: Optional[dict]        # Closest-to-ideal entry (overall winner)


def _eps_dominates(a: Sequence[float], b: Sequence[float], eps: float) -> bool:
    return all(x <= (1.0 + eps) * y for x, y in zip(a, b))


def _strictly_dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def apex_pareto_filter(candidates: List[dict], eps: float = 0.0) -> Tuple[
        List[dict], List[dict]]:
    """Return (pareto, dominated) partition under ε-dominance."""
    pareto = []
    dominated = []
    for cand in candidates:
        cv = cand["cost_vector"]
        if eps > 0.0:
            is_dom = any(
                _eps_dominates(other["cost_vector"], cv, eps)
                and other is not cand
                for other in candidates
            )
        else:
            is_dom = any(
                _strictly_dominates(other["cost_vector"], cv)
                for other in candidates if other is not cand
            )
        (dominated if is_dom else pareto).append(cand)
    return pareto, dominated


def _normalise(costs: np.ndarray) -> np.ndarray:
    mn = costs.min(axis=0)
    mx = costs.max(axis=0)
    span = np.where(mx - mn > 1e-9, mx - mn, 1.0)
    return (costs - mn) / span


def apex_rank(candidates: List[dict], eps: float = 0.0,
              weights: Sequence[float] = (1.0, 1.0, 1.0)) -> ApexRanking:
    """Run the full APEX ranking pipeline.

    ``weights`` applies after normalisation to let callers bias toward
    one objective when computing the distance-to-ideal score.
    """
    if not candidates:
        return ApexRanking(pareto=[], dominated=[],
                           ideal=(0.0, 0.0, 0.0), nadir=(0.0, 0.0, 0.0),
                           best=None)

    pareto, dominated = apex_pareto_filter(candidates, eps=eps)

    source = pareto if pareto else candidates
    costs = np.asarray([c["cost_vector"] for c in source], dtype=np.float64)
    ideal = tuple(float(v) for v in costs.min(axis=0))
    nadir = tuple(float(v) for v in costs.max(axis=0))

    norm = _normalise(costs)
    w = np.asarray(weights, dtype=np.float64)
    w = w / (w.sum() if w.sum() > 0 else 1.0)
    distances = np.sqrt(((norm * w) ** 2).sum(axis=1))

    for cand, dist, nrow in zip(source, distances, norm):
        cand["apex_distance_to_ideal"] = float(dist)
        cand["apex_normalised_costs"] = [float(x) for x in nrow]

    ranked = sorted(source, key=lambda c: c["apex_distance_to_ideal"])
    for rank, cand in enumerate(ranked, start=1):
        cand["apex_rank"] = rank

    best = ranked[0] if ranked else None
    return ApexRanking(pareto=pareto, dominated=dominated,
                       ideal=ideal, nadir=nadir, best=best)


def format_ranking_summary(ranking: ApexRanking) -> str:
    lines = []
    lines.append("APEX Pareto analysis")
    lines.append(f"  Pareto-optimal candidates : {len(ranking.pareto)}")
    lines.append(f"  Dominated candidates      : {len(ranking.dominated)}")
    lines.append(f"  Ideal   (construction, environmental, geometry): "
                 f"{tuple(round(v, 2) for v in ranking.ideal)}")
    lines.append(f"  Nadir                                         : "
                 f"{tuple(round(v, 2) for v in ranking.nadir)}")
    # When the Pareto frontier collapses to a single point, ideal == nadir
    # and every normalised distance is 0.0 by construction. Surface that
    # fact so the report isn't misread as "perfect match".
    degenerate = (len(ranking.pareto) <= 1 or
                  all(abs(a - b) < 1e-9
                      for a, b in zip(ranking.ideal, ranking.nadir)))
    if degenerate and ranking.best is not None:
        lines.append("  (single Pareto survivor — distances are 0 by "
                     "construction, not a true 'perfect' score)")
    if ranking.best is None:
        lines.append("  No candidates produced.")
        return "\n".join(lines)

    lines.append("")
    lines.append("  Ranked Pareto candidates (closest-to-ideal first):")
    source = ranking.pareto if ranking.pareto else ranking.dominated
    for cand in sorted(source, key=lambda c: c.get("apex_rank", 10_000)):
        cv = cand["cost_vector"]
        tag = " <-- BEST" if cand is ranking.best else ""
        lines.append(
            f"    rank {cand.get('apex_rank', '?'):>2}  "
            f"id={cand.get('candidate_id', '?'):>2}  "
            f"constr={cv[0]:.1f}  env={cv[1]:.1f}  geo={cv[2]:.1f}  "
            f"dist={cand.get('apex_distance_to_ideal', float('nan')):.3f}"
            f"{tag}"
        )
    return "\n".join(lines)
