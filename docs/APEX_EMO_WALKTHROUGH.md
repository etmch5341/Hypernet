# A*pex + EMO: Hyperloop Route Optimization

## Overview

This system combines **A* pathfinding** with **Evolutionary Multi-Objective Optimization (EMO)** to find optimal Hyperloop routes that balance competing objectives like speed, comfort, and cost.

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Weight Config│────▶│ A* Pathfinder│────▶│ Route +      │
│   α, β, γ    │     │              │     │ Metrics      │
└──────────────┘     └──────────────┘     └──────┬───────┘
       ▲                                         │
       │                                         ▼
       │                                  ┌──────────────┐
       └──────────────────────────────────│   NSGA-II    │
                                          │  Evolution   │
                                          └──────┬───────┘
                                                 │
                                                 ▼
                                          ┌──────────────┐
                                          │ Pareto-Opt   │
                                          │   Routes     │
                                          └──────────────┘
```

---

## The Problem

A Hyperloop route must optimize multiple conflicting objectives:

| Objective | Description | Trade-off |
|-----------|-------------|-----------|
| **Time** | Travel duration | Faster = straighter = more tunneling |
| **Turn Angle** | Curvature penalty | Less turning = longer routes |
| **Jerk** | Smoothness (Δ curvature) | Smoother = slower speed changes |
| **Distance** | Route length | Shorter = more obstacles |

No single "best" route exists—only **trade-offs**.

---

## Solution Architecture

### Phase 1: A*pex Seeding ([apex_emo_test.py](file:///Users/pranavkrishnan/Development/Hypernet/src/apex_emo_test.py))

Generates diverse initial routes by running A* with different weight configurations.

**The Cost Function** (from your PG/OE spec):
```
c_k = α·t_k + β·θ_k + γ·J_k
```
Where:
- `t_k` = time on edge (based on curvature-limited speed)
- `θ_k` = turn angle at node
- `J_k` = jerk (change in turn angle)

**Key Classes**:
- `HyperloopObjectives`: Computes edge costs with terrain penalties
- `APexSeeder`: Generates 20 seeds with varied weights

### Phase 2: Evolutionary Optimization ([apex_nsga3.py](file:///Users/pranavkrishnan/Development/Hypernet/src/apex_nsga3.py))

Uses **NSGA-II** to evolve better weight configurations.

**What Evolves**: The 3 weights `[α, β, γ]` (not the paths directly)

**Fitness Evaluation**: Each weight set runs A* → gets 4 objective scores

---

## How Evolution Works

### Per Generation (20 generations total):

1. **Population**: 20 individuals, each is `[α, β, γ]`

2. **Evaluate**: Run A* for each → get `[Time, Turn, Jerk, Distance]`

3. **Rank**: Non-dominated sorting (Pareto ranking)
   - Rank 1: Not dominated by anyone
   - Rank 2: Dominated only by Rank 1

4. **Select Parents**: Tournament selection (better rank wins)

5. **Crossover (90% probability)**:
   ```
   Parent A: [0.95, 0.30, 0.29]
   Parent B: [0.98, 0.01, 0.30]
              ↓ SBX blend ↓
   Child:    [0.96, 0.15, 0.29]
   ```

6. **Mutate (30% probability)**: Add small random noise

7. **Survivor Selection**: Keep best 20 from parents + children

---

## Terrain Cost Model

Routes can traverse any terrain with cost penalties:

| Terrain | Multiplier | Configuration |
|---------|------------|---------------|
| Roads | 1.0x | Preferred path |
| Open Land | 5.0x | Possible but costly |
| Protected Areas | 50.0x | Very expensive |

---

## Output

The system produces a **Pareto front** of ~20 non-dominated solutions:

| Solution | Time | Distance | Turn | Jerk |
|----------|------|----------|------|------|
| Fast | 10.4s | 404m | 1.57 rad | 3.14 |
| Short | 11.1s | 389m | 2.36 rad | 3.93 |

Engineers can then choose the trade-off that best fits project constraints.

---

## Files

| File | Purpose |
|------|---------|
| [apex_emo_test.py](file:///Users/pranavkrishnan/Development/Hypernet/src/apex_emo_test.py) | A* + cost functions + seeding |
| [apex_nsga3.py](file:///Users/pranavkrishnan/Development/Hypernet/src/apex_nsga3.py) | NSGA-II optimization loop |
| [hyperloop_astar_from_npz.py](file:///Users/pranavkrishnan/Development/Hypernet/src/simple_astar_results/advanced/hyperloop_astar_from_npz.py) | Multi-goal A* for test sets |
