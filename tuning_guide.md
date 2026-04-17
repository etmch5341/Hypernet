# A*pex Hyperloop Tuning Guide

This guide explains how to adjust the "knobs" in `src/run_apex_custom.py` to refine your hyperloop routes.

## 1. Road Discount (`road`)
**What it is:** The "Infrastructure Loyalty" slider.
- **Range:** `0.0` (Free travel on roads) to `1.0` (No preference for roads).
- **Physical Impact:** 
    - At **0.3 (30%)**, the algorithm will take significant detours just to stay on a highway.
    - At **1.0 (100%)**, it ignores roads entirely and cuts the shortest geometric path.
- **When to tune:** If the route is crossing too much protected land, lower this value.

## 2. Turn Stiffness (`stiff`)
**What it is:** The "G-Force" slider.
- **Range:** `1.0` (Jagged/Natural) to `50.0+` (Glass-smooth).
- **Physical Impact:** 
    - At **5.0**, the route follows the natural curves of the landscape.
    - At **40.0**, the algorithm treats a turn as "physically painful," forcing long, beautiful straights.
- **When to tune:** If the `routes_overlay.png` looks "wiggly" or "stair-stepped," turn this up.

## 3. Epsilon (`eps_vector`)
**What it is:** The "Diversity vs. Speed" slider.
- **Default:** `(0.3, 0.3, 0.3, 0.3)` (30% pruning threshold).
- **Physical Impact:** 
    - A **low epsilon (0.05)** will find thousands of nearly identical routes, slowing down the search.
    - A **high epsilon (0.4)** will find only the most distinct "macro-corridors" (e.g., North vs. South).
- **When to tune:** If you want more variety in your routes, lower this. If the search is too slow, raise it.

## 4. Heuristic Weight (`h`)
**What it is:** The "Search Beam" slider.
- **Range:** `1.0` (Admissible, slow) to `1.5` (Aggressive, fast).
- **Physical Impact:** 
    - Higher values pull the search toward the goal like a magnet. 
    - If too high, it might "blind" the algorithm to a great highway that starts in the wrong direction.
- **When to tune:** Use **1.25** for deep exploration, and **1.4** for quick results.
