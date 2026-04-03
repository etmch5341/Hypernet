#!/usr/bin/env python3
"""
Hyperloop Pathfinding Algorithm Presentation Generator
Produces a self-contained PDF using only matplotlib (no pandoc/LaTeX needed).

Run from the advanced/ directory:
    python generate_presentation.py

Output: hyperloop_algorithm_presentation.pdf
"""

import os
import math
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.image as mpimg
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_BASE   = os.path.join(SCRIPT_DIR, "comparison_all_austin")
OUT_PDF    = os.path.join(SCRIPT_DIR, "hyperloop_algorithm_presentation.pdf")

# ── Color palette ─────────────────────────────────────────────────────────────
C = dict(
    bg      = "#F7F9FC",
    header  = "#1A2C4E",
    accent  = "#2E86AB",
    green   = "#27AE60",
    orange  = "#E67E22",
    red     = "#C0392B",
    purple  = "#8E44AD",
    yellow  = "#F39C12",
    teal    = "#16A085",
    light   = "#ECF0F1",
    text    = "#2C3E50",
    subtext = "#7F8C8D",
    astar   = "#4A90E2",
    ara     = "#FFD700",
    mha     = "#9B59B6",
    namoa   = "#FF8C00",
    theta   = "#27AE60",
    bidir   = "#E74C3C",
    emoa    = "#16A085",
    alt     = "#8E44AD",
)

ALGO_COLORS = {
    "A*":         C["astar"],
    "ARA*":       C["ara"],
    "MHA*":       C["mha"],
    "NAMOA*-dr":  C["namoa"],
    "Theta*":     C["theta"],
    "BiA*":       C["bidir"],
    "EMOA*":      C["emoa"],
    "ALT":        C["alt"],
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def blank_fig(bg=C["bg"]):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(bg)
    return fig


def header_bar(fig, title, subtitle=None, color=C["header"]):
    """Draw a colored header bar at the top of a page."""
    ax = fig.add_axes([0, 0.88, 1, 0.12])
    ax.set_facecolor(color)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.04, 0.62, title, color="white", fontsize=20,
            fontweight="bold", va="center", transform=ax.transAxes)
    if subtitle:
        ax.text(0.04, 0.20, subtitle, color="#BDC3C7", fontsize=11,
                va="center", transform=ax.transAxes)
    return ax


def footer(fig, page_num, total=16):
    ax = fig.add_axes([0, 0, 1, 0.04])
    ax.set_facecolor(C["header"])
    ax.axis("off")
    ax.text(0.5, 0.5, f"Guadaloop Hyperloop  ·  Pathfinding Algorithm Analysis  ·  Page {page_num}/{total}",
            color="#BDC3C7", fontsize=8, ha="center", va="center", transform=ax.transAxes)


def body_ax(fig, rect=(0.04, 0.06, 0.92, 0.80)):
    ax = fig.add_axes(rect)
    ax.set_facecolor(C["bg"])
    ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    return ax


def colored_box(ax, x, y, w, h, color, text, text_color="white",
                fontsize=10, fontweight="bold", radius=0.015, va="center", ha="left",
                padding=(0.015, 0.008)):
    """Draw a rounded rectangle with text inside."""
    box = FancyBboxPatch((x, y), w, h,
                          boxstyle=f"round,pad=0",
                          facecolor=color, edgecolor="none",
                          transform=ax.transAxes, clip_on=False)
    ax.add_patch(box)
    ax.text(x + padding[0], y + h / 2, text, color=text_color,
            fontsize=fontsize, fontweight=fontweight,
            va=va, ha=ha, transform=ax.transAxes, wrap=True)


def stat_card(ax, x, y, w, h, label, value, color):
    """Metric card: colored top strip, big value, label below."""
    strip_h = h * 0.30
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0",
                          facecolor="white", edgecolor=color, linewidth=2,
                          transform=ax.transAxes)
    ax.add_patch(box)
    strip = FancyBboxPatch((x, y + h - strip_h), w, strip_h,
                            boxstyle="round,pad=0",
                            facecolor=color, edgecolor="none",
                            transform=ax.transAxes)
    ax.add_patch(strip)
    ax.text(x + w / 2, y + h * 0.55, value, color=C["text"],
            fontsize=13, fontweight="bold", ha="center", va="center",
            transform=ax.transAxes)
    ax.text(x + w / 2, y + h * 0.22, label, color=C["subtext"],
            fontsize=8, ha="center", va="center", transform=ax.transAxes)


def algo_header(ax, x, y, w, h, name, color):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0",
                          facecolor=color, edgecolor="none",
                          transform=ax.transAxes)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, name, color="white",
            fontsize=15, fontweight="bold", ha="center", va="center",
            transform=ax.transAxes)


def bullet(ax, x, y, text, color=C["accent"], fontsize=10, indent=0.03):
    ax.text(x + indent, y, "•", color=color, fontsize=fontsize + 2,
            va="top", transform=ax.transAxes)
    ax.text(x + indent + 0.025, y, text, color=C["text"], fontsize=fontsize,
            va="top", wrap=True, transform=ax.transAxes)


def section_title(ax, x, y, text, color=C["accent"]):
    ax.text(x, y, text, color=color, fontsize=13, fontweight="bold",
            va="top", transform=ax.transAxes)
    # Underline
    ax.plot([x, x + 0.95], [y - 0.035, y - 0.035], color=color,
            linewidth=1.5, transform=ax.transAxes)


def load_img(path):
    """Safely load image; return None if missing."""
    if os.path.exists(path):
        try:
            return mpimg.imread(path)
        except Exception:
            return None
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

def page_title(pdf):
    fig = blank_fig()

    # Big gradient-ish background block
    bg_ax = fig.add_axes([0, 0, 1, 1])
    bg_ax.set_facecolor(C["header"])
    bg_ax.axis("off")

    # Decorative accent stripe
    stripe = FancyBboxPatch((0, 0.38), 1, 0.005,
                             boxstyle="square,pad=0",
                             facecolor=C["accent"], edgecolor="none",
                             transform=bg_ax.transAxes)
    bg_ax.add_patch(stripe)

    bg_ax.text(0.5, 0.76, "Hyperloop Pathfinding", color="white",
               fontsize=32, fontweight="bold", ha="center", va="center",
               transform=bg_ax.transAxes)
    bg_ax.text(0.5, 0.65, "Algorithm Analysis", color=C["accent"],
               fontsize=28, fontweight="bold", ha="center", va="center",
               transform=bg_ax.transAxes)

    bg_ax.text(0.5, 0.53, "Current Algorithms · Limitations · Recommendations · Testing Plan",
               color="#BDC3C7", fontsize=13, ha="center", va="center",
               transform=bg_ax.transAxes)

    # Stats row
    labels = ["Test Map", "Raster Size", "Algorithms Compared", "Best Speed"]
    values = ["Austin, TX", "540 × 864 px", "4 (+ 4 recommended)", "MHA*  1.01s"]
    colors = [C["accent"], C["green"], C["orange"], C["purple"]]
    for i, (lbl, val, col) in enumerate(zip(labels, values, colors)):
        xc = 0.13 + i * 0.19
        box = FancyBboxPatch((xc - 0.07, 0.18), 0.14, 0.14,
                              boxstyle="round,pad=0.01",
                              facecolor=col, edgecolor="none",
                              transform=bg_ax.transAxes)
        bg_ax.add_patch(box)
        bg_ax.text(xc, 0.29, val, color="white", fontsize=10,
                   fontweight="bold", ha="center", va="center",
                   transform=bg_ax.transAxes)
        bg_ax.text(xc, 0.21, lbl, color="#ECF0F1", fontsize=8,
                   ha="center", va="center", transform=bg_ax.transAxes)

    bg_ax.text(0.5, 0.09,
               f"Guadaloop Student Organization  ·  University of Texas at Austin  ·  {datetime.date.today().strftime('%B %Y')}",
               color="#7F8C8D", fontsize=9, ha="center", va="center",
               transform=bg_ax.transAxes)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_problem_setup(pdf, pn):
    fig = blank_fig()
    header_bar(fig, "Problem Setup", "What are we routing, and how is the map structured?")
    footer(fig, pn)
    ax = body_ax(fig)

    section_title(ax, 0, 0.95, "The Raster Grid")
    bullet(ax, 0, 0.87, "Map is a 2D raster where each cell = 1 pixel of geographic terrain")
    bullet(ax, 0, 0.80, "Cell value 1 = road/rail corridor  |  Cell value 0 = off-road terrain")
    bullet(ax, 0, 0.73, "Movement: 8-directional (N, NE, E, SE, S, SW, W, NW) with diagonal support")

    section_title(ax, 0, 0.63, "Cost Model")
    costs = [
        ("On-road traversal",   "1.0",  C["green"],  "Standard movement along road network"),
        ("Off-road traversal",  "5.0",  C["red"],    "5× penalty for leaving the road corridor"),
        ("Terrain roughness",   "0.12–1.0", C["orange"], "Local road density (5×5 window), precomputed"),
    ]
    for i, (name, val, col, desc) in enumerate(costs):
        y = 0.54 - i * 0.11
        colored_box(ax, 0.00, y, 0.22, 0.08, col, name, fontsize=9)
        colored_box(ax, 0.23, y, 0.10, 0.08, "#2C3E50", val, fontsize=11)
        ax.text(0.35, y + 0.04, desc, color=C["text"], fontsize=9.5,
                va="center", transform=ax.transAxes)

    section_title(ax, 0, 0.28, "Austin Test Configuration")
    cfg = [
        ("Start Station", "station1  →  pixel (829, 34)  →  row=34, col=829"),
        ("End Station",   "station2  →  pixel (152, 510) →  row=510, col=152"),
        ("Map Bounds",    "Lat 30.238–30.285  |  Lon -97.795 to -97.704  (EPSG:3083)"),
        ("Map Size",      "864 wide × 540 tall  =  466,560 total cells"),
        ("Goals tracked via", "Bitmask: 00=neither, 01=station1 visited, 11=both visited (done)"),
    ]
    for i, (k, v) in enumerate(cfg):
        y = 0.19 - i * 0.065
        ax.text(0.01, y, k + ":", color=C["accent"], fontsize=9.5,
                fontweight="bold", va="top", transform=ax.transAxes)
        ax.text(0.23, y, v, color=C["text"], fontsize=9.5,
                va="top", transform=ax.transAxes)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_overview_table(pdf, pn):
    fig = blank_fig()
    header_bar(fig, "Current Algorithms — At a Glance",
               "Four algorithms benchmarked on the same Austin test raster")
    footer(fig, pn)
    ax = body_ax(fig, rect=(0.02, 0.06, 0.96, 0.80))

    cols = ["Algorithm", "Type", "Objectives", "Heuristics", "Expansions", "Time", "Path (wp)"]
    rows = [
        ["A*",         "Optimal",  "1 (distance)",   "1 admissible\n(Euclidean+MST)",         "211,023",    "2.60 s",   "979"],
        ["ARA*",       "Anytime",  "1 (distance)",   "1 admissible\n(inflated ε·h)",           "1,276,755",  "10.30 s",  "909"],
        ["MHA*",       "Bounded",  "1 (distance)",   "3 (anchor+2\ninadmissible)",             "79,775",     "1.01 s",   "924"],
        ["NAMOA*-dr",  "Pareto",   "3 (dist,\noffroad, rough)", "3 vector\n(BFS+Dijkstra)",   "1,154,829",  "361.54 s", "902\n(5 solutions)"],
    ]
    algo_cols = [C["astar"], C["ara"], C["mha"], C["namoa"]]

    col_widths = [0.12, 0.10, 0.14, 0.16, 0.13, 0.10, 0.14]
    col_starts = [sum(col_widths[:i]) + 0.01 for i in range(len(col_widths))]
    row_h = 0.10
    header_y = 0.87

    # Header row
    for j, (col, cx) in enumerate(zip(cols, col_starts)):
        colored_box(ax, cx, header_y, col_widths[j] - 0.005, 0.07,
                    C["header"], col, fontsize=8.5)

    # Data rows
    for i, (row, acol) in enumerate(zip(rows, algo_cols)):
        y = header_y - (i + 1) * (row_h + 0.005)
        row_bg = "#FAFAFA" if i % 2 == 0 else "white"
        bg = FancyBboxPatch((0.01, y), 0.98, row_h, boxstyle="square,pad=0",
                             facecolor=row_bg, edgecolor="#E0E0E0", linewidth=0.5,
                             transform=ax.transAxes)
        ax.add_patch(bg)

        for j, (cell, cx) in enumerate(zip(row, col_starts)):
            color = "white" if j == 0 else C["text"]
            fw    = "bold"  if j == 0 else "normal"
            bg_c  = acol    if j == 0 else row_bg
            if j == 0:
                colored_box(ax, cx, y, col_widths[j] - 0.005, row_h, acol, cell,
                            fontsize=8.5, va="center")
            else:
                ax.text(cx + 0.005, y + row_h / 2, cell, color=C["text"],
                        fontsize=8, va="center", transform=ax.transAxes)

    # Key insight callout
    y_note = header_y - 5 * (row_h + 0.005) - 0.02
    colored_box(ax, 0.01, y_note, 0.97, 0.085, C["accent"],
                "Key Insights:   MHA* is 4× faster than A* and 350× faster than NAMOA*-dr.  "
                "NAMOA*-dr is the only algorithm that returns multiple trade-off paths (Pareto front).",
                fontsize=9, fontweight="normal")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_algo_deepdive(pdf, pn, name, color, tagline,
                        how_it_works, heuristics, params, strengths, weaknesses, stats):
    fig = blank_fig()
    header_bar(fig, name, tagline, color=color)
    footer(fig, pn)
    ax = body_ax(fig)

    # Left column: How it works + heuristics
    section_title(ax, 0, 0.95, "How It Works", color=color)
    y = 0.87
    for line in how_it_works:
        bullet(ax, 0, y, line, color=color, fontsize=9.5)
        y -= 0.075

    section_title(ax, 0, y - 0.01, "Heuristics", color=color)
    y -= 0.08
    for line in heuristics:
        bullet(ax, 0, y, line, color=color, fontsize=9.5)
        y -= 0.07

    # Right column: Stats + strengths/weaknesses
    rx = 0.52
    section_title(ax, rx, 0.95, "Austin Results", color=color)

    for i, (lbl, val) in enumerate(stats):
        sy = 0.83 - i * 0.125
        stat_card(ax, rx + (i % 2) * 0.23, sy, 0.20, 0.10, lbl, val, color)
        if i % 2 == 1:
            sy -= 0.13

    sw_y = 0.83 - math.ceil(len(stats) / 2) * 0.13 - 0.02

    section_title(ax, rx, sw_y, "Strengths", color=C["green"])
    y2 = sw_y - 0.08
    for s in strengths:
        bullet(ax, rx, y2, s, color=C["green"], fontsize=9)
        y2 -= 0.065

    section_title(ax, rx, y2 - 0.01, "Weaknesses", color=C["red"])
    y2 -= 0.08
    for w in weaknesses:
        bullet(ax, rx, y2, w, color=C["red"], fontsize=9)
        y2 -= 0.065

    # Param strip at bottom
    param_str = "   |   ".join(f"{k}: {v}" for k, v in params.items())
    colored_box(ax, 0, 0.02, 1.0, 0.06, C["header"],
                "Parameters:  " + param_str, fontsize=8.5, fontweight="normal")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_image(pdf, pn, img_path, title, subtitle, caption):
    fig = blank_fig()
    header_bar(fig, title, subtitle)
    footer(fig, pn)

    img = load_img(img_path)
    if img is not None:
        img_ax = fig.add_axes([0.03, 0.10, 0.94, 0.74])
        img_ax.imshow(img, aspect="auto")
        img_ax.axis("off")
    else:
        ax = body_ax(fig)
        ax.text(0.5, 0.5, f"[Image not found]\n{img_path}",
                color=C["red"], fontsize=12, ha="center", va="center",
                transform=ax.transAxes)

    cap_ax = fig.add_axes([0.03, 0.05, 0.94, 0.05])
    cap_ax.axis("off")
    cap_ax.text(0.5, 0.5, caption, color=C["subtext"], fontsize=9,
                ha="center", va="center", transform=cap_ax.transAxes,
                style="italic")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_pareto(pdf, pn):
    fig = blank_fig()
    header_bar(fig, "NAMOA*-dr — Pareto Front",
               "Five trade-off paths found; user selects based on priorities")
    footer(fig, pn)
    ax = body_ax(fig)

    section_title(ax, 0, 0.95, "The 5 Pareto-Optimal Solutions")

    solutions = [
        (1, 1025.1, 45, 641.84, 865,  False, "Shortest path but crosses off-road 45 times"),
        (2, 1061.7, 15, 637.88, 902,  True,  "SELECTED — balanced trade-off across all 3 objectives"),
        (3, 1064.9, 10, 643.08, 906,  False, "10 off-road crossings, slight roughness increase"),
        (4, 1070.6,  3, 646.72, 913,  False, "Almost entirely on-road, slightly longer path"),
        (5, 1072.6,  1, 648.56, 915,  False, "Most on-road (1 crossing), longest path"),
    ]
    cols = ["#", "Distance\n(units)", "Off-Road\nCells", "Roughness\nSum", "Waypoints", "Status"]
    col_ws = [0.04, 0.12, 0.12, 0.12, 0.12, 0.45]
    col_xs = [sum(col_ws[:i]) + 0.01 for i in range(len(col_ws))]

    # Table header
    for j, (col, cx) in enumerate(zip(cols, col_xs)):
        colored_box(ax, cx, 0.82, col_ws[j] - 0.005, 0.07, C["header"], col, fontsize=8)

    for i, (n, dist, offrd, rough, wp, selected, note) in enumerate(solutions):
        y = 0.82 - (i + 1) * 0.085
        bg = C["namoa"] if selected else ("#FAFAFA" if i % 2 == 0 else "white")
        txt_col = "white" if selected else C["text"]
        box = FancyBboxPatch((0.01, y), 0.98, 0.075, boxstyle="square,pad=0",
                              facecolor=bg, edgecolor="#DDDDDD", linewidth=0.5,
                              transform=ax.transAxes)
        ax.add_patch(box)
        vals = [str(n), f"{dist:.1f}", str(offrd), f"{rough:.2f}", str(wp), note]
        for j, (val, cx) in enumerate(zip(vals, col_xs)):
            ax.text(cx + 0.005, y + 0.037, val, color=txt_col, fontsize=8.5,
                    va="center", transform=ax.transAxes,
                    fontweight="bold" if selected else "normal")

    # Explanation
    section_title(ax, 0, 0.12, "How to Read the Pareto Front", color=C["namoa"])
    ax.text(0.01, 0.04,
            "No single solution is 'best' — each is optimal for a different priority. "
            "A path with fewer off-road crossings is always longer. "
            "The 'balanced' selection (Sol. 2) minimizes the normalized distance to the ideal point (0, 0, 0).",
            color=C["text"], fontsize=9.5, va="top", wrap=True,
            transform=ax.transAxes)

    # Load the pareto graph on right side
    img = load_img(os.path.join(IMG_BASE, "namoa_data", "namoa_pareto_front_graph.png"))
    if img is not None:
        img_ax = fig.add_axes([0.60, 0.12, 0.38, 0.68])
        img_ax.imshow(img, aspect="auto")
        img_ax.axis("off")
        # Shrink the text area
        ax.set_xlim(0, 0.60)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_limitations(pdf, pn):
    fig = blank_fig()
    header_bar(fig, "Limitations of Current Algorithms",
               "What these four algorithms cannot do well — and why it matters for hyperloop")
    footer(fig, pn)
    ax = body_ax(fig)

    lims = [
        (
            "Grid-Constrained Movement",
            C["red"],
            [
                "All 4 algorithms snap to 8 grid directions: N, NE, E, SE, S, SW, W, NW",
                "Real hyperloop tubes travel in any continuous direction — not 45° increments",
                "Result: staircase-like paths with unnecessary turns, 5-10% longer than necessary",
            ],
            "A* path:  ↗ ↗ → → → ↘ ↘ ↘ → →\nTheta*:   ─────────────────────────",
        ),
        (
            "Scale — Austin → Dallas",
            C["orange"],
            [
                "Current test raster: 540×864 = 466,560 cells  (suburb-scale)",
                "Austin → Dallas: ~195 miles. At same resolution → ~10M+ cells",
                "A* would need tens of millions of expansions; NAMOA*-dr would be infeasible",
            ],
            "Current: 540×864  →  466K cells\nA→D map: ~3,000×4,000  →  12M cells",
        ),
        (
            "NAMOA*-dr Runtime",
            C["namoa"],
            [
                "361 seconds (6 min) on a small 540×864 map with ε-dominance at 0.15",
                "Multi-objective label explosion grows exponentially with map size",
                "Not practical for interactive planning or larger city-pair corridors",
            ],
            "Austin small map:  361 s\nEstimated A→D map:  hours",
        ),
        (
            "MHA* Greedy Road Heuristic Underperforms",
            C["mha"],
            [
                "The 3rd heuristic (Greedy Road-Biased) fired 0 times out of 79,775 expansions",
                "Manhattan+MST always won the expansion race — GreedyRoad never got a turn",
                "Opportunity: better inadmissible heuristic design could improve further",
            ],
            "Anchor:      54,949 expansions\nManhattan:   24,826 expansions\nGreedyRoad:       0 expansions",
        ),
    ]

    y = 0.93
    for (title, color, bullets, aside) in lims:
        colored_box(ax, 0, y, 0.58, 0.065, color, title, fontsize=10)
        for i, b in enumerate(bullets):
            bullet(ax, 0.01, y - 0.06 - i * 0.055, b, color=color, fontsize=8.5)
        # Aside box
        aside_box = FancyBboxPatch((0.62, y - len(bullets) * 0.055 - 0.01),
                                    0.36, len(bullets) * 0.055 + 0.07,
                                    boxstyle="round,pad=0.01",
                                    facecolor=color, edgecolor="none", alpha=0.12,
                                    transform=ax.transAxes)
        ax.add_patch(aside_box)
        ax.text(0.64, y + 0.025, aside, color=color, fontsize=8,
                va="top", transform=ax.transAxes, family="monospace")
        y -= len(bullets) * 0.055 + 0.095

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_recommendation(pdf, pn, name, color, tagline,
                          problem, how_works, how_works_detail,
                          benefits, effort, compare_note, formula=None):
    fig = blank_fig()
    header_bar(fig, f"Recommendation: {name}", tagline, color=color)
    footer(fig, pn)
    ax = body_ax(fig)

    # Problem solved
    colored_box(ax, 0, 0.92, 1.0, 0.055, "#F0F0F0",
                "Problem Solved:   " + problem, fontsize=9.5,
                text_color=C["text"], fontweight="normal")

    section_title(ax, 0, 0.84, "How It Works", color=color)
    y = 0.76
    for i, line in enumerate(how_works):
        bullet(ax, 0, y, line, color=color, fontsize=9.5)
        y -= 0.07
    if formula:
        colored_box(ax, 0.05, y - 0.01, 0.50, 0.065, C["light"],
                    formula, text_color=C["text"], fontsize=10, fontweight="bold")
        y -= 0.085

    section_title(ax, 0, y - 0.02, "Worked Example", color=color)
    ax.text(0.02, y - 0.09, how_works_detail, color=C["text"], fontsize=9.5,
            va="top", wrap=True, transform=ax.transAxes, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=C["light"], edgecolor=color, linewidth=1))

    # Right column
    rx = 0.55
    section_title(ax, rx, 0.84, "Expected Benefits", color=color)
    yb = 0.76
    for b in benefits:
        bullet(ax, rx, yb, b, color=C["green"], fontsize=9.5)
        yb -= 0.07

    section_title(ax, rx, yb - 0.02, "Implementation Effort", color=C["orange"])
    colored_box(ax, rx, yb - 0.10, 0.44, 0.065, C["orange"], effort, fontsize=9.5)

    section_title(ax, rx, yb - 0.18, "vs. Current Algorithms", color=color)
    ax.text(rx + 0.01, yb - 0.26, compare_note, color=C["text"], fontsize=9,
            va="top", wrap=True, transform=ax.transAxes)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_testing_plan(pdf, pn):
    fig = blank_fig()
    header_bar(fig, "Testing Plan",
               "How to implement, benchmark, and validate each recommended algorithm")
    footer(fig, pn)
    ax = body_ax(fig)

    section_title(ax, 0, 0.95, "Implementation Pattern (same for every new algorithm)")
    steps = [
        ("Create",    "hyperloop_<name>_from_npz.py  (same CLI interface as existing algorithms)"),
        ("Register",  "Add run_<name>() to compare_all_algorithms.py and run_all_tests.py"),
        ("Animate",   "Extend astar_animator.py — new panel in the 2×2 (or 3×2) grid GIF"),
        ("Run Austin","python compare_all_algorithms.py --input austin_test_raster.npz  (fast feedback)"),
        ("Scale",     "Run on Seattle and Portland once Austin results look correct"),
    ]
    y = 0.87
    for i, (label, desc) in enumerate(steps):
        colored_box(ax, 0.00, y, 0.08, 0.055, C["accent"], str(i+1), fontsize=14)
        colored_box(ax, 0.09, y, 0.14, 0.055, C["header"], label, fontsize=9)
        ax.text(0.25, y + 0.027, desc, color=C["text"], fontsize=9.5,
                va="center", transform=ax.transAxes)
        y -= 0.068

    section_title(ax, 0, 0.49, "Metrics to Record per Algorithm")
    metrics = [
        ("Expansions",        "Total nodes popped from priority queue"),
        ("Wall-clock time",   "End-to-end runtime including heuristic precomputation"),
        ("Path length",       "Number of waypoints in final path"),
        ("Path smoothness",   "Count of direction changes (fewer = smoother = better for hyperloop)"),
        ("Path cost",         "Sum of edge weights along the final path"),
        ("Off-road cells",    "Number of raster cells in path with road_bitmap=0 (NAMOA / EMOA only)"),
    ]
    y = 0.41
    for lbl, desc in metrics:
        ax.text(0.02, y, lbl + ":", color=C["accent"], fontsize=9.5,
                fontweight="bold", va="top", transform=ax.transAxes)
        ax.text(0.25, y, desc, color=C["text"], fontsize=9.5,
                va="top", transform=ax.transAxes)
        y -= 0.055

    section_title(ax, 0, y - 0.01, "Visual Validation Checklist")
    checks = [
        "Path connects station1 (top-right) to station2 (bottom-left) — no disconnected blobs",
        "Exploration heatmap covers a corridor between the two stations — not a random blob",
        "Path color distinct in the 2×2 grid animation (no color overlap between panels)",
        "NAMOA* / EMOA* Pareto front plot shows increasing trade-off (not all solutions identical)",
        "Theta* paths visibly smoother than A* paths — fewer diagonal staircase segments",
    ]
    y2 = y - 0.09
    for c in checks:
        ax.text(0.02, y2, "☐  " + c, color=C["text"], fontsize=9, va="top",
                transform=ax.transAxes)
        y2 -= 0.055

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_summary(pdf, pn):
    fig = blank_fig()
    header_bar(fig, "Summary & Recommendation",
               "Which algorithm to use and when — for Austin–Dallas hyperloop routing")
    footer(fig, pn)
    ax = body_ax(fig)

    section_title(ax, 0, 0.95, "Current Algorithm Scorecard")
    curr = [
        ("A*",        C["astar"],  "✓ Optimal",   "✗ Grid-constrained paths",      "Baseline — replace with Theta*"),
        ("ARA*",      C["ara"],    "✓ Anytime",   "✗ 1.27M expansions, 10s",       "Good for time-limited search"),
        ("MHA*",      C["mha"],    "✓ Fastest (1s)", "✗ Single-objective only",    "Best speed; add Theta* variant"),
        ("NAMOA*-dr", C["namoa"],  "✓ Pareto front","✗ 361s — very slow",          "Replace with EMOA* for scale"),
    ]
    y = 0.87
    for name, col, pro, con, verdict in curr:
        colored_box(ax, 0.00, y, 0.13, 0.065, col, name, fontsize=9)
        ax.text(0.14, y + 0.032, pro,     color=C["green"],   fontsize=9, va="center", transform=ax.transAxes)
        ax.text(0.34, y + 0.032, con,     color=C["red"],     fontsize=9, va="center", transform=ax.transAxes)
        ax.text(0.60, y + 0.032, verdict, color=C["subtext"], fontsize=9, va="center",
                transform=ax.transAxes, style="italic")
        y -= 0.075

    section_title(ax, 0, 0.44, "Recommended Additions (Priority Order)")
    recs = [
        ("1",  "Theta*",        C["theta"],  "Any-angle",  "LOW",    "Drop-in A* replacement. Smoother paths, same runtime. Most impactful for infrastructure planning."),
        ("2",  "Bidirectional A*", C["bidir"],"Speed",     "MEDIUM", "Search from both ends simultaneously. ~50–70% fewer expansions for long Austin→Dallas corridors."),
        ("3",  "EMOA*",         C["emoa"],   "Multi-obj",  "MEDIUM", "Replaces NAMOA*-dr. Same Pareto front guarantee with dramatically fewer expansions."),
        ("4",  "ALT",           C["alt"],    "Scale",      "MEDIUM", "Landmark-based heuristic precomputation. Enables scaling to full city-pair rasters."),
    ]
    y = 0.36
    for rank, name, col, cat, effort, desc in recs:
        colored_box(ax, 0.00, y, 0.04, 0.065, col, rank, fontsize=12)
        colored_box(ax, 0.05, y, 0.13, 0.065, col, name, fontsize=9)
        colored_box(ax, 0.19, y, 0.09, 0.065, C["header"], cat, fontsize=8)
        eff_col = C["green"] if effort == "LOW" else C["orange"]
        colored_box(ax, 0.29, y, 0.09, 0.065, eff_col, effort + " effort", fontsize=8)
        ax.text(0.40, y + 0.032, desc, color=C["text"], fontsize=8.5,
                va="center", transform=ax.transAxes)
        y -= 0.075

    colored_box(ax, 0, 0.01, 1.0, 0.055, C["accent"],
                "Bottom line: Implement Theta* first — it improves every other algorithm and requires minimal code change. "
                "Then EMOA* to fix the NAMOA*-dr runtime problem.",
                fontsize=9, fontweight="normal")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    os.chdir(SCRIPT_DIR)
    print(f"Generating presentation → {OUT_PDF}")

    with PdfPages(OUT_PDF) as pdf:

        # P1 — Title
        print("  Page 1: Title")
        page_title(pdf)

        # P2 — Problem Setup
        print("  Page 2: Problem Setup")
        page_problem_setup(pdf, 2)

        # P3 — Overview table
        print("  Page 3: Overview Table")
        page_overview_table(pdf, 3)

        # P4 — A* deep dive
        print("  Page 4: A*")
        page_algo_deepdive(
            pdf, 4,
            name="A*  (A-Star)",
            color=C["astar"],
            tagline="Optimal single-source shortest path with admissible heuristic",
            how_it_works=[
                "Maintains a priority queue ordered by f(n) = g(n) + h(n)",
                "g(n) = exact cost from start to n  |  h(n) = heuristic lower bound to goal",
                "Expands the node with lowest f first; guarantees optimal path when h is admissible",
                "Multi-goal handled with bitmask: state = (position, visited_mask)",
            ],
            heuristics=[
                "h_anchor: Euclidean distance to nearest unvisited goal",
                "Plus MST cost of remaining goals (Prim's algorithm)",
                "Combined: h = min_dist_to_goal + mst_cost(remaining_goals)",
                "Always admissible (never overestimates) → guarantees optimal solution",
            ],
            params={"Diagonal": "Yes (8-dir)", "On-road cost": "1.0",
                    "Off-road cost": "5.0", "Max expansions": "5,000,000"},
            strengths=["Provably optimal path", "Simple, well-understood", "Fast heuristic (Euclidean+MST)"],
            weaknesses=["Explores too broadly near obstacles", "Grid-constrained (staircase) paths",
                        "No anytime behavior — must finish before returning answer"],
            stats=[("Expansions", "211,023"), ("Time", "2.60 s"),
                   ("Path", "979 wp"), ("Optimality", "Guaranteed")],
        )

        # P5 — ARA* deep dive
        print("  Page 5: ARA*")
        page_algo_deepdive(
            pdf, 5,
            name="ARA*  (Anytime Repairing A*)",
            color=C["ara"],
            tagline="Returns a good solution fast, then refines it toward optimality over time",
            how_it_works=[
                "Runs weighted A* with inflated heuristic: f = g + ε·h  (ε starts high)",
                "High ε → finds a path quickly but suboptimal (up to ε× optimal cost)",
                "Decreases ε each iteration and repairs the solution using INCONS set",
                "Stops when ε=1 (optimal) or time budget exceeded — returns best found so far",
            ],
            heuristics=[
                "Same admissible heuristic as A* (Euclidean + MST)",
                "Inflated by ε: f = g + ε·h  gives bounded suboptimality guarantee",
                "ε=3.0 → path ≤ 3× optimal  |  ε=1.0 → path is optimal",
                "Reuses g-values across iterations — no wasted work",
            ],
            params={"ε start": "3.0", "ε end": "1.0", "ε step": "−0.5",
                    "Max iterations": "10", "Max expansions/iter": "2,000,000"},
            strengths=["Returns first path in <1s (ε=3)", "Provable suboptimality bound at each step",
                       "Final result matches A* optimal"],
            weaknesses=["1.27M total expansions — heaviest search", "Later iterations do redundant work",
                        "No multi-objective support"],
            stats=[("Expansions", "1,276,755"), ("Time", "10.30 s"),
                   ("Path", "909 wp"), ("Iterations", "10 (3 improving)")],
        )

        # P6 — MHA* deep dive
        print("  Page 6: MHA*")
        page_algo_deepdive(
            pdf, 6,
            name="MHA*  (Multi-Heuristic A*)",
            color=C["mha"],
            tagline="Three parallel searches sharing state — fastest algorithm in the comparison",
            how_it_works=[
                "Maintains 3 separate OPEN lists, one per heuristic (anchor + 2 inadmissible)",
                "Shared g-values and came_from: any heuristic can benefit from another's expansions",
                "Expansion rule: expand from inadmissible i if key_i ≤ w2 × anchor_min_key",
                "Otherwise expand anchor — this preserves the w1-suboptimality guarantee",
            ],
            heuristics=[
                "H0 Anchor (admissible): Euclidean + MST  — guarantees w1-bounded optimality",
                "H1 Manhattan+MST (inadmissible): more aggressive, ignores diagonal savings",
                "H2 Greedy Road-Biased: 2×(Euclidean+MST), discounted 0.8× when on road cell",
                "w1=2.0 (suboptimality bound)  |  w2=1.2 (inadmissible expansion threshold)",
            ],
            params={"w1": "2.0", "w2": "1.2", "Heuristics": "3", "Max expansions": "5,000,000"},
            strengths=["Fastest: 79,775 expansions, 1.01s", "Multiple heuristics explore diverse corridors",
                       "Bounded suboptimality guarantee (2× optimal)"],
            weaknesses=["Single-objective only (distance)", "Greedy Road heuristic fired 0 times — needs tuning",
                        "w2 tuning sensitive: too high → Manhattan dominates; too low → only Anchor runs"],
            stats=[("Expansions", "79,775"), ("Time", "1.01 s"),
                   ("Path", "924 wp"), ("Anchor expns", "54,949")],
        )

        # P7 — NAMOA*-dr deep dive
        print("  Page 7: NAMOA*-dr")
        page_algo_deepdive(
            pdf, 7,
            name="NAMOA*-dr  (Non-dominated Multi-Objective A* — Delayed Reopening)",
            color=C["namoa"],
            tagline="Finds ALL Pareto-optimal paths across 3 objectives simultaneously",
            how_it_works=[
                "Each state holds a set of non-dominated cost vectors (labels), not a single g-value",
                "A label g1 dominates g2 if g1[i] ≤ g2[i] for all i and g1[j] < g2[j] for some j",
                "Delayed reopening: dominated labels batched and reopened periodically (not immediately)",
                "ε-dominance (ε=0.15): g1 ε-dominates g2 if g1[i] ≤ (1+ε)·g2[i] — controls label explosion",
            ],
            heuristics=[
                "H_distance: Euclidean + MST (same as A*) — lower bound on remaining distance",
                "H_offroad: precomputed via reverse 0-1 BFS from each goal — exact min off-road cells",
                "H_roughness: precomputed via reverse Dijkstra from each goal — exact min roughness sum",
                "All three components are admissible → guarantees complete Pareto front",
            ],
            params={"ε": "0.15", "Roughness window": "5×5", "Max expansions": "1,500,000",
                    "Objectives": "3 (dist, offroad, rough)"},
            strengths=["Only algorithm returning multiple trade-off paths", "Complete Pareto front with ε-approximation",
                       "Precomputed heuristics give tight bounds"],
            weaknesses=["361 seconds (6 min) on small map", "Label explosion even with ε-dominance",
                        "Impractical for larger Austin→Dallas scale without major improvement"],
            stats=[("Expansions", "1,154,829"), ("Time", "361.54 s"),
                   ("Path (selected)", "902 wp"), ("Pareto solutions", "5")],
        )

        # P8 — Visual comparison
        print("  Page 8: Visual Comparison")
        page_image(
            pdf, 8,
            img_path=os.path.join(IMG_BASE, "four_algorithm_comparison.png"),
            title="Visual Comparison — All Four Algorithms",
            subtitle="Austin test raster | 2×2 panel layout | each algorithm in its own panel",
            caption="Blue = A* explored | Yellow = ARA* explored | Purple = MHA* explored | Orange = NAMOA*-dr explored"
                    "  ·  Bright path lines show final selected route for each algorithm",
        )

        # P9 — Pareto front
        print("  Page 9: Pareto Front")
        page_pareto(pdf, 9)

        # P10 — Limitations
        print("  Page 10: Limitations")
        page_limitations(pdf, 10)

        # P11 — Theta*
        print("  Page 11: Theta*")
        page_recommendation(
            pdf, 11,
            name="Theta*",
            color=C["theta"],
            tagline="Any-angle pathfinding — removes the grid-direction constraint entirely",
            problem="All current algorithms snap to 8 directions, producing staircase paths 5–10% longer than necessary.",
            how_works=[
                "During neighbor relaxation, try shortcutting through the grandparent node",
                "If line_of_sight(grandparent, neighbor) is clear → connect directly",
                "Result: paths follow natural straight lines through open terrain",
            ],
            how_works_detail=(
                "Standard A*:\n"
                "  relax(parent → current → neighbor)\n\n"
                "Theta* change:\n"
                "  if line_of_sight(grandparent, neighbor):\n"
                "      relax(grandparent → neighbor)   ← skip intermediate cells\n"
                "  else:\n"
                "      relax(parent → neighbor)         ← fallback to A*"
            ),
            formula="f(n) = g(parent_or_grandparent) + dist(parent_or_grandparent, n) + h(n)",
            benefits=[
                "Paths 5–10% shorter in Euclidean distance",
                "Dramatically fewer direction changes (critical for hyperloop construction cost)",
                "Same time complexity as A* — no meaningful slowdown",
                "Lazy Theta* variant even faster on dense grids",
            ],
            effort="LOW — ~30 line change to A* neighbor loop + line_of_sight() helper",
            compare_note=(
                "vs A*: same optimality class, shorter and smoother paths\n"
                "vs MHA*: can be combined → Multi-Heuristic Theta*\n"
                "vs NAMOA*: can extend to multi-objective Theta* with cost vectors"
            ),
        )

        # P12 — Bidirectional A*
        print("  Page 12: Bidirectional A*")
        page_recommendation(
            pdf, 12,
            name="Bidirectional A*",
            color=C["bidir"],
            tagline="Meet-in-the-middle search — halves the effective search depth",
            problem="Single-direction search must traverse the entire Austin–Dallas corridor. "
                    "For long-distance routes, this wastes the majority of expansions on empty space.",
            how_works=[
                "Run two simultaneous A* frontiers: Forward (from Austin) and Backward (from Dallas)",
                "Each iteration expands the frontier with the smaller f-value",
                "Stop when the two frontiers overlap; the meeting point yields the optimal path",
            ],
            how_works_detail=(
                "Forward search:   Austin  →  meeting point\n"
                "Backward search:  Dallas  →  meeting point\n\n"
                "Search space comparison:\n"
                "  One-way A*:     O(b^d)       b=branching, d=depth\n"
                "  Bidirectional:  O(b^(d/2))   ← square root reduction\n\n"
                "For d=1000 expansions needed:\n"
                "  A*:   1,000 expansions\n"
                "  BiA*: ~32 expansions from each end = ~64 total"
            ),
            formula="Stop when: g_fwd(u) + g_bwd(u) ≤ best_path_cost  for any node u",
            benefits=[
                "~50–70% fewer expansions on long corridors",
                "Pairs with any heuristic — use same Euclidean+MST",
                "Can combine with Theta* → Bidirectional Theta*",
                "Critical for scaling to full Austin → Dallas distance",
            ],
            effort="MEDIUM — two open lists, backward heuristic, termination condition",
            compare_note=(
                "vs A* (211K expns): expect ~70–120K expansions\n"
                "vs MHA* (79K expns): comparable, different trade-off\n"
                "Combine both: Bidirectional MHA* could be even faster"
            ),
        )

        # P13 — EMOA*
        print("  Page 13: EMOA*")
        page_recommendation(
            pdf, 13,
            name="EMOA*",
            color=C["emoa"],
            tagline="Enhanced Multi-Objective A* (2022) — same Pareto front, far fewer expansions",
            problem="NAMOA*-dr takes 361 seconds on a small map due to label explosion. "
                    "This makes multi-objective routing infeasible at scale.",
            how_works=[
                "Labels processed in sorted order of first objective (distance) using a priority queue",
                "Tighter dominance pruning: uses G-closed sets indexed by first objective",
                "Detects and discards dominated labels earlier — before they enter the open list",
                "Result: exponentially fewer labels survive to expansion",
            ],
            how_works_detail=(
                "NAMOA*-dr label check (per expansion):\n"
                "  Compare new label against ALL labels in closed set\n"
                "  → O(|closed|) per expansion\n\n"
                "EMOA* label check:\n"
                "  Closed set indexed by objective 1 (sorted)\n"
                "  Binary search prunes majority of dominated labels\n"
                "  → O(log|closed|) per expansion\n\n"
                "Benchmark results (literature):\n"
                "  NAMOA*-dr: 100% baseline\n"
                "  EMOA*:     10–40% of NAMOA* expansions on road networks"
            ),
            formula="key(label) = (g[0], g[1], g[2])  sorted by g[0] first",
            benefits=[
                "Same complete Pareto front guarantee as NAMOA*-dr",
                "10–40% of NAMOA*-dr expansions (literature benchmarks on road networks)",
                "Same ε-dominance extension possible for approximate Pareto",
                "Published 2022 — state-of-the-art multi-objective pathfinding",
            ],
            effort="MEDIUM — replaces NAMOA*-dr; same data interface and output format",
            compare_note=(
                "vs NAMOA*-dr (361s): expect 40–150s on same Austin map\n"
                "Same 5 Pareto solutions, same output format\n"
                "Direct drop-in replacement in compare_all_algorithms.py"
            ),
        )

        # P14 — ALT
        print("  Page 14: ALT")
        page_recommendation(
            pdf, 14,
            name="ALT  (A* + Landmarks + Triangle Inequality)",
            color=C["alt"],
            tagline="Precomputed landmark distances give tight heuristics — critical for large maps",
            problem="Euclidean distance heuristic is weak on maps with obstacles (rivers, protected areas). "
                    "For a full Austin→Dallas raster (12M+ cells), A* expansions would be unacceptable.",
            how_works=[
                "Choose ~16 landmark nodes spread across the map (corners, key waypoints)",
                "Precompute exact shortest distances from every cell to every landmark (Dijkstra×16)",
                "At query time: h(u, goal) ≥ |d(landmark, goal) − d(landmark, u)|  for each landmark",
                "Take the maximum across all landmarks → tightest admissible lower bound",
            ],
            how_works_detail=(
                "Triangle inequality:\n"
                "  d(u, goal) ≥ d(L, goal) − d(L, u)   for any landmark L\n"
                "  d(u, goal) ≥ d(u, L) − d(goal, L)\n\n"
                "Example on Austin map:\n"
                "  Euclidean h:  200 cells  (weak lower bound)\n"
                "  ALT h:        195 cells  (tighter — avoids known detours)\n\n"
                "Preprocessing:\n"
                "  Run 16 Dijkstras once  → save distance arrays\n"
                "  Reuse for every future query on same map\n"
                "  Preprocessing: ~minutes  |  Query speedup: 2–5×"
            ),
            formula="h_ALT(u, goal) = max over all landmarks L of |d(L, goal) − d(L, u)|",
            benefits=[
                "2–5× fewer A* expansions vs Euclidean heuristic on obstacle-heavy maps",
                "One-time preprocessing — reusable for all Austin↔Dallas queries",
                "Works with any single-objective algorithm (A*, ARA*, MHA*, Theta*)",
                "Standard technique in OpenStreetMap (OSRM) and geographic routing",
            ],
            effort="MEDIUM — precomputation script + modified heuristic function",
            compare_note=(
                "vs A* (211K expns): expect 50–100K with good landmarks\n"
                "vs MHA*: can replace all 3 heuristics with ALT variants\n"
                "Most impactful when scaling to Austin→Dallas full raster"
            ),
        )

        # P15 — Testing plan
        print("  Page 15: Testing Plan")
        page_testing_plan(pdf, 15)

        # P16 — Summary
        print("  Page 16: Summary")
        page_summary(pdf, 16)

        # PDF metadata
        d = pdf.infodict()
        d["Title"]   = "Hyperloop Pathfinding Algorithm Analysis"
        d["Author"]  = "Guadaloop — University of Texas at Austin"
        d["Subject"] = "A*, ARA*, MHA*, NAMOA*-dr + Theta*, BiA*, EMOA*, ALT recommendations"
        d["CreationDate"] = datetime.datetime.now()

    print(f"\nDone! PDF saved to:\n  {OUT_PDF}")
    return OUT_PDF


if __name__ == "__main__":
    main()
