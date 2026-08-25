"""
Plots for the cubics f(a) = a^3 - 6a^2 + 11a - 6 and g(a) = a^3 - 6a^2 + 11a,
and the geometry of the equation g(a) = 6.

Key identity used throughout: with u = a - 2,
    f(a) = u^3 - u        and      g(a) = u^3 - u + 6 = f(a) + 6
so g is f translated up by exactly 6, and g(a) = 6  <=>  f(a) = 0.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# --- design tokens (validated categorical slots 1 and 2, light surface) -------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e1"
AXIS = "#c9c8c2"
SERIES_1 = "#2a78d6"   # slot 1, blue   -> f(a)
SERIES_2 = "#eb6834"   # slot 2, orange -> g(a)
GUIDE = "#8b8a84"      # reference lines are annotation, never a series colour

A_MIN, A_MAX = 0.3, 3.7
ROOTS = (1.0, 2.0, 3.0)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "axes.labelcolor": INK_2,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "axes.linewidth": 0.9,
})


def f(a):
    return a ** 3 - 6 * a ** 2 + 11 * a - 6


def g(a):
    return a ** 3 - 6 * a ** 2 + 11 * a


def new_axes(figsize=(7.0, 3.3)):
    """A recessive frame: no top/right spines, muted grid beneath the marks."""
    fig, ax = plt.subplots(figsize=figsize)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.set_xlim(A_MIN, A_MAX)
    ax.set_xlabel("a")
    return fig, ax


def root_markers(ax, ys, colour, labels, dy=0.16, ha="center"):
    """Roots get a marker with a 2px surface ring plus a direct label."""
    for a, y, label in zip(ROOTS, ys, labels):
        ax.plot([a], [y], marker="o", markersize=8, color=colour,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=5,
                clip_on=False, linestyle="none")
        ax.annotate(label, xy=(a, y), xytext=(0, dy * 72), textcoords="offset points",
                    ha=ha, va="bottom" if dy > 0 else "top", fontsize=9.5,
                    color=INK, zorder=6)


def finish(fig, ax, title, subtitle, path):
    ax.set_title(title, color=INK, fontsize=12.5, pad=32, loc="left", weight="bold")
    ax.annotate(subtitle, xy=(0, 1), xycoords="axes fraction", xytext=(0, 9),
                textcoords="offset points", fontsize=9.5, color=INK_2,
                va="bottom", ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("wrote", path)


OUT = "figs"
a = np.linspace(A_MIN, A_MAX, 1400)

# --- Figure 1: f(a) = a^3 - 6a^2 + 11a - 6, cutting the axis at 1, 2, 3 ------
fig, ax = new_axes()
ax.axhline(0, color=GUIDE, linewidth=1.1, linestyle=(0, (5, 4)), zorder=2)
ax.plot(a, f(a), color=SERIES_1, linewidth=2.2, solid_capstyle="round", zorder=4)
root_markers(ax, [0, 0, 0], SERIES_1, ["a = 1", "a = 2", "a = 3"], dy=0.20)
ax.annotate("y = 0", xy=(A_MAX, 0), xytext=(-4, 6), textcoords="offset points",
            ha="right", fontsize=9, color=INK_2)
ax.set_ylabel("f(a)")
finish(fig, ax,
       "f(a) = a³ − 6a² + 11a − 6",
       "Three simple real zeros — the curve crosses y = 0 at a = 1, 2 and 3",
       f"{OUT}/fig1_f.png")

# --- Figure 2: g(a) = a^3 - 6a^2 + 11a meeting the line y = 6 ----------------
fig, ax = new_axes()
ax.axhline(6, color=GUIDE, linewidth=1.1, linestyle=(0, (5, 4)), zorder=2)
ax.plot(a, g(a), color=SERIES_2, linewidth=2.2, solid_capstyle="round", zorder=4)
root_markers(ax, [6, 6, 6], SERIES_2, ["a = 1", "a = 2", "a = 3"], dy=0.20)
ax.annotate("y = 6", xy=(A_MAX, 6), xytext=(-4, 6), textcoords="offset points",
            ha="right", fontsize=9, color=INK_2)
ax.set_ylabel("g(a)")
finish(fig, ax,
       "g(a) = a³ − 6a² + 11a",
       "The line y = 6 meets the curve exactly three times: a = 1, 2 and 3",
       f"{OUT}/fig2_g.png")

# --- Figure 3: the two curves together — g is f lifted by 6 ------------------
fig, ax = new_axes(figsize=(7.0, 3.9))
ax.axhline(0, color=GUIDE, linewidth=1.1, linestyle=(0, (5, 4)), zorder=2)
ax.axhline(6, color=GUIDE, linewidth=1.1, linestyle=(0, (5, 4)), zorder=2)
ax.plot(a, f(a), color=SERIES_1, linewidth=2.2, solid_capstyle="round",
        zorder=4, label="f(a) = a³ − 6a² + 11a − 6")
ax.plot(a, g(a), color=SERIES_2, linewidth=2.2, solid_capstyle="round",
        zorder=4, label="g(a) = a³ − 6a² + 11a")
for a0 in ROOTS:
    ax.plot([a0, a0], [0, 6], color=AXIS, linewidth=1.0, linestyle=(0, (2, 3)), zorder=3)
    for y0, colour in ((0.0, SERIES_1), (6.0, SERIES_2)):
        ax.plot([a0], [y0], marker="o", markersize=8, color=colour,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=5, linestyle="none")
    ax.annotate(f"a = {a0:.0f}", xy=(a0, 6), xytext=(0, 12), textcoords="offset points",
                ha="center", fontsize=9.5, color=INK, zorder=6)
ax.annotate("y = 6", xy=(A_MIN, 6), xytext=(4, 6), textcoords="offset points",
            fontsize=9, color=INK_2)
ax.annotate("y = 0", xy=(A_MIN, 0), xytext=(4, 6), textcoords="offset points",
            fontsize=9, color=INK_2)
ax.set_ylabel("y")
ax.set_ylim(-3.8, 10.4)
leg = ax.legend(loc="lower right", frameon=False, fontsize=9.5,
                handlelength=1.6, borderpad=0.2)
for text in leg.get_texts():
    text.set_color(INK_2)
finish(fig, ax,
       "The same cubic, six units apart",
       "g(a) = f(a) + 6, so solving g(a) = 6 is exactly solving f(a) = 0",
       f"{OUT}/fig3_both.png")

# --- numbers quoted in the document, recomputed rather than asserted ---------
crit = [2 - 1 / np.sqrt(3), 2 + 1 / np.sqrt(3)]
print("f at roots:", [round(float(f(r)), 12) for r in ROOTS])
print("g at roots:", [round(float(g(r)), 12) for r in ROOTS])
print("turning points a =", [round(c, 6) for c in crit],
      "f =", [round(float(f(c)), 6) for c in crit],
      "g =", [round(float(g(c)), 6) for c in crit])
print("numpy roots of f:", sorted(np.roots([1, -6, 11, -6]).real.round(12)))
print("numpy roots of g:", np.roots([1, -6, 11, 0]))
