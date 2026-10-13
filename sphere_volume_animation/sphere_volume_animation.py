#!/usr/bin/env python3
"""
Animated derivation of the volume of a sphere using calculus (disk method).

    V = integral from -R to R of pi * (R^2 - x^2) dx = 4/3 * pi * R^3

The animation walks through five steps:

  1. Revolve   Spin the half-circle y = sqrt(R^2 - x^2) about the x-axis.
  2. Slice     A slice perpendicular to the x-axis is a disk of radius
               r = sqrt(R^2 - x^2), so its area is A(x) = pi * (R^2 - x^2).
  3. Sum       Stack n disks of thickness dx. Their volumes form a Riemann
               sum that approaches the true volume as n grows.
  4. Limit     As dx -> 0 the sum becomes an integral; watch the volume
               accumulate as the area under A(x).
  5. Evaluate  Work out the integral to get V = 4/3 * pi * R^3.

Usage:
    pip install numpy matplotlib     # ffmpeg is required for .mp4 output

    python sphere_volume_animation.py                        # sphere_volume.mp4
    python sphere_volume_animation.py -o sphere_volume.gif   # animated GIF
    python sphere_volume_animation.py --show                 # live window
    python sphere_volume_animation.py --frame 300 -o f.png   # single still
"""

import argparse
import shutil
import sys
from pathlib import Path

import matplotlib
import numpy as np

if "--show" not in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

# ----------------------------------------------------------------------------
# Constants and styling
# ----------------------------------------------------------------------------

R = 1.0  # sphere radius; every number shown on screen is in units of R
EXACT = 4 / 3 * np.pi * R**3
FPS = 24

BG = "#0e1117"
PANEL = "#151a24"
FG = "#e6e8ef"
MUTED = "#7d8597"
GRID = "#2a3140"
GOLD = "#ffd166"
RED = "#ff6b6b"
CYAN = "#22d3ee"

# Slices are coloured by their x-position so a disk in 3-D and its bar
# under the A(x) graph always share the same colour.
CMAP = LinearSegmentedColormap.from_list(
    "slices", ["#22d3ee", "#818cf8", "#e879f9", "#fb923c"]
)
LIGHT = np.array([0.35, -0.55, 0.75])
LIGHT /= np.linalg.norm(LIGHT)

ELEV, AZIM = 20, -58

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "text.color": FG,
    }
)


# ----------------------------------------------------------------------------
# Maths
# ----------------------------------------------------------------------------


def radius(x):
    """Radius of the cross-section disk at x: r = sqrt(R^2 - x^2)."""
    return np.sqrt(np.clip(R**2 - np.asarray(x, float) ** 2, 0, None))


def area(x):
    """Cross-section area A(x) = pi * r^2 = pi * (R^2 - x^2)."""
    return np.pi * (R**2 - np.asarray(x, float) ** 2)


def volume_up_to(t):
    """V(t) = integral from -R to t of A(x) dx = pi (R^2 t - t^3/3 + 2R^3/3)."""
    return np.pi * (R**2 * t - t**3 / 3 + 2 * R**3 / 3)


def riemann_sum(n):
    """Midpoint Riemann sum of A(x) on [-R, R] with n slices."""
    edges = np.linspace(-R, R, n + 1)
    mids = (edges[:-1] + edges[1:]) / 2
    return edges, mids, float(np.sum(area(mids)) * (2 * R / n))


def ease(p):
    p = np.clip(p, 0.0, 1.0)
    return p * p * (3 - 2 * p)


def stage(p, start, end):
    """Eased progress of p within the sub-interval [start, end]."""
    return ease((p - start) / (end - start))


# ----------------------------------------------------------------------------
# 3-D geometry helpers
# ----------------------------------------------------------------------------


def x_color(x):
    return np.asarray(CMAP((np.asarray(x) + R) / (2 * R)))[..., :3]


def revolve(x, r, theta):
    """Rotate the profile (x, r(x)) about the x-axis -> X, Y, Z grids.

    theta = 0 puts the profile in the upright x-z plane, so it looks just
    like the 2-D graph of y = r(x).
    """
    X, T = np.meshgrid(np.asarray(x, float), theta, indexing="ij")
    rr = np.broadcast_to(np.asarray(r, float)[:, None], X.shape)
    return X, rr * np.sin(T), rr * np.cos(T)


def quads(X, Y, Z):
    """Split a parametric surface grid into an (N, 4, 3) array of quads."""
    P = np.stack([X, Y, Z], axis=-1)
    q = np.stack([P[:-1, :-1], P[1:, :-1], P[1:, 1:], P[:-1, 1:]], axis=2)
    return q.reshape(-1, 4, 3)


def shade(q, base, ambient=0.38):
    """Simple Lambert shading of each quad, two-sided."""
    n = np.cross(q[:, 2] - q[:, 0], q[:, 3] - q[:, 1])
    length = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.where(length == 0, 1, length)
    k = ambient + (1 - ambient) * np.abs(n @ LIGHT)
    base = np.broadcast_to(np.asarray(base, float), (len(q), 3))
    return np.clip(base * k[:, None], 0, 1)


def solid_surface(x, r, theta, color=None):
    q = quads(*revolve(x, r, theta))
    base = x_color(q[:, :, 0].mean(axis=1)) if color is None else to_rgb(color)
    return q, shade(q, base)


def disk(x0, x1, r, n_theta=40, color=None, ambient=0.38):
    """A thin cylinder (side + both caps) between x0 and x1."""
    th = np.linspace(0, 2 * np.pi, n_theta + 1)
    q = np.concatenate(
        [
            quads(*revolve([x0, x1], [r, r], th)),
            quads(*revolve([x0, x0], [0, r], th)),
            quads(*revolve([x1, x1], [0, r], th)),
        ]
    )
    base = x_color((x0 + x1) / 2) if color is None else to_rgb(color)
    return q, shade(q, base, ambient)


def add_polys(ax, q, colors, alpha=1.0, edge_darken=None, zorder=2):
    rgba = np.column_stack([colors, np.full(len(colors), alpha)])
    if edge_darken is None:
        edges, lw = rgba, 0.3  # paint edges with the face colour to hide seams
    else:
        edges = np.column_stack([colors * edge_darken, np.full(len(colors), alpha)])
        lw = 0.5
    coll = Poly3DCollection(q, facecolors=rgba, edgecolors=edges, linewidths=lw)
    coll.set_zorder(zorder)
    ax.add_collection3d(coll)


def ghost_sphere(ax, alpha=0.25):
    """Faint wireframe of the full sphere for reference."""
    phi = np.linspace(0, np.pi, 15)
    X, Y, Z = revolve(-R * np.cos(phi), R * np.sin(phi), np.linspace(0, 2 * np.pi, 25))
    ax.plot_wireframe(X, Y, Z, color=MUTED, linewidth=0.5, alpha=alpha, zorder=1)


def x_axis(ax):
    ax.plot([-1.55 * R, 1.55 * R], [0, 0], [0, 0], color=FG, lw=1.0, alpha=0.75,
            ls=(0, (4, 3)), zorder=5)
    ax.text(1.65 * R, 0, 0, "x", color=FG, fontsize=14, zorder=5)


# ----------------------------------------------------------------------------
# The animation
# ----------------------------------------------------------------------------


class SphereVolumeAnimation:
    STEPS = ["Revolve", "Slice", "Sum", "Limit", "Evaluate"]
    RIEMANN_NS = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48]
    FRAMES_PER_N = 16

    def __init__(self):
        self.fig = fig = plt.figure(figsize=(12.8, 7.2), facecolor=BG)
        self.ax3d = fig.add_axes([-0.02, -0.02, 0.58, 0.9], projection="3d")
        self.ax2d = fig.add_axes([0.61, 0.52, 0.36, 0.33])
        self.axtx = fig.add_axes([0.585, 0.02, 0.41, 0.44])

        fig.text(0.03, 0.935, "Volume of a Sphere with Calculus", fontsize=22,
                 fontweight="bold", color=FG)
        fig.text(0.03, 0.895, "The disk method:  slice  →  sum  →  take the limit",
                 fontsize=12, color=MUTED)
        self.step_labels = [
            fig.text(0.575 + 0.08 * i, 0.935, f"{i + 1} {name}", fontsize=11, color=MUTED)
            for i, name in enumerate(self.STEPS)
        ]

        riemann_frames = len(self.RIEMANN_NS) * self.FRAMES_PER_N + 24
        self.scenes = [
            (self.scene_revolve, 84),
            (self.scene_slice, 108),
            (self.scene_sum, riemann_frames),
            (self.scene_limit, 96),
            (self.scene_evaluate, 144),
        ]
        self.n_frames = sum(length for _, length in self.scenes)

    # -- frame dispatch ------------------------------------------------------

    def draw(self, frame):
        for step, (scene, length) in enumerate(self.scenes):
            if frame < length or step == len(self.scenes) - 1:
                break
            frame -= length
        for i, label in enumerate(self.step_labels):
            active = i == step
            label.set_color(GOLD if active else MUTED)
            label.set_fontweight("bold" if active else "normal")

        self.reset_3d()
        scene(frame, min(frame / (length - 1), 1.0))
        return []

    def reset_3d(self, azim=AZIM):
        ax = self.ax3d
        ax.cla()
        ax.set_axis_off()
        ax.set_facecolor(BG)
        ax.computed_zorder = False
        lim = 1.1 * R
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_box_aspect((1, 1, 1), zoom=1.3)
        ax.view_init(elev=ELEV, azim=azim)

    # -- 2-D panel helpers ---------------------------------------------------

    def reset_2d(self, title, ylim, yticks, equal=False):
        ax = self.ax2d
        ax.cla()
        ax.set_facecolor(PANEL)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=10)
        ax.set_xlim(-1.2 * R, 1.2 * R)
        ax.set_ylim(*ylim)
        ax.set_xticks([-R, 0, R], ["−R", "0", "R"])
        ax.set_yticks(*yticks)
        ax.set_title(title, color=FG, fontsize=12, loc="left", pad=8)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_aspect("equal" if equal else "auto", adjustable="box")
        return ax

    def area_graph(self, title):
        ax = self.reset_2d(title, (0, 1.25 * np.pi * R**2), ([0, np.pi * R**2], ["0", r"$\pi R^2$"]))
        x = np.linspace(-R, R, 300)
        ax.plot(x, area(x), color=FG, lw=2, zorder=4)
        ax.set_xlabel("x", color=MUTED, fontsize=11, labelpad=0)
        return ax

    @staticmethod
    def shade_area(ax, a, b, segments=160):
        """Fill under A(x) on [a, b] with the slice colour gradient."""
        if b - a < 1e-6:
            return
        x = np.linspace(a, b, segments + 1)
        y = area(x)
        verts = [[(x[i], 0), (x[i], y[i]), (x[i + 1], y[i + 1]), (x[i + 1], 0)]
                 for i in range(segments)]
        colors = x_color((x[:-1] + x[1:]) / 2)
        ax.add_collection(PolyCollection(verts, facecolors=colors, edgecolors=colors,
                                         linewidths=0.3, alpha=0.85, zorder=2))

    # -- text panel helper ---------------------------------------------------

    def write(self, header, lines, visible=None):
        """lines: list of (text, fontsize, gap-before) tuples."""
        ax = self.axtx
        ax.cla()
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.text(0, 0.98, header, color=GOLD, fontsize=15, fontweight="bold", va="top")
        y = 0.86
        for i, (text, size, gap) in enumerate(lines):
            y -= gap
            if visible is not None and i >= visible:
                break
            boxed = text.startswith("BOX:")
            ax.text(0.02, y, text[4:] if boxed else text, fontsize=size, va="center",
                    color=FG,
                    bbox=dict(boxstyle="round,pad=0.45", fc=PANEL, ec=GOLD, lw=1.6)
                    if boxed else None)
            y -= 0.02

    # -- Step 1: build the sphere as a solid of revolution -------------------

    def scene_revolve(self, f, p):
        ax = self.ax3d
        phi = np.linspace(0, np.pi, 61)
        x, r = -R * np.cos(phi), R * np.sin(phi)

        angle = 2 * np.pi * stage(p, 0.12, 0.85)
        if angle > 1e-3:
            theta = np.linspace(0, angle, max(3, int(56 * angle / (2 * np.pi)) + 1))
            add_polys(ax, *solid_surface(x, r, theta), alpha=0.95)
        x_axis(ax)
        # The generating half-circle, carried around by the rotation.
        ax.plot(x, r * np.sin(angle), r * np.cos(angle), color=GOLD, lw=3, zorder=6)
        # A little arc showing the direction of rotation.
        arc = np.linspace(0, max(angle, 1e-3), 60)
        ax.plot(np.full_like(arc, 1.3 * R), 0.3 * R * np.sin(arc), 0.3 * R * np.cos(arc),
                color=RED, lw=2, zorder=6)
        ax.scatter([1.3 * R], [0.3 * R * np.sin(angle)], [0.3 * R * np.cos(angle)],
                   color=RED, s=25, zorder=7)
        ax.text(1.3 * R, 0, 0.45 * R, f"{np.degrees(angle):3.0f}°", color=RED,
                fontsize=11, zorder=7)

        ax2 = self.reset_2d("Generating curve  " + r"$y=\sqrt{R^2-x^2}$", (-0.12, 1.25 * R),
                            ([0, R], ["0", "R"]), equal=True)
        t = np.linspace(0, np.pi, 200)
        ax2.fill_between(-R * np.cos(t), R * np.sin(t), color=GOLD, alpha=0.08)
        ax2.plot(-R * np.cos(t), R * np.sin(t), color=GOLD, lw=2.5)
        ax2.annotate("", xy=(1.17 * R, 0), xytext=(-1.17 * R, 0),
                     arrowprops=dict(arrowstyle="->", color=FG, lw=1.2))
        ax2.text(0, 0.06 * R, "axis of rotation (x-axis)", color=MUTED, fontsize=9, ha="center")

        self.write("Step 1 · Build the sphere by revolution", [
            (r"Take the upper half of the circle  $x^2+y^2=R^2$:", 12.5, 0.02),
            (r"$y=\sqrt{R^2-x^2},\quad -R\leq x\leq R$", 16, 0.10),
            ("Spin it 360° around the x-axis.", 12.5, 0.13),
            ("Every point of the curve sweeps out a circle,", 12.5, 0.09),
            ("and together they form a sphere of radius R.", 12.5, 0.07),
        ])

    # -- Step 2: one slice is a disk -----------------------------------------

    def scene_slice(self, f, p):
        ax = self.ax3d
        if p < 0.55:
            xs = -0.8 * R + 1.6 * R * stage(p, 0.0, 0.55)
        else:
            xs = 0.8 * R - 0.35 * R * stage(p, 0.55, 0.75)
        r = float(radius(xs))
        dx = 0.06 * R

        ghost_sphere(ax, alpha=0.35)
        add_polys(ax, *disk(xs - dx / 2, xs + dx / 2, r, n_theta=48, color=GOLD,
                               ambient=0.65))
        x_axis(ax)
        # The right triangle O -> (x, 0) -> P on the sphere, in the x-z plane.
        ax.plot([0, xs], [0, 0], [0, 0], color=CYAN, lw=3, zorder=6)
        ax.plot([xs, xs], [0, 0], [0, r], color=RED, lw=3, zorder=6)
        ax.plot([0, xs], [0, 0], [0, r], color=FG, lw=2, zorder=6)
        ax.text(xs / 2, 0, -0.17 * R, "x", color=CYAN, fontsize=13, zorder=7)
        ax.text(xs + 0.06 * R, 0, r / 2, "r", color=RED, fontsize=13, zorder=7)
        ax.text(xs / 2 - 0.12 * R, 0, r / 2 + 0.06 * R, "R", color=FG, fontsize=13, zorder=7)

        ax2 = self.reset_2d("Cross-section through the axis", (-0.12, 1.25 * R),
                            ([0, R], ["0", "R"]), equal=True)
        t = np.linspace(0, np.pi, 200)
        ax2.plot(-R * np.cos(t), R * np.sin(t), color=MUTED, lw=1.5)
        ax2.fill_between([xs - dx / 2, xs + dx / 2], [r, r], color=GOLD, alpha=0.5)
        ax2.plot([0, xs], [0, 0], color=CYAN, lw=3)
        ax2.plot([xs, xs], [0, r], color=RED, lw=3)
        ax2.plot([0, xs], [0, r], color=FG, lw=2)
        ax2.scatter([xs], [r], color=GOLD, s=30, zorder=5)
        side = 1 if xs >= 0 else -1
        ax2.text(xs / 2, -0.09, "x", color=CYAN, fontsize=12, ha="center", va="center")
        ax2.text(xs + side * 0.07, r / 2, "r", color=RED, fontsize=12, ha="center")
        ax2.text(xs / 2 - side * 0.09, r / 2 + 0.05, "R", color=FG, fontsize=12, ha="center")
        ax2.text(-1.15 * R, 1.13 * R,
                 f"x = {xs:+.2f}R    r = {r:.2f}R    A = {area(xs):.2f}R²",
                 color=GOLD, fontsize=10)

        self.write("Step 2 · Slice it into thin disks", [
            ("Cut perpendicular to the x-axis at position x.", 12.5, 0.02),
            ("The slice is a disk. By Pythagoras:", 12.5, 0.08),
            (r"$r^2+x^2=R^2\ \ \Rightarrow\ \ r=\sqrt{R^2-x^2}$", 15, 0.10),
            ("So the area of the slice is", 12.5, 0.11),
            (r"$A(x)=\pi r^2=\pi\,(R^2-x^2)$", 15, 0.10),
            (r"and a slice of thickness $dx$ has volume $dV=A(x)\,dx$", 12.5, 0.11),
        ])

    # -- Step 3: Riemann sum of disks ----------------------------------------

    def scene_sum(self, f, p):
        ax = self.ax3d
        n = self.RIEMANN_NS[min(f // self.FRAMES_PER_N, len(self.RIEMANN_NS) - 1)]
        edges, mids, approx = riemann_sum(n)

        ghost_sphere(ax, alpha=0.2)
        parts = [disk(a, b, float(radius(m)), n_theta=36) for a, b, m in
                 zip(edges[:-1], edges[1:], mids)]
        add_polys(ax, np.concatenate([q for q, _ in parts]),
                  np.concatenate([c for _, c in parts]),
                  edge_darken=0.55 if n <= 16 else None)
        x_axis(ax)

        ax2 = self.area_graph(r"Cross-section area  $A(x)=\pi\,(R^2-x^2)$")
        width = 2 * R / n
        ax2.bar(mids, area(mids), width=width, color=x_color(mids), alpha=0.85,
                edgecolor=BG if n <= 24 else "none", linewidth=0.8, zorder=2)
        ax2.text(-1.15 * R, 1.13 * np.pi * R**2, f"{n} bars  =  {n} disks", color=GOLD,
                 fontsize=10)

        err = (approx - EXACT) / EXACT * 100
        self.write("Step 3 · Add up the slices (Riemann sum)", [
            (rf"Use $n={n}$ disks of thickness $\Delta x = 2R/n$:", 12.5, 0.02),
            (r"$V\ \approx\ \sum_{i=1}^{n}\ \pi\,(R^2-x_i^2)\,\Delta x$", 16, 0.13),
            ("Each disk is one bar under the graph of A(x).", 12.5, 0.14),
            (f"n = {n:<3d}  V ≈ {approx:.4f} R³    exact {EXACT:.4f} R³",
             13, 0.10),
            (f"error {err:+.2f}%   →  more slices, better fit", 12.5, 0.08),
        ])

    # -- Step 4: limit -> integral, volume accumulates -----------------------

    def scene_limit(self, f, p):
        ax = self.ax3d
        t = -R + 2 * R * stage(p, 0.06, 0.86)
        phi_t = float(np.arccos(np.clip(-t / R, -1, 1)))
        theta = np.linspace(0, 2 * np.pi, 49)

        ghost_sphere(ax, alpha=0.25)
        if phi_t > 1e-3:
            phi = np.linspace(0, phi_t, max(3, int(60 * phi_t / np.pi) + 1))
            q, c = solid_surface(-R * np.cos(phi), R * np.sin(phi), theta)
            rt = float(radius(t))
            if rt > 1e-3:  # gold cap = the current cross-section A(t)
                cap = quads(*revolve([t, t], [0, rt], theta))
                q = np.concatenate([q, cap])
                c = np.concatenate([c, shade(cap, to_rgb(GOLD), ambient=0.65)])
            add_polys(ax, q, c)
        x_axis(ax)

        ax2 = self.area_graph(r"Volume so far = shaded area under $A(x)$")
        self.shade_area(ax2, -R, t)
        ax2.plot([t, t], [0, area(t)], color=GOLD, lw=2.5, zorder=5)
        ax2.scatter([t], [area(t)], color=GOLD, s=30, zorder=6)
        ax2.text(-1.15 * R, 1.13 * np.pi * R**2, f"t = {t:+.2f}R", color=GOLD, fontsize=10)

        self.write("Step 4 · Let Δx → 0: sum → integral", [
            (r"$V=\lim_{n\to\infty}\ \sum_{i=1}^{n}\ \pi\,(R^2-x_i^2)\,\Delta x$", 15, 0.04),
            (r"$\ \ \ =\ \int_{-R}^{\,R}\ \pi\,(R^2-x^2)\,dx$", 16, 0.15),
            ("Volume swept from −R up to x = t:", 12.5, 0.14),
            (r"$V(t)=\int_{-R}^{\,t}A(x)\,dx$" + f"  =  {volume_up_to(t):.3f} R³",
             15, 0.10),
        ])

    # -- Step 5: evaluate the integral ---------------------------------------

    def scene_evaluate(self, f, p):
        self.reset_3d(azim=AZIM + 360 * p)
        ax = self.ax3d
        phi = np.linspace(0, np.pi, 61)
        add_polys(ax, *solid_surface(-R * np.cos(phi), R * np.sin(phi),
                                     np.linspace(0, 2 * np.pi, 49)))
        x_axis(ax)

        ax2 = self.area_graph(r"Total area under $A(x)$ = volume of the sphere")
        self.shade_area(ax2, -R, R)
        ax2.text(0, 0.42 * np.pi * R**2, r"$\frac{4}{3}\pi R^3$", color=BG, fontsize=18,
                 ha="center", fontweight="bold", zorder=5)

        lines = [
            (r"$V=\int_{-R}^{\,R}\pi\,(R^2-x^2)\,dx$", 15, 0.03),
            (r"$\ \ \ =\pi\left[R^2x-\frac{x^3}{3}\right]_{-R}^{\,R}$", 15, 0.11),
            (r"$\ \ \ =\pi\left[\left(R^3-\frac{R^3}{3}\right)-"
             r"\left(-R^3+\frac{R^3}{3}\right)\right]$", 15, 0.115),
            (r"$\ \ \ =\pi\left(\frac{2R^3}{3}+\frac{2R^3}{3}\right)$", 15, 0.115),
            (r"BOX:$V=\frac{4}{3}\pi R^3\ \approx\ 4.18879\,R^3$", 17, 0.135),
        ]
        visible = 1 + int(f / 16)
        self.write("Step 5 · Evaluate the integral", lines, visible=visible)


# ----------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--output", default="sphere_volume.mp4",
                        help="output file: .mp4 or .gif (or .png with --frame)")
    parser.add_argument("--dpi", type=int, default=100,
                        help="resolution; 100 gives 1280x720 (default: 100)")
    parser.add_argument("--show", action="store_true",
                        help="play the animation in a window instead of saving it")
    parser.add_argument("--frame", type=int, default=None,
                        help="render only this frame number to --output as a still image")
    args = parser.parse_args()

    anim = SphereVolumeAnimation()

    if args.frame is not None:
        anim.draw(args.frame)
        anim.fig.savefig(args.output, dpi=args.dpi, facecolor=BG)
        print(f"Saved frame {args.frame} of {anim.n_frames} to {args.output}")
        return

    movie = FuncAnimation(anim.fig, anim.draw, frames=anim.n_frames,
                          interval=1000 / FPS, blit=False)
    if args.show:
        plt.show()
        return

    out = Path(args.output)
    if shutil.which("ffmpeg"):
        extra = ["-pix_fmt", "yuv420p"] if out.suffix.lower() == ".mp4" else []
        writer = FFMpegWriter(fps=FPS, bitrate=4000, extra_args=extra)
    elif out.suffix.lower() == ".gif":
        writer = PillowWriter(fps=FPS)
    else:
        sys.exit("ffmpeg was not found. Install it, or save a GIF instead: "
                 "-o sphere_volume.gif")

    def progress(i, total):
        if i % 24 == 0 or i == total - 1:
            print(f"\rRendering frame {i + 1}/{total}", end="", flush=True)

    movie.save(out, writer=writer, dpi=args.dpi, savefig_kwargs={"facecolor": BG},
               progress_callback=progress)
    print(f"\nSaved {anim.n_frames / FPS:.0f}s animation to {out}")


if __name__ == "__main__":
    main()
