"""Shared matplotlib styling helpers.

Every chart in the project is built through these helpers so that the
palette, grid weight and label treatment stay consistent.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

from . import config as C


def apply_style() -> None:
    """Install the project's matplotlib defaults: recessive grid, thin marks."""
    mpl.rcParams.update(
        {
            "figure.facecolor": C.SURFACE,
            "axes.facecolor": C.SURFACE,
            "savefig.facecolor": C.SURFACE,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "figure.dpi": 110,
            "font.size": 10,
            "font.family": "sans-serif",
            "text.color": C.INK_PRIMARY,
            "axes.labelcolor": C.INK_SECONDARY,
            "axes.edgecolor": C.GRID,
            "axes.linewidth": 0.8,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlecolor": C.INK_PRIMARY,
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": C.GRID,
            "grid.linewidth": 0.7,
            "grid.alpha": 1.0,
            "xtick.color": C.INK_SECONDARY,
            "ytick.color": C.INK_SECONDARY,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "legend.fontsize": 9,
            "legend.labelcolor": C.INK_SECONDARY,
            "lines.linewidth": 1.6,
            "lines.solid_capstyle": "round",
            "figure.autolayout": False,
        }
    )


def color(ticker: str) -> str:
    """Colour for a ticker; colour follows the entity, never its rank."""
    return C.SERIES_COLORS.get(ticker, C.INK_SECONDARY)


def titled(
    ax: plt.Axes, title: str, subtitle: str | None = None, wrap: int = 96
) -> None:
    """Left-aligned title with an optional muted subtitle beneath it.

    The title is padded upward to clear the subtitle, which is drawn just above
    the axes; without the pad the two render on top of each other.
    """
    if not subtitle:
        ax.set_title(title, loc="left")
        return

    import textwrap

    lines = textwrap.wrap(subtitle, wrap)
    ax.set_title(title, loc="left", pad=14 + 12 * len(lines))
    ax.text(
        0.0,
        1.015,
        "\n".join(lines),
        transform=ax.transAxes,
        fontsize=9,
        color=C.INK_MUTED,
        va="bottom",
        ha="left",
        linespacing=1.35,
    )


def label_line_end(ax: plt.Axes, x, y, text: str, col: str, dx: float = 8.0) -> None:
    """Direct-label the end of a line.

    Two of the four series sit below 3:1 contrast on the light surface, so the
    relief rule applies: identity is never carried by colour alone.
    """
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(dx, 0),
        textcoords="offset points",
        va="center",
        ha="left",
        fontsize=9,
        color=C.INK_SECONDARY,
        fontweight="normal",
        annotation_clip=False,
    )


def label_line_ends(
    ax: plt.Axes, items: list[tuple[float, float, str, str]], min_gap_frac: float = 0.055
) -> None:
    """Direct-label several line ends, nudging them apart so none overlap.

    `items` is a list of (x, y, text, colour). Labels are placed at their true y
    where there is room and pushed apart vertically where there is not, so the
    reading order still matches the line order.
    """
    if not items:
        return
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * min_gap_frac

    ordered = sorted(items, key=lambda it: it[1])
    placed: list[float] = []
    for _, y, _, _ in ordered:
        if placed and y - placed[-1] < gap:
            placed.append(placed[-1] + gap)
        else:
            placed.append(y)

    # If pushing upward overflowed the axes, shift the whole stack back down.
    overflow = placed[-1] - hi
    if overflow > 0:
        placed = [p - overflow for p in placed]

    for (x, _, text, _), y_label in zip(ordered, placed):
        ax.annotate(
            text,
            xy=(x, y_label),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=9,
            color=C.INK_SECONDARY,
            fontweight="normal",
            annotation_clip=False,
        )


def shade_crises(ax: plt.Axes, label: bool = False) -> None:
    """Shade the crisis windows defined in config on a date axis."""
    import pandas as pd

    for start, end, tag in C.CRISIS_PERIODS:
        ax.axvspan(
            pd.Timestamp(start),
            pd.Timestamp(end),
            color=C.INK_PRIMARY,
            alpha=0.05,
            lw=0,
            zorder=0,
        )
        if label:
            ax.text(
                pd.Timestamp(start),
                ax.get_ylim()[1],
                f" {tag}",
                fontsize=7.5,
                color=C.INK_MUTED,
                va="top",
                ha="left",
                rotation=90,
            )


def save(fig: plt.Figure, filename: str) -> str:
    """Save a figure into outputs/figures and return its path as a string."""
    path = C.FIGURES / filename
    fig.savefig(path)
    plt.close(fig)
    return str(path)
