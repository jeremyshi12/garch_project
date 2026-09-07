"""Helpers for writing result tables to disk in both CSV and Markdown."""

from __future__ import annotations

import pandas as pd

from . import config as C


def save_table(df: pd.DataFrame, name: str, floatfmt: str = "%.4f") -> str:
    """Write a table to outputs/tables as CSV plus a Markdown twin."""
    csv_path = C.TABLES / f"{name}.csv"
    md_path = C.TABLES / f"{name}.md"
    df.to_csv(csv_path, float_format=floatfmt.replace("%", "%"))
    with open(md_path, "w") as fh:
        fh.write(df.to_markdown(floatfmt=floatfmt.lstrip("%")))
        fh.write("\n")
    return str(csv_path)


def show(df: pd.DataFrame, decimals: int = 4):
    """Round for display without mutating the stored precision."""
    return df.round(decimals)
