"""Project-wide configuration: tickers, date range, paths and plotting palette."""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUTS = PROJECT_ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"
REPORTS = PROJECT_ROOT / "reports"

for _d in (DATA_RAW, DATA_PROCESSED, FIGURES, TABLES, REPORTS):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Universe
# --------------------------------------------------------------------------
# Ticker -> display name. Order is fixed and drives the colour assignment,
# so a chart that drops an index never repaints the ones that remain.
INDICES: dict[str, str] = {
    "^GSPC": "S&P 500",
    "^STOXX50E": "EURO STOXX 50",
    "^N225": "Nikkei 225",
    "^FTSE": "FTSE 100",
}

REGION: dict[str, str] = {
    "^GSPC": "United States",
    "^STOXX50E": "Euro area",
    "^N225": "Japan",
    "^FTSE": "United Kingdom",
}

START_DATE = "2000-01-01"
END_DATE = None  # None -> today

# Trading days per year, used to annualise daily volatility.
TRADING_DAYS = 252

# --------------------------------------------------------------------------
# Modelling
# --------------------------------------------------------------------------
# Specifications fitted to every index. Each entry is a kwargs dict for
# arch.arch_model(); "label" is what shows up in the comparison tables.
MODEL_SPECS: list[dict] = [
    {"label": "GARCH(1,1)-Normal", "vol": "GARCH", "p": 1, "q": 1, "dist": "normal"},
    {"label": "GARCH(1,1)-t", "vol": "GARCH", "p": 1, "q": 1, "dist": "t"},
    {"label": "GARCH(1,1)-skewt", "vol": "GARCH", "p": 1, "q": 1, "dist": "skewt"},
    {"label": "GJR-GARCH(1,1)-t", "vol": "GARCH", "p": 1, "o": 1, "q": 1, "dist": "t"},
    {"label": "EGARCH(1,1)-t", "vol": "EGARCH", "p": 1, "q": 1, "dist": "t"},
]

# The headline specification the project is built around.
BASELINE_LABEL = "GARCH(1,1)-Normal"

# VaR backtest settings
VAR_LEVELS = (0.01, 0.05)
VAR_OOS_START = "2016-01-01"  # first day of the out-of-sample window
VAR_REFIT_EVERY = 10  # refit the model every N trading days

# Crisis windows highlighted on the conditional-volatility charts.
CRISIS_PERIODS: list[tuple[str, str, str]] = [
    ("2007-07-01", "2009-06-30", "Global financial crisis"),
    ("2011-07-01", "2012-06-30", "Euro sovereign debt crisis"),
    ("2020-02-15", "2020-06-30", "COVID-19 crash"),
    ("2022-01-01", "2022-10-31", "Inflation / rates shock"),
]

# --------------------------------------------------------------------------
# Palette (validated with the data-viz six checks, light surface #fcfcfb:
# all-pairs CVD dE 13.0, normal-vision dE 19.6). Magenta and yellow fall below
# 3:1 against the surface, so every chart using them carries direct labels or a
# companion table -- see outputs/tables/.
# --------------------------------------------------------------------------
SERIES_COLORS: dict[str, str] = {
    "^GSPC": "#2a78d6",  # blue
    "^STOXX50E": "#008300",  # green
    "^N225": "#e87ba4",  # magenta
    "^FTSE": "#eda100",  # yellow
}

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8880"
SURFACE = "#fcfcfb"
GRID = "#e4e3df"

# Status colours, reserved -- never reused as a series colour.
STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}


def name(ticker: str) -> str:
    """Display name for a ticker, falling back to the ticker itself."""
    return INDICES.get(ticker, ticker)
