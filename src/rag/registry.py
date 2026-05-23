"""Curated ASX annual-report allowlist + ``indexed`` probe.

This module is the source of truth for the company allowlist. The fetch
script and on-demand ingest both import :data:`ASX_ANNUAL_REPORTS` from
here. URLs are reviewed manually — corpus sources are part of the demo's
audit trail.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_CORPUS_ROOT = Path("data/corpus/asx")


@dataclass(frozen=True)
class AnnualReport:
    """One ASX annual-report entry in the curated allowlist.

    Attributes:
        ticker: Uppercase ASX ticker.
        name: Display name (e.g. ``Woolworths Group``).
        url: Direct HTTPS URL to the FY annual-report PDF.
        fy_label: Short FY label such as ``FY25``.
    """

    ticker: str
    name: str
    url: str
    fy_label: str


@dataclass(frozen=True)
class CompanyInfo:
    """One allowlist entry projected for API/UI consumption.

    Attributes:
        ticker: Uppercase ASX ticker.
        name: Display name.
        fy_label: Short FY label such as ``FY25``.
        indexed: True when the annual-report PDF is present on disk.
    """

    ticker: str
    name: str
    fy_label: str
    indexed: bool


# Ticker -> annual-report metadata. Curated allowlist of 8 ASX 50 names.
# URLs reviewed manually; bumping FY is a manual PR.
#
# BHP and WES were evaluated and dropped: both issuers' CDNs (bhp.com,
# wesfarmers.gcs-web.com) reject programmatic HTTPS clients via TLS
# fingerprinting / bot protection, so on-demand ingest cannot fetch their
# annual reports without a browser-impersonation HTTP layer.
ASX_ANNUAL_REPORTS: dict[str, AnnualReport] = {
    "WOW": AnnualReport(
        ticker="WOW",
        name="Woolworths Group",
        url=(
            "https://www.woolworthsgroup.com.au/content/dam/wwg/sustainability/"
            "reports/f25/Woolworths%20Group%20Annual%20Report%202025%20.pdf"
        ),
        fy_label="FY25",
    ),
    "CBA": AnnualReport(
        ticker="CBA",
        name="Commonwealth Bank of Australia",
        url=(
            "https://www.commbank.com.au/content/dam/commbank-assets/investors/"
            "docs/results/fy25/2025-annual-report.pdf"
        ),
        fy_label="FY25",
    ),
    "TLS": AnnualReport(
        ticker="TLS",
        name="Telstra Group",
        url=(
            "https://www.telstra.com.au/content/dam/tcom/about-us/investors/"
            "pdf-g/telstra-annual-report-2025.pdf"
        ),
        fy_label="FY25",
    ),
    "CSL": AnnualReport(
        ticker="CSL",
        name="CSL Limited",
        url=(
            "https://www.csl.com/-/media/shared/documents/annual-report/csl-annual-report-2025.pdf"
        ),
        fy_label="FY25",
    ),
    "NAB": AnnualReport(
        ticker="NAB",
        name="National Australia Bank",
        url=(
            "https://www.nab.com.au/content/dam/nab/documents/reports/corporate/"
            "2025-annual-report.pdf"
        ),
        fy_label="FY25",
    ),
    "ANZ": AnnualReport(
        ticker="ANZ",
        name="ANZ Group",
        url=(
            "https://www.anz.com.au/content/dam/anzcom/shareholder/"
            "2025-annual-report/anz-2025-annual-report.pdf"
        ),
        fy_label="FY25",
    ),
    "RIO": AnnualReport(
        ticker="RIO",
        name="Rio Tinto",
        url=(
            "https://cdn-rio.dataweavers.io/-/media/content/documents/invest/"
            "reports/annual-reports/2025-annual-report.pdf"
        ),
        fy_label="FY25",
    ),
    "MQG": AnnualReport(
        ticker="MQG",
        name="Macquarie Group",
        url=(
            "https://www.macquarie.com/assets/macq/investor/reports/2025/"
            "macquarie-group-fy25-annual-report.pdf"
        ),
        fy_label="FY25",
    ),
}


def _pdf_path(entry: AnnualReport) -> Path:
    """Return the on-disk PDF path for one allowlist entry."""
    return _CORPUS_ROOT / entry.ticker.lower() / f"{entry.ticker}-annual-report.pdf"


def is_indexed(ticker: str) -> bool:
    """Return True when the ticker's annual-report PDF exists on disk.

    Args:
        ticker: Allowlist ticker (case-insensitive).

    Returns:
        True when the cached PDF is present and non-empty.
    """
    entry = ASX_ANNUAL_REPORTS.get(ticker.upper())
    if entry is None:
        return False
    path = _pdf_path(entry)
    return path.exists() and path.stat().st_size > 0


def list_companies() -> list[CompanyInfo]:
    """Return all allowlist entries with their indexed flag.

    Returns:
        Companies in allowlist iteration order.
    """
    return [
        CompanyInfo(
            ticker=entry.ticker,
            name=entry.name,
            fy_label=entry.fy_label,
            indexed=is_indexed(entry.ticker),
        )
        for entry in ASX_ANNUAL_REPORTS.values()
    ]


def get_entry(ticker: str) -> AnnualReport:
    """Look up one allowlist entry.

    Args:
        ticker: Allowlist ticker (case-insensitive).

    Returns:
        The :class:`AnnualReport` entry.

    Raises:
        KeyError: When the ticker is not in the allowlist.
    """
    return ASX_ANNUAL_REPORTS[ticker.upper()]


def pdf_path_for(ticker: str) -> Path:
    """Return the on-disk PDF path for the given ticker.

    Args:
        ticker: Allowlist ticker (case-insensitive).

    Returns:
        Filesystem path under ``data/corpus/asx/<ticker>/``.

    Raises:
        KeyError: When the ticker is not in the allowlist.
    """
    return _pdf_path(get_entry(ticker))


def source_label(ticker: str) -> str:
    """Return the chunk-metadata ``source`` label for the given ticker.

    Args:
        ticker: Allowlist ticker (case-insensitive).

    Returns:
        Label of the form ``ASX-<TICKER>``.
    """
    return f"ASX-{ticker.upper()}"
