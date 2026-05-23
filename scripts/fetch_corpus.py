"""Download the AuditCopilot corpus PDFs.

Three categories, each cached under ``data/corpus/``:

* AUASB ASA standards (primary Australian corpus).
* IAASB ISA reference (international, ASA is derived from this).
* The configured ASX annual report (default: Woolworths Group FY25).

URLs are kept in this script as a deliberate, reviewable allowlist — corpus
sources are part of the demo's audit trail.

Run with::

    uv run python scripts/fetch_corpus.py [--ticker WOW]
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import requests

from src.core.config import settings
from src.core.logging_config import configure_logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CorpusSource:
    """One downloadable PDF.

    Attributes:
        label: Short identifier, used in log lines and on-disk filenames.
        url: Direct HTTPS URL to the PDF.
        category: ``auasb``, ``iaasb``, or ``asx`` — picks the output folder.
    """

    label: str
    url: str
    category: str


# Curated list of standards likely to be cited in fraud / journal-entry demos.
# Expand over time; keep the list small so the demo index stays focused.
# URLs verified 2026-05-22 via the AUASB Standards Portal detail pages.
AUASB_STANDARDS: tuple[CorpusSource, ...] = (
    CorpusSource(
        label="ASA-240-fraud",
        url="https://www.auasb.gov.au/media/g2npxdg3/asa240_12-23.pdf",
        category="auasb",
    ),
    CorpusSource(
        label="ASA-315-risk",
        url="https://www.auasb.gov.au/media/1ppnfkvx/asa315_12-23-1.pdf",
        category="auasb",
    ),
    CorpusSource(
        label="ASA-330-response",
        url="https://www.auasb.gov.au/media/ne4dxsd5/asa_330_12_21.pdf",
        category="auasb",
    ),
)

# IAASB ISA standards are documented as a future addition. The AUASB ASAs are
# substantively derived from the ISAs, so the primary corpus is already covered
# by the AUASB downloads above. Add ISA PDFs here once a stable public URL is
# confirmed (the IFAC site uses session-scoped download links).
IAASB_STANDARDS: tuple[CorpusSource, ...] = ()

# Ticker -> annual report PDF URL.
# Defaulting to Woolworths FY25. Add more entries here when expanding the demo.
# URL verified 2026-05-22 (10.4 MB).
ASX_ANNUAL_REPORTS: dict[str, str] = {
    "WOW": (
        "https://www.woolworthsgroup.com.au/content/dam/wwg/sustainability/"
        "reports/f25/Woolworths%20Group%20Annual%20Report%202025%20.pdf"
    ),
}


def _output_path(root: Path, source: CorpusSource, ticker: str | None) -> Path:
    """Return the on-disk path for a corpus source.

    Args:
        root: Corpus root directory.
        source: Source descriptor.
        ticker: ASX ticker, used when ``source.category == 'asx'``.

    Returns:
        Path under ``root/<category>/[<ticker>/]<label>.pdf``.
    """
    if source.category == "asx" and ticker is not None:
        folder = root / "asx" / ticker.lower()
    else:
        folder = root / source.category
    return folder / f"{source.label}.pdf"


def download(source: CorpusSource, out_path: Path, *, user_agent: str) -> bool:
    """Download a single PDF if it isn't cached already.

    Args:
        source: Source descriptor.
        out_path: Destination file path.
        user_agent: Value for the ``User-Agent`` header.

    Returns:
        True when a new download occurred, False when the cached file was kept.
    """
    if out_path.exists() and out_path.stat().st_size > 0:
        logger.info("corpus cached", extra={"label": source.label, "path": str(out_path)})
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("corpus downloading", extra={"label": source.label, "url": source.url})
    response = requests.get(source.url, headers={"User-Agent": user_agent}, timeout=60, stream=True)
    response.raise_for_status()
    with out_path.open("wb") as fp:
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if chunk:
                fp.write(chunk)
    logger.info(
        "corpus downloaded", extra={"label": source.label, "bytes": out_path.stat().st_size}
    )
    return True


def build_source_list(ticker: str) -> list[CorpusSource]:
    """Assemble the full source list for a given ticker.

    Args:
        ticker: ASX ticker for the annual report.

    Returns:
        Concatenated AUASB + IAASB + ASX (one entry) sources.

    Raises:
        KeyError: When the ticker has no registered annual-report URL.
    """
    annual_url = ASX_ANNUAL_REPORTS[ticker.upper()]
    asx_source = CorpusSource(
        label=f"{ticker.upper()}-annual-report",
        url=annual_url,
        category="asx",
    )
    return [*AUASB_STANDARDS, *IAASB_STANDARDS, asx_source]


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(description="Fetch AuditCopilot corpus PDFs.")
    parser.add_argument(
        "--ticker",
        default=settings.demo_ticker,
        help="ASX ticker for the annual report (default from settings).",
    )
    parser.add_argument(
        "--out",
        default="data/corpus",
        help="Output root directory.",
    )
    args = parser.parse_args(argv)

    configure_logging()
    root = Path(args.out)
    try:
        sources = build_source_list(args.ticker)
    except KeyError:
        logger.error("ticker not registered", extra={"ticker": args.ticker})
        return 2

    new = 0
    for source in sources:
        out_path = _output_path(root, source, ticker=args.ticker)
        if download(source, out_path, user_agent=settings.http_user_agent):
            new += 1
    logger.info("corpus fetch complete", extra={"new_downloads": new, "total": len(sources)})
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
