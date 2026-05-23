"""Generate a synthetic general-ledger CSV with seeded fraud-like anomalies.

The output is deterministic for a given ``--seed`` so demos reproduce. Anomaly
patterns reflect ASA 240 / journal-entry-testing concerns:

* ``round_amount``: implausibly round dollar values posted to revenue/expense.
* ``after_hours`` / ``weekend``: postings outside normal business hours.
* ``unusual_user_account``: a user touching an account they normally don't.
* ``near_duplicate``: two near-identical entries within minutes.
* ``benford_violation``: amounts whose first digit deviates from Benford's law
  (cluster of first-digit ``9`` postings by one user).

Schema (CSV columns)::

    tx_id, date, account, debit, credit, user, posting_ts,
    description, is_anomaly, anomaly_type

Run::

    uv run python scripts/gen_journal_entries.py
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from faker import Faker

from src.core.config import settings
from src.core.logging_config import configure_logging

logger = logging.getLogger(__name__)

ACCOUNTS: tuple[str, ...] = (
    "1000-Cash",
    "1100-AccountsReceivable",
    "1200-Inventory",
    "2000-AccountsPayable",
    "2100-AccruedLiabilities",
    "3000-Equity",
    "4000-Revenue",
    "5000-COGS",
    "6000-OperatingExpense",
    "6100-Travel",
    "6200-Marketing",
    "7000-InterestExpense",
)

USERS: tuple[str, ...] = tuple(f"user_{i:03d}" for i in range(20))

ANOMALY_RATE: float = 0.05  # ~5% of rows are anomalous


@dataclass(frozen=True)
class GenConfig:
    """Parameters controlling synthetic GL generation.

    Attributes:
        rows: Total number of journal-entry rows to produce.
        seed: RNG seed for full reproducibility.
        start_date: Inclusive start of the posting window.
        end_date: Exclusive end of the posting window.
    """

    rows: int
    seed: int
    start_date: datetime
    end_date: datetime


def _business_timestamp(rng: random.Random, day: datetime) -> datetime:
    """Return a timestamp within standard business hours on ``day``."""
    hour = rng.randint(9, 17)
    minute = rng.randint(0, 59)
    second = rng.randint(0, 59)
    return day.replace(hour=hour, minute=minute, second=second)


def _after_hours_timestamp(rng: random.Random, day: datetime) -> datetime:
    """Return a timestamp outside business hours on ``day``."""
    hour = rng.choice([0, 1, 2, 3, 4, 22, 23])
    minute = rng.randint(0, 59)
    return day.replace(hour=hour, minute=minute, second=rng.randint(0, 59))


def _normal_amount(rng: random.Random) -> float:
    """Sample a plausible, non-round transaction amount."""
    # Mixture: small purchases, medium invoices, occasional large entries.
    bucket = rng.random()
    if bucket < 0.6:
        return round(rng.uniform(10, 5_000), 2)
    if bucket < 0.95:
        return round(rng.uniform(5_000, 50_000), 2)
    return round(rng.uniform(50_000, 250_000), 2)


def _round_amount(rng: random.Random) -> float:
    """Sample a suspiciously round amount (multiple of 1000 or 10000)."""
    base = rng.choice([1_000, 5_000, 10_000, 50_000, 100_000])
    multiplier = rng.randint(1, 9)
    return float(base * multiplier)


def _benford_violation_amount(rng: random.Random) -> float:
    """Return an amount whose first digit is 9 (Benford under-represents 9)."""
    magnitude = rng.choice([100, 1_000, 10_000])
    return round(9 * magnitude + rng.uniform(0, magnitude * 0.999), 2)


def _normal_user_account(rng: random.Random) -> tuple[str, str]:
    """Sample a user and an account they commonly touch.

    Users are partitioned into rough functional groups so most postings are
    'expected' user x account pairs. Returns the chosen pair.
    """
    user = rng.choice(USERS)
    # Group users by index modulo bucket count
    bucket = int(user.split("_")[1]) % 4
    if bucket == 0:  # AR / cash
        account = rng.choice(["1000-Cash", "1100-AccountsReceivable", "4000-Revenue"])
    elif bucket == 1:  # AP / expenses
        account = rng.choice(["2000-AccountsPayable", "6000-OperatingExpense", "6100-Travel"])
    elif bucket == 2:  # COGS / inventory
        account = rng.choice(["1200-Inventory", "5000-COGS", "2100-AccruedLiabilities"])
    else:  # finance / equity
        account = rng.choice(["3000-Equity", "7000-InterestExpense", "6200-Marketing"])
    return user, account


def _unusual_user_account(rng: random.Random) -> tuple[str, str]:
    """Sample a user paired with an account from a different functional group."""
    user, normal_account = _normal_user_account(rng)
    other_accounts = [a for a in ACCOUNTS if a != normal_account]
    return user, rng.choice(other_accounts)


def _build_row(
    *,
    rng: random.Random,
    faker: Faker,
    day: datetime,
    is_anomaly: bool,
    anomaly_type: str,
) -> dict[str, object]:
    """Build a single GL row dict for the chosen anomaly mode."""
    if anomaly_type == "round_amount":
        amount = _round_amount(rng)
        user, account = _normal_user_account(rng)
        ts = _business_timestamp(rng, day)
    elif anomaly_type == "after_hours":
        amount = _normal_amount(rng)
        user, account = _normal_user_account(rng)
        ts = _after_hours_timestamp(rng, day)
    elif anomaly_type == "weekend":
        amount = _normal_amount(rng)
        user, account = _normal_user_account(rng)
        # Force a weekend by shifting day forward to nearest Sat/Sun
        shift = (5 - day.weekday()) % 7
        weekend_day = day + timedelta(days=shift if shift > 0 else 1)
        ts = _business_timestamp(rng, weekend_day)
    elif anomaly_type == "unusual_user_account":
        amount = _normal_amount(rng)
        user, account = _unusual_user_account(rng)
        ts = _business_timestamp(rng, day)
    elif anomaly_type == "benford_violation":
        amount = _benford_violation_amount(rng)
        user, account = _normal_user_account(rng)
        ts = _business_timestamp(rng, day)
    else:  # normal
        amount = _normal_amount(rng)
        user, account = _normal_user_account(rng)
        ts = _business_timestamp(rng, day)

    debit = amount if rng.random() < 0.5 else 0.0
    credit = 0.0 if debit > 0 else amount

    return {
        "tx_id": str(uuid.UUID(int=rng.getrandbits(128))),
        "date": ts.date().isoformat(),
        "account": account,
        "debit": debit,
        "credit": credit,
        "user": user,
        "posting_ts": ts.isoformat(timespec="seconds"),
        "description": faker.sentence(nb_words=6),
        "is_anomaly": is_anomaly,
        "anomaly_type": anomaly_type if is_anomaly else "",
    }


def generate(cfg: GenConfig) -> pd.DataFrame:
    """Generate the synthetic GL as a pandas DataFrame.

    Args:
        cfg: Generator configuration.

    Returns:
        DataFrame with one row per journal entry, including ground-truth
        ``is_anomaly`` and ``anomaly_type`` columns.
    """
    rng = random.Random(cfg.seed)
    faker = Faker()
    faker.seed_instance(cfg.seed)

    span_days = max(1, (cfg.end_date - cfg.start_date).days)
    anomaly_modes = (
        "round_amount",
        "after_hours",
        "weekend",
        "unusual_user_account",
        "benford_violation",
    )

    rows: list[dict[str, object]] = []
    for _ in range(cfg.rows):
        day = cfg.start_date + timedelta(days=rng.randrange(span_days))
        # Push the base day to a weekday; anomaly cases may override.
        while day.weekday() >= 5:
            day += timedelta(days=1)
        is_anomaly = rng.random() < ANOMALY_RATE
        anomaly_type = rng.choice(anomaly_modes) if is_anomaly else ""
        rows.append(
            _build_row(
                rng=rng,
                faker=faker,
                day=day,
                is_anomaly=is_anomaly,
                anomaly_type=anomaly_type,
            )
        )

    # Inject near-duplicate pairs by cloning a small fraction of rows.
    duplicates_target = max(1, cfg.rows // 200)
    for _ in range(duplicates_target):
        src_idx = rng.randrange(len(rows))
        original = rows[src_idx]
        original_ts = datetime.fromisoformat(str(original["posting_ts"]))
        dup_ts = original_ts + timedelta(minutes=rng.randint(1, 10))
        dup = {
            **original,
            "tx_id": str(uuid.UUID(int=rng.getrandbits(128))),
            "posting_ts": dup_ts.isoformat(timespec="seconds"),
            "is_anomaly": True,
            "anomaly_type": "near_duplicate",
        }
        rows.append(dup)

    df = pd.DataFrame(rows)
    df.sort_values("posting_ts", inplace=True, ignore_index=True)
    return df


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(description="Generate synthetic GL with seeded anomalies.")
    parser.add_argument("--out", default="data/gl/journal_entries.csv")
    parser.add_argument("--rows", type=int, default=settings.gl_row_count)
    parser.add_argument("--seed", type=int, default=settings.anomaly_seed)
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2024-12-31")
    args = parser.parse_args(argv)

    configure_logging()
    cfg = GenConfig(
        rows=args.rows,
        seed=args.seed,
        start_date=datetime.fromisoformat(args.start),
        end_date=datetime.fromisoformat(args.end),
    )

    df = generate(cfg)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(
        "gl generated",
        extra={
            "path": str(out_path),
            "rows": len(df),
            "anomalies": int(df["is_anomaly"].sum()),
        },
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
