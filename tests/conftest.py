"""Shared pytest fixtures and configuration."""

from __future__ import annotations

import random

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _seed_everything() -> None:
    """Make every test deterministic by seeding random sources."""
    random.seed(0)
    np.random.seed(0)
    try:
        import torch

        torch.manual_seed(0)
    except ImportError:  # pragma: no cover
        pass
