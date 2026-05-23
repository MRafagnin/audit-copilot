# ADR 005 — Native Windows runtime (no Docker locally)

* **Status**: Accepted
* **Date**: 2026-05-22
* **Owner**: Matheus Rafagnin

## Context

The development laptop is Intune-managed and **blocks WSL2 and Docker
Desktop**. The demo must run on the same machine it is built on, in front
of an interviewer, with zero "works on my machine" risk.

## Decision

Run the entire stack as **native Windows processes**:

* Python 3.11 via `uv`-managed virtualenv (`./.venv`).
* Ollama installed as a Windows service on `localhost:11434`.
* ChromaDB in **embedded mode** with a persistent on-disk store at
  `data/chroma/` — no Chroma server process.
* FastAPI via `uvicorn` on port 8000; Streamlit on port 8501; both started
  by `scripts/run_dev.ps1`.

`uv` provides the reproducibility guarantee that containers usually provide:
locked dependency versions, deterministic install, single command bootstrap.

## Alternatives considered

| Option | Why not |
|---|---|
| Docker Desktop locally | Blocked by Intune policy. |
| WSL2 + Linux containers | Blocked by Intune policy. |
| Podman on Windows | Same kernel constraints as Docker; not approved. |
| Cloud development environment (GitHub Codespaces) | Requires network; defeats the offline-demo property. |
| Vagrant / Hyper-V VM | Heavyweight for a portfolio repo; doesn't help the demo machine. |

## Consequences

**Positive**

* Bootstrap is a single PowerShell session: install `uv`, `uv sync`,
  `ollama pull`, `make bootstrap`. No virtualisation layer to debug.
* Native I/O performance — Chroma + embedding model load fast.
* Reproducible across two Windows laptops by re-running `make bootstrap`.

**Negative / accepted trade-offs**

* Platform-specific scripts (`run_dev.ps1` is PowerShell). A Bash twin is
  trivial to add when the project leaves the Windows machine.
* No production-parity container locally. Mitigated by the Azure mapping —
  containerisation happens at the deploy boundary, not the dev boundary.

## Azure mapping

The same code is **packaged into a container at deploy time** and run on
**Azure Container Apps** (FastAPI) or **Azure App Service** (Streamlit).
Local development stays native; the production image is built in CI from
the same `pyproject.toml`. ChromaDB embedded becomes either **Azure AI
Search** (preferred) or a managed vector DB; Ollama is replaced by **Azure
OpenAI** (see ADR 001).
