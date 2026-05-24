# Prerequisites — Before We Start Building

> Complete every item in this checklist before we begin Phase 0. Each step has copy-pasteable PowerShell commands and a verification command so you can confirm it worked.

Estimated total time: **30–45 minutes**, most of it unattended downloads.

---

## 1. System requirements check

Before installing anything, confirm your machine can run the stack.

| Requirement | Minimum | Why |
|---|---|---|
| OS | Windows 10/11 (you have this) | Ollama + Python support |
| RAM | 16 GB (32 GB recommended) | `qwen2.5:7b-instruct` needs ~6 GB RAM at runtime; Python + embedded Chroma use the rest |
| Disk | 15 GB free | Model weights (~5 GB) + corpus + Python deps + buffer |
| CPU | Any modern x64 | GPU optional; CPU inference works for the demo |

> **Note:** This project runs **natively** — no Docker required. Decision driven by Intune policy on the dev machine blocking WSL2 (`Wsl/ERROR_ACCESS_DISABLED_BY_POLICY`). ChromaDB runs in embedded mode and FastAPI/Streamlit run directly via `uv`. This is a defensible architecture choice for a local-first audit demo — see `PLAN.md` for the trade-off discussion.

**Verify** (PowerShell):

```powershell
# RAM (look for "Total Physical Memory")
systeminfo | Select-String "Total Physical Memory"
Output: Total Physical Memory:         32,480 MB

# Free disk space on C:
Get-PSDrive C | Select-Object Used, Free
Output: Used 93711818752 / Free 412620361728
```

---

## 2. Install Git (skip if `git --version` already works)

```powershell
winget install --id Git.Git -e --source winget
```

**Verify**:

```powershell
git --version
# Expected: git version 2.x.x
```

---

## 3. Install Python 3.11

We pin to 3.11 — it's the sweet spot for PyTorch, `sentence-transformers`, and ChromaDB compatibility as of mid-2026.

```powershell
winget install --id Python.Python.3.11 -e --source winget
```

After install, **close and reopen PowerShell** so `PATH` refreshes.

**Verify**:

```powershell
python --version
# Expected: Python 3.11.x

py -3.11 --version
# Expected: Python 3.11.x
```

If `python` resolves to a different version, use `py -3.11` everywhere in the project.

---

## 4. Install `uv` (fast Python package manager)

We'll use `uv` instead of `pip`/`poetry` — it's 10–100x faster and handles virtual environments cleanly. Showing modern tooling is also a small but real signal on a portfolio project.

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close and reopen PowerShell.

**Verify**:

```powershell
uv --version
# Expected: uv 0.x.x
```

---

## 5. ~~Install Docker Desktop~~ — skipped

Docker Desktop requires WSL2 or Hyper-V, both blocked by Intune policy on this machine. The project runs natively instead. No action needed here.

---

## 6. Install Ollama (native, runs as a Windows service)

Ollama runs as a background service on port 11434. The FastAPI service (running natively via `uv`) will talk to it over `http://localhost:11434`.

```powershell
winget install --id Ollama.Ollama -e --source winget
```

After install, Ollama runs as a background service on port 11434.

**Verify**:

```powershell
ollama --version
# Expected: ollama version 0.x.x

# Check the daemon is serving
Invoke-WebRequest http://localhost:11434/api/tags | Select-Object -ExpandProperty Content
# Expected: {"models":[]} (empty list is fine — we haven't pulled any yet)
```

**Pull the model** (this is the big download — ~4.7 GB, do it now over good wifi):

```powershell
ollama pull qwen2.5:7b-instruct
```

**Verify**:

```powershell
ollama list
# Expected: qwen2.5:7b-instruct listed with a size around 4.7 GB

# Smoke test the model
ollama run qwen2.5:7b-instruct "Reply with just the word OK."
# Expected: OK
```

---

## 7. Install Visual Studio Code extensions (if missing)

You already use VS Code with Copilot. Make sure these are installed for the best dev loop:

```powershell
code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
code --install-extension charliermarsh.ruff
code --install-extension tamasfe.even-better-toml
```

---

## 8. Create a GitHub repository (empty, public)

Public is intentional — this is a portfolio project meant to be visible to your panel.

1. Go to https://github.com/new
2. **Repository name**: `audit-copilot`
3. **Description**: `Local-first AI assistant for Audit & Assurance — RAG over auditing standards + journal-entry anomaly detection.`
4. **Visibility**: Public
5. **Do not** initialize with README, .gitignore, or license — we'll create those locally.
6. Click **Create repository**.
7. Clone URL (HTTPS): `https://github.com/MRafagnin/audit-copilot.git` — Phase 0 will use this.

**Verify** your git identity is configured:

```powershell
git config --global user.name "Matheus Rafagnin"
git config --global user.email "mprafagn@gmail.com"  # use the email tied to your GitHub account
git config --global init.defaultBranch main
```

We're using **HTTPS** (not SSH). Auth happens on the first `git push` via Git Credential Manager — it pops a browser sign-in to GitHub and caches the token. No SSH key setup needed.

---

## 9. Decide your HTTP `User-Agent` string

We fetch public PDFs (AUASB standards, the chosen ASX annual report). Most hosts don't strictly require it, but identifying the caller is good practice and avoids being blocked as a bot. Format is `Name email@domain`. Pick one now:

- **Option A (recommended for a portfolio repo)**: a personal email — `"Matheus Rafagnin matheus.rafagnin@gmail.com"`. Keeps the repo decoupled from your employer.
- **Option B**: your work email. Fine, but if the repo goes public it leaks your work email to scrapers.

Write your choice down — you'll paste it into `.env` during Phase 0 as `HTTP_USER_AGENT`. **Do not commit this to git** (`.env` will be in `.gitignore`).

---

## 10. Pick the ASX annual report for the demo

The RAG corpus is grounded against AUASB ASA standards plus one real Australian annual report at a time. `WOW` (Woolworths Group, FY25) is the **bootstrap default** — it is fetched, chunked, and indexed automatically by `uv run make bootstrap`. The other allowlist tickers are fetched on demand the first time you select them from the Streamlit sidebar (the UI calls `POST /companies/{ticker}/ingest`).

Current 8-ticker allowlist (`src/rag/registry.py`, all FY25):

| Ticker | Company                          | Indexed by bootstrap? |
|--------|----------------------------------|------------------------|
| `WOW`  | Woolworths Group                 | yes (default)          |
| `CBA`  | Commonwealth Bank of Australia   | on demand              |
| `TLS`  | Telstra Group                    | on demand              |
| `CSL`  | CSL Limited                      | on demand              |
| `NAB`  | National Australia Bank          | on demand              |
| `ANZ`  | ANZ Group                        | on demand              |
| `RIO`  | Rio Tinto                        | on demand              |
| `MQG`  | Macquarie Group                  | on demand              |

> `BHP` and `WES` were evaluated and dropped — both issuers' CDNs reject programmatic HTTPS clients via TLS fingerprinting, so on-demand ingest cannot fetch their annual reports without a browser-impersonation HTTP layer.

**Default pick: `WOW`** — the FY25 Annual Report has substantial risk and internal-controls content, the retail framing is intuitive, and "Woolworths" lands instantly with a Sydney audience. No action needed; bootstrap handles it.
---

## 11. Set up a project folder

Choose where the repo will live. Recommendation: somewhere short and **outside OneDrive** to avoid file-sync conflicts with the Chroma persistent store and `.venv`.

```powershell
# Recommended
New-Item -ItemType Directory -Path C:\dev -Force | Out-Null
cd C:\dev
```

Do **not** clone yet — Phase 0 will do that.

---

## Pre-flight checklist

Tick each box before we start Phase 0:

- [ ] 16 GB+ RAM confirmed, 15 GB+ free disk
- [ ] `git --version` works
- [ ] `python --version` shows 3.11.x
- [ ] `uv --version` works
- [ ] `ollama list` shows `qwen2.5:7b-instruct`
- [ ] `ollama run qwen2.5:7b-instruct "Reply with just the word OK."` returns OK
- [ ] VS Code extensions installed
- [ ] Empty `audit-copilot` repo created on GitHub (HTTPS: `https://github.com/MRafagnin/audit-copilot.git`)
- [ ] `git config --global user.name` and `user.email` set

When every box is ticked, reply "ready" and I'll kick off Phase 0 (repo bootstrap).

---

## Troubleshooting

**`winget` not recognized**: install App Installer from the Microsoft Store, then reopen PowerShell.

**Ollama port 11434 already in use**: another Ollama instance or process is bound. `Get-Process ollama | Stop-Process` then restart from the Start menu.

**`ollama pull` is slow/stalls**: it's a single ~4.7 GB blob — let it run. If it fails partway, rerun the same command; it resumes.

**Corporate proxy blocking downloads**: set `HTTPS_PROXY` and `HTTP_PROXY` in your PowerShell session before retrying. If `huggingface.co` or `ollama.com` are blocked, you'll need to run this project from a personal network.
