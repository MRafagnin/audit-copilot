# Starts FastAPI (uvicorn) and Streamlit side-by-side in two child processes.
# Ollama is expected to already be running as a Windows service on port 11434.
# Press Ctrl+C in this window to stop both.

$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $root

Write-Host "starting fastapi on http://localhost:8000" -ForegroundColor Cyan
$api = Start-Process -FilePath "uv" `
    -ArgumentList "run", "uvicorn", "src.api.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000" `
    -NoNewWindow -PassThru

Write-Host "starting streamlit on http://localhost:8501" -ForegroundColor Cyan
$ui = Start-Process -FilePath "uv" `
    -ArgumentList "run", "streamlit", "run", "app.py", "--server.port", "8501" `
    -NoNewWindow -PassThru

try {
    Wait-Process -Id $api.Id, $ui.Id
}
finally {
    Write-Host "`nshutting down dev servers" -ForegroundColor Yellow
    foreach ($p in @($api, $ui)) {
        if ($p -and !$p.HasExited) {
            try { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue } catch {}
        }
    }
}
