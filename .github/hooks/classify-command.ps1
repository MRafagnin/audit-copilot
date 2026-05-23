# =============================================================================
# classify-command.ps1 -- PreToolUse hook
#
# Reads a tool invocation JSON payload from stdin, extracts the proposed shell
# command, and emits a permission decision on stdout classifying it into one
# of four buckets:
#
#   [READ-ONLY]   just looking, no harm done              -> allow
#   [MUTATION]    changes things, but recoverable         -> ask
#   [DESTRUCTIVE] irreversible, think twice               -> ask (loud)
#   [SYSTEM]      packages / permissions / global state   -> ask
#
# Order of evaluation: DESTRUCTIVE -> SYSTEM -> READ-ONLY -> MUTATION (default).
# A command that matches any DESTRUCTIVE pattern is ALWAYS gated, even if it
# also matches a read-only pattern.
#
# Exit 0 always (the decision is carried in the JSON body, not the exit code).
# =============================================================================

$ErrorActionPreference = 'Stop'

function Write-Decision {
    param(
        [Parameter(Mandatory)][ValidateSet('allow','ask','deny')] [string] $Decision,
        [Parameter(Mandatory)][string] $Reason
    )
    $payload = @{
        hookSpecificOutput = @{
            hookEventName            = 'PreToolUse'
            permissionDecision       = $Decision
            permissionDecisionReason = $Reason
        }
    } | ConvertTo-Json -Depth 5 -Compress
    Write-Output $payload
}

# -----------------------------------------------------------------------------
# 1. Read stdin and pull out the command string from common payload shapes.
# -----------------------------------------------------------------------------
$raw = [Console]::In.ReadToEnd()
if ([string]::IsNullOrWhiteSpace($raw)) {
    Write-Decision -Decision 'ask' -Reason '[MUTATION] Empty hook payload; deferring to user.'
    exit 0
}

try {
    $payload = $raw | ConvertFrom-Json -ErrorAction Stop
} catch {
    Write-Decision -Decision 'ask' -Reason '[MUTATION] Unparseable hook payload; deferring to user.'
    exit 0
}

$cmd = $null
foreach ($path in @(
    'toolInput.command',
    'tool_input.command',
    'input.command',
    'params.command',
    'arguments.command'
)) {
    $obj = $payload
    foreach ($seg in $path.Split('.')) {
        if ($null -eq $obj) { break }
        $prop = $obj.PSObject.Properties[$seg]
        if ($null -eq $prop) { $obj = $null; break }
        $obj = $prop.Value
    }
    if ($obj) { $cmd = [string]$obj; break }
}

if ([string]::IsNullOrWhiteSpace($cmd)) {
    Write-Decision -Decision 'ask' -Reason '[MUTATION] No command string found in payload.'
    exit 0
}

# -----------------------------------------------------------------------------
# 2. Pattern banks. Single source of truth -- keep these short and obvious.
#    All matching is case-insensitive.
# -----------------------------------------------------------------------------
$destructive = @(
    '\brm\s+-[a-z]*r[a-z]*f\b',          # rm -rf, rm -fr, rm -Rf, etc.
    '\brm\s+-[a-z]*f[a-z]*r\b',
    '\bRemove-Item\b.*\s-(Recurse|Force)\b',
    '\brmdir\s+/s\b',
    '\bdel\s+/[sqf]\b',
    '\bgit\s+push\b.*(--force\b|--force-with-lease\b|\s-f\b)',
    '\bgit\s+reset\s+--hard\b',
    '\bgit\s+clean\s+-[a-z]*f\b',
    '\bgit\s+checkout\s+--\s',
    '\bgit\s+branch\s+-D\b',
    '\bgit\s+branch\s+--delete\s+--force\b',
    '\bgit\s+tag\s+-d\b',
    '\bgit\s+rebase\b',
    '\bgit\s+commit\s+--amend\b',
    '\bgit\s+filter-(branch|repo)\b',
    '\bgit\s+stash\s+(drop|clear)\b',
    '--no-verify\b',
    '\bDROP\s+(TABLE|DATABASE|SCHEMA)\b',
    '\bTRUNCATE\s+TABLE\b',
    '\bDELETE\s+FROM\b',
    '\baz\s+boards\s+work-item\s+delete\b',
    '\baz\s+(group|resource)\s+delete\b',
    '\bFormat-Volume\b',
    '\bRemove-PSDrive\b'
)

$system = @(
    '\bInstall-Module\b',
    '\bInstall-Package\b',
    '\bUninstall-(Module|Package)\b',
    '\bSet-ExecutionPolicy\b',
    '\bStart-Service\b',
    '\bStop-(Service|Process|Computer)\b',
    '\bRestart-(Service|Computer)\b',
    '\bpip3?\s+(install|uninstall)\b',
    '\bpython\s+-m\s+pip\s+(install|uninstall)\b',
    '\bnpm\s+(install|i|uninstall|remove|rm|update|upgrade)\b',
    '\bchoco\s+(install|uninstall|upgrade)\b',
    '\bwinget\s+(install|uninstall|upgrade)\b',
    '\bscoop\s+(install|uninstall|update)\b',
    '\bsudo\b',
    '\brunas\b'
)

$readonly = @(
    '^\s*(Get-ChildItem|Get-Content|Get-Location|Get-Item|Get-ItemProperty|Get-Command|Get-Module|Get-Process|Get-Date|Test-Path|Select-Object|Where-Object|Measure-Object|Format-Table|Format-List|Out-String|Write-Output|Write-Host)\b',
    '^\s*(echo|type|cat|ls|pwd|dir|head|tail|wc|find)\b',
    '^\s*git\s+(status|diff|log|show|branch|remote|config\s+--get|rev-parse|ls-files|blame|describe)\b',
    '^\s*git\s+fetch\s+--dry-run\b',
    '^\s*python\s+(--version|-V)\b',
    '^\s*python\s+-m\s+pip\s+(show|list|freeze)\b',
    '^\s*pip3?\s+(show|list|freeze|--version)\b',
    '^\s*az\s+(account\s+show|account\s+list|boards\s+work-item\s+show|boards\s+query|devops\s+project\s+(show|list))\b',
    '^\s*npm\s+(list|ls|view|outdated|--version|-v)\b',
    '^\s*node\s+--version\b'
)

function Test-AnyMatch {
    param([string] $Text, [string[]] $Patterns)
    foreach ($p in $Patterns) {
        if ([System.Text.RegularExpressions.Regex]::IsMatch(
                $Text, $p,
                [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)) {
            return $p
        }
    }
    return $null
}

# -----------------------------------------------------------------------------
# 3. Classify. DESTRUCTIVE wins over everything else.
# -----------------------------------------------------------------------------
$preview = if ($cmd.Length -gt 120) { $cmd.Substring(0, 120) + '...' } else { $cmd }

$hit = Test-AnyMatch -Text $cmd -Patterns $destructive
if ($hit) {
    Write-Decision -Decision 'ask' `
        -Reason "[DESTRUCTIVE] Irreversible -- think twice. Matched /$hit/. Command: $preview"
    exit 0
}

$hit = Test-AnyMatch -Text $cmd -Patterns $system
if ($hit) {
    Write-Decision -Decision 'ask' `
        -Reason "[SYSTEM] Touches packages / permissions / global state. Matched /$hit/. Command: $preview"
    exit 0
}

$hit = Test-AnyMatch -Text $cmd -Patterns $readonly
if ($hit) {
    Write-Decision -Decision 'allow' `
        -Reason "[READ-ONLY] Inspection only. Matched /$hit/."
    exit 0
}

Write-Decision -Decision 'ask' `
    -Reason "[MUTATION] Changes things, but recoverable. Command: $preview"
exit 0
