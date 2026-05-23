# =============================================================================
# inject-guards-legend.ps1 -- SessionStart + UserPromptSubmit hook
#
# Emits a systemMessage reminding the agent of the four Command Guard tiers.
# The legend matches .github/copilot-instructions.md verbatim so there is
# exactly one source of truth; if this message drifts from the instructions,
# update both.
# =============================================================================

$legend = @'
### Command Guards

Classify every command before running it:

[READ-ONLY] -- just looking, no harm done
[MUTATION] -- changes things, but recoverable
[DESTRUCTIVE] -- irreversible, think twice
[SYSTEM] -- touching system-level stuff like packages and permissions

Enforcement: a PreToolUse hook classifies every terminal command and returns
allow for [READ-ONLY] or ask for everything else. State the tier in square
brackets before running any non-trivial command.
'@

$out = @{
    systemMessage = $legend
    continue      = $true
} | ConvertTo-Json -Depth 3 -Compress

Write-Output $out
exit 0
