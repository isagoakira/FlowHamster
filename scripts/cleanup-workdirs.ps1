# FlowHamster Agent Workdir Cleanup Script
# Usage: .\scripts\cleanup-workdirs.ps1 [-DryRun]
# Removes completed agent workdirs older than 7 days to prevent disk bloat.
param(
    [switch]$DryRun
)

$workspaceBase = "C:\Users\11436\multica_workspaces_desktop-api.multica.ai\3b277bc2-38bb-4425-a3da-86154d244202"
$cutoffDate = (Get-Date).AddDays(-7)

if (!(Test-Path $workspaceBase)) {
    Write-Error "Workspace base not found: $workspaceBase"
    exit 1
}

$workdirs = Get-ChildItem $workspaceBase -Directory | Where-Object {
    $workdirPath = Join-Path $_.FullName "workdir"
    Test-Path $workdirPath
}

Write-Host "Found $($workdirs.Count) workdirs under $workspaceBase" -ForegroundColor Cyan

$removed = 0
$skipped = 0

foreach ($dir in $workdirs) {
    $workdirPath = Join-Path $dir.FullName "workdir"
    $lastWrite = (Get-Item $workdirPath).LastWriteTime

    if ($lastWrite -lt $cutoffDate) {
        $ageDays = [math]::Floor(((Get-Date) - $lastWrite).TotalDays)
        Write-Host "[$ageDays days old] $($dir.Name)" -ForegroundColor Yellow
        if (!$DryRun) {
            Remove-Item $dir.FullName -Recurse -Force
            Write-Host "  -> REMOVED" -ForegroundColor Red
        } else {
            Write-Host "  -> Would remove (dry-run)" -ForegroundColor DarkGray
        }
        $removed++
    } else {
        $skipped++
    }
}

Write-Host "" -ForegroundColor Cyan
Write-Host "Summary: $removed removed, $skipped kept (cutoff: $($cutoffDate.ToString('yyyy-MM-dd')))"
if ($DryRun) {
    Write-Host "This was a dry-run. Re-run without -DryRun to actually delete." -ForegroundColor Magenta
}
