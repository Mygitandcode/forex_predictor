# AutoGluon Training Monitor
# Usage: powershell -ExecutionPolicy Bypass -File monitor_training.ps1

param(
    [int]$TrainPID = 1884,
    [string]$ModelDir = "D:\dapanAllThings\codes\forex_predictor\autogluon_models\EURUSD_15m",
    [int]$IntervalSec = 10
)

$lastCpu = 0
$round = 0
$startTime = $null
$memHistory = @()

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  AutoGluon Training Monitor  (PID $TrainPID)" -ForegroundColor Cyan
Write-Host "  Press Ctrl+C to exit" -ForegroundColor Gray
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

while ($true) {
    $round++
    $now = Get-Date -Format "HH:mm:ss"
    $proc = Get-Process -Id $TrainPID -ErrorAction SilentlyContinue

    if (-not $proc) {
        Clear-Host
        Write-Host "============================================" -ForegroundColor Red
        Write-Host "  [$now] PROCESS $TrainPID GONE!" -ForegroundColor Red
        Write-Host "  Training ended, failed, or killed." -ForegroundColor White
        Write-Host "============================================" -ForegroundColor Red
        break
    }

    if (-not $startTime) { $startTime = $proc.StartTime }

    $cpu = [math]::Round($proc.CPU, 1)
    $mem = [math]::Round($proc.WorkingSet64 / 1MB, 0)
    $elapsedMin = [math]::Round(((Get-Date) - $startTime).TotalMinutes, 1)
    $cpuDelta = if ($lastCpu -gt 0) { [math]::Round($cpu - $lastCpu, 1) } else { 0 }
    $lastCpu = $cpu

    # CPU status
    $cpuStatus = if ($cpuDelta -gt 1) { "[HIGH] CPU surging" }
    elseif ($cpuDelta -gt 0.1) { "[OK]   CPU growing normally" }
    elseif ($cpuDelta -gt 0) { "[LOW]  CPU slight growth" }
    else { "[ZERO] CPU not increasing" }
    $cpuColor = if ($cpuDelta -gt 0.1) { "Green" } elseif ($cpuDelta -gt 0) { "Yellow" } else { "Red" }

    # Memory trend
    $memHistory += $mem
    if ($memHistory.Count -gt 6) { $memHistory = $memHistory[-6..-1] }
    if ($memHistory.Count -ge 3) {
        $memDiff = $mem - $memHistory[0]
        $memStatus = if ($memDiff -gt 500) { "[UP]   +${memDiff}MB trending up" }
        elseif ($memDiff -lt -500) { "[DOWN] ${memDiff}MB trending down" }
        else { "[OK]   stable" }
    } else {
        $memStatus = "[...]  sampling..."
    }

    # Disk activity
    $files = Get-ChildItem $ModelDir -Recurse -ErrorAction SilentlyContinue | Where-Object { -not $_.PSIsContainer }
    $newestFile = $files | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    $newestAge = if ($newestFile) { [math]::Round(((Get-Date) - $newestFile.LastWriteTime).TotalSeconds) } else { -1 }
    if ($newestAge -ge 0 -and $newestAge -lt 60) { $diskStatus = "[OK]   disk active (${newestAge}s ago)"; $diskColor = "Green" }
    elseif ($newestAge -lt 300) { $diskStatus = "[...] disk quiet (${newestAge}s ago)"; $diskColor = "Yellow" }
    elseif ($newestAge -lt 900) { $diskStatus = "[NORM] no disk write (${newestAge}s, normal for in-memory phase)"; $diskColor = "Cyan" }
    else { $diskStatus = "[WARN] disk silent >15min! Check if stuck."; $diskColor = "Red" }

    # Progress bar
    $totalSec = 7200
    $pct = [math]::Min([math]::Round($elapsedMin / ($totalSec / 60) * 100, 1), 100)
    $remainMin = [math]::Round($totalSec / 60 - $elapsedMin, 1)
    $barLen = 28
    $filled = [math]::Round($barLen * $pct / 100)
    $bar = "[" + ("=" * $filled) + ("-" * ($barLen - $filled)) + "]"

    Clear-Host
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host "  AutoGluon Monitor  [$now]  Round $round" -ForegroundColor Cyan
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  PROCESS     PID=$TrainPID  |  CPU total: ${cpu}s  |  Mem: ${mem} MB" -ForegroundColor White
    Write-Host "  CPU delta   $cpuStatus" -ForegroundColor $cpuColor
    Write-Host "  Mem trend   $memStatus  (now $mem MB)" -ForegroundColor White
    Write-Host "  Disk        $diskStatus" -ForegroundColor $diskColor
    Write-Host ""
    Write-Host "  PROGRESS    $bar  $pct%  ($elapsedMin / 120 min)" -ForegroundColor Yellow
    if ($remainMin -gt 0) { Write-Host "  ETA         about $remainMin min remaining" -ForegroundColor Gray }
    Write-Host ""
    Write-Host "  Newest file : $($newestFile.Name)  @ $($newestFile.LastWriteTime.ToString('HH:mm:ss'))" -ForegroundColor DarkGray
    Write-Host "  File count  : $($files.Count)" -ForegroundColor DarkGray
    Write-Host ""
    Write-Host "  Ctrl+C to exit" -ForegroundColor DarkGray

    Start-Sleep -Seconds $IntervalSec
}
