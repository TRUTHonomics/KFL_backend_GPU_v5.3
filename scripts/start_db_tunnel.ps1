# Start SSH tunnel naar kflhyper op VM120 (10.10.10.3).
# REASON: Docker gebruikt host.docker.internal -> host IP; tunnel moet op 0.0.0.0:15432 luisteren.
# Gebruik: .\scripts\start_db_tunnel.ps1   of  powershell -File scripts\start_db_tunnel.ps1

$RemoteHost = "10.10.10.3"
$RemoteUser = "bart"
$LocalPort = 15432
$RemotePort = 5432

# Windows OpenSSH: expliciet pad zodat we de juiste ssh gebruiken (niet WSL)
$ssh = Get-Command ssh -ErrorAction SilentlyContinue
if (-not $ssh) {
    Write-Host "ERROR: ssh niet gevonden. Installeer OpenSSH Client (Windows Settings -> Apps -> Optional features)." -ForegroundColor Red
    exit 1
}

$existing = Get-NetTCPConnection -LocalPort $LocalPort -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Poort $LocalPort is al in gebruik. Tunnel draait mogelijk al." -ForegroundColor Yellow
    Write-Host "Om opnieuw te starten: stop eerst het ssh-proces dat poort $LocalPort gebruikt." -ForegroundColor Yellow
    exit 0
}

Write-Host "Start SSH tunnel: 0.0.0.0:${LocalPort} -> ${RemoteUser}@${RemoteHost}:${RemotePort}" -ForegroundColor Cyan
Write-Host "Container kan verbinden via host.docker.internal:${LocalPort}" -ForegroundColor Cyan
Write-Host "Keep-alive actief: tunnel herstart automatisch bij uitval." -ForegroundColor Gray

# REASON: Keep-alive loop: als SSH stopt (netwerk, reboot remote, etc.) direct herstart.
while ($true) {
    $proc = Start-Process -FilePath $ssh.Source `
        -ArgumentList "-N -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes -L `"0.0.0.0:${LocalPort}:localhost:${RemotePort}`" `"${RemoteUser}@${RemoteHost}`"" `
        -PassThru -NoNewWindow
    Write-Host "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') Tunnel gestart (PID $($proc.Id))" -ForegroundColor Green
    $proc.WaitForExit()
    $exit = $proc.ExitCode
    Write-Host "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') Tunnel gestopt (exit $exit). Herstart over 5s..." -ForegroundColor Yellow
    Start-Sleep -Seconds 5
}
