# Registreer geplande taak: start SSH DB-tunnel bij aanmelding.
# Eenmalig uitvoeren als Administrator: powershell -ExecutionPolicy Bypass -File scripts\install_db_tunnel_task.ps1

$TaskName = "KFL_DB_Tunnel_15432"
$ScriptPath = Join-Path $PSScriptRoot "start_db_tunnel.ps1"

if (-not (Test-Path $ScriptPath)) {
    Write-Host "ERROR: start_db_tunnel.ps1 niet gevonden: $ScriptPath" -ForegroundColor Red
    exit 1
}

$Action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$ScriptPath`""
$Trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
$Principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited

Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Principal $Principal | Out-Null

Write-Host "Taak geregistreerd: $TaskName" -ForegroundColor Green
Write-Host "  Trigger: bij aanmelding ($env:USERNAME)" -ForegroundColor Gray
Write-Host "  Tunnel start automatisch. Handmatig starten: .\scripts\start_db_tunnel.ps1" -ForegroundColor Gray
Write-Host "  Of taak nu uitvoeren: Start-ScheduledTask -TaskName $TaskName" -ForegroundColor Gray
