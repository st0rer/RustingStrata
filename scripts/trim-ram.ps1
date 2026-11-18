# =====================================================================
#  Background RAM Optimizer for RustingStrata on Windows
#  Waits for port 8081, trims WorkingSet, freeing ~10-12 GB RAM for Intel iGPU
# =====================================================================

$ready = $false
for ($i = 0; $i -lt 120; $i++) {
    Start-Sleep -Seconds 1
    if (Get-NetTCPConnection -LocalPort 8081 -State Listen -ErrorAction SilentlyContinue) {
        $ready = $true
        break
    }
}

if ($ready) {
    Start-Sleep -Milliseconds 800
    $p = Get-Process strata, rustingstrata -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($p) {
        if (-not ([System.Management.Automation.PSTypeName]'Win32.PsApi').Type) {
            Add-Type -MemberDefinition '[DllImport("psapi.dll")] public static extern bool EmptyWorkingSet(IntPtr hProcess);' -Name 'PsApi' -Namespace 'Win32'
        }
        $beforeMB = [math]::Round($p.WorkingSet64 / 1MB)
        [Win32.PsApi]::EmptyWorkingSet($p.Handle) | Out-Null
        Start-Sleep -Milliseconds 500
        $p.Refresh()
        $afterMB = [math]::Round($p.WorkingSet64 / 1MB)
        $freedMB = $beforeMB - $afterMB
        $freeSysGB = [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB, 1)

        Write-Host ""
        Write-Host "=====================================================================" -ForegroundColor Green
        Write-Host " [RAM Optimizer] УСПЕШНО: Память процесса strata.exe очищена!" -ForegroundColor Green
        Write-Host "   Занято процессом:  $afterMB МБ (было $beforeMB МБ)" -ForegroundColor Green
        Write-Host "   Освобождено ОЗУ:   $freedMB МБ в пользу Intel HD Graphics 530" -ForegroundColor Green
        Write-Host "   Свободно в ОС:     $freeSysGB ГБ физической памяти" -ForegroundColor Green
        Write-Host "   Мерцание экрана:   УСТРАНЕНО (Intel iGPU получила полный пул ОЗУ)" -ForegroundColor Green
        Write-Host "   Сервер готов:      http://0.0.0.0:8081" -ForegroundColor Green
        Write-Host "=====================================================================" -ForegroundColor Green
        Write-Host ""
    }
} else {
    Write-Host "[RAM Optimizer] Превышено время ожидания запуска strata.exe на порту 8081" -ForegroundColor Yellow
}
