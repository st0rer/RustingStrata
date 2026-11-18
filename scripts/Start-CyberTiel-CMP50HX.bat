@echo off
setlocal
title RustingStrata Cyber-Tiel-Coder 35B MoE (CMP 50HX 20GB, 128K Ctx, MTP-2)

echo =====================================================================
echo  Starting RustingStrata - Cyber-Tiel-Coder 35B-A3B (sm_75, MTP-2)
echo  Port: 8081 / Context: 131,072 (128K) / Chunk: 4,096 / KV: Q8
echo  MTP: 2 / Anti-Loop: ON / 100%% Experts in VRAM (0 CPU misses)
echo  Auto-Trim RAM: ON (Frees ~11 GB WorkingSet for Intel HD Graphics)
echo  Anti-Loop Calibrated: Temp 0.6 / Min-P 0.05 / Rep-Pen 1.05 / Rep-Last-N 64 (Pres/Freq 0.0 for pure CoT)
echo =====================================================================

:: 1. Apply GPU lock (165W, 1750 MHz)
nvidia-smi -pm 1 >nul 2>&1
nvidia-smi -pl 165 >nul 2>&1
nvidia-smi -lgc 0,1750 >nul 2>&1

:: 2. Stop any existing strata or llama-server instances
taskkill /F /IM strata.exe >nul 2>&1
taskkill /F /IM rustingstrata.exe >nul 2>&1
taskkill /F /IM llama-server.exe >nul 2>&1
timeout /t 1 /nobreak >nul 2>&1

:: 2.5 Start embedded background RAM trimmer (waits for port 8081, then frees ~11 GB WorkingSet for Intel iGPU)
start "" /b powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0trim-ram.ps1"

:: 3. Launch Strata OpenAI-compatible server on http://0.0.0.0:8081
:: Note: For full 256K context, set --ctx 262144 --chunk 2048 --vram-margin 128
echo Launching Strata OpenAI API on http://0.0.0.0:8081 ...
"C:\cmp\RustingStrata\target\release\strata.exe" serve ^
  --model "C:\models\cybertiel\Cyber-Tiel-Coder-35B-A3B-MTP.gguf" ^
  --gpu ^
  --mtp 2 ^
  --ctx 131072 ^
  --chunk 4096 ^
  --vram-margin 512 ^
  --kv q8 ^
  --temp 0.6 ^
  --top-k 20 ^
  --top-p 0.95 ^
  --addr 0.0.0.0:8081

pause
