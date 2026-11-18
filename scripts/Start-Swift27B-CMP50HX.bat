@echo off
setlocal
title RustingStrata Swift 27B Dense (CMP 50HX 20GB, 65K Ctx, MTP-2)

echo =====================================================================
echo  Starting RustingStrata - Swift 27B Dense (sm_75 Native, MTP-2)
echo  Port: 8081 / Context: 65,536 (64K) / Chunk: 4,096 / KV: Q8
echo  MTP: 2 / Anti-Loop: ON (Temp 0.6 / Top-K 20 / Top-P 0.95)
echo  Auto-Trim RAM: ON (Frees ~11 GB WorkingSet for Intel HD Graphics)
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
:: Note: For 100K context, change --ctx to 100000 (takes ~18.3 GB VRAM)
echo Launching Strata OpenAI API on http://0.0.0.0:8081 ...
"C:\cmp\RustingStrata\target\release\strata.exe" serve ^
  --model "C:\models\swift\Swift-1.5-Qwen3.8-27B-Uncensored-VAL-APEX-I-MiniPlus-V3.gguf" ^
  --gpu ^
  --mtp 2 ^
  --ctx 65536 ^
  --chunk 4096 ^
  --vram-margin 512 ^
  --kv q8 ^
  --temp 0.6 ^
  --top-k 20 ^
  --top-p 0.95 ^
  --addr 0.0.0.0:8081

pause
