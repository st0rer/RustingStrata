@echo off
setlocal
title RustingStrata - Cyber-Tiel-Coder 35B MoE (CMP 50HX 20GB, MTP-2)

echo =====================================================================
echo  RustingStrata: Cyber-Tiel-Coder 35B-A3B (Turing sm_75, MTP-2)
echo  Context: 131,072 (128K) / Chunk: 4,096 / KV: Q8
echo  MTP: 2 (Pure Neural Lookahead) / 100%% Experts in VRAM (0 CPU misses)
echo  Port: 8081 (OpenAI-compatible)
echo =====================================================================

:: 1. Apply GPU lock (165W TDP, 1750 MHz memory clock)
nvidia-smi -pm 1 >nul 2>&1
nvidia-smi -pl 165 >nul 2>&1
nvidia-smi -lgc 0,1750 >nul 2>&1

:: 2. Terminate existing strata processes
taskkill /F /IM strata.exe >nul 2>&1
taskkill /F /IM rustingstrata.exe >nul 2>&1

:: 3. Launch server
rustingstrata.exe serve ^
  --model "C:\models\cybertiel\Cyber-Tiel-Coder-35B-A3B-MTP.gguf" ^
  --gpu ^
  --mtp 2 ^
  --ctx 131072 ^
  --chunk 4096 ^
  --vram-margin 512 ^
  --kv q8 ^
  --temp 0.2 ^
  --top-k 20 ^
  --top-p 0.95 ^
  --addr 0.0.0.0:8081

pause
