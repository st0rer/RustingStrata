@echo off
setlocal
title RustingStrata - Swift-1.5 Qwen3.8 27B Dense (CMP 50HX 20GB, MTP-2)

echo =====================================================================
echo  RustingStrata: Swift-1.5-Qwen3.8-27B (Turing sm_75, MTP-2)
echo  Context: 65,536 (64K) / Chunk: 4,096 / KV: Q8
echo  MTP: 2 (Pure Neural Lookahead) / 100%% Weights in VRAM
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
  --model "C:\models\swift\Swift-1.5-Qwen3.8-27B-Uncensored-VAL-APEX-I-MiniPlus-V3.gguf" ^
  --gpu ^
  --mtp 2 ^
  --ctx 65536 ^
  --kv q8 ^
  --addr 0.0.0.0:8081

pause
