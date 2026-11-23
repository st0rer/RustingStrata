@echo off
rem ==============================================================================
rem Optimal Benchmarking Script for Inkling-2-Mini + DSpark Speculative Decoding
rem Target Hardware: NVIDIA CMP 50HX 20GB (TCC mode, SM 75, Driver 617.14)
rem Sweet Spot: --spec-draft-n-max 3 (85.2% - 93.3% Acceptance, 80+ tok/s Decode)
rem ==============================================================================

setlocal
set "BIN=C:\cmp\RustingStrata\target\release\strata.exe"
if not exist "%BIN%" set "BIN=target\release\strata.exe"
if not exist "%BIN%" set "BIN=target\release\rustingstrata.exe"

set "MODEL=C:\models\inkling\Inkling-2-Mini-Abliterated-APEX-I-MiniPlus-V2.1.gguf"
set "DRAFT=C:\models\inkling\dspark-Inkling-2-Mini-Q4_0.gguf"

rem Check GPU settings
nvidia-smi -pm 1 >nul 2>&1
nvidia-smi -pl 165 >nul 2>&1
nvidia-smi -lgc 0,1750 >nul 2>&1

echo ==============================================================================
echo Starting Inkling-2-Mini Benchmark with DSpark GPU Drafter (Affinity 0xE)
echo Model: %MODEL%
echo Draft: %DRAFT%
echo ==============================================================================

start /affinity 0xE /b /wait "%BIN%" bench ^
  --model "%MODEL%" ^
  --draft-model "%DRAFT%" ^
  --spec-draft-n-max 3 ^
  --gpu ^
  --chunk 2048 ^
  --ctx 512000 ^
  --kv q8 ^
  --vram-margin 512 ^
  --pp 128,1024,4096 ^
  --tg 64 ^
  --reps 1

echo Benchmark completed with exit code %ERRORLEVEL%.
