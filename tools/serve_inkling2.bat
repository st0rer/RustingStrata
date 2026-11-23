@echo off
rem ==============================================================================
rem Optimal Production Server Script for Inkling-2-Mini + DSpark Speculative Decoding
rem Target Hardware: NVIDIA CMP 50HX 20GB (TCC mode, SM 75, Driver 617.14)
rem Port: 0.0.0.0:8081 (OpenAI Compatible API)
rem ==============================================================================

setlocal
set "BIN=C:\cmp\RustingStrata\target\release\strata.exe"
if not exist "%BIN%" set "BIN=target\release\strata.exe"
if not exist "%BIN%" set "BIN=target\release\rustingstrata.exe"

set "MODEL=C:\models\inkling\Inkling-2-Mini-Abliterated-APEX-I-MiniPlus-V2.1.gguf"
set "DRAFT=C:\models\inkling\dspark-Inkling-2-Mini-Q4_0.gguf"

set LOG=C:\cmp\logs\strata-inkling2-8081.log
if not exist C:\cmp\logs mkdir C:\cmp\logs

nvidia-smi -pm 1 >nul 2>&1
nvidia-smi -pl 165 >nul 2>&1
nvidia-smi -lgc 0,1750 >nul 2>&1
taskkill /F /IM strata.exe >nul 2>&1

echo [%date% %time%] Starting Inkling-2-Mini + DSpark server on 0.0.0.0:8081 >> "%LOG%"

start /affinity 0xE "" "%BIN%" serve ^
  --model "%MODEL%" ^
  --draft-model "%DRAFT%" ^
  --spec-draft-n-max 3 ^
  --gpu ^
  --chunk 2048 ^
  --ctx 65536 ^
  --vram-margin 512 ^
  --addr 0.0.0.0:8081 ^
  --temp 0.2 ^
  --top-p 0.95 ^
  --top-k 40 ^
  --repeat-penalty 1.05 ^
  --repeat-last-n 64

echo Strata server launched with affinity 0xE on port 8081.
