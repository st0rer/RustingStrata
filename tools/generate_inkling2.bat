@echo off
rem ==============================================================================
rem Generation Utility for Inkling-2-Mini + DSpark Speculative Decoding
rem Target Hardware: NVIDIA CMP 50HX 20GB (TCC mode, SM 75, Driver 617.14)
rem ==============================================================================

setlocal
set "BIN=C:\cmp\RustingStrata\target\release\strata.exe"
if not exist "%BIN%" set "BIN=target\release\strata.exe"
if not exist "%BIN%" set "BIN=target\release\rustingstrata.exe"

set "MODEL=C:\models\inkling\Inkling-2-Mini-Abliterated-APEX-I-MiniPlus-V2.1.gguf"
set "DRAFT=C:\models\inkling\dspark-Inkling-2-Mini-Q4_0.gguf"

if "%~1"=="" (
    set "PROMPT=Write a Python function to compute the nth Fibonacci number:"
) else (
    set "PROMPT=%~1"
)

powershell.exe -Command "& '%BIN%' generate --model '%MODEL%' --draft-model '%DRAFT%' --spec-draft-n-max 3 --gpu --chunk 2048 --ctx 512000 --kv q8 -n 128 --temp 0 --prompt '%PROMPT%'"
