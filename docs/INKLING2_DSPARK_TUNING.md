# Inkling-2-Mini + DSpark Speculative Decoding Tuning Guide

Target Architecture: NVIDIA CMP 50HX 20GB (TCC mode, SM 75 / Turing, Driver 617.14)  
Engine: RustingStrata (with GPU Argmax, Mutex elimination, and Paired `gemvq2` kernels)

---

## 1. Optimal Launch Parameters

### Benchmarking
```cmd
start /affinity 0xE /b /wait strata.exe bench ^
  --model "C:\models\inkling\Inkling-2-Mini-Abliterated-APEX-I-MiniPlus-V2.1.gguf" ^
  --draft-model "C:\models\inkling\dspark-Inkling-2-Mini-Q4_0.gguf" ^
  --spec-draft-n-max 3 ^
  --gpu ^
  --chunk 2048 ^
  --vram-margin 512 ^
  --pp 128,1024,4096 ^
  --tg 64 ^
  --reps 3
```

### Production Server (OpenAI-compatible REST API)
```cmd
start /affinity 0xE "" strata.exe serve ^
  --model "C:\models\inkling\Inkling-2-Mini-Abliterated-APEX-I-MiniPlus-V2.1.gguf" ^
  --draft-model "C:\models\inkling\dspark-Inkling-2-Mini-Q4_0.gguf" ^
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
```

---

## 2. Key Parameter Recommendations

| Parameter | Recommended Value | Reason |
|---|---|---|
| `--spec-draft-n-max` | **`3`** | **Sweet spot:** yields 85.2% - 93.3% acceptance rate while keeping LM-head memory bus reads under 4.0 GB per step. |
| `--chunk` | **`2048`** | **Prefill peak:** delivers **750+ tok/s** on 1024-token contexts without thrashing the L2 cache. For ultra-long inputs, chunks scale up to 65,536 tokens. |
| `--vram-margin` | **`512`** | Retains sufficient buffer for dynamic scratch expansions without triggering out-of-memory errors on 20GB cards. |
| `--temp` | **`0.0 - 0.2`** | For coding and deterministic agent workflows; maintains acceptance rate at >85% and decode speed at **80+ tok/s**. |
| CPU Affinity | **`0xE`** (Cores 1, 2, 3) | Keeps Core 0 free for Windows Desktop Window Manager (DWM) and PCIe driver dispatch. |

---

## 3. Benchmark Progression

| Optimization Stage | Prefill (pp128) | Decode (tg64) | Acceptance Rate |
|---|---|---|---|
| Baseline (CPU Drafter) | 420.0 tok/s | 29.5 tok/s | 68.0% |
| DSpark GPU Drafter (V1) | 435.0 tok/s | 74.2 tok/s | 84.1% |
| GPU Argmax & Zero-Alloc Sort (`4888f9e`) | 451.1 tok/s | 78.5 tok/s | 85.2% |
| Mutex Lock Elimination (`ca94029`) | 446.7 tok/s | 79.9 tok/s | 85.2% |
| Paired Projections `gemvq2` (`54481cf`) | 462.6 tok/s | 80.1 tok/s | 85.2% |
| DSpark Kernel Pairing & Zero-Alloc (`084aa96`) | 445.4 tok/s | **80.2 tok/s** (peak 80.4) | **93.3%** |
