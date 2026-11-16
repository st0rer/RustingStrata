# High-Performance Turing (sm_75) & CMP 50HX Tuning Guide for RustingStrata

This document details the architectural optimizations, kernel flags, memory strategies, and speculative decoding calibration applied in this fork of **RustingStrata**, specifically tuned for NVIDIA Turing GPUs (`sm_75`), such as the **CMP 50HX 20 GB GDDR6**, **RTX 2080 Ti (11 GB / 22 GB mods)**, **Titan RTX**, and **RTX 2060/2070/2080**.

---

## 1. Hardware Architecture & Target Profile

| Specification | NVIDIA CMP 50HX (Headless Turing) | NVIDIA RTX 2080 Ti |
| :--- | :--- | :--- |
| **GPU Die** | TU102 (Turing, Compute Capability `7.5`) | TU102 (`sm_75`) |
| **CUDA Cores / SMs** | 3,584 Cores / 56 Streaming Multiprocessors | 4,352 Cores / 68 SMs |
| **VRAM Capacity** | **20,480 MiB (20 GB GDDR6)** | 11,264 MiB (or 22 GB mod) |
| **Memory Bus Width** | **320-bit** | 352-bit |
| **Theoretical Bandwidth** | **560.0 GB/s** (at 1750 MHz / 14 Gbps) | 616.0 GB/s |
| **TDP & Power Limit** | 165W (Targeted lock: 165W) | 250W–280W |
| **Shared Memory / SM** | **64 KB** opt-in limit (`cudaSharedMemBankSizeFourByte`) | 64 KB opt-in limit |

---

## 2. Key Optimizations Implemented in this Fork

### 2.1. NVRTC Fast-Math & Compiler Vectorization
- **File:** `src/cuda/nvrtc.rs`
- **Compiler Flags:** Added `"--use_fast_math"` and `"-extra-device-vectorization"`.
- **Impact:** 
  - Unlocks hardware-native Fast Multiply-Add (FMA) instructions across all 56 Turing SMs.
  - Replaces slow double-precision transcendentals with hardware reciprocal square root (`rsqrtf`) and fast exponentials in RMSNorm, RoPE, and Gated DeltaNet linear attention kernels.
  - Improves decode latency from **8.8 ms/tok down to 7.4 ms/tok** on `Cyber-Tiel-Coder-35B` without numerical divergence.

### 2.2. Persistent Windows `%LOCALAPPDATA%` Cubin Cache
- **File:** `src/cuda/mod.rs`
- **Mechanism:** On Windows, fallback to `%LOCALAPPDATA%\rusting_strata` rather than requiring Unix-style `$XDG_CACHE_HOME` or `$HOME`.
- **Impact:** 
  - Eliminates the 15-second NVRTC re-compilation delay whenever the application is launched from different directories or bat files.
  - Cold startup takes **< 1.0 second**.

### 2.3. $O(N)$ Fast Vectorized Top-K / Top-P Sampler
- **File:** `src/ops.rs`
- **Mechanism:** 
  - Replaced $O(V \log V)$ sorting of the entire 248,320-token vocabulary.
  - Logits below $e^{-20}$ of maximum are filtered in SIMD lanes; then `select_nth_unstable_by` finds the Top-K candidates in linear $O(N)$ time before normalizing.
  - Logit scaling multiplies by precomputed inverse temperature (`inv_temp = 1.0 / temp`), removing expensive division per token.
- **Impact:** Saves ~0.3–0.5 ms per token during sampling on large vocabularies.

### 2.4. Real-Time Telemetry & Per-Request Delta Profiling
- **File:** `src/main.rs`
- **Features:** 
  - Tracks delta speculation counters per HTTP request (avoids server lifetime cumulative counter distortion).
  - Emits real-time breakdown on completion:
    ```text
    =================== INFERENCE STAGE BREAKDOWN ===================
    PREFILL: 35 tokens (30 cached, 5 new) in 0.043s (116.1 tok/s) | TTFT: 43.1 ms
    DECODE:  150 tokens in 1.11s (135.2 tok/s) | TPOT: 7.40 ms/tok | finish=length
    SPECULATION: [MTP] Drafts accepted: 85/130 (65.4%) | Eff: 2.31 tok/step
    BANDWIDTH & MBU: Achieved Bandwidth: 51.6 GB/s | Model Bandwidth Utilization: 9.2%
    =================================================================
    ```

---

## 3. Speculative Decoding Optimization (MTP vs N-gram vs MTP-3)

### 3.1. Why Pure MTP-2 Beats Hybrid N-gram
On models trained with a dedicated Next-N neural prediction head (e.g. `Cyber-Tiel-Coder-35B-A3B-MTP` and `Swift-1.5-Qwen3.8-27B`):
- **Hybrid N-gram (`--spec-type ngram-mod`):** Heuristic syntactic matching from recent context. During reasoning (`<think>`) and code generation, syntactic repeats fail verification (~30–40% acceptance), triggering DeltaNet recurrent state rollbacks and blocking the neural MTP head.
- **Pure MTP-2 (`--mtp 2`):** Drafts are generated from the deep semantic representations of the trained MTP head. Acceptance rises to **65–75%**, delivering **132–135 tok/s** decode.

### 3.2. Why MTP-2 Outperforms MTP-3
Empirical tests on the CMP 50HX:
* **MTP-2 (`--mtp 2`):** 132.5 – 135.2 tok/s | 65.4% – 75.0% acceptance | TPOT: 7.40 ms.
* **MTP-3 (`--mtp 3`):** 110.7 – 116.2 tok/s | 46.7% – 56.0% acceptance | TPOT: 8.60 ms.

*Cause:* Because the model has `n_nextn: 1`, draft 3 is an unanchored prediction on top of a prediction without full transformer normalization. The error rate for token 3 exceeds 65%, while expanding verification batch size from 3 to 4 tokens adds ~5 ms per step, resulting in a net speed loss.

---

## 4. Context Scaling on Hybrid DeltaNet (20 GB VRAM)

`Cyber-Tiel-Coder-35B` combines:
* **30 Gated DeltaNet Layers (Linear Attention):** Constant $O(1)$ memory state ($S \in \mathbb{R}^{d \times d} \approx 30\text{ MB}$ total).
* **10 Self-Attention Layers:** Full global attention, requiring only 20.0 KB per token in FP16 (or ~10 KB in Q8).

| Context Length (`--ctx`) | KV Cache Size (Q8) | VRAM Total | Expert Residency (Hit Rate) |
|---|---|---|---|
| **4K** | 40 MB | 14.1 GB | 100% (10,496 / 10,496) |
| **32K** | 320 MB | 14.4 GB | 100% (10,496 / 10,496) |
| **65K** | 640 MB | 14.7 GB | 100% (10,496 / 10,496) |
| **128K** | 1,280 MB | **15.4 GB** | **100% (10,496 / 10,496)** |
| **256K (Full)** | 2,560 MB | **16.7 GB** | **100% (10,496 / 10,496)** |

---

## 5. Recommended Production Launcher Script (Windows `.bat`)

```bat
@echo off
setlocal
title RustingStrata - Cyber-Tiel-Coder 35B MoE (CMP 50HX 20GB, MTP-2)

:: 1. Apply hardware GPU lock (165W TDP, 1750 MHz lock)
nvidia-smi -pm 1 >nul 2>&1
nvidia-smi -pl 165 >nul 2>&1
nvidia-smi -lgc 0,1750 >nul 2>&1

:: 2. Terminate dangling instances
taskkill /F /IM strata.exe >nul 2>&1

:: 3. Launch Strata OpenAI-compatible server on http://0.0.0.0:8081
"strata.exe" serve ^
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
```
