# [Draft issue] qwen4exp: decode cost grows ~linearly with context — QSA top-k applied as a dense mask, indexer re-pools the whole cache every step

Filed against: llama.cpp `b10671` / `b19cbe925` (also reproduces on the PR #27742 branch build)
Model: `unsloth/Qwen3.8-Flash-Next-GGUF` UD-Q6_K_XL (`general.architecture = qwen4exp`, 48 layers,
`full_attention_interval = 4` → 12 QSA layers + 36 Gated DeltaNet, `indexer.top_k = 2048`)

## Summary

Generated-token latency scales with the **number of tokens already in the context**, even though
QSA is supposed to bound attention work to `indexer_top_k` cells. On my hardware:

| tokens in context | measured tg | ms/token |
|---|---|---|
| ~1k | 31.1 | 32.2 |
| ~46k | 14.8 | 67.6 |
| ~122k | 10.0 | 100.0 |
| ~160k | 8.5 | 117.2 |
| ~176k | 8.2 | 122.0 |

Fit: `step_time ≈ 31 ms + 0.55 µs × n_ctx_used`. Per QSA layer that is ~46 ns per cached token per
step — roughly **80× the bytes actually required** to read one token's K/V (2 heads × 256 dim,
q8_0 ≈ 512 B). Prompt processing decays the same way: 630 tok/s at 2k → 425 tok/s at 122k.

During decode **neither GPU nor CPU is saturated**: SM utilisation 21–22 %, memory utilisation 3–7 %,
~1.4 CPU cores busy of 20. The step is serialised on work that should not exist at decode time.

## Cause 1 — the sparse selection is expressed as a mask, so FA is still dense

`src/models/qwen4exp.cpp`, `build_attn_qsa()`:

```cpp
ggml_fill(kq_mask, -INFINITY);                                   // l.642  writes all n_kv cells
ggml_set_rows(ctx0, kq_mask, zero, top_k);                       // l.647  writes 0.0 at the ~2048 winners
auto cur = ggml_flash_attn_ext(ctx0, q, k, v, kq_mask, ...);     // l.660  walks the full n_kv span
```

`ggml_flash_attn_ext()` cannot skip cells because of a mask *value* — it still loads K/V and
computes `QK^T` for every cached token, then discards all but `top_k`. So QSA currently bounds what
the model *sees*, not what is *computed*. Comment in the code: *"Dense GQA self-attention restricted
to the cells that top_k names."*

## Cause 2 — the indexer rebuilds every block key on every step

`src/models/qwen4exp.cpp`, `build_qsa_top_k()`: for each of the 12 QSA layers, per step:

1. `ggml_get_rows()` over **all** `n_kv` indexer keys,
2. reshape and mean-pool with `r` separate `ggml_cont()` slice copies + `r-1` `ggml_add()`,
3. `build_norm()` (RMS) over all `n_blocks`,
4. **`ggml_rope_multi()` re-RoPEs every block key**,
5. `ggml_mul_mat` score, `relu`, `permute+cont`, `sum_rows`,
6. `ggml_get_rows()` to expand block scores back to `n_kv`, add the mask (with an `f16→f32` cast),
7. `ggml_top_k` over `n_kv`.

Only the newest completed block ever changes, yet ~0.5 GB of traffic and ~180 extra graph nodes per
layer-step are spent recomputing the rest — i.e. ~6 GB/step and ~180 extra kernel launches on top of
the 7.6 GB of weights genuinely required per token.

## Negative controls (so the obvious suspects can be ruled out)

| suspect | test | result |
|---|---|---|
| GPU clocks / DVFS | `nvidia-smi -pm 1 -lgc 1695,1695`: SM 1140 → 1470 MHz | kernel-launch latency 16.0 → **2.5 µs**, decode **±0 %** (8.53 → 8.21 t/s while ctx grew) |
| power limit | cumulative throttle counters since boot | `SW Power Capping 0 µs`, `HW Thermal 0 µs`, 61–70 W of 250 W |
| HBM bandwidth | measured 783 GB/s D2D; memory util 3–7 % | not bandwidth-bound (also independently stated by the 170tune project: "single-stream decode is not bandwidth-bound") |
| PCIe / interconnect | Gen2 ×4, 1.62 GB/s H2D, **no P2P** | per-token activations are ~10 KB; the 9.8 µs small-transfer latency cannot account for 100 ms/token |
| CPU-side offload | 54.4 GB `per_layer_token_embd` on CPU | only 1.4 cores busy; PLE gathers are O(1) per token |
| KV quantisation | `-ctk q8_0 -ctv q8_0` already on | since Cause 1 makes FA read all KV every step, KV bytes are direct per-token cost — `f16` K would be *worse*, not free |

## Suggested directions

1. **Make the top-k selection a gather, not a mask.** Materialise the selected `top_k` cells
   (compact K/V, or a gather inside the FA kernel) so the attention kernel cost is O(top_k), not
   O(n_kv). `GGML_OP_FLASH_ATTN_EXT` already registers a `TOP_K` extra buffer that is unused here.
2. **Cache pooled block keys.** Compute mean-pool + RMS-norm + RoPE once per *completed* block,
   store them in the indexer cache, and per step do only the newest block plus the score/top-k pass.
   Re-RoPE-ing the entire block cache every token is the single most avoidable term.
3. **Fuse the pooling** (`r` `ggml_cont` slices + adds → one kernel), which also cuts node count and
   launch overhead for the 4× and 12× multiplication across layers.
4. **Consider a narrower indexer cache type** — the cache is currently f32 (see the "in its own
   memory type" path in `llama-model.cpp` ~l.2503), which doubles/triples the O(n_kv) traffic in
   steps 1–5 above.

## Environment / reproduction

3× GA100-class 64 GB (sm_80, 70 SM, PCIe Gen2 ×4, no P2P) in a KVM guest, driver 610.43.03,
`--jinja --flash-attn on --fit off -ngl all --split-mode layer --tensor-split 1,1,1 -c 262144
--parallel 1 -b 4096 -ub 1024 -ctk q8_0 -ctv q8_0 --no-context-shift`.
Repro: measure `timings.predicted_per_second` on `/completion` for the same `n_predict` at
increasing prompt lengths (prompt cache on, so only the decode part differs).

Happy to test patches or run a profiling trace on this hardware — the failure mode is very clean and
the box is dedicated to this model.
