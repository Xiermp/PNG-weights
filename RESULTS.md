# Results

Metrics: **KL** = mean KL(original ‖ PNG model) over token positions, **top-1** = how often the
argmax token agrees with the original. These measure *fidelity*. Perplexity is shown too, but a PPL
below the original is quantization noise acting as regularization on out-of-domain text, not a
better model — read KL first.

Raw JSON for every table is in [`results/`](results/).

## TinyMoE-200M-2x16 (MoE, 16 experts, 170M of 200M params are experts)

fp32 checkpoint 799 MB, base PPL ≈ 107.2. 16k tokens wikitext-2, ctx 512.

| variant | MB | PPL | KL | top-1 | notes |
|---|---|---|---|---|---|
| old_v2 (first version) | 135.6 | 109.16 | 0.0272 | 88.3 % | q4 g64 absmax, everything else fp16 |
| + clip search, emb/head q8 | **117.0** | 105.83 | 0.0260 | 87.7 % | smallest q4 |
| + group 32 | 122.7 | 106.12 | 0.0223 | 89.8 % | |
| + u8 scales (`q4` preset) | **119.6** | 106.05 | **0.0222** | **90.1 %** | best q4 |
| + Hadamard rotation | 119.6 | 107.03 | 0.0215 | 89.7 % | no effect: experts have no outliers |
| q3 g32 | 104.6 | 111.19 | 0.0814 | 77.9 % | |
| q3 g32 + rotation + u8 scales | 98.4 | 110.47 | 0.0847 | 79.3 % | 3.8× the KL of q4 for −18 % size |

## GPT-2 small (dense, 124M)

fp32 498 MB, base PPL 23.59 (wikitext-2) / 56.30 (tinyshakespeare).
24 windows × 1024 tokens; KL on 4 windows. Linear layers quantized, embeddings as noted.

| variant | MB | PPL wiki | PPL shakesp. | KL | top-1 |
|---|---|---|---|---|---|
| old_v2: q4 g64, fp16 rest, lm_head stored twice | 182.7 | 27.94 | 63.98 | 0.150 | 77.9 % |
| clip search + tied lm_head + byte shuffle | 109.1 | 26.42 | 56.85 | 0.153 | 78.0 % |
| + emb q8 | 80.5 | 26.75 | 57.10 | 0.166 | 75.4 % |
| + group 32 | 83.5 | 25.80 | 57.32 | 0.119 | 79.7 % |
| + u8 scales | 82.0 | 25.79 | 56.75 | 0.118 | 79.6 % |
| + rotation (`q4-rot`) | 82.3 | 25.32 | 52.77 | **0.110** | **81.1 %** |
| q3 g64 | 72.4 | 48.27 | 84.89 | 0.710 | 57.1 % |
| q3 g64 + rotation | 70.1 | 45.24 | 77.05 | 0.608 | 61.4 % |
| q3 g32 + rotation + u8 scales | 71.7 | 41.62 | 64.24 | 0.501 | 62.9 % |
| **emb in q4** | 61.3 | **919** | **6216** | 4.12 | 23.9 % |

## What worked, what did not

| idea | expected | measured | verdict |
|---|---|---|---|
| clip search per group | lower error | rel. error 0.113 → 0.100, PPL better, KL ≈ same | ✅ small |
| group 32 instead of 64 | better quality | KL −20…−25 % on both models | ✅ |
| store tied weights once | −size | GPT-2 −77 MB | ✅ (model-specific) |
| embeddings 8-bit | −size | MoE −19 MB at same KL | ✅ |
| embeddings 4-bit | −size | GPT-2 destroyed (PPL 919) | ❌ never |
| u8 "double-quant" scales | ~−0.4 bit/weight | −1.5…−3 MB, quality unchanged | ✅ small |
| rANS entropy coding of codes | ~−7 % | **−0.5 %** — zlib on packed nibbles already got it | ❌ |
| one code per byte, let PNG/zlib entropy-code | ~−7 % | **+7 %** larger | ❌ |
| Hadamard rotation | fewer outliers | dense: KL −7 %, top-1 +1.4 pt; MoE: nothing | ✅ dense only |
| 3-bit without calibration | usable | KL ×4 (MoE) / ×4.5 (dense) vs q4 | ❌ needs GPTQ-style calibration |

The headline observation: **at identical bits, MoE experts lose ~5× less fidelity than dense
layers** (KL 0.022 vs 0.110), and rotation — which fixes outliers — does nothing for them.
That is followed up in a separate research repo.
