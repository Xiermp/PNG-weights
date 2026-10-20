# pngweights file format (v2)

## Container

A standard PNG: `IHDR` (width **1024**, height = rows, bit depth 8, color type 0 gray, no interlace),
optional `zTXt` with keyword `pngweights`, one `IDAT` (zlib), `IEND`. Every scanline uses filter 0.

Pixel bytes, row-major:

```
u64 BE  payload_length
bytes   payload
zeros   padding to a whole row
```

The `zTXt` text (zlib-compressed UTF-8 JSON) is the **manifest**:

```json
{"format": "pngweights", "version": 2, "model": "...", "preset": "q4", "config": {...},
 "tensors": [{"name": "...", "shape": [..], "kind": "rtnx", "bits": 4, "group": 32, ...,
              "transpose": false, "bytes": 123, "sha256": "..."}]}
```

v1 files (first version) have no `zTXt`; their manifest lives in a separate JSON file and is passed
explicitly (`codec.load(png, manifest_json)`, `pngweights verify f.png --manifest m.json`).

## Payload

Concatenation, in manifest order, of `u32 LE length` + tensor blob. All integers little-endian.
A blob decodes to a 2-D array `(R, C)`; it is transposed if `transpose` is set, then reshaped to `shape`.

### `raw`
```
u32 R, u32 C, u32 dtype      dtype: 0 fp16 | 1 fp32 | 2 fp16 byte-shuffled | 3 fp32 byte-shuffled
values                       shuffled = all byte-0s, then all byte-1s, ...
```

### `rtn` (v1-compatible)
```
u32 R, C, bits, group, N     N = R*C rounded up to a multiple of group
fp16 scale[N/group]
codes                         N codes, `bits` each, MSB-first bit-packed
w = (code - half) * scale     half = (2^bits - 1) // 2
```

### `rtnx`
```
u32 R, C, bits, group, N, enc, rot, seed, sgroup
scales   sgroup == 0 : fp16 scale[N/group]
         sgroup  > 0 : fp16 (min, max)[ceil(ns/sgroup)] then u8 q[ns];  s = min + q/255 * (max-min)
codes    enc 0 : bit-packed as in rtn
         enc 1 : one u8 per code
         enc 2 : rANS (constriction, Categorical with p = (count+1)/Σ): u32 count[2^bits], u32 nwords, u32 words[]
x = (code - half) * scale, reshaped to (R, C)
if rot > 0 : w = D · H_rot(x) along the last dim, H = normalized Walsh-Hadamard in blocks of `rot`,
             D = diag(±1) from numpy default_rng(seed).integers(0, 2, C)
```

### `alias`
```
u32 source_index, u32 transpose     tensor = (transposed) copy of an earlier tensor
```
