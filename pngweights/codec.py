"""Tensor codecs and the PNG container.

The container is a valid grayscale 8-bit PNG (width 1024, filter 0, one IDAT).
Pixel bytes = u64 BE payload length + payload + zero padding.
The manifest (JSON) is embedded in a zTXt chunk with keyword ``pngweights``,
so a single .png file is self-contained. Files written before the manifest was
embedded (format v1) can still be read by passing the JSON manifest explicitly.

Payload = concatenation of ``[u32 LE length][tensor blob]`` in manifest order.

Tensor kinds
------------
raw    [R][C][dtype]  dtype: 0 fp16, 1 fp32, 2 fp16 byte-shuffled, 3 fp32 byte-shuffled
rtn    [R][C][bits][group][N] + fp16 scales + MSB-first packed codes      (symmetric, grouped)
rtnx   [R][C][bits][group][N][enc][rot][seed][sgroup] + scales + codes
         enc    0 packed bits | 1 one code per byte | 2 rANS (needs `constriction`)
         rot    0 none | b>0 randomized block-Hadamard of block b along the last dim
         sgroup 0 fp16 scales | k>0 u8 scales, fp16 min/max per k groups ("double quant")
alias  [src index][transpose]   tied weights stored once

Reconstruction for rtn/rtnx: w = (code - half) * scale, half = (2**bits - 1) // 2
(the top code 2**bits - 1 is never produced; kept for compatibility with v1 files).
"""
import hashlib
import json
import struct
import zlib

import numpy as np

FORMAT = 'pngweights'
VERSION = 2
WIDTH = 1024
MANIFEST_KEY = b'pngweights'
CLIP_GRID = np.array([1.0, .97, .94, .91, .88, .85, .82, .79, .76, .73, .70], np.float32)


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def u32s(*v):
    return struct.pack('<' + 'I' * len(v), *v)


# ----------------------------------------------------------------- PNG container
def _chunk(tag, data):
    return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff)


def png_write(path, payload, manifest=None, level=9):
    """Write payload bytes as a gray8 PNG. `manifest` (dict) is embedded as zTXt."""
    body = struct.pack('>Q', len(payload)) + payload
    rows = (len(body) + WIDTH - 1) // WIDTH
    body += b'\x00' * (rows * WIDTH - len(body))
    a = np.frombuffer(body, np.uint8).reshape(rows, WIDTH)
    raw = np.concatenate([np.zeros((rows, 1), np.uint8), a], 1).tobytes()
    chunks = [_chunk(b'IHDR', struct.pack('>IIBBBBB', WIDTH, rows, 8, 0, 0, 0, 0))]
    if manifest is not None:
        txt = json.dumps(manifest, separators=(',', ':')).encode('utf-8')
        chunks.append(_chunk(b'zTXt', MANIFEST_KEY + b'\x00\x00' + zlib.compress(txt, 9)))
    chunks += [_chunk(b'IDAT', zlib.compress(raw, level)), _chunk(b'IEND', b'')]
    with open(path, 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n' + b''.join(chunks))


def png_read(path):
    """-> (payload bytes, embedded manifest dict or None)"""
    data = open(path, 'rb').read()
    if data[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError(f'{path}: not a PNG')
    pos, idat, manifest = 8, [], None
    while pos < len(data):
        ln = struct.unpack('>I', data[pos:pos + 4])[0]
        tag, body = data[pos + 4:pos + 8], data[pos + 8:pos + 8 + ln]
        if tag == b'IHDR':
            w, h, depth, ctype = struct.unpack('>IIBB', body[:10])
            if (w, depth, ctype) != (WIDTH, 8, 0):
                raise ValueError(f'{path}: not a pngweights atlas (w={w}, depth={depth}, color={ctype})')
        elif tag == b'zTXt' and body.startswith(MANIFEST_KEY + b'\x00'):
            manifest = json.loads(zlib.decompress(body[len(MANIFEST_KEY) + 2:]))
        elif tag == b'IDAT':
            idat.append(body)
        pos += 12 + ln
    raw = np.frombuffer(zlib.decompress(b''.join(idat)), np.uint8).reshape(-1, WIDTH + 1)
    if not (raw[:, 0] == 0).all():
        raise ValueError('only PNG filter type 0 is supported')
    body = raw[:, 1:].tobytes()
    plen = struct.unpack('>Q', body[:8])[0]
    return body[8:8 + plen], manifest


def split_tensors(payload):
    out, pos = [], 0
    while pos < len(payload):
        (ln,) = struct.unpack('<I', payload[pos:pos + 4])
        out.append(payload[pos + 4:pos + 4 + ln])
        pos += 4 + ln
    return out


# ----------------------------------------------------------------- bit packing
def pack_codes(codes, bits):
    c = np.asarray(codes, np.uint8).reshape(-1)
    if bits == 8:
        return c.tobytes()
    if bits == 4:
        if c.size % 2:
            c = np.concatenate([c, np.zeros(1, np.uint8)])
        return ((c[0::2] << 4) | c[1::2]).astype(np.uint8).tobytes()
    b = ((c[:, None] >> np.arange(bits - 1, -1, -1, dtype=np.uint8)) & 1).astype(np.uint8)
    return np.packbits(b.reshape(-1), bitorder='big').tobytes()


def unpack_codes(buf, bits, n):
    b = np.frombuffer(buf, np.uint8)
    if bits == 8:
        return b[:n]
    if bits == 4:
        out = np.empty(b.size * 2, np.uint8)
        out[0::2], out[1::2] = b >> 4, b & 15
        return out[:n]
    out, step = np.empty(n, np.uint8), 1 << 20
    allbits = np.unpackbits(b, bitorder='big')
    w = (1 << np.arange(bits - 1, -1, -1)).astype(np.uint8)
    for s in range(0, n, step):
        e = min(n, s + step)
        out[s:e] = (allbits[s * bits:e * bits].reshape(-1, bits) * w).sum(1, dtype=np.uint16)
    return out


# ----------------------------------------------------------------- rotation
def hadamard_block(C, max_block=1024):
    """largest power of two <= max_block that divides C"""
    b = 1
    while C % (b * 2) == 0 and b * 2 <= max_block:
        b *= 2
    return b


def _fwht(X, b):
    M, C = X.shape
    Y = X.reshape(M, C // b, b).astype(np.float32, copy=True)
    h = 1
    while h < b:
        Y = Y.reshape(M, C // b, b // (2 * h), 2, h)
        a, c = Y[..., 0, :].copy(), Y[..., 1, :].copy()
        Y[..., 0, :], Y[..., 1, :] = a + c, a - c
        h *= 2
    return (Y.reshape(M, C) / np.sqrt(b)).astype(np.float32)


def _signs(C, seed):
    return np.where(np.random.default_rng(seed).integers(0, 2, C) == 1, 1.0, -1.0).astype(np.float32)


def rotate(X, b, seed):
    return _fwht(X * _signs(X.shape[1], seed), b)


def unrotate(X, b, seed):
    return _fwht(X, b) * _signs(X.shape[1], seed)


# ----------------------------------------------------------------- scale double-quant
def _dq_scales(s, sgroup):
    ns = s.size
    S = np.concatenate([s, np.repeat(s[-1:], (-ns) % sgroup)]).reshape(-1, sgroup)
    lo = S.min(1).astype(np.float16).astype(np.float32)
    hi = S.max(1).astype(np.float16).astype(np.float32)
    rng = np.maximum(hi - lo, 1e-12)
    q = np.clip(np.rint((S - lo[:, None]) / rng[:, None] * 255), 0, 255).astype(np.uint8)
    s_hat = np.maximum((lo[:, None] + q / 255 * rng[:, None]).reshape(-1)[:ns], 1e-12).astype(np.float32)
    return s_hat, np.stack([lo, hi], 1).astype('<f2').tobytes() + q.reshape(-1)[:ns].tobytes()


def _dq_decode(b, ns, sgroup):
    nsg = (ns + sgroup - 1) // sgroup
    lh = np.frombuffer(b[:nsg * 4], '<f2').astype(np.float32).reshape(nsg, 2)
    q = np.frombuffer(b[nsg * 4:nsg * 4 + ns], np.uint8).astype(np.float32)
    g = np.arange(ns) // sgroup
    return np.maximum(lh[g, 0] + q / 255 * (lh[g, 1] - lh[g, 0]), 1e-12), nsg * 4 + ns


# ----------------------------------------------------------------- rANS (optional)
def _rans_model(cnt):
    import constriction
    p = (cnt.astype(np.float64) + 1) / (cnt + 1).sum()
    return constriction.stream.model.Categorical(p, perfect=False)


def _rans_encode(codes, bits):
    import constriction
    cnt = np.bincount(codes, minlength=2 ** bits).astype(np.int64)
    enc = constriction.stream.stack.AnsCoder()
    enc.encode_reverse(codes.astype(np.int32), _rans_model(cnt))
    w = enc.get_compressed().astype('<u4')
    return u32s(*cnt.astype(np.uint32).tolist()) + u32s(w.size) + w.tobytes()


def _rans_decode(b, bits, n):
    import constriction
    K = 2 ** bits
    cnt = np.frombuffer(b[:4 * K], '<u4').astype(np.int64)
    (nw,) = struct.unpack('<I', b[4 * K:4 * K + 4])
    w = np.frombuffer(b[4 * K + 4:4 * K + 4 + 4 * nw], '<u4').astype(np.uint32)
    return constriction.stream.stack.AnsCoder(w).decode(_rans_model(cnt), n).astype(np.uint8)


# ----------------------------------------------------------------- tensor codecs
def _as2d(W):
    W = np.ascontiguousarray(W, np.float32)
    return W.reshape(1, -1) if W.ndim < 2 else W.reshape(-1, W.shape[-1])


def encode_raw(W, dtype='fp16', shuffle=True):
    W2 = _as2d(W)
    arr = W2.reshape(-1).astype('<f2' if dtype == 'fp16' else '<f4')
    code = (0 if dtype == 'fp16' else 1) + (2 if shuffle else 0)
    body = arr.view(np.uint8).reshape(-1, arr.itemsize).T.tobytes() if shuffle else arr.tobytes()
    return u32s(*W2.shape, code) + body


def decode_raw(b):
    R, C, code = struct.unpack('<III', b[:12])
    isz, dt = (2, '<f2') if code in (0, 2) else (4, '<f4')
    body = np.frombuffer(b[12:], np.uint8)
    if code >= 2:
        body = body.reshape(isz, -1).T.copy()
    return body.view(dt).astype(np.float32).reshape(R, C)


def _search_scales(G, bits, clip, fp16_scales):
    qmax = 2 ** bits - 1
    half = qmax // 2
    absmax = np.abs(G).max(1)
    best_s = best_err = None
    for a in (CLIP_GRID if clip else CLIP_GRID[:1]):
        s = np.where(absmax > 0, absmax * a / half, 1.0).astype(np.float32)
        if fp16_scales:
            s = s.astype(np.float16).astype(np.float32)
        q = np.clip(np.rint(G / s[:, None]), -half, qmax - half)
        err = ((q * s[:, None] - G) ** 2).sum(1)
        if best_s is None:
            best_s, best_err = s.copy(), err
        else:
            m = err < best_err
            best_s[m], best_err[m] = s[m], err[m]
    return best_s


def encode_rtn(W, bits=4, group=64, clip=True):
    """v1-compatible grouped symmetric RTN (fp16 scale per group, packed codes)."""
    W2 = _as2d(W)
    R, C = W2.shape
    flat = W2.reshape(-1)
    flat = np.concatenate([flat, np.zeros((-flat.size) % group, np.float32)])
    G = flat.reshape(-1, group)
    half = (2 ** bits - 1) // 2
    s = _search_scales(G, bits, clip, True)
    codes = (np.clip(np.rint(G / s[:, None]), -half, 2 ** bits - 1 - half) + half).astype(np.uint8)
    return u32s(R, C, bits, group, G.size) + s.astype('<f2').tobytes() + pack_codes(codes, bits)


def decode_rtn(b):
    R, C, bits, group, N = struct.unpack('<IIIII', b[:20])
    ns, off = N // group, 20
    s = np.frombuffer(b[off:off + ns * 2], '<f2').astype(np.float32)
    off += ns * 2
    codes = unpack_codes(b[off:off + (N * bits + 7) // 8], bits, N)
    half = (2 ** bits - 1) // 2
    return ((codes.astype(np.float32) - half).reshape(ns, group) * s[:, None]).reshape(-1)[:R * C].reshape(R, C)


def encode_rtnx(W, bits=4, group=32, clip=True, enc=0, rot=0, seed=0, sgroup=0):
    """rot=-1 picks the largest power-of-two Hadamard block dividing the last dim."""
    W2 = _as2d(W)
    R, C = W2.shape
    rb = hadamard_block(C) if rot == -1 else rot
    X = rotate(W2, rb, seed) if rb else W2
    flat = X.reshape(-1)
    flat = np.concatenate([flat, np.zeros((-flat.size) % group, np.float32)])
    G = flat.reshape(-1, group)
    half = (2 ** bits - 1) // 2
    s = _search_scales(G, bits, clip, fp16_scales=not sgroup)
    if sgroup:
        s, sblob = _dq_scales(s, sgroup)
    else:
        sblob = s.astype('<f2').tobytes()
    codes = (np.clip(np.rint(G / s[:, None]), -half, 2 ** bits - 1 - half) + half).astype(np.uint8).reshape(-1)
    cblob = pack_codes(codes, bits) if enc == 0 else codes.tobytes() if enc == 1 else _rans_encode(codes, bits)
    return u32s(R, C, bits, group, G.size, enc, rb, seed, sgroup) + sblob + cblob


def decode_rtnx(b):
    R, C, bits, group, N, enc, rb, seed, sgroup = struct.unpack('<9I', b[:36])
    ns, off = N // group, 36
    if sgroup:
        s, used = _dq_decode(b[off:], ns, sgroup)
        off += used
    else:
        s = np.frombuffer(b[off:off + 2 * ns], '<f2').astype(np.float32)
        off += 2 * ns
    if enc == 0:
        codes = unpack_codes(b[off:], bits, N)
    elif enc == 1:
        codes = np.frombuffer(b[off:off + N], np.uint8)
    else:
        codes = _rans_decode(b[off:], bits, N)
    half = (2 ** bits - 1) // 2
    X = ((codes.astype(np.float32) - half).reshape(ns, group) * s[:, None]).reshape(-1)[:R * C].reshape(R, C)
    return unrotate(X, rb, seed) if rb else X


# ----------------------------------------------------------------- atlas
_SPEC_KEYS = {'rtn': ('bits', 'group', 'clip'),
              'rtnx': ('bits', 'group', 'clip', 'enc', 'rot', 'seed', 'sgroup'),
              'raw': ('dtype', 'shuffle')}


def encode_tensor(W, spec):
    k = spec['kind']
    src = W.T if spec.get('transpose') else W
    args = {key: spec[key] for key in _SPEC_KEYS[k] if key in spec}
    return {'rtn': encode_rtn, 'rtnx': encode_rtnx, 'raw': encode_raw}[k](src, **args)


def build(tensors, plan, meta=None):
    """tensors: iterable of (name, ndarray); plan: name -> spec dict.
    spec kinds: {'kind':'raw', dtype, shuffle} | {'kind':'rtn'|'rtnx', bits, group, ...,
    'transpose': bool} | {'kind':'alias', 'of': name, 'transpose': bool}
    -> (payload, manifest dict)"""
    tensors = sorted(tensors, key=lambda t: plan[t[0]]['kind'] == 'alias')
    parts, entries, index = [], [], {}
    for name, W in tensors:
        spec = plan[name]
        if spec['kind'] == 'alias':
            blob = u32s(index[spec['of']], int(spec.get('transpose', False)))
        else:
            blob = encode_tensor(np.asarray(W, np.float32), spec)
        index[name] = len(entries)
        entries.append({'name': name, 'shape': list(np.shape(W)), **spec,
                        'bytes': len(blob), 'sha256': sha256(blob)})
        parts.append(u32s(len(blob)) + blob)
    manifest = {'format': FORMAT, 'version': VERSION, **(meta or {}), 'tensors': entries}
    return b''.join(parts), manifest


def save(path, tensors, plan, meta=None):
    payload, manifest = build(tensors, plan, meta)
    png_write(path, payload, manifest)
    return manifest


def load(path, manifest=None, verify=True):
    """-> (dict name -> float32 ndarray, manifest). `manifest` (dict or JSON path) is only
    needed for v1 files without an embedded manifest."""
    payload, embedded = png_read(path)
    if manifest is None:
        manifest = embedded
    elif isinstance(manifest, str):
        manifest = json.load(open(manifest))
    if manifest is None:
        raise ValueError(f'{path} has no embedded manifest; pass the JSON manifest of this v1 file')
    entries = manifest['tensors']
    parts = split_tensors(payload)
    if len(parts) != len(entries):
        raise ValueError(f'tensor count mismatch: {len(parts)} blobs vs {len(entries)} manifest entries')
    out, order = {}, []
    for e, blob in zip(entries, parts):
        if verify and sha256(blob) != e['sha256']:
            raise ValueError(f'sha256 mismatch for {e["name"]}')
        k = e['kind']
        if k == 'alias':
            src, tr = struct.unpack('<II', blob)
            a = out[order[src]]
            a = a.T if tr else a
        else:
            a = {'rtn': decode_rtn, 'rtnx': decode_rtnx, 'raw': decode_raw}[k](blob)
            if e.get('transpose'):
                a = a.T
        out[e['name']] = np.array(a.reshape(e['shape']), np.float32, order='C')   # keeps 0-d shape
        order.append(e['name'])
    return out, manifest
