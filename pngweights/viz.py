"""Render a weight tensor as a *viewable* picture (the atlas itself is just bytes)."""
import struct
import zlib

import numpy as np


def write_png(path, img):
    """img: uint8 array (H, W) gray or (H, W, 3) RGB."""
    img = np.ascontiguousarray(img, np.uint8)
    h, w = img.shape[:2]
    ctype = 2 if img.ndim == 3 else 0
    raw = np.concatenate([np.zeros((h, 1), np.uint8), img.reshape(h, -1)], 1).tobytes()

    def chunk(tag, data):
        return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff)
    with open(path, 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, ctype, 0, 0, 0))
                + chunk(b'IDAT', zlib.compress(raw, 6)) + chunk(b'IEND', b''))


def tensor_image(W, max_side=1024):
    """Diverging colormap: blue = negative, white = 0, red = positive (clipped at 99.5th pct)."""
    W = np.asarray(W, np.float32)
    W = W.reshape(-1, W.shape[-1]) if W.ndim > 1 else W[None]
    sy, sx = max(1, -(-W.shape[0] // max_side)), max(1, -(-W.shape[1] // max_side))
    if sy > 1 or sx > 1:                                   # mean-pool down to max_side
        H, Wd = W.shape[0] // sy * sy, W.shape[1] // sx * sx
        W = W[:H, :Wd].reshape(H // sy, sy, Wd // sx, sx).mean((1, 3))
    lim = np.percentile(np.abs(W), 99.5) + 1e-12
    x = np.clip(W / lim, -1, 1)
    r = np.where(x > 0, 1.0, 1.0 + x)
    b = np.where(x < 0, 1.0, 1.0 - x)
    g = 1.0 - np.abs(x)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)
