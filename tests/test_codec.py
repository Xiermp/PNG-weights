import json

import numpy as np
import pytest

from pngweights import codec


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.mark.parametrize('bits', [2, 3, 4, 5, 8])
def test_pack_roundtrip(rng, bits):
    c = rng.integers(0, 2 ** bits, 10007).astype(np.uint8)
    assert np.array_equal(codec.unpack_codes(codec.pack_codes(c, bits), bits, c.size), c)


def test_rotation_is_orthogonal(rng):
    X = rng.standard_normal((16, 768)).astype(np.float32)
    b = codec.hadamard_block(768)
    assert b == 256
    Y = codec.rotate(X, b, 7)
    assert np.allclose(np.linalg.norm(Y), np.linalg.norm(X), rtol=1e-5)
    assert np.allclose(codec.unrotate(Y, b, 7), X, atol=1e-5)


def test_rtnx_plain_equals_rtn(rng):
    W = (rng.standard_t(3, (256, 384)) * 0.02).astype(np.float32)
    a = codec.decode_rtn(codec.encode_rtn(W, 4, 32))
    b = codec.decode_rtnx(codec.encode_rtnx(W, 4, 32))
    assert np.array_equal(a, b)


@pytest.mark.parametrize('cfg', [dict(), dict(sgroup=16), dict(rot=-1), dict(rot=-1, sgroup=16, enc=1)])
def test_rtnx_error(rng, cfg):
    W = (rng.standard_normal((128, 512)) * 0.02).astype(np.float32)
    Wh = codec.decode_rtnx(codec.encode_rtnx(W, 4, 32, **cfg))
    rel = np.linalg.norm(Wh - W) / np.linalg.norm(W)
    assert Wh.shape == W.shape and rel < 0.12


def test_raw_roundtrip(rng):
    W = rng.standard_normal((7, 33)).astype(np.float32)
    for dt in ('fp16', 'fp32'):
        for sh in (False, True):
            out = codec.decode_raw(codec.encode_raw(W, dt, sh))
            assert np.allclose(out, W, atol=1e-3 if dt == 'fp16' else 0)


def test_atlas_roundtrip(tmp_path, rng):
    W = rng.standard_normal((64, 96)).astype(np.float32) * 0.05
    tensors = [('lm_head', W.T.copy()), ('emb', W), ('ln', np.ones(96, np.float32)),
               ('experts', rng.standard_normal((4, 32, 64)).astype(np.float32)), ('scalar', np.float32(3.0))]
    plan = {'emb': {'kind': 'rtn', 'bits': 8, 'group': 32},
            'lm_head': {'kind': 'alias', 'of': 'emb', 'transpose': True},
            'ln': {'kind': 'raw', 'dtype': 'fp16'},
            'experts': {'kind': 'rtnx', 'bits': 4, 'group': 32, 'rot': -1, 'sgroup': 16, 'seed': 1},
            'scalar': {'kind': 'raw', 'dtype': 'fp32'}}
    p = tmp_path / 'm.png'
    codec.save(str(p), tensors, plan, {'model': 'test'})
    w, man = codec.load(str(p))
    assert man['format'] == 'pngweights' and man['version'] == codec.VERSION and man['model'] == 'test'
    assert np.array_equal(w['lm_head'], w['emb'].T)
    assert w['experts'].shape == (4, 32, 64) and w['scalar'].shape == ()
    assert float(w['scalar']) == 3.0


def test_v1_file_with_external_manifest(tmp_path, rng):
    """v1 files: no embedded manifest, JSON next to the PNG."""
    W = rng.standard_normal((32, 64)).astype(np.float32)
    payload, man = codec.build([('w', W)], {'w': {'kind': 'rtn', 'bits': 4, 'group': 64}})
    p, j = tmp_path / 'old.png', tmp_path / 'old.json'
    codec.png_write(str(p), payload, manifest=None)
    j.write_text(json.dumps(man))
    with pytest.raises(ValueError):
        codec.load(str(p))
    w, _ = codec.load(str(p), str(j))
    assert w['w'].shape == (32, 64)


def test_corruption_detected(tmp_path, rng):
    payload, man = codec.build([('w', rng.standard_normal((32, 64)).astype(np.float32))],
                               {'w': {'kind': 'raw', 'dtype': 'fp16'}})
    man['tensors'][0]['sha256'] = '0' * 64
    p = tmp_path / 'bad.png'
    codec.png_write(str(p), payload, man)
    with pytest.raises(ValueError, match='sha256'):
        codec.load(str(p))
