"""Hugging Face / PyTorch helpers: pack a transformers model into a PNG, load it back.

Loading builds the model from its config only (config.json, a few KB) and fills every
parameter from the PNG, so the original fp32/fp16 checkpoint is not needed.
"""
import numpy as np

# presets for "big" weight matrices (linear layers and MoE expert tensors)
PRESETS = {
    'q4':       dict(kind='rtnx', bits=4, group=32, clip=True, enc=0, rot=0, sgroup=16),   # best size/quality
    'q4-fast':  dict(kind='rtn',  bits=4, group=64, clip=True),                            # v1-compatible
    'q4-rot':   dict(kind='rtnx', bits=4, group=32, clip=True, enc=0, rot=-1, sgroup=16),  # dense models w/ outliers
    'q3':       dict(kind='rtnx', bits=3, group=32, clip=True, enc=0, rot=-1, sgroup=16),
    'q8':       dict(kind='rtn',  bits=8, group=64, clip=False),
}
EMBED_SPEC = dict(kind='rtn', bits=8, group=64, clip=True)        # embeddings / lm_head: q4 breaks them
OTHER_SPEC = dict(kind='raw', dtype='fp16', shuffle=True)          # norms, biases, routers, small tensors


def make_plan(model, preset='q4', min_numel=4096, embed_spec=EMBED_SPEC, other_spec=OTHER_SPEC):
    """Decide a codec for every parameter of a transformers model."""
    import torch.nn as nn
    big = PRESETS[preset] if isinstance(preset, str) else preset
    params = dict(model.named_parameters(remove_duplicate=False))
    ids = {}
    plan = {}
    # GPT-2 style Conv1D stores weights as [in, out]: quantize along the input dim -> transpose
    conv1d = {f'{n}.weight' for n, m in model.named_modules() if type(m).__name__ == 'Conv1D'}
    emb = {f'{n}.weight' for n, m in model.named_modules() if isinstance(m, nn.Embedding)}
    out_emb = model.get_output_embeddings() if hasattr(model, 'get_output_embeddings') else None
    out_name = next((n for n, p in params.items() if out_emb is not None and p is out_emb.weight), None)
    for n, p in params.items():
        key = (p.data_ptr(), tuple(p.shape))
        if key in ids:                                    # tied / shared parameter
            plan[n] = {'kind': 'alias', 'of': ids[key]}
            continue
        ids[key] = n
        if n in emb or n == out_name:
            plan[n] = dict(embed_spec)
        elif p.dim() >= 2 and p.numel() >= min_numel:
            plan[n] = dict(big, transpose=n in conv1d)
            if big['kind'] == 'rtnx':
                plan[n]['seed'] = 1234
        else:
            plan[n] = dict(other_spec)
    return plan


def pack(model_or_id, path, preset='q4', meta=None, **plan_kw):
    """Quantize a transformers model (instance, hub id or local dir) into one PNG file."""
    from . import codec
    model, model_id = _resolve(model_or_id)
    plan = make_plan(model, preset, **plan_kw)
    tensors = [(n, p.detach().float().cpu().numpy())
               for n, p in model.named_parameters(remove_duplicate=False)]
    info = {'model': model_id, 'preset': preset if isinstance(preset, str) else 'custom',
            'config': model.config.to_dict()}
    return codec.save(path, tensors, plan, {**info, **(meta or {})})


def load(path, model_id=None, device='cpu', dtype=None):
    """Rebuild a transformers model from a PNG atlas. The architecture comes from the
    config stored in the PNG (or from `model_id` if given)."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    from . import codec
    weights, manifest = codec.load(path)
    if model_id is not None:
        cfg = AutoConfig.from_pretrained(model_id)
    else:
        c = dict(manifest['config'])
        cfg = AutoConfig.for_model(c.pop('model_type'), **c)
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32)
    sd = {k: torch.from_numpy(v) for k, v in weights.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        raise ValueError(f'unexpected tensors in atlas: {unexpected[:5]}')
    missing = [k for k in missing if not k.endswith('inv_freq')]
    if missing:
        raise ValueError(f'tensors missing from atlas: {missing[:5]}')
    model = model.eval().to(device)
    return model.to(dtype) if dtype is not None else model


def _resolve(model_or_id):
    if isinstance(model_or_id, str):
        import torch
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(model_or_id, dtype=torch.float32)
        return m.eval(), model_or_id
    return model_or_id, getattr(model_or_id.config, '_name_or_path', '') or 'unknown'
