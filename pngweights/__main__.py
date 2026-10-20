"""python -m pngweights <command> ...

  pack     MODEL OUT.png [--preset q4|q4-fast|q4-rot|q3|q8]   quantize a HF model into a PNG
  info     FILE.png                                           size breakdown per codec
  verify   FILE.png [--manifest M.json]                       decode everything, check sha256
  generate FILE.png "prompt" [--tokenizer ID] [-n 40]         run the model straight from the PNG
  preview  FILE.png OUT.png [--tensor NAME]                   render a weight matrix as an image
"""
import argparse
import os
import sys
import time
from collections import defaultdict


def main(argv=None):
    ap = argparse.ArgumentParser(prog='pngweights', description='Neural network weights packed into a PNG.')
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('pack'); p.add_argument('model'); p.add_argument('out')
    p.add_argument('--preset', default='q4')
    p = sub.add_parser('info'); p.add_argument('file'); p.add_argument('--manifest')
    p = sub.add_parser('verify'); p.add_argument('file'); p.add_argument('--manifest')
    p = sub.add_parser('generate'); p.add_argument('file'); p.add_argument('prompt')
    p.add_argument('--tokenizer'); p.add_argument('-n', type=int, default=40)
    p.add_argument('--device', default='cuda' if _has_cuda() else 'cpu')
    p = sub.add_parser('preview'); p.add_argument('file'); p.add_argument('out')
    p.add_argument('--tensor'); p.add_argument('--manifest')
    a = ap.parse_args(argv)

    from . import codec
    if a.cmd == 'pack':
        from . import hf
        t = time.time()
        man = hf.pack(a.model, a.out, a.preset)
        n = sum(_numel(e['shape']) for e in man['tensors'] if e['kind'] != 'alias')
        mb = os.path.getsize(a.out) / 1e6
        print(f'{a.out}: {mb:.1f} MB, {n / 1e6:.1f}M params, {mb * 8e6 / n:.2f} bits/param, {time.time() - t:.0f}s')

    elif a.cmd == 'info':
        _, man = codec.png_read(a.file)
        man = man or _load_json(a.manifest)
        by = defaultdict(lambda: [0, 0, 0])
        for e in man['tensors']:
            k = e['kind'] if e['kind'] in ('raw', 'alias') else f"{e['kind']} q{e['bits']} g{e['group']}"
            by[k][0] += 1; by[k][1] += _numel(e['shape']); by[k][2] += e['bytes']
        print(f"{a.file}: {os.path.getsize(a.file) / 1e6:.1f} MB on disk | model={man.get('model', '?')} "
              f"preset={man.get('preset', '?')} format v{man.get('version', 1)}")
        for k, (cnt, n, b) in sorted(by.items(), key=lambda x: -x[1][2]):
            bpw = f'{b * 8 / n:6.2f}' if k != 'alias' else '     -'
            print(f'  {k:16s} {cnt:4d} tensors {n / 1e6:9.2f}M params {b / 1e6:8.1f} MB  {bpw} bits/param')

    elif a.cmd == 'verify':
        t = time.time()
        w, man = codec.load(a.file, a.manifest)
        print(f'OK: {len(w)} tensors, {sum(x.size for x in w.values()) / 1e6:.1f}M params, '
              f'all sha256 match, {time.time() - t:.1f}s')

    elif a.cmd == 'generate':
        import torch
        from transformers import AutoTokenizer
        from . import hf
        _, man = codec.png_read(a.file)
        tok = AutoTokenizer.from_pretrained(a.tokenizer or man['model'])
        model = hf.load(a.file, device=a.device)
        ids = tok(a.prompt, return_tensors='pt').input_ids.to(a.device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=a.n, do_sample=False, pad_token_id=tok.eos_token_id)
        print(tok.decode(out[0], skip_special_tokens=True))

    elif a.cmd == 'preview':
        from . import viz
        w, _ = codec.load(a.file, a.manifest)
        name = a.tensor or max((k for k in w if w[k].ndim >= 2), key=lambda k: w[k].size)
        viz.write_png(a.out, viz.tensor_image(w[name]))
        print(f'{a.out}: {name} {list(w[name].shape)}')


def _numel(shape):
    n = 1
    for s in shape:
        n *= s
    return n


def _load_json(p):
    import json
    if not p:
        sys.exit('no embedded manifest (v1 file): pass --manifest')
    return json.load(open(p))


def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == '__main__':
    main()
