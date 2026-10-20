"""Compare a model against its PNG versions: perplexity, KL(original || png) and top-1 agreement.

  python examples/eval.py MODEL a.png b.png ... [--text corpus.txt] [--tokens 16384] [--ctx 512]

Without --text, wikitext-2-raw test split is used (needs `datasets`).
KL / top-1 are computed on the first --kl-windows windows (logits of the original are kept in RAM).
"""
import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pngweights import hf


def corpus(tok, path, n):
    if path:
        text = open(path, encoding='utf-8').read()
    else:
        from datasets import load_dataset
        text = '\n\n'.join(load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')['text'])
    return tok(text[: n * 8], return_tensors='pt').input_ids[0, :n]


@torch.no_grad()
def run(model, ids, ctx, keep):
    dev = next(model.parameters()).device
    nll, kept = [], []
    for i in range(0, len(ids) - ctx, ctx):
        x = ids[i:i + ctx][None].to(dev)
        lp = torch.log_softmax(model(x).logits[0].float(), -1)
        nll.append(-lp[:-1].gather(1, x[0, 1:, None]).mean().item())
        if len(kept) < keep:
            kept.append(lp.cpu())
    return float(np.exp(np.mean(nll))), kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model'); ap.add_argument('pngs', nargs='+')
    ap.add_argument('--text'); ap.add_argument('--tokens', type=int, default=16384)
    ap.add_argument('--ctx', type=int, default=512); ap.add_argument('--kl-windows', type=int, default=8)
    ap.add_argument('--out', default='eval_results.json')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    tok = AutoTokenizer.from_pretrained(a.model)
    ids = corpus(tok, a.text, a.tokens)
    base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval().to(dev)
    bppl, blp = run(base, ids, a.ctx, a.kl_windows)
    del base
    R = {'model': a.model, 'tokens': int(len(ids)), 'ctx': a.ctx, 'base_ppl': round(bppl, 3), 'variants': {}}
    print(f'base ppl {bppl:.3f}')
    for p in a.pngs:
        m = hf.load(p, device=dev)
        ppl, lp = run(m, ids, a.ctx, a.kl_windows)
        kl = np.mean([(b.exp() * (b - q)).sum(-1).mean().item() for b, q in zip(blp, lp)])
        t1 = np.mean([(b.argmax(-1) == q.argmax(-1)).float().mean().item() for b, q in zip(blp, lp)])
        R['variants'][os.path.basename(p)] = r = {
            'mb': round(os.path.getsize(p) / 1e6, 1), 'ppl': round(ppl, 3), 'ppl_ratio': round(ppl / bppl, 4),
            'kl': round(float(kl), 4), 'top1': round(float(t1), 4)}
        print(os.path.basename(p), r)
        del m
    json.dump(R, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
