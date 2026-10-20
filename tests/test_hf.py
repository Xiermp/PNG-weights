"""End-to-end with tiny random transformers models (no download). Skipped without torch."""
import pytest

torch = pytest.importorskip('torch')
transformers = pytest.importorskip('transformers')

from pngweights import hf  # noqa: E402


def _check(model, tmp_path, preset, tol):
    p = str(tmp_path / 'm.png')
    hf.pack(model, p, preset)
    m2 = hf.load(p)
    x = torch.randint(0, model.config.vocab_size, (1, 32))
    with torch.no_grad():
        a, b = model(x).logits, m2(x).logits
    assert (a - b).abs().max().item() < tol
    return m2


def test_gpt2_conv1d_and_tied(tmp_path):
    torch.manual_seed(0)
    cfg = transformers.GPT2Config(n_layer=2, n_embd=128, n_head=4, vocab_size=500, n_positions=64)
    model = transformers.GPT2LMHeadModel(cfg).eval()
    plan = hf.make_plan(model, 'q4')
    assert plan['lm_head.weight']['kind'] == 'alias'
    assert plan['transformer.h.0.attn.c_attn.weight']['transpose'] is True
    _check(model, tmp_path, 'q4', 0.2)


@pytest.mark.parametrize('preset,tol', [('q4', 0.3), ('q4-fast', 0.3), ('q3', 0.6)])
def test_moe(tmp_path, preset, tol):
    torch.manual_seed(0)
    cfg = transformers.Qwen3MoeConfig(vocab_size=500, hidden_size=128, intermediate_size=256,
                                      moe_intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                                      num_key_value_heads=2, num_experts=8, num_experts_per_tok=2,
                                      tie_word_embeddings=False)
    model = transformers.AutoModelForCausalLM.from_config(cfg).eval()
    _check(model, tmp_path, preset, tol)
