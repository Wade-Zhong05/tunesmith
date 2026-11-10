# Student guide: the parts you implement

Two files are skeletons. Everything else in the project already calls them, and each has
a test file that acts as an autograder. Tests are **skipped** while a function still
raises `NotImplementedError` and **pass** once it is correct.

```bash
pixi run pytest tests/test_transformer.py tests/test_sampling_filters.py -q -rs
```

Until you finish, the app and the pipeline work with the GRU composer or the baselines,
and generation falls back to plain sampling. Nothing is blocked.

## 1. `src/tunesmith/model/transformer.py` (main part)

A GPT-style decoder over the token sequence `BOS <7 control tokens> SEP <REMI music
tokens> EOS`. Suggested order:

| Step | What | Test that turns green |
|---|---|---|
| 1 | `MLP`: Linear → GELU → Linear → Dropout | (indirect) |
| 2 | `CausalSelfAttention` without a cache: Q/K/V projection, split heads, scaled dot-product with a causal mask, merge heads, output projection | `test_attention_shape` |
| 3 | `Block`: pre-LayerNorm residual block | (indirect) |
| 4 | `Composer.__init__`: embeddings, blocks, final LayerNorm, tied LM head, `self.apply(self._init_weights)` | `test_forward_shapes`, `test_weight_tying` |
| 5 | `Composer.forward`: positions, blocks, logits; raise `ValueError` beyond `max_seq_len` | `test_causality`, `test_max_seq_len_is_enforced`, `test_can_overfit_one_sequence`, `test_base_config_size` |
| 6 (optional) | KV cache: concatenate past keys/values; positions continue from the past length | `test_kv_cache_matches_full_forward` |
| 7 (optional) | RoPE (`positional: rope`) | `test_rope_option_if_implemented` |

**Shapes to keep in mind:**
- `x`: (B, T, d_model);
- `q`, `k`, `v` after the head split: (B, n_heads, T, head_dim);
- attention weights: (B, n_heads, T, T_past + T);
- cache per layer: a tuple `(k, v)`, each (B, n_heads, T_total, head_dim).

**After the tests pass:**
1. Smoke-train on CPU: `pixi run smoke --model-type transformer`.
2. Train for real: `pixi run -e cuda train-composer --config configs/composer_base.yaml`.
3. Compare against the GRU with `pixi run evaluate` (see SETUP.md §6).

Things to be ready to explain in the technical walkthrough:
- why the mask is causal;
- why the LM head is tied to the embedding;
- what attention looks at in this data (e.g. the control tokens at the start of the
  sequence);
- what the KV cache saves.

## 2. `src/tunesmith/model/sampling_filters.py`

| Function | Behaviour | Tests |
|---|---|---|
| `apply_temperature(logits, t)` | `logits / t` | `test_temperature` |
| `top_k_filter(logits, k)` | keep the k largest per row, the rest -inf | `test_top_k_keeps_k_largest_per_row` |
| `top_p_filter(logits, p)` | smallest set of tokens whose cumulative probability reaches p; the top token always survives; masked tokens stay masked | `test_top_p_smallest_nucleus`, `test_filters_keep_masked_tokens_masked` |

`sample_next` (provided) chains them; `test_sample_next_uses_filters` checks the chain.

## 3. Afterwards

- Update the "Written by the student" line in ATTRIBUTION.md, and the module docstrings
  of the two files.
- Good experiments for the report:
  - temperature × top-p grid with the music metrics;
  - Transformer vs GRU at equal parameter count;
  - a learned vs RoPE positional ablation, if you implement RoPE.
