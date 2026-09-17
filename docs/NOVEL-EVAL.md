# Novel prediction accuracy

Honest generalization numbers: production base models (Gutenberg training)
measured on unseen novels neither tier trained on. Recorded 2026-09-17,
Qwen2.5-1.5B base Q4_K_M, gate −1.5.

Corpora (`corpora/novel_en.txt`, `corpora/novel_de.txt`, committed):
- EN: *Pride and Prejudice*, Jane Austen (Gutenberg 1342) — 128k tokens.
- DE: *Die Leiden des jungen Werther*, Goethe (projekt-gutenberg.org,
  boilerplate stripped) — 35k tokens.

## Word tier (100-sentence samples, `eval_novel`)

| corpus | top-1 | top-3 | savings |
|---|---|---|---|
| EN novel | 0.255 | 0.389 | 0.142 |
| DE novel | 0.196 | 0.329 | 0.096 |

Compare train-text numbers (top-1 0.897, savings 0.722): the gap is
memorization vs generalization, as expected. ~0.2 top-1 on unseen literary
text from pure n-grams is a working baseline, not a product.

## Sentence tier (same runs)

| corpus | shown / triggers | acceptance | wrong rate | TTFT p50 |
|---|---|---|---|---|
| EN novel | 1 / 100 | 0.000 | 1.000 | 172 ms |
| DE novel | 10 / 100 | 0.030 | 0.700 | 2288 ms |

The gate (−1.5, tuned on short simple sentences) fires on almost nothing
literary — mean logprob does not transfer across domains. Wrong rate on
what passes is high. Takeaway, not failure: ghost-text UX tolerates
 silence better than noise, but the gate needs per-domain calibration
 (open: better confidence features, M5 style priors help here).

## Personal tier, temporal on novels (stub backend, word-only)

Learn first 75%, test held-out 300 sentences:

| corpus | held-out words | baseline | personal | delta |
|---|---|---|---|---|
| EN novel | 2289 | 7507 (0.236) | 7110 (0.276) | **+397** |
| DE novel | 5461 | 9410 (0.645) | 8598 (0.676) | **+812** |

Personal n-grams generalize strongly on realistic text (author vocabulary
and refrains recur across halves) — an order of magnitude more than on the
tiny sample corpus (+6), where cross-boundary repetition barely exists.

## Reproduce

```sh
cargo run -p predict-eval --example eval_novel -- <model.gguf> corpora/novel_en.txt 100
cargo run -p predict-eval --example eval_temporal -- none -1.5 0.7 corpora/novel_en.txt 300
```
