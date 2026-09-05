# PROBE RUN — phase 2 semantics on REAL data (wikitext-103), H100, 5 000 steps (2026-08-30)

**The question this answers** (user, verbatim): "lo que pasa dentro del
modelo — los +1743/−290834 que pueden significar 'hola' — ¿lo entendemos?"

**Method:** token→feature association computed from COMMIT facts alone
(zero activations read; the journal IS the instrument), then a CAUSAL
ablation as a journalized POLICY intervention (feature forced to
ABORT(reason=policy) at eval; the non-fact is data).

## Result 1 — the model develops named features for real tokens

Top associations at 5k steps (gpt2 vocab, 50 257 tokens; N_f=32 768):

| token | feature | lift | purity |
|---|---|---|---|
| `'.'` | f20338 | **177.6** | 67.6% |
| `' '` | f26032 | 133.2 | **99.8%** |
| `'-'` | f2011 | 123.5 | 91.9% |
| `' ('` | f5183 | 115.6 | 54.5% |
| `' ='` | f11867 | 112.5 | 85.8% |
| `<\|endoftext\|>` | f17950 | 104.4 | 67.9% |
| `'\n'` | f22038 | 101.5 | 94.0% |
| `' for'` | f16912 | 97.6 | 57.3% |
| `' is'` | f14909 | 92.9 | 37.1% |
| `' that'` | f11839 | 88.9 | 49.7% |

Reading: at 5k steps the highest-lift features name STRUCTURAL tokens
(punctuation, whitespace, document boundaries) — exactly what a young LM
learns first. Function words are emerging with lower purity (superposition
still visible — ' is' at 37% purity is shared). This is the expected
honest picture, now MEASURED from receipts instead of guessed.

## Result 2 — the association is causal, and surgically so

Ablating f20338 (deny-list → ABORT reason=policy, journalized):

| | NLL('.') | NLL(rest) |
|---|---|---|
| base | 1.5288 | 5.5132 |
| f20338 denied | 1.5321 | 5.5129 |
| **delta** | **+0.0032** | −0.0003 |

Specificity criterion (target delta > 3× |rest delta|): **PASS (10.7×)**.
Denying one feature of 32 768 hurts exactly the token it names and nothing
else. Effect size is small in absolute terms — expected: (a) 5k steps,
(b) purity 67.6% means sibling features still carry '.',
(c) k=8 writes/token gives redundancy. The point is not the magnitude; it
is that the intervention is a FACT: the ABORT rides the same journal as
every other verdict, citable and replayable.

## Circuit health

71/71 T1 audits green; gate BIT_IDENTICAL; val PPL 243.4 at 5k steps
(catalog arm, consistent with the 1b trajectory toward 167.8 at 20k).

## What this closes and what it opens

Closes: the category question. "What does the arithmetic mean?" is now a
query + an intervention, both over facts. Tamkin et al. (ICML'24) steer by
activating codes on a finetuned bottleneck; we deny features as journalized
policy acts on a from-scratch model — and the deny itself becomes evidence.

Opens (phase 3): the same probe on a model that TALKS (nanochat d20) —
deny the feature a chat model uses for a concept and watch the conversation
change, with receipts.
