# NER model: why `en_core_web_sm` and not `en_core_web_lg`

`nlp/scripts/clean_plots.py` uses spaCy's named-entity recogniser to find character names and replace them with
`PERSON`. spaCy ships English pipelines in several sizes. We compared the two most relevant ones.

## The two models

| | `en_core_web_sm` | `en_core_web_lg` |
|---|---|---|
| Size | ~12 MB | ~400 MB (adds pretrained static word vectors) |
| Architecture | Same CNN pipeline (tagger, parser, NER) | Same pipeline, plus 300-dim word vectors as input features |
| Reported NER F-score (OntoNotes) | ~0.85 | ~0.85–0.86 |
| Used in the course | Yes (week 7) | No |

The difference is the word vectors. They help `lg` recognise rarer words as names, because it
knows from the vectors that a word looks like a first name even if it never saw it labelled.

## What we measured

Both models were run on all 1,789 plots with **the same masking rules and thresholds**:
- rule 1: spaCy PERSON entities
- rule 2: spreading a name within a plot
- rule 3: a corpus-wide name lexicon

Each was scored on the 40 hand-labelled sentences in `nlp/validation/ner_gold_sample.tsv` (95 name tokens).

| Model | Rules | Precision | Recall |
|---|---|---|---|
| sm | rule 1 only | 0.955 | 0.663 |
| sm | rules 1–3 | 0.966 | **0.884** |
| lg | rule 1 only | 1.000 | 0.747 |
| lg | rules 1–3 | 0.988 | **0.884** |

Across all plots, `sm` masks 112,062 name tokens and `lg` masks 115,121 (+3%).

*Note:* this comparison was run before the 45 sequels were removed (1,789 → 1,744 films). On the
1,744-film set, 2 of the 40 sentences belong to removed films (The Purge: Anarchy, The Purge: Election Year),
so `03_nlp.ipynb` scores `sm` on 38 sentences (86 name tokens): precision 0.963, recall 0.919. The recall rises
only because those two sentences held several of the misses. The masking itself did not change.

## Why we kept `sm`

1. **Same recall.** Recall is what matters most here: a name left in the text creates fake
   similarity between films. With the full rules, both models catch the same 84 of 95 name tokens.
2. **The models miss the same 11 words.** All of them are names that are also ordinary words,
   months or places:
   - Capable, Toast, Cheedo and Dag in *Mad Max: Fury Road*
   - Big Daddy in *The Purge: Anarchy*
   - March (Holland March) in *The Nice Guys*
   - Georgia in *Beginners*

   A bigger model does not fix this kind of error.
3. **`lg`'s advantage disappears once the rules are added.** `lg` is clearly better on its own
   (recall 0.75 vs 0.66), but rules 2 and 3 close the gap. The only remaining gain is two fewer
   false masks ("Oahu", "Odyssey"), which is within noise on 40 sentences.
4. **Practical reasons.** `sm` is the model from the course exercises, it is 30× smaller for the
   rest of the group to install, and every output in `nlp/` was already built with it.

## Limitations

- The gold sample is small (95 tokens), so a recall of 0.88 could plausibly be anywhere between
  about 0.82 and 0.94.
- We did not try the transformer model (`en_core_web_trf`). It would likely miss the same
  "common word" names.

## For the report

> Character names were masked with spaCy (`en_core_web_sm`) plus two rules: spreading a name within
> its plot, and a corpus-wide name lexicon. On 40 hand-labelled sentences this gave 0.97 precision
> and 0.88 recall. The larger `en_core_web_lg` gave the same recall (0.99 precision), so we kept the
> small model. The remaining misses are names that are also common words (Capable, Big Daddy,
> March).

## Reproducing

To compare the models again:
1. Install the large model: `py -3.12 -m spacy download en_core_web_lg`
2. Set `MODEL = "en_core_web_lg"` in `nlp/scripts/clean_plots.py` and run it.
3. Rerun cell 1.4 of `nlp/notebooks/03_nlp.ipynb`.

Both parses are cached in `nlp/cache/`. Set `MODEL` back to `en_core_web_sm` afterwards and rerun
the script, so the outputs match this choice.
