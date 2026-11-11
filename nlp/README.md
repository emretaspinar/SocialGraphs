# NLP: plot text, similarity and validation (Person B)

This folder turns the Wikipedia plot of every film into numbers, so that films can be compared by *what
happens in them*. Those numbers can then be compared with the collaboration network.

## If you only need the similarity matrices (Person A and C)

```python
import numpy as np
S = np.load("nlp/similarity/sim_embed.npy")   # 1744 x 1744, S[i, j] = plot similarity of films i and j
```

Row and column `i` is the `i`-th film of `network/films_nodes.tsv`, so the order is films sorted by `node_id`.
`nlp/representations/node_ids.txt` lists the same order. The diagonal is 1.

| file | method | what "similar" means |
|---|---|---|
| `sim_tfidf.npy` | TF-IDF (course definition, 6.2) | the plots share distinctive words |
| `sim_embed.npy` | sentence encoder all-mpnet-base-v2 (7.2) | the plots mean similar things, even in different words |
| `sim_minilm.npy` | sentence encoder all-MiniLM-L6-v2 (7.2) | the same, with a smaller encoder |
| `sim_lda.npy` | LDA topic mixtures, K = 20 (6.5) | the plots mix the same topics. Unstable across seeds, use with care |

Each method uses its own scale (encoders: mostly 0.3–0.8; TF-IDF: mostly 0–0.1). So compare
**ranks**, not raw values, across methods.

## Topics and distinctive words (Person C)

```python
import pandas as pd
topics = pd.read_csv("nlp/topics/topics.tsv", sep="	")        # node_id, dominant_topic, topic_0 .. topic_19
labels = pd.read_csv("nlp/topics/topic_labels.tsv", sep="	")  # topic, label, share, coherent, top_words

import sys; sys.path.insert(0, "nlp/scripts")
from distinctive_words import distinctive_words
table = distinctive_words(dict(zip(partition.node_id, partition.community)), top_n=10)
```

`topic_t` is the share of a film's plot assigned to topic t (each row sums to 1). Topics with `coherent = False`
mix unrelated themes; do not interpret them. `distinctive_words` compares every group with all other films
(weighted log-odds, Monroe et al. 2008) and returns group, rank, word, z, films_in_group, group_size.

## Folder layout

```
nlp/
├── scripts/                     code that builds everything (run these)
│   ├── paths.py                 every file location, in one place
│   ├── clean_plots.py           step 1: clean the plots, mask character names
│   ├── build_similarity.py      step 2: vectors and similarity matrices for all methods
│   ├── write_topics.py          topic shares per film + topic labels, for Person C
│   └── distinctive_words.py     distinctive words of any grouping of films (for Person C's communities)
├── notebooks/                   analyses; they only read files
│   ├── 03_nlp.ipynb             1. cleaning + validation of the masking   2. comparing the four methods
│   └── 03b_nlp_analyses.ipynb   what the measures pick up: shared words, collaborators vs. plot neighbours,
│                                leftover names, LDA coherence and stability, plot map,
│                                rated film pairs (hand-labelled validation), topic/word files
├── data/                        step 1 output
│   ├── plots_clean.tsv          1744 rows: node_id, text_masked, tokens, tokens_names_kept, n_names_masked
│   └── person_mentions.tsv      every PERSON entity spaCy found (inspection only)
├── representations/             step 2 output: one row per film and method
│   ├── tfidf.npz, tfidf_names_kept.npz   sparse 1744 x vocabulary
│   ├── lda.npz                  1744 x 20 (topic mixtures)
│   ├── minilm.npz, mpnet.npz    1744 x 384, 1744 x 768
│   ├── node_ids.txt             the film of every row
│   ├── *_terms.txt              the word of every TF-IDF column
│   └── lda_topics.tsv           top words of every LDA topic
├── similarity/                  the shared 1744 x 1744 matrices (see above)
├── topics/                      topics.tsv (topic shares per film), topic_labels.tsv
├── validation/                  hand labels, never overwritten by code
│   ├── ner_gold_sample.tsv      40 sentences with the character names marked (done)
│   ├── pair_ratings.tsv         63 film pairs, each rater gives 1-5 (to do)
│   └── pair_ratings_plots.md    the start of every plot in that sheet, for the raters
├── docs/
│   └── ner_model_choice.md      why spaCy's small model is used
└── cache/                       spaCy parse (not in git, rebuilt automatically)
```

## How to rebuild everything

From the repository root:

```bash
python nlp/scripts/clean_plots.py        # step 1: ~6 min the first time (spaCy), seconds afterwards
python nlp/scripts/build_similarity.py   # step 2: ~25 min (the mpnet encoder on CPU); TF-IDF and LDA take seconds
python nlp/scripts/build_similarity.py --only tfidf lda   # rebuild only the fast methods
python nlp/scripts/write_topics.py       # topic files; stops if LDA changed and the labels need redoing
```

Open the notebooks from `nlp/notebooks/`. They find the scripts through `../scripts`.
`build_similarity.py` needs `sentence-transformers` (see `requirements.txt`); the notebooks do not.

## The pipeline in one picture

```
network/film_plots.zip ──clean_plots.py──▶ data/plots_clean.tsv ──build_similarity.py──▶ representations/*.npz
   (1744 raw plots)        mask names         text_masked, tokens        5 methods              (1744 x d each)
                                                                                                       │
                                                                                              X @ X.T  ▼
                                                                                  similarity/sim_*.npy (1744 x 1744)
```

## Key decisions (details in the notebooks)

- **Only PERSON names are masked.** Places and organisations are story content. The masking uses spaCy plus two
  rules. Against 40 hand-labelled sentences it reaches precision 0.97 and recall 0.88. A larger spaCy model gives
  the same recall (`docs/ner_model_choice.md`).
- **TF-IDF follows the course definition** (tf = count/|d|, idf = ln(N/df)). It ignores words in fewer than 2 films
  or more than half of them.
- **Encoders read passages of at most 180 words** (whole sentences), because a plot is longer than their input
  limit. A film's vector is the mean of its passage vectors.
- **LDA uses K = 20.** Coherence peaks at K = 15 but is flat from 5 to 20. The model is unstable across seeds, so
  its topics are better for *describing* groups of films than as a similarity measure.
- **Leftover names** that the masking misses (e.g. "chucky", "bishop") affect a few sequel pairs but barely change
  the overall results (`03b_nlp_analyses.ipynb` section 1.4).
