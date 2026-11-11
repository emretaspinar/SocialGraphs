"""Distinctive words of groups of films: weighted log-odds with an informative Dirichlet prior (Monroe et al. 2008).

Use it for any grouping of the films: communities, genres, decades. Each group is compared with all other films
("one against the rest"). A word is distinctive for a group if the group uses it much more often than the rest,
by more than chance would explain.

Why not just compare word frequencies? A word used twice in a small group gets a huge frequency ratio by luck.
The log-odds with a prior shrinks rare words towards the corpus average, and dividing by the standard error turns
the difference into a z-score: large only when the difference is big AND backed by many occurrences.

The method (Monroe, Colaresi & Quinn 2008, "Fightin' Words", Political Analysis 16(4), section 3.5):
    y_w      count of word w in the group           n      total words in the group
    r_w      count of word w in the rest            m      total words in the rest
    a_w      prior count of w: its count in the whole corpus, scaled so the prior adds up to PRIOR_SIZE words
    a0       sum of all a_w (= PRIOR_SIZE)

    delta_w  = log((y_w + a_w) / (n + a0 - y_w - a_w))  -  log((r_w + a_w) / (m + a0 - r_w - a_w))
    var_w    = 1 / (y_w + a_w)  +  1 / (r_w + a_w)
    z_w      = delta_w / sqrt(var_w)                    > 1.96: used significantly more in the group

Input text: the `tokens` column of nlp/data/plots_clean.tsv (lemmatised, stopwords and character names removed),
so names do not show up as "distinctive". A word must also occur in at least MIN_GROUP_FILMS films of the group,
so a single film's leftover name cannot top the list.

In Python (e.g. from a notebook):
    import sys; sys.path.insert(0, "../nlp/scripts")     # path to this folder
    from distinctive_words import distinctive_words
    groups = dict(zip(partition.node_id, partition.community))      # node_id -> group label
    table = distinctive_words(groups, top_n=10)

From the command line, with any TSV that has a node_id column and a group column:
    python nlp/scripts/distinctive_words.py communities/partitions.tsv --column community \\
        --where scheme=default --where seed=0
"""
import argparse

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

import paths

MIN_DF = 5               # ignore words that occur in fewer than 5 films overall (as for LDA)
MIN_GROUP_FILMS = 3      # a distinctive word must occur in at least 3 films of the group
PRIOR_SIZE = 10_000      # strength of the prior, in words; larger = more shrinkage towards the corpus average


def word_counts(min_df=MIN_DF):
    """Word counts per film from plots_clean.tsv.

    Returns (node_ids, counts, vocab):
        node_ids   list of 1744 node_ids (films_nodes.tsv order), the rows of counts
        counts     sparse 1744 x V matrix, counts[i, w] = how often word w occurs in film i's plot
        vocab      array of the V words, the columns of counts
    """
    clean = pd.read_csv(paths.PLOTS_CLEAN, sep="\t", quoting=3, keep_default_na=False)
    vec = CountVectorizer(analyzer=str.split, min_df=min_df)      # tokens are already space-separated words
    counts = vec.fit_transform(clean.tokens)
    return list(clean.node_id), counts.tocsr(), vec.get_feature_names_out()


def log_odds_z(in_group, rest, prior):
    """The z-score of every word, for one group against the rest.

    in_group, rest, prior: arrays of length V (word counts in the group, in the rest, prior counts).
    Returns an array of length V; see the module docstring for the formula.
    """
    n, m, a0 = in_group.sum(), rest.sum(), prior.sum()
    delta = (np.log((in_group + prior) / (n + a0 - in_group - prior))
             - np.log((rest + prior) / (m + a0 - rest - prior)))
    variance = 1 / (in_group + prior) + 1 / (rest + prior)
    return delta / np.sqrt(variance)


def distinctive_words(groups, top_n=10, min_group_films=MIN_GROUP_FILMS, prior_size=PRIOR_SIZE, counts=None):
    """The top_n most distinctive words of every group.

    groups: dict or Series, node_id -> group label. Films not in it are left out entirely (they are neither in
            a group nor in "the rest").
    counts: optional output of word_counts(), to avoid recounting when calling this function many times.

    Returns a DataFrame with one row per (group, word), sorted by group and rank:
        group, rank, word, z, films_in_group (films of the group containing the word), group_size
    """
    node_ids, X, vocab = counts if counts is not None else word_counts()
    groups = pd.Series(groups)
    row_of = {node: i for i, node in enumerate(node_ids)}
    unknown = set(groups.index) - set(row_of)
    if unknown:
        raise ValueError(f"{len(unknown)} node_ids not in plots_clean.tsv, e.g. {sorted(unknown)[:3]}")

    rows = np.array([row_of[node] for node in groups.index])        # row of every grouped film
    labels = groups.to_numpy()
    X = X[rows]                                                     # only the grouped films: n_films x V
    total = np.asarray(X.sum(axis=0)).ravel()                       # V corpus counts
    prior = total / total.sum() * prior_size                        # V prior counts, adding up to prior_size
    present = (X > 0).astype(np.int32)                              # 1 where a film contains a word

    out = []
    for label in sorted(set(labels), key=str):
        mask = labels == label
        in_group = np.asarray(X[mask].sum(axis=0)).ravel()          # V counts inside the group
        films_with_word = np.asarray(present[mask].sum(axis=0)).ravel()
        z = log_odds_z(in_group, total - in_group, prior)
        z[films_with_word < min_group_films] = -np.inf               # too few films of the group use it
        for rank, w in enumerate(np.argsort(-z)[:top_n], start=1):
            if np.isfinite(z[w]):
                out.append({"group": label, "rank": rank, "word": vocab[w], "z": round(float(z[w]), 2),
                            "films_in_group": int(films_with_word[w]), "group_size": int(mask.sum())})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("groups_file", help="TSV with a node_id column and a group column")
    ap.add_argument("--column", required=True, help="the group column, e.g. community")
    ap.add_argument("--where", action="append", default=[], metavar="COL=VALUE",
                    help="keep only rows where COL equals VALUE (repeatable), e.g. --where seed=0")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--out", help="write the table to this TSV instead of printing it")
    args = ap.parse_args()

    table = pd.read_csv(args.groups_file, sep="\t", comment="#", dtype=str, keep_default_na=False)
    for condition in args.where:
        col, value = condition.split("=", 1)
        table = table[table[col] == value]
    result = distinctive_words(dict(zip(table.node_id, table[args.column])), top_n=args.top)
    if args.out:
        result.to_csv(args.out, sep="\t", index=False)
        print(f"wrote {args.out}")
    else:
        for group, words in result.groupby("group", sort=False):
            print(f"{group} ({words.group_size.iloc[0]} films): " + ", ".join(words.word))


if __name__ == "__main__":
    main()
