"""STEP 2 - Turn every cleaned plot into a vector, and compare films by the cosine of their vectors.

Input
    nlp/data/plots_clean.tsv   from clean_plots.py (1744 rows, films_nodes.tsv order)

Five methods. Each turns the 1744 plots into a matrix with one row per film:

    method             input column         one film becomes ...                                course   matrix shape
    tfidf              tokens               TF-IDF weights of its words (course definition)     6.2      1744 x 11,829 (sparse)
    tfidf_names_kept   tokens_names_kept    the same, but names not masked (robustness only)    6.2      1744 x 14,459 (sparse)
    lda                tokens               its mixture over K = 20 LDA topics                  6.5      1744 x 20
    minilm             text_masked          a sentence-embedding (all-MiniLM-L6-v2)             7.2      1744 x 384
    mpnet              text_masked          a sentence-embedding (all-mpnet-base-v2)            7.2      1744 x 768

Every row is scaled to length 1. Then the cosine similarity of films i and j is simply the dot product of
rows i and j, and the full film-film similarity matrix is S = X @ X.T, shape 1744 x 1744.

Output
    nlp/representations/<method>.npz      the matrix X of each method (rows = films)
    nlp/representations/node_ids.txt      the film of every row (1744 lines)
    nlp/representations/<method>_terms.txt  the word of every TF-IDF column
    nlp/representations/lda_topics.tsv    the top words of every LDA topic
    nlp/similarity/sim_<name>.npy         S for each method, 1744 x 1744 float32; the files shared with
                                          Person A and C: sim_tfidf, sim_embed (= mpnet), sim_minilm, sim_lda

How other code uses the results
    S = np.load("nlp/similarity/sim_embed.npy")      # S[i, j] = plot similarity of films i and j, S[i, i] = 1
    # or, from Python code that imports this file:
    S = similarity("mpnet")

Usage (from the repository root):
    python nlp/scripts/build_similarity.py               # all methods (~25 min, mostly the mpnet encoder)
    python nlp/scripts/build_similarity.py --only lda    # rebuild one method (TF-IDF and LDA take seconds)
    python nlp/scripts/build_similarity.py --only lda --lda-k 15
"""
import argparse
import re
import time

import numpy as np
import pandas as pd
from scipy import sparse
import scipy.sparse.linalg  # noqa: F401  (makes sparse.linalg.norm available)

import paths

# ==================================================================================================
# SETTINGS
# ==================================================================================================
METHODS = ["tfidf", "tfidf_names_kept", "lda", "minilm", "mpnet"]

# The files promised to Person A and C in the task division: method -> similarity/<name>.npy
SHARED = {"tfidf": "sim_tfidf", "mpnet": "sim_embed", "minilm": "sim_minilm", "lda": "sim_lda"}

# TF-IDF: ignore words in fewer than 2 films (typos, leftover names: they cannot make two films similar
# anyway) and words in more than half of all films (too common to tell films apart).
TFIDF_MIN_DF, TFIDF_MAX_DF = 2, 0.5

# LDA: the vocabulary filter of exercise 6.5, K = 20 topics (coherence is flat for K = 5-20, see
# notebooks/03b_nlp_analyses.ipynb section 3), a fixed seed, and 50 passes over the data.
LDA_MIN_DF, LDA_MAX_DF = 5, 0.5
LDA_K, LDA_SEED, LDA_ITER = 20, 0, 50

# Sentence encoders (Hugging Face model names)
ENCODERS = {"minilm": "sentence-transformers/all-MiniLM-L6-v2",
            "mpnet": "sentence-transformers/all-mpnet-base-v2"}
# An encoder reads at most 256 (MiniLM) or 384 (mpnet) word pieces and silently cuts off the rest.
# A plot has ~630 words, so it is split into passages of at most 180 words (as in 7.2), which fit.
PASSAGE_WORDS = 180
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")    # split after . ! or ? followed by whitespace


# ==================================================================================================
# LOADING (also used by the notebooks)
# ==================================================================================================

def load_clean():
    """plots_clean.tsv as a DataFrame: 1744 rows, columns node_id, text_masked, tokens, tokens_names_kept,
    n_names_masked. keep_default_na=False keeps strings like "NA" as text instead of turning them into NaN."""
    return pd.read_csv(paths.PLOTS_CLEAN, sep="\t", quoting=3, keep_default_na=False)


def node_ids():
    """The film of every row: list of 1744 node_ids (films_nodes.tsv order)."""
    return paths.NODE_IDS.read_text(encoding="utf-8").split("\n")


def load(method):
    """The matrix X of one method, shape 1744 x d, every row of length 1.

    TF-IDF is a scipy sparse matrix (most words do not occur in most plots); the others are numpy arrays.
    Returns (node_ids, X).
    """
    path = paths.representation_file(method)
    X = sparse.load_npz(path) if method.startswith("tfidf") else np.load(path)["X"]
    return node_ids(), X


def similarity(method):
    """The film-film cosine similarity matrix of one method: numpy array 1744 x 1744, float32.

    Because every row of X has length 1, cosine(i, j) = X[i] . X[j], so all pairs at once are X @ X.T.
    """
    _, X = load(method)
    S = X @ X.T                                            # (1744 x d) @ (d x 1744) = 1744 x 1744
    return np.asarray(S.todense() if sparse.issparse(S) else S, dtype=np.float32)


def load_shared(method):
    """The saved similarity matrix of one method from nlp/similarity/ (1744 x 1744 float32)."""
    return np.load(paths.similarity_file(SHARED[method]))


def l2(X):
    """Scale every row of a dense matrix to length 1. X: n x d -> n x d."""
    return X / np.linalg.norm(X, axis=1, keepdims=True)


# ==================================================================================================
# METHOD 1: TF-IDF (exercise 6.2)
# ==================================================================================================

def build_tfidf(texts):
    """TF-IDF exactly as defined on the course page: tf = count / |d|, idf = ln(N / df).

        count   how often the word occurs in the plot
        |d|     number of tokens in the plot
        N       number of plots (1744)
        df      number of plots that contain the word

    A word gets a high weight if it is frequent in this plot but rare in the others ("heist", "zombie").

    scikit-learn's TfidfVectorizer always computes idf = ln(N / df) + 1, so the +1 is subtracted by hand,
    as in 6.2. Afterwards every row is scaled to length 1. That does not change any cosine, and it makes
    the division by |d| irrelevant (it only rescales a row).

    texts: 1744 strings of space-separated tokens (already lemmatised and filtered by clean_plots.py)
    Returns X (sparse 1744 x V, V = vocabulary size) and info (dict with the vocabulary).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(analyzer=str.split,              # tokens are ready: just split on spaces
                          min_df=TFIDF_MIN_DF, max_df=TFIDF_MAX_DF,
                          smooth_idf=False,                # idf = ln(N/df) + 1 instead of ln((1+N)/(1+df)) + 1
                          norm=None)                       # no row scaling yet
    vec.fit(texts)
    vec.idf_ = vec.idf_ - 1                                # ln(N/df) + 1  ->  ln(N/df)
    X = vec.transform(texts)                               # sparse 1744 x V: count x idf
    row_length = sparse.linalg.norm(X, axis=1)             # 1744 values
    X = sparse.diags(1 / row_length) @ X                   # divide row i by its length -> rows of length 1
    return X.astype(np.float32).tocsr(), {"vocabulary": len(vec.vocabulary_),
                                          "terms": vec.get_feature_names_out()}


# ==================================================================================================
# METHOD 2: LDA topic model (exercise 6.5)
# ==================================================================================================

def fit_lda(texts, k, seed):
    """Fit one LDA topic model with the project settings.

    LDA assumes each plot is a mixture of K topics, and each topic is a probability distribution over
    words. It learns both from the word counts alone.

    texts: 1744 token strings. k: number of topics. seed: random seed.
    Returns a dict:
        counts      sparse 1744 x V   word counts per plot (V = vocabulary after the df filter, ~6,800)
        theta       1744 x K          each plot's mixture over topics; every row sums to 1
        phi         K x V             each topic's distribution over words; every row sums to 1
        vocab       V words           the column names of counts and phi
        perplexity  float             how surprised the model is by the data (lower = better fit)
    Also used by the notebook for refits with another K or seed.
    """
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer
    vec = CountVectorizer(analyzer=str.split, min_df=LDA_MIN_DF, max_df=LDA_MAX_DF)
    counts = vec.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=k, random_state=seed, max_iter=LDA_ITER,
                                    learning_method="batch", n_jobs=-1).fit(counts)
    # lda.components_ is K x V of unnormalised weights; dividing each row by its sum gives probabilities
    phi = lda.components_ / lda.components_.sum(axis=1, keepdims=True)
    return {"counts": counts, "theta": lda.transform(counts), "phi": phi,
            "vocab": vec.get_feature_names_out(), "perplexity": lda.perplexity(counts)}


def build_lda(texts, k, seed):
    """LDA representation: each film's topic mixture theta (1744 x K), rows scaled to length 1, so the
    cosine of two films compares their topic mixtures. Also returns the raw theta and a topic table."""
    m = fit_lda(texts, k, seed)
    theta, phi, vocab = m["theta"], m["phi"], m["vocab"]
    topics = pd.DataFrame({
        "topic": range(k),
        "share": theta.mean(axis=0).round(4),                                    # average weight over all films
        "top_words": [" ".join(vocab[np.argsort(-phi[t])[:15]]) for t in range(k)],   # 15 most probable words
    })
    return l2(theta).astype(np.float32), {"theta": theta.astype(np.float32), "topics": topics,
                                          "vocabulary": len(vocab), "perplexity": m["perplexity"]}


# ==================================================================================================
# METHODS 3-4: sentence encoders (exercise 7.2)
# ==================================================================================================

def split_into_passages(text, max_words=PASSAGE_WORDS):
    """Group whole sentences into passages of at most max_words words (7.2).

    A sentence is never cut in half; a single sentence longer than max_words becomes its own passage.
    Returns a list of strings (a plot of ~630 words gives ~4 passages).
    """
    passages, current, n_words = [], [], 0
    for sentence in SENTENCE_END.split(text):
        words = len(sentence.split())
        if current and n_words + words > max_words:        # adding this sentence would be too long:
            passages.append(" ".join(current))              # close the current passage
            current, n_words = [], 0
        current.append(sentence)
        n_words += words
    if current:
        passages.append(" ".join(current))                  # the last, unfinished passage
    return passages


def build_encoder(texts, model_name):
    """Sentence-embedding representation of every plot.

    A sentence encoder is a neural network that maps a passage of text to one vector, such that passages
    with similar meaning get similar vectors, even without shared words.

    1. split every plot into passages            1744 plots -> ~7,000 passages
    2. encode every passage                      ~7,200 x d  (d = 384 for MiniLM, 768 for mpnet)
    3. average the passages of each film         1744 x d
    4. scale every row to length 1               1744 x d

    texts: 1744 masked plot texts. Returns X (1744 x d) and info (passage counts).
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device="cpu")

    per_film = [split_into_passages(t) for t in texts]     # list of 1744 lists of passages
    passages = [p for film in per_film for p in film]      # one flat list, so all are encoded in one call
    too_long = sum(len(model.tokenizer(p)["input_ids"]) > model.max_seq_length for p in passages)

    vectors = model.encode(passages, batch_size=32, normalize_embeddings=True, show_progress_bar=True)

    # Film f owns the passages bounds[f] .. bounds[f+1]-1 of the flat list
    bounds = np.cumsum([0] + [len(f) for f in per_film])   # 1790 values
    X = np.stack([vectors[a:b].mean(axis=0) for a, b in zip(bounds[:-1], bounds[1:])])   # 1744 x d
    return l2(X).astype(np.float32), {"passages": len(passages), "max_seq_length": model.max_seq_length,
                                      "passages_truncated": too_long}


# ==================================================================================================
# MAIN
# ==================================================================================================

def write_shared():
    """Write similarity/sim_<name>.npy (1744 x 1744 float32) for every method in SHARED that has been built."""
    paths.SIMILARITY.mkdir(parents=True, exist_ok=True)
    for method, name in SHARED.items():
        if paths.representation_file(method).exists():
            np.save(paths.similarity_file(name), similarity(method))
            print(f"wrote nlp/similarity/{name}.npy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=METHODS, default=METHODS, help="methods to (re)build")
    ap.add_argument("--lda-k", type=int, default=LDA_K, help="number of LDA topics")
    ap.add_argument("--lda-seed", type=int, default=LDA_SEED)
    args = ap.parse_args()

    clean = load_clean()
    paths.REPRESENTATIONS.mkdir(parents=True, exist_ok=True)
    paths.NODE_IDS.write_text("\n".join(clean.node_id), encoding="utf-8")

    for method in args.only:
        t0 = time.time()
        if method.startswith("tfidf"):
            column = "tokens_names_kept" if method == "tfidf_names_kept" else "tokens"
            X, info = build_tfidf(clean[column])
            sparse.save_npz(paths.representation_file(method), X)
            paths.terms_file(method).write_text("\n".join(info.pop("terms")), encoding="utf-8")
        elif method == "lda":
            X, info = build_lda(clean.tokens, args.lda_k, args.lda_seed)
            np.savez_compressed(paths.representation_file(method), X=X, theta=info.pop("theta"))
            info.pop("topics").to_csv(paths.LDA_TOPICS, sep="\t", index=False)
            info["k"] = args.lda_k
        else:
            X, info = build_encoder(clean.text_masked.tolist(), ENCODERS[method])
            np.savez_compressed(paths.representation_file(method), X=X)

        summary = ", ".join(f"{k} {v:,.0f}" if isinstance(v, (int, float)) else f"{k} {v}" for k, v in info.items())
        print(f"{method:<17} {X.shape[0]} x {X.shape[1]:<6} {time.time() - t0:5.0f}s   {summary}")

    write_shared()


if __name__ == "__main__":
    main()
