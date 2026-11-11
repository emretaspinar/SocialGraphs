"""Every file location used by the NLP part, in one place.

Scripts and notebooks import this file instead of writing paths themselves, so a folder can be moved by
changing one line here.

    nlp/
    ├── scripts/          code that builds the files below, plus distinctive_words.py
    ├── notebooks/        analysis notebooks; they only read files
    ├── data/             cleaned plot text (step 1)
    ├── representations/  one vector per film and method (step 2)
    ├── similarity/       1744 x 1744 film-film similarity matrices, shared with Person A and C
    ├── topics/           LDA topic shares per film and topic labels, for Person C
    ├── validation/       hand-labelled files (never overwritten by code)
    ├── docs/             short notes on decisions
    └── cache/            spaCy parse, rebuilt automatically, not in git

Row order: every per-film file and every matrix uses the order of network/films_nodes.tsv
(films sorted by node_id). Row i of every file is the same film.
"""
from pathlib import Path

NLP = Path(__file__).resolve().parent.parent     # .../SocialGraphs/nlp
ROOT = NLP.parent                                # .../SocialGraphs (the repository)

# ---- inputs, written by build_network.py ----------------------------------------------------------
NETWORK = ROOT / "network"
FILMS_NODES = NETWORK / "films_nodes.tsv"        # one row per film (1744 rows): node_id, name, genres, ...
FILMS_EDGES = NETWORK / "films_edges.tsv"        # one row per linked film pair (11,465 rows)
PLOTS_ZIP = NETWORK / "film_plots.zip"           # one Wikipedia plot text per film
PARTITIONS = ROOT / "communities" / "partitions.tsv"   # Person C's communities (may not exist yet)

# ---- step 1: clean_plots.py -----------------------------------------------------------------------
DATA = NLP / "data"
PLOTS_CLEAN = DATA / "plots_clean.tsv"           # 1744 rows: node_id, text_masked, tokens, tokens_names_kept, ...
PERSON_MENTIONS = DATA / "person_mentions.tsv"   # every PERSON entity spaCy found, for inspection
CACHE = NLP / "cache"                            # spaCy parse of all plots (~14 MB, ~6 min to rebuild)

# ---- step 2: build_similarity.py ------------------------------------------------------------------
REPRESENTATIONS = NLP / "representations"
NODE_IDS = REPRESENTATIONS / "node_ids.txt"      # 1744 lines: the film of every row
LDA_TOPICS = REPRESENTATIONS / "lda_topics.tsv"  # K rows: each topic's top words
SIMILARITY = NLP / "similarity"                  # sim_<name>.npy, each 1744 x 1744 float32

# ---- validation (hand labels) ---------------------------------------------------------------------
VALIDATION = NLP / "validation"
NER_GOLD = VALIDATION / "ner_gold_sample.tsv"                  # 40 sentences with hand-labelled names
PAIR_RATINGS = VALIDATION / "pair_ratings.tsv"                 # ~60 (query, candidate) film pairs rated 1-5
PAIR_PLOTS = VALIDATION / "pair_ratings_plots.md"               # short plot of every film in that sheet, for raters

# ---- topics for Person C: write_topics.py ---------------------------------------------------------
TOPICS_DIR = NLP / "topics"
TOPICS = TOPICS_DIR / "topics.tsv"               # 1744 rows: node_id, dominant_topic, topic_0 .. topic_19 (shares)
TOPIC_LABELS = TOPICS_DIR / "topic_labels.tsv"   # 20 rows: topic, label, share, top_words


def representation_file(method):
    """representations/<method>.npz, e.g. representations/mpnet.npz"""
    return REPRESENTATIONS / f"{method}.npz"


def terms_file(method):
    """representations/<method>_terms.txt: the word of every TF-IDF column"""
    return REPRESENTATIONS / f"{method}_terms.txt"


def similarity_file(name):
    """similarity/<name>.npy, e.g. similarity/sim_embed.npy"""
    return SIMILARITY / f"{name}.npy"
