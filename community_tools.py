import argparse
import sys
from datetime import date
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import network_tools as nt  # noqa: E402

COMM = HERE / "communities"
SIM = HERE / "nlp" / "similarity"
SCHEMES = ["default", "actors", "directors", "dir1", "dir2", "dir4", "dir10", "billing"]
N_SEEDS = 50


# --------------------------------------------------------------------------- loading
def load_similarity(kind, nodes=None):
    """B's similarity matrix nlp/similarity/sim_<kind>.npy (tfidf, embed, minilm, lda), as float64."""
    S = np.load(SIM / f"sim_{kind}.npy").astype(np.float64)
    if nodes is not None:
        assert S.shape == (len(nodes), len(nodes)), S.shape
    return S


def scheme_graph(scheme, nodes):
    """Weighted networkx graph (integer nodes 0..N-1) for a scheme; 'default' = films_edges.tsv."""
    W = nt.weight_matrix(nt.load_edges(None if scheme == "default" else scheme), nodes)
    return nx.from_numpy_array(W)


def load_partitions(path=None):
    return pd.read_csv(path or COMM / "partitions.tsv", sep="\t", comment="#")


def labels_of(table, nodes, **where):
    """Label array (row order = nodes) from a partitions/consensus table filtered by column=value."""
    t = table
    for k, v in where.items():
        t = t[t[k] == v]
    return t.set_index("node_id").loc[nodes.node_id, "community"].to_numpy()


def primary_genre(nodes, genre_counts=None):
    """One genre per film. IMDb lists genres alphabetically, so the first listed genre means nothing;
    we take the film's *rarest* genre in the dataset (the most specific label: Horror over Drama)."""
    if genre_counts is None:
        genre_counts = pd.Series([g for s in nodes.genre_set for g in s]).value_counts()
    return nodes.genre_set.map(lambda s: min(s, key=lambda g: (genre_counts[g], g))).to_numpy()


# --------------------------------------------------------------------------- louvain
def relabel_by_size(labels):
    """Relabel so that 0 is the largest community (ties broken by smallest member index)."""
    labels = np.asarray(labels)
    uniq, first, counts = np.unique(labels, return_index=True, return_counts=True)
    order = np.lexsort((first, -counts))
    mapping = np.empty(len(uniq), dtype=int)
    mapping[order] = np.arange(len(uniq))
    return mapping[np.searchsorted(uniq, labels)]


def louvain(G, seed, weight="weight", resolution=1.0):
    comms = nx.community.louvain_communities(G, weight=weight, seed=seed, resolution=resolution)
    labels = np.empty(G.number_of_nodes(), dtype=int)
    for c, members in enumerate(comms):
        labels[list(members)] = c
    return relabel_by_size(labels)


def modularity(G, labels, weight="weight"):
    comms = pd.Series(np.arange(len(labels))).groupby(labels).apply(set).tolist()
    return nx.community.modularity(G, comms, weight=weight)


def louvain_runs(G, seeds=range(N_SEEDS)):
    return np.array([louvain(G, s) for s in seeds])


def coassignment(runs):
    """Fraction of runs in which each pair of films is in the same community (N x N)."""
    n = runs.shape[1]
    D = np.zeros((n, n))
    for lab in runs:
        C = sparse.csr_matrix((np.ones(n), (np.arange(n), lab)))
        D += (C @ C.T).toarray()
    return D / len(runs)


def consensus_partition(runs, tau=0.5, max_iter=20, seed=0):
    """Consensus clustering (Lancichinetti & Fortunato 2012): build the co-assignment matrix, drop entries
    below tau, run Louvain on it again with the same number of seeds; repeat until all runs agree.
    Returns (labels, iterations)."""
    for it in range(1, max_iter + 1):
        D = coassignment(runs)
        np.fill_diagonal(D, 0)
        D[D < tau] = 0
        G = nx.from_numpy_array(D)
        runs = np.array([louvain(G, seed + 1000 * it + s) for s in range(len(runs))])
        if (runs == runs[0]).all():
            return runs[0], it
    return runs[0], max_iter


# --------------------------------------------------------------------------- stability
def pairwise_scores(runs, score=normalized_mutual_info_score):
    """Score (NMI by default) for every pair of partitions in runs (list/array of label arrays)."""
    return np.array([score(a, b) for a, b in combinations(runs, 2)])


def compare(a, b):
    return {"NMI": normalized_mutual_info_score(a, b), "AMI": adjusted_mutual_info_score(a, b),
            "ARI": adjusted_rand_score(a, b)}


# --------------------------------------------------------------------------- coherence
def _onehot(labels):
    n = len(labels)
    return sparse.csr_matrix((np.ones(n), (np.arange(n), labels)))


def within_between(S, labels):
    """Mean similarity over within-community pairs and over between-community pairs (i < j)."""
    labels = np.asarray(labels)
    n = len(labels)
    M = np.asarray(_onehot(labels).T @ S)            # M[c, i] = sum of S[j, i] over films j in community c
    sizes = np.bincount(labels)
    total = (S.sum() - np.trace(S)) / 2
    within = (M[labels, np.arange(n)].sum() - np.trace(S)) / 2
    n_within = (sizes * (sizes - 1)).sum() / 2
    n_total = len(labels) * (len(labels) - 1) / 2
    w = within / n_within
    b = (total - within) / (n_total - n_within)
    return w, b


def label_permutation_test(S, labels, n_perm=999, seed=0):
    """Permutation test of (within - between) mean similarity: shuffle community labels across films,
    which keeps every community's size. Returns dict with observed values, null mean/sd, z and p."""
    rng = np.random.default_rng(seed)
    w, b = within_between(S, labels)
    null = np.empty(n_perm)
    for k in range(n_perm):
        wp, bp = within_between(S, rng.permutation(labels))
        null[k] = wp - bp
    obs = w - b
    return {"within": w, "between": b, "diff": obs, "null_mean": null.mean(), "null_sd": null.std(ddof=1),
            "z": (obs - null.mean()) / null.std(ddof=1), "p": (1 + (null >= obs).sum()) / (n_perm + 1)}


def silhouette(S, labels, min_size=5, n_perm=0, seed=0):
    """Silhouette on cosine distance 1 - S, over films in communities of at least min_size films.
    With n_perm > 0 also returns the silhouettes of size-preserving label shuffles (the null)."""
    from sklearn.metrics import silhouette_score
    labels = np.asarray(labels)
    keep = np.isin(labels, np.flatnonzero(np.bincount(labels) >= min_size))
    D = np.clip(1 - S[np.ix_(keep, keep)], 0, None)
    np.fill_diagonal(D, 0)
    lab = labels[keep]
    obs = silhouette_score(D, lab, metric="precomputed")
    if not n_perm:
        return obs
    rng = np.random.default_rng(seed)
    null = np.array([silhouette_score(D, rng.permutation(lab), metric="precomputed") for _ in range(n_perm)])
    return obs, null


# --------------------------------------------------------------------------- genre
def genre_assortativity(G, nodes, genres=None):
    """Per genre: attribute assortativity of the binary 'film has genre g' label on the network, plus the
    observed fraction of edges joining two films that share at least one genre."""
    genres = genres or sorted({g for s in nodes.genre_set for g in s})
    H = G.copy()
    rows = []
    for g in genres:
        nx.set_node_attributes(H, {i: int(g in s) for i, s in enumerate(nodes.genre_set)}, "has")
        rows.append({"genre": g, "films": int(sum(g in s for s in nodes.genre_set)),
                     "assortativity": nx.attribute_assortativity_coefficient(H, "has")})
    return pd.DataFrame(rows)


def genre_overlap_of_edges(edges_ij, nodes):
    """Fraction of the given pairs (i, j arrays) whose films share at least one genre, and mean Jaccard."""
    gs = nodes.genre_set.to_numpy()
    jac = np.array([len(gs[a] & gs[b]) / len(gs[a] | gs[b]) for a, b in zip(*edges_ij)])
    return (jac > 0).mean(), jac.mean()


def genre_enrichment(labels, nodes, communities, genres=None):
    """Community x genre enrichment, multi-label (a film counts for each of its genres).
    For every (community, genre): observed films, expected films, log2(obs/exp), hypergeometric
    upper-tail p (over-representation) and Benjamini-Hochberg q over the whole table."""
    genres = genres or sorted({g for s in nodes.genre_set for g in s})
    N = len(labels)
    rows = []
    for c in communities:
        in_c = labels == c
        n_c = in_c.sum()
        for g in genres:
            has = nodes.genre_set.map(lambda s: g in s).to_numpy()
            K, k = has.sum(), (has & in_c).sum()
            exp = n_c * K / N
            rows.append({"community": c, "genre": g, "obs": int(k), "exp": exp,
                         "log2_enrich": np.log2((k + 0.5) / (exp + 0.5)),
                         "p": stats.hypergeom.sf(k - 1, N, K, n_c)})
    df = pd.DataFrame(rows)
    df["q"] = stats.false_discovery_control(df.p.to_numpy())
    return df


def knn_graph(S, k=10):
    """Symmetric k-nearest-neighbour graph from a similarity matrix: i - j if j is among i's k most similar
    films or vice versa; edge weight = the similarity."""
    S = S.copy()
    np.fill_diagonal(S, -np.inf)
    nn = np.argpartition(-S, k, axis=1)[:, :k]
    G = nx.Graph()
    G.add_nodes_from(range(len(S)))
    for i, row in enumerate(nn):
        for j in row:
            G.add_edge(i, int(j), weight=float(S[i, j]))
    return G


# --------------------------------------------------------------------------- pairs
def pair_table(labels, nodes, W, sims, pgenre):
    """Every film pair (i < j) once, as a DataFrame with i, j, same_comm, linked, same_primary,
    genre_jaccard and one column per similarity matrix in sims (dict name -> matrix)."""
    i, j = np.triu_indices(len(nodes), 1)
    gs = nodes.genre_set.to_numpy()
    sizes = np.array([len(s) for s in gs])
    # genre Jaccard via a binary genre matrix
    genres = sorted({g for s in gs for g in s})
    B = np.array([[g in s for g in genres] for s in gs], dtype=np.int8)
    inter = (B @ B.T)[i, j]
    df = pd.DataFrame({"i": i, "j": j, "same_comm": labels[i] == labels[j], "linked": W[i, j] > 0,
                       "same_primary": pgenre[i] == pgenre[j],
                       "genre_jaccard": inter / (sizes[i] + sizes[j] - inter)})
    for name, S in sims.items():
        df[name] = S[i, j]
    return df


# --------------------------------------------------------------------------- command line
def write_partitions(seeds=N_SEEDS):
    COMM.mkdir(exist_ok=True)
    nodes = nt.load_nodes()
    rows, cons_rows, summary = [], [], []
    for scheme in SCHEMES:
        G = scheme_graph(scheme, nodes)
        runs = louvain_runs(G, range(seeds))
        for s, lab in enumerate(runs):
            rows.append(pd.DataFrame({"node_id": nodes.node_id, "scheme": scheme, "seed": s, "community": lab}))
        cons, iters = consensus_partition(runs)
        cons_rows.append(pd.DataFrame({"node_id": nodes.node_id, "scheme": scheme, "community": cons}))
        nmi = pairwise_scores(list(runs))
        summary.append(f"# {scheme}: {len(np.unique(cons))} consensus communities ({iters} iterations), "
                       f"Q = {modularity(G, cons):.3f}, mean pairwise NMI of seeds = {nmi.mean():.3f}")
        print(summary[-1][2:], flush=True)

    head = (f"# Louvain communities (Theme C, {date.today()}): networkx louvain_communities, weight = edge weight, "
            f"resolution 1.\n# One row per film, weighting scheme and seed (0..{seeds - 1}). Community 0 = largest "
            "community of that run. scheme 'default' = network/films_edges.tsv,\n# the others = "
            "network/weightings/edges_<scheme>.tsv. Isolated films are their own community.\n")
    with open(COMM / "partitions.tsv", "w") as f:
        f.write(head)
        pd.concat(rows).to_csv(f, sep="\t", index=False)
    with open(COMM / "consensus.tsv", "w") as f:
        f.write(f"# Consensus partition per weighting scheme (Theme C, {date.today()}), Lancichinetti & Fortunato "
                f"2012 over the {seeds} Louvain runs in partitions.tsv\n# (co-assignment threshold 0.5). "
                "Community 0 = largest.\n" + "\n".join(summary) + "\n")
        pd.concat(cons_rows).to_csv(f, sep="\t", index=False)
    print("wrote", COMM / "partitions.tsv", "and", COMM / "consensus.tsv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("partitions", help="write communities/partitions.tsv and consensus.tsv")
    p.add_argument("--seeds", type=int, default=N_SEEDS)
    args = ap.parse_args()
    if args.cmd == "partitions":
        write_partitions(args.seeds)
