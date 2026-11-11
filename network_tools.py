"""Network structure and null models (Theme A): shared helpers for 02_network_analysis.ipynb.

Everything works on dense N x N numpy matrices whose row order is films_nodes.tsv sorted by
node_id (the project-wide convention), and on the upper triangle of those matrices
(N (N-1) / 2 = 1,599,366 film pairs for N = 1789).

Contents
    loading       load_nodes, load_edges, weight_matrix, load_similarity
    topology      topology_summary, top_centrality, ccdf
    SQ1 stats     hop_matrix, hop_classes, sim_by_hop, linked_vs_unlinked, mantel
    null models   rewire (degree-preserving double-edge swaps), genre_year_pools,
                  genre_year_null (genre- and year-matched random pairs)
    regression    mrqap (OLS with double-semi-partialling permutation p-values)
    backbone      disparity_filter

Command line: write the degree-preserving rewired networks shared with Theme C
    python network_tools.py rewire --n 100          # -> network/null/rewired_<i>.tsv
Network i is rewire(..., seed=i), so the files are exactly the first n networks of the
null ensemble used in the notebook.
"""
import argparse
import urllib.parse
import warnings
import zipfile
from datetime import date
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.sparse.csgraph import connected_components, shortest_path

HERE = Path(__file__).parent
NET = HERE / "network"
HOP_LABELS = ["1", "2", "3", "4+", "disc."]


# --------------------------------------------------------------------------- loading
def read_tsv(path):
    return pd.read_csv(path, sep="\t", comment="#", quoting=3)


def load_nodes():
    nodes = read_tsv(NET / "films_nodes.tsv").sort_values("node_id").reset_index(drop=True)
    nodes["genre_set"] = nodes.genres.str.split(",").map(frozenset)
    return nodes


def load_edges(scheme=None):
    """Edge list for a weighting scheme (network/weightings/edges_<scheme>.tsv);
    None gives the default network/films_edges.tsv (director weight 2)."""
    path = NET / "films_edges.tsv" if scheme is None else NET / "weightings" / f"edges_{scheme}.tsv"
    return read_tsv(path)


def edge_index(edges, nodes):
    idx = pd.Series(np.arange(len(nodes)), index=nodes.node_id)
    return idx[edges.source].to_numpy(), idx[edges.target].to_numpy()


def weight_matrix(edges, nodes):
    """Symmetric dense weight matrix W (0 = no edge)."""
    i, j = edge_index(edges, nodes)
    W = np.zeros((len(nodes), len(nodes)))
    W[i, j] = W[j, i] = edges.weight.to_numpy()
    return W


def to_graph(W, nodes=None):
    G = nx.from_numpy_array(W)
    if nodes is not None:
        G = nx.relabel_nodes(G, dict(enumerate(nodes.node_id)))
    for _, _, d in G.edges(data=True):
        d["distance"] = 1 / d["weight"]
    return G


def provisional_tfidf(nodes):
    """Stand-in for Theme B's nlp/sim_tfidf.npy: plain TF-IDF cosine on the raw plots
    (no NER removal, no lemmatization). Only so the pipeline runs before B delivers."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    with zipfile.ZipFile(NET / "film_plots.zip") as z:
        plots = [z.read(f"film_plots/{urllib.parse.quote(n, safe='')}.txt").decode()
                 for n in nodes.node_id]
    X = TfidfVectorizer(sublinear_tf=True, stop_words="english", min_df=2,
                        max_df=0.5).fit_transform(plots)
    S = (X @ X.T).toarray()
    np.fill_diagonal(S, 1.0)
    return S


def load_similarity(kind, nodes):
    """B's similarity matrix nlp/sim_<kind>.npy (kind = 'tfidf' or 'embed').
    Falls back to provisional_tfidf for 'tfidf' and returns None for a missing 'embed'."""
    path = HERE / "nlp" / f"sim_{kind}.npy"
    if path.exists():
        S = np.load(path)
        assert S.shape == (len(nodes), len(nodes)), f"{path} has shape {S.shape}"
        return S
    if kind == "tfidf":
        warnings.warn(f"{path} not found: using provisional TF-IDF (raw plots, no NER removal)")
        return provisional_tfidf(nodes)
    warnings.warn(f"{path} not found: skipping '{kind}'")
    return None


def upper(M):
    """Upper-triangle vector of a square matrix (all film pairs once)."""
    return M[np.triu_indices(len(M), 1)]


# --------------------------------------------------------------------------- topology
def ccdf(x):
    """Complementary CDF P(X >= x) on the sorted distinct values of x (x > 0)."""
    x = np.sort(np.asarray(x)[np.asarray(x) > 0])
    vals, first = np.unique(x, return_index=True)
    return vals, 1 - first / len(x)


def topology_summary(W):
    """One row of the network statistics table for a weight matrix W."""
    A = (W > 0).astype(float)
    G = to_graph(W)
    k, s = A.sum(1), W.sum(1)
    n_comp, lab = connected_components(sparse.csr_matrix(A), directed=False)
    sizes = np.bincount(lab)
    gcc = np.flatnonzero(lab == sizes.argmax())
    Wg = W[np.ix_(gcc, gcc)]
    hops = shortest_path(sparse.csr_matrix(Wg > 0), unweighted=True, directed=False)
    D = np.where(Wg > 0, 1 / np.where(Wg > 0, Wg, 1), 0)
    dist = shortest_path(sparse.csr_matrix(D), directed=False)
    iu = np.triu_indices(len(gcc), 1)
    wnorm = W.max()
    for _, _, d in G.edges(data=True):  # networkx weighted clustering expects w / max(w)
        d["wnorm"] = d["weight"] / wnorm
    return {
        "nodes": len(W), "edges": int(A.sum() / 2), "density": A.sum() / (len(W) * (len(W) - 1)),
        "mean degree": k.mean(), "max degree": int(k.max()), "mean strength": s.mean(),
        "isolates": int((k == 0).sum()), "components (size>1)": int((sizes > 1).sum()),
        "giant component %": 100 * len(gcc) / len(W),
        "clustering (unweighted)": nx.average_clustering(G),
        "clustering (weighted)": nx.average_clustering(G, weight="wnorm"),
        "transitivity": nx.transitivity(G),
        "degree assortativity": nx.degree_assortativity_coefficient(G),
        "avg shortest path (hops, GCC)": hops[iu].mean(),
        "avg shortest path (1/w, GCC)": dist[iu].mean(),
        "diameter (hops, GCC)": int(hops[iu].max()),
    }


def top_centrality(W, nodes, n=10):
    """Top-n films by weighted betweenness (distance = 1/weight) and eigenvector centrality."""
    G = to_graph(W, nodes)
    bc = nx.betweenness_centrality(G, weight="distance")
    gcc = G.subgraph(max(nx.connected_components(G), key=len))
    ev = nx.eigenvector_centrality_numpy(gcc, weight="weight")
    name = dict(zip(nodes.node_id, nodes.name))
    top = lambda c: [f"{name[f]} ({v:.3f})" for f, v in sorted(c.items(), key=lambda x: -x[1])[:n]]
    return pd.DataFrame({"betweenness": top(bc), "eigenvector": top(ev)},
                        index=pd.RangeIndex(1, n + 1, name="rank"))


# --------------------------------------------------------------------------- SQ1 statistics
def hop_matrix(W):
    """Unweighted shortest-path length between every film pair (inf = disconnected)."""
    return shortest_path(sparse.csr_matrix(W > 0), unweighted=True, directed=False)


def hop_classes(H):
    """Map hop distances to classes 0..4 = HOP_LABELS (1, 2, 3, 4+, disconnected)."""
    c = np.clip(H, 1, 4) - 1
    c[~np.isfinite(H)] = 4
    return c.astype(np.int8)


def sim_by_hop(s, c):
    """Mean similarity per hop class (vectors over pairs); NaN for an empty class."""
    tot = np.bincount(c, weights=s, minlength=5)
    cnt = np.bincount(c, minlength=5)
    with np.errstate(invalid="ignore"):
        return tot / cnt


def linked_vs_unlinked(s, linked):
    """Effect size of plot similarity for linked vs. unlinked pairs."""
    a, b = s[linked], s[~linked]
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                     / (len(a) + len(b) - 2))
    u = stats.mannwhitneyu(a, b, alternative="greater")
    return {"mean linked": a.mean(), "mean unlinked": b.mean(), "ratio": a.mean() / b.mean(),
            "cohen d": (a.mean() - b.mean()) / pooled,
            "AUC (P[linked > unlinked])": u.statistic / (len(a) * len(b))}


def _corr(x, y):
    x, y = x - x.mean(), y - y.mean()
    return float(x @ y / np.sqrt((x @ x) * (y @ y)))


def mantel(S, M, n_perm=999, method="spearman", seed=0):
    """Mantel test between two symmetric matrices: correlation of their upper triangles,
    p-value from jointly permuting the rows and columns of M (film pairs are not
    independent, so ordinary correlation p-values are far too small). One-sided (r > 0)."""
    n = len(S)
    iu = np.triu_indices(n, 1)
    x, y = S[iu], M[iu]
    if method == "spearman":
        x, y = stats.rankdata(x), stats.rankdata(y)
        R = np.zeros((n, n))
        R[iu] = y
        M = R + R.T
    r = _corr(x, y)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for b in range(n_perm):
        p = rng.permutation(n)
        null[b] = _corr(x, M[np.ix_(p, p)][iu])
    return {"r": r, "p": (1 + (null >= r).sum()) / (n_perm + 1),
            "null mean": null.mean(), "null sd": null.std()}


# --------------------------------------------------------------------------- null models
def rewire(i, j, n_nodes, seed, swaps_per_edge=10):
    """Degree-preserving randomisation by double-edge swaps (Maslov & Sneppen 2002).

    (a, b), (c, d) -> (a, d), (c, b) unless that creates a self-loop or a multi-edge.
    Each edge keeps its attributes when it moves (we return a permutation `src` such that
    new edge e carries the attributes of old edge src[e]), so degrees and the weight
    distribution are preserved exactly; strengths only approximately.
    Returns new endpoint arrays (i, j) with i < j and the attribute index array.
    """
    rng = np.random.default_rng(seed)
    a, b = np.minimum(i, j).astype(np.int64), np.maximum(i, j).astype(np.int64)
    m = len(a)
    present = set((a * n_nodes + b).tolist())
    n_try = swaps_per_edge * m
    e1, e2 = rng.integers(0, m, n_try), rng.integers(0, m, n_try)
    flip = rng.random(n_try) < 0.5
    a, b = a.tolist(), b.tolist()
    done = 0
    for x, y, f in zip(e1.tolist(), e2.tolist(), flip.tolist()):
        if x == y:
            continue
        u, v = a[x], b[x]
        p, q = (b[y], a[y]) if f else (a[y], b[y])
        # (u, v), (p, q) -> (u, q), (p, v)
        if u == q or p == v:
            continue
        k1 = min(u, q) * n_nodes + max(u, q)
        k2 = min(p, v) * n_nodes + max(p, v)
        if k1 in present or k2 in present or k1 == k2:
            continue
        present.discard(a[x] * n_nodes + b[x])
        present.discard(a[y] * n_nodes + b[y])
        present.add(k1)
        present.add(k2)
        a[x], b[x] = min(u, q), max(u, q)
        a[y], b[y] = min(p, v), max(p, v)
        done += 1
    # edge x keeps its own attributes; only its endpoints changed
    return np.array(a), np.array(b), np.arange(m), done / n_try


def rewired_weight_matrix(i, j, w, n_nodes, seed):
    a, b, src, _ = rewire(i, j, n_nodes, seed)
    W = np.zeros((n_nodes, n_nodes))
    W[a, b] = W[b, a] = w[src]
    return W


def genre_year_pools(nodes, min_pool=3):
    """For each film, the films it can be swapped for in the genre/year null.

    Tiers (first with >= min_pool candidates): identical IMDb genre set and |year diff| <= 1;
    identical genre set, any year; >= 2 shared genres (or the single genre) and |year diff| <= 1;
    >= 1 shared genre, any year (last resort, so every film has a pool).
    Returns (flat candidate array, offsets, sizes, tier per film)."""
    y = nodes.year.to_numpy()
    n = len(nodes)
    G = nodes.genres.str.get_dummies(sep=",").to_numpy()
    overlap, size = G @ G.T, G.sum(1)
    same = (overlap == size[:, None]) & (overlap == size[None, :])
    shared = overlap >= np.minimum(2, size)[:, None]
    near = np.abs(y[:, None] - y[None, :]) <= 1
    np.fill_diagonal(same, False)
    any_shared = overlap >= 1
    for m in (same, shared, any_shared):
        np.fill_diagonal(m, False)
    pools, tier = [], []
    for f in range(n):
        for t, mask in enumerate([same[f] & near[f], same[f], shared[f] & near[f], any_shared[f]]):
            cand = np.flatnonzero(mask)
            if len(cand) >= min_pool or t == 3:
                pools.append(cand)
                tier.append(t)
                break
    sizes = np.array([len(p) for p in pools])
    offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    return np.concatenate(pools), offsets, sizes, np.array(tier)


def genre_year_null(S, A, pi, pj, pools, rng, max_tries=10):
    """One draw of the genre/year null: every pair (i, j) becomes (i, j'), with j' a random
    film of the same genre set and year as j that is not i and not linked to i.
    Returns the matched similarities (NaN where no valid j' was found)."""
    flat, off, size, _ = pools
    out = np.full(len(pi), np.nan)
    todo = np.arange(len(pi))
    for _ in range(max_tries):
        j = pj[todo]
        jj = flat[off[j] + (rng.random(len(todo)) * size[j]).astype(np.int64)]
        ok = (jj != pi[todo]) & (A[pi[todo], jj] == 0)
        out[todo[ok]] = S[pi[todo[ok]], jj[ok]]
        todo = todo[~ok]
        if len(todo) == 0:
            break
    return out


def zscore(obs, null):
    null = np.asarray(null)
    return (obs - np.nanmean(null, 0)) / np.nanstd(null, 0)


# --------------------------------------------------------------------------- regression
def _ols(X, y):
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    sigma2 = resid @ resid / (len(y) - X.shape[1])
    return beta, beta / np.sqrt(sigma2 * np.diag(XtX_inv))


def mrqap(Y, Xs, n_perm=999, seed=0):
    """Multiple regression QAP with double-semi-partialling (Dekker, Krackhardt & Snijders 2007).

    Y: N x N matrix; Xs: dict name -> N x N matrix. Predictors are standardised, so the
    coefficients are comparable. For each predictor, regress it on the others, permute the
    rows/columns of the residual matrix and refit; p = share of |t_perm| >= |t_obs|."""
    n = len(Y)
    iu = np.triu_indices(n, 1)
    y = Y[iu]
    names = list(Xs)
    X = np.column_stack([Xs[k][iu] for k in names])
    X = (X - X.mean(0)) / X.std(0)
    X1 = np.column_stack([np.ones(len(y)), X])
    beta, t = _ols(X1, y)
    rng = np.random.default_rng(seed)
    p = {}
    for k in range(len(names)):
        others = np.delete(X1, k + 1, axis=1)
        resid = X1[:, k + 1] - others @ np.linalg.lstsq(others, X1[:, k + 1], rcond=None)[0]
        R = np.zeros((n, n))
        R[iu] = resid
        R = R + R.T
        tk = np.empty(n_perm)
        for b in range(n_perm):
            perm = rng.permutation(n)
            Xp = X1.copy()
            Xp[:, k + 1] = R[np.ix_(perm, perm)][iu]
            tk[b] = _ols(Xp, y)[1][k + 1]
        p[names[k]] = (1 + (np.abs(tk) >= abs(t[k + 1])).sum()) / (n_perm + 1)
    r2 = 1 - ((y - X1 @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    table = pd.DataFrame({"beta (per SD)": beta[1:], "t": t[1:], "p (MRQAP)": [p[k] for k in names]},
                         index=names)
    return table, r2


# --------------------------------------------------------------------------- backbone
def disparity_filter(W, alpha=0.05):
    """Disparity-filter backbone (Serrano, Boguna & Vespignani 2009).
    Edge (i, j) is kept if it is significant for at least one endpoint:
    (1 - w_ij / s_i) ** (k_i - 1) < alpha. Nodes of degree 1 never make an edge significant.
    Returns a boolean matrix of kept edges."""
    k = (W > 0).sum(1)
    s = W.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        pij = np.where(W > 0, W / s[:, None], 0)
        a = (1 - pij) ** (k[:, None] - 1)
    sig = (W > 0) & (k[:, None] > 1) & (a < alpha)
    return sig | sig.T


# --------------------------------------------------------------------------- CLI
def write_rewired(n, out=NET / "null", scheme=None):
    nodes = load_nodes()
    edges = load_edges(scheme)
    i, j = edge_index(edges, nodes)
    out.mkdir(parents=True, exist_ok=True)
    cols = [c for c in ["weight", "shared_directors", "shared_actors", "billing_score"] if c in edges]
    today = date.today().isoformat()
    for seed in range(n):
        a, b, src, acc = rewire(i, j, len(nodes), seed)
        df = pd.DataFrame({"source": nodes.node_id.to_numpy()[a], "target": nodes.node_id.to_numpy()[b],
                           **{c: edges[c].to_numpy()[src] for c in cols}})
        df = df.sort_values(["source", "target"])
        path = out / f"rewired_{seed}.tsv"
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(f"# Degree-preserving rewired film network {seed} ({today}), "
                    f"from {'films_edges.tsv' if scheme is None else f'edges_{scheme}.tsv'}.\n")
            f.write(f"# network_tools.rewire(seed={seed}): {10 * len(edges)} double-edge swap attempts, "
                    f"{acc:.1%} accepted. Edges keep their attributes (weight, shared_*) when moved,\n")
            f.write("# so every film keeps its degree and the weight distribution is unchanged. "
                    "Take the node set from films_nodes.tsv.\n")
            df.to_csv(f, sep="\t", index=False, quoting=3)
    print(f"wrote {n} rewired networks -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rewire", help="write network/null/rewired_<i>.tsv")
    r.add_argument("--n", type=int, default=100)
    r.add_argument("--scheme", default=None, help="weighting scheme (default: films_edges.tsv)")
    args = ap.parse_args()
    write_rewired(args.n, scheme=args.scheme)
