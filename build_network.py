"""Build the weighted film collaboration network in the course's Marvel format.

Nodes are films. Two films are linked when they share at least one star
(first 3 billed actors) or director. Edge weight:

    weight = DIRECTOR_WEIGHT * shared_directors + ACTOR_WEIGHT * shared_actors

People are matched within the same role only (director-director, actor-actor).

Outputs (in network/), same layout as week1_nodes.tsv / week1_edges.tsv /
marvel_pages.zip:
    films_nodes.tsv   node_id, name, wikidata_id, url, description + film metadata
    films_edges.tsv   source, target, weight, shared_directors, shared_actors, ...
    film_plots.zip    film_plots/<url-encoded node_id>.txt  (Wikipedia plot section)

node_id is the film's English Wikipedia article name, like the Marvel node ids.
The network is undirected: each pair appears once in films_edges.tsv.

Weighting schemes (SQ2) are written to network/weightings/edges_<scheme>.tsv,
one edge list per scheme, keeping only pairs with weight > 0 (see SCHEMES):

    actors      shared_actors                       (director links dropped)
    directors   shared_directors                    (actor links dropped)
    dir1..dir10 d * shared_directors + shared_actors, d = 1, 2, 4, 10
    billing     2 * shared_directors + billing_score, where each shared actor adds
                the mean of 1/position in the two films (lead-lead 1.0,
                lead-3rd 0.67, 3rd-3rd 0.33), so shared leads count more

Usage:
    python build_network.py                      # director 2, actor 1
    python build_network.py --director-weight 3
"""
import argparse
import itertools
import re
import urllib.parse
import zipfile
from collections import defaultdict
from datetime import date
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
WIKI = "https://en.wikipedia.org/wiki/"


def split_ids(s):
    return [x for x in s.split("|") if x] if isinstance(s, str) else []


def load_films(path):
    films = pd.read_csv(path)
    films["node_id"] = films.article.str.replace(WIKI, "", regex=False).map(urllib.parse.unquote)
    assert films.node_id.is_unique
    return films


# scheme -> (description, weight function of the edge table)
SCHEMES = {
    "actors": ("shared_actors",
               lambda e: e.shared_actors),
    "directors": ("shared_directors",
                  lambda e: e.shared_directors),
    **{f"dir{d}": (f"{d} * shared_directors + shared_actors",
                   lambda e, d=d: d * e.shared_directors + e.shared_actors)
       for d in (1, 2, 4, 10)},
    "billing": ("2 * shared_directors + billing_score "
                "(each shared actor adds mean(1/position) over the two films)",
                lambda e: 2 * e.shared_directors + e.billing_score),
}
# director weight relative to actors, for plotting correspondence vs. director weight
SCHEME_DIRECTOR_WEIGHT = {"actors": 0, "dir1": 1, "dir2": 2, "dir4": 4, "dir10": 10,
                          "directors": float("inf"), "billing": 2}


def build_edges(films, director_weight, actor_weight):
    pairs = defaultdict(lambda: {"shared_directors": 0, "shared_actors": 0, "billing_score": 0.0,
                                 "director_names": [], "actor_names": []})
    for role, id_col, name_col in [("directors", "director_ids", "directors"),
                                   ("actors", "star_ids", "stars")]:
        person_films, person_name = defaultdict(dict), {}  # pid -> {node: billing position}
        for node, ids, names in zip(films.node_id, films[id_col], films[name_col]):
            for pos, (pid, name) in enumerate(zip(split_ids(ids), split_ids(names)), start=1):
                person_films[pid][node] = pos
                person_name[pid] = name
        for pid, nodes in person_films.items():
            for a, b in itertools.combinations(sorted(nodes), 2):
                pairs[(a, b)][f"shared_{role}"] += 1
                pairs[(a, b)][f"{role[:-1]}_names"].append(person_name[pid])
                if role == "actors":
                    pairs[(a, b)]["billing_score"] += (1 / nodes[a] + 1 / nodes[b]) / 2

    edges = pd.DataFrame([{"source": a, "target": b, **e} for (a, b), e in pairs.items()])
    edges["weight"] = director_weight * edges.shared_directors + actor_weight * edges.shared_actors
    edges["billing_score"] = edges.billing_score.round(4)
    for c in ["director_names", "actor_names"]:
        edges[c] = edges[c].str.join("|")
    cols = ["source", "target", "weight", "shared_directors", "shared_actors", "billing_score",
            "director_names", "actor_names"]
    return edges[cols].sort_values(["source", "target"]).reset_index(drop=True)


def write_schemes(edges, n_nodes, out, today):
    """One edge list per weighting scheme in out/weightings/edges_<scheme>.tsv."""
    (out / "weightings").mkdir(exist_ok=True)
    for name, (desc, fn) in SCHEMES.items():
        e = edges.assign(weight=fn(edges).round(4))
        e = e[e.weight > 0]
        write_tsv(e, out / "weightings" / f"edges_{name}.tsv", [
            f"Film collaboration network ({today}), weighting scheme '{name}'.",
            f"weight = {desc}. Only pairs with weight > 0.",
            f"{n_nodes} nodes (take them from films_nodes.tsv), {len(e)} edges.",
        ])
        print(f"  {name:10s} {len(e):6d} edges")


def first_sentence(text):
    return re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0]


def write_tsv(df, path, header_lines):
    with open(path, "w", encoding="utf-8", newline="") as f:
        for line in header_lines:
            f.write(f"# {line}\n")
        df.to_csv(f, sep="\t", index=False, quoting=3)  # QUOTE_NONE, like the course files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--films", default=HERE / "films_final.csv")
    ap.add_argument("--director-weight", type=float, default=2.0)
    ap.add_argument("--actor-weight", type=float, default=1.0)
    ap.add_argument("--out", default=HERE / "network")
    args = ap.parse_args()

    films = load_films(args.films)
    edges = build_edges(films, args.director_weight, args.actor_weight)
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    today = date.today().isoformat()

    nodes = pd.DataFrame({
        "node_id": films.node_id,
        "name": films.title,
        "wikidata_id": films.qid,
        "url": films.article,
        "description": films["plot"].map(first_sentence),
        "imdb_id": films.imdb,
        "year": films.year,
        "genres": films.genres,
        "directors": films.directors.fillna(""),
        "stars": films.stars,
        "num_votes": films.numVotes,
        "rating": films.averageRating,
        "runtime": films.runtimeMinutes,
        "plot_words": films.plot_words,
    }).sort_values("node_id")
    for c in ["name", "description"]:  # QUOTE_NONE: no tabs or newlines allowed in fields
        nodes[c] = nodes[c].str.replace(r"[\t\n\r]+", " ", regex=True)

    write_tsv(nodes, out / "films_nodes.tsv", [
        f"Film collaboration network ({today})",
        f"{len(nodes)} US films 2010-2019 (top by IMDb votes, one film per franchise, no animation).",
        "node_id = English Wikipedia article name; matches films_edges.tsv and film_plots.zip.",
    ])
    write_tsv(edges, out / "films_edges.tsv", [
        f"Film collaboration network ({today}): undirected, one row per film pair.",
        "Edge when two films share a director or one of the first 3 billed actors (same role only).",
        f"weight = {args.director_weight:g} * shared_directors + {args.actor_weight:g} * shared_actors. "
        "billing_score = sum over shared actors of mean(1/billing position) in the two films. "
        f"{len(nodes)} nodes, {len(edges)} edges.",
        f"{len(set(nodes.node_id) - set(edges.source) - set(edges.target))} films have no edge "
        "- take the full node set from films_nodes.tsv.",
    ])

    write_schemes(edges, len(nodes), out, today)

    with zipfile.ZipFile(out / "film_plots.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("film_plots/README.txt",
                   f"Film collaboration network ({today})\n"
                   "Plot section of the English Wikipedia article for each film in films_nodes.tsv.\n"
                   "One file per film. The filename is the node_id, URL-encoded, because some titles\n"
                   "contain characters a filesystem will not take (Mad_Max:_Fury_Road).\n"
                   "urllib.parse.unquote the stem and it matches node_id exactly.\n")
        for node, plot in zip(films.node_id, films["plot"]):
            z.writestr(f"film_plots/{urllib.parse.quote(node, safe='')}.txt", plot)

    print(f"{len(nodes)} nodes, {len(edges)} edges -> {out}")


if __name__ == "__main__":
    main()
