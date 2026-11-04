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


def build_edges(films, director_weight, actor_weight):
    pairs = defaultdict(lambda: {"shared_directors": 0, "shared_actors": 0,
                                 "director_names": [], "actor_names": []})
    for role, id_col, name_col in [("directors", "director_ids", "directors"),
                                   ("actors", "star_ids", "stars")]:
        person_films, person_name = defaultdict(set), {}
        for node, ids, names in zip(films.node_id, films[id_col], films[name_col]):
            for pid, name in zip(split_ids(ids), split_ids(names)):
                person_films[pid].add(node)
                person_name[pid] = name
        for pid, nodes in person_films.items():
            for a, b in itertools.combinations(sorted(nodes), 2):
                pairs[(a, b)][f"shared_{role}"] += 1
                pairs[(a, b)][f"{role[:-1]}_names"].append(person_name[pid])

    edges = pd.DataFrame([{"source": a, "target": b, **e} for (a, b), e in pairs.items()])
    edges["weight"] = director_weight * edges.shared_directors + actor_weight * edges.shared_actors
    for c in ["director_names", "actor_names"]:
        edges[c] = edges[c].str.join("|")
    cols = ["source", "target", "weight", "shared_directors", "shared_actors",
            "director_names", "actor_names"]
    return edges[cols].sort_values(["source", "target"])


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
        f"{len(nodes)} nodes, {len(edges)} edges.",
        f"{len(set(nodes.node_id) - set(edges.source) - set(edges.target))} films have no edge "
        "- take the full node set from films_nodes.tsv.",
    ])

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
