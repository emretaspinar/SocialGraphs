"""Build the weighted film collaboration network in the course's Marvel format.

Nodes are films. Two films are linked when they share a person, in one of three ways:

    shared director    the same person directed both films                          weight 3
    shared actor       the same person is one of the first 3 billed actors in both   weight 1
    cross-role         the person directed one film and is a lead actor in the other weight 2

    weight = 3 * shared_directors + 2 * shared_cross + 1 * shared_actors

Every shared person counts, so two films with the same director and two shared actors get 3 + 1 + 1 = 5.
A person counts as cross-role for a pair only if they do not already share the same role in it (a director
who also stars in both films counts as a shared director and a shared actor, not a third time).

Sequels: films_final.csv was meant to have one film per franchise, but small series slipped through
(Sinister + Sinister 2). These were removed from films_final.csv itself (1789 -> 1744 films): films linked by
Wikidata "follows" / "followed by" (P155 / P156) form a group, and only the group's film with the most IMDb votes
was kept. So the csv and the network contain the same films.

Outputs (in network/), same layout as week1_nodes.tsv / week1_edges.tsv / marvel_pages.zip:
    films_nodes.tsv     node_id, name, wikidata_id, url, description + film metadata
    films_edges.tsv     source, target, weight, shared_directors, shared_actors, shared_cross, ...
    film_plots.zip      film_plots/<url-encoded node_id>.txt  (Wikipedia plot section)

node_id is the film's English Wikipedia article name, like the Marvel node ids.
The network is undirected: each pair appears once in films_edges.tsv.

Usage:
    python build_network.py                          # director 3, cross-role 2, actor 1
    python build_network.py --director-weight 4
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
    """'nm1|nm2|nm3' -> ['nm1', 'nm2', 'nm3']; empty or missing -> []"""
    return [x for x in s.split("|") if x] if isinstance(s, str) else []


def load_films(path):
    films = pd.read_csv(path)
    films["node_id"] = films.article.str.replace(WIKI, "", regex=False).map(urllib.parse.unquote)
    assert films.node_id.is_unique
    return films


# --------------------------------------------------------------------------------------------- edges

def people_by_film(films, id_col, name_col):
    """For one role: person id -> set of node_ids, and person id -> name."""
    person_films, person_name = defaultdict(set), {}
    for node, ids, names in zip(films.node_id, films[id_col], films[name_col]):
        for pid, name in zip(split_ids(ids), split_ids(names)):
            person_films[pid].add(node)
            person_name[pid] = name
    return person_films, person_name


def build_edges(films, director_weight, actor_weight, cross_weight):
    """One row per linked film pair, with the number and names of shared people of each kind."""
    pairs = defaultdict(lambda: {"shared_directors": 0, "shared_actors": 0, "shared_cross": 0,
                                 "director_names": [], "actor_names": [], "cross_names": []})
    directed, director_name = people_by_film(films, "director_ids", "directors")
    starred, actor_name = people_by_film(films, "star_ids", "stars")

    # Same role: every pair of films with this director (or this actor)
    for role, person_films, person_name in [("directors", directed, director_name), ("actors", starred, actor_name)]:
        for pid, nodes in person_films.items():
            for a, b in itertools.combinations(sorted(nodes), 2):
                pairs[(a, b)][f"shared_{role}"] += 1
                pairs[(a, b)][f"{role[:-1]}_names"].append(person_name[pid])

    # Cross-role: the person directed film a and acts in film b (or the other way round). Skipped if the person
    # already shares a role in that pair, so nobody is counted twice for the same pair.
    for pid in set(directed) & set(starred):
        cross_pairs = {tuple(sorted((a, b))) for a in directed[pid] for b in starred[pid] if a != b}
        for a, b in cross_pairs:
            same_role = {a, b} <= directed[pid] or {a, b} <= starred[pid]
            if not same_role:
                pairs[(a, b)]["shared_cross"] += 1
                pairs[(a, b)]["cross_names"].append(director_name[pid])

    edges = pd.DataFrame([{"source": a, "target": b, **e} for (a, b), e in pairs.items()])
    edges["weight"] = (director_weight * edges.shared_directors + cross_weight * edges.shared_cross
                       + actor_weight * edges.shared_actors)
    for c in ["director_names", "actor_names", "cross_names"]:
        edges[c] = edges[c].str.join("|")
    cols = ["source", "target", "weight", "shared_directors", "shared_actors", "shared_cross",
            "director_names", "actor_names", "cross_names"]
    return edges[cols].sort_values(["source", "target"])


# --------------------------------------------------------------------------------------------- output

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
    ap.add_argument("--director-weight", type=float, default=3.0)
    ap.add_argument("--cross-weight", type=float, default=2.0)
    ap.add_argument("--actor-weight", type=float, default=1.0)
    ap.add_argument("--out", default=HERE / "network")
    args = ap.parse_args()

    films = load_films(args.films)
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    edges = build_edges(films, args.director_weight, args.actor_weight, args.cross_weight)
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
        f"{len(nodes)} US films 2010-2019 (top by IMDb votes, one film per franchise, no animation, sequels removed).",
        "node_id = English Wikipedia article name; matches films_edges.tsv and film_plots.zip.",
    ])
    write_tsv(edges, out / "films_edges.tsv", [
        f"Film collaboration network ({today}): undirected, one row per film pair.",
        "Edge when two films share a director or one of the first 3 billed actors (same role), "
        "or one film's director is a lead actor in the other (cross-role).",
        f"weight = {args.director_weight:g} * shared_directors + {args.cross_weight:g} * shared_cross + "
        f"{args.actor_weight:g} * shared_actors. {len(nodes)} nodes, {len(edges)} edges.",
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
