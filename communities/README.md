# Communities (Person C)

Built by `python community_tools.py partitions` (about 3 min). The analyses are in `notebooks/04_communities.ipynb`, and the results are in `results/communities/`.

| file | rows | columns |
|---|---|---|
| `partitions.tsv` | film × scheme × seed (8 × 50) | `node_id, scheme, seed, community` |
| `consensus.tsv` | film × scheme | `node_id, scheme, community`: the consensus of the 50 seeds (use this one) |

* Weighted Louvain (networkx, resolution 1). Community 0 is the largest community of that run.
* `scheme`: `default` = `network/films_edges.tsv`. The others (`actors, directors, dir1, dir2, dir4, dir10, billing`) come from `network/weightings/edges_<scheme>.tsv`.
* Isolated films are their own community.

```python
import pandas as pd
cons = pd.read_csv("communities/consensus.tsv", sep="\t", comment="#")
labels = cons[cons.scheme == "default"].set_index("node_id").community
```
