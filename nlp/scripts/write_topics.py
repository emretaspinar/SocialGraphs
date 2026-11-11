"""Write the LDA topic shares per film and the topic labels, the topic files for Person C.

Input (written by build_similarity.py --only lda):
    nlp/representations/lda.npz           theta: 1744 x 20, each film's mixture over the 20 topics (rows sum to 1)
    nlp/representations/lda_topics.tsv    the 15 most probable words of every topic

Output (nlp/topics/):
    topics.tsv          one row per film (1744 rows, films_nodes.tsv order):
                            node_id, dominant_topic, topic_0 ... topic_19
                        topic_t = share of the film's plot assigned to topic t; each row sums to 1.
                        dominant_topic = the topic with the largest share.
    topic_labels.tsv    one row per topic (20 rows): topic, label, share, coherent, top_words
                        share = average share over all films; coherent = False for mixed topics that have no
                        clear theme (read their top words before using them in the paper).

The labels below were written by reading each topic's top words and its 6 films with the largest share. They
belong to this exact LDA fit (K = 20, seed 0). If LDA is rebuilt with other settings, the topics change and the
check in main() stops the script: relabel the topics by hand, then rerun.

Usage:
    python nlp/scripts/write_topics.py
"""
import numpy as np
import pandas as pd

import paths

# topic -> (label, coherent, the first 3 top words at the time of labelling, used to detect a changed fit)
LABELS = {
    0:  ("film industry and journalism",           True,  "film new work"),
    1:  ("family survival: war, farms, camps",      True,  "child family camp"),
    2:  ("space, aliens and invasion",              True,  "alien earth destroy"),
    3:  ("writers, books and hotels",               True,  "hotel book write"),
    4:  ("school and sports teams",                 True,  "school student game"),
    5:  ("spies, military and terrorism",           True,  "agent team cia"),
    6:  ("sea and survival",                        True,  "crew boat ship"),
    7:  ("ghosts, possession and the supernatural", True,  "body father memory"),
    8:  ("horror: attack in a house",               True,  "house car escape"),
    9:  ("mixed: chases and killings",              False, "pope shoot zoo"),
    10: ("mixed: zombies and epidemics",            False, "zombie family infected"),
    11: ("fantasy: kings, battles and witches",     True,  "king attack fight"),
    12: ("murder investigation",                    True,  "murder police killer"),
    13: ("music, bands and performing",             True,  "perform band drug"),
    14: ("crime: police, drugs and gangs",          True,  "police man money"),
    15: ("relationships and family life",          True,  "father friend mother"),
    16: ("competition: sports, jobs and contests",  True,  "win fight team"),
    17: ("mixed: expeditions, caves and tunnels",   False, "creature team voice"),
    18: ("money and business",                      True,  "money company pay"),
    19: ("mixed: escape and rescue missions",       False, "escape san bishop"),
}


def main():
    ids = paths.NODE_IDS.read_text(encoding="utf-8").split("\n")          # 1744 node_ids, the row order
    theta = np.load(paths.representation_file("lda"))["theta"]           # 1744 x 20, rows sum to 1
    topics = pd.read_csv(paths.LDA_TOPICS, sep="\t")                     # 20 rows: topic, share, top_words
    k = theta.shape[1]

    # Stop if the LDA fit is not the one the labels were written for
    changed = [t for t in range(k)
               if t not in LABELS or not topics.top_words[t].startswith(LABELS[t][2] + " ")]
    if len(LABELS) != k or changed:
        raise SystemExit(f"LDA topics changed since labelling (topics {changed}): update LABELS in write_topics.py")

    # Per film: the share of every topic (rounded; rows still sum to 1 within 0.001)
    shares = pd.DataFrame(theta.round(4), columns=[f"topic_{t}" for t in range(k)])
    shares.insert(0, "dominant_topic", theta.argmax(axis=1))
    shares.insert(0, "node_id", ids)

    labels = pd.DataFrame({
        "topic": range(k),
        "label": [LABELS[t][0] for t in range(k)],
        "share": topics.share,
        "coherent": [LABELS[t][1] for t in range(k)],
        "top_words": topics.top_words,
    })

    paths.TOPICS_DIR.mkdir(exist_ok=True)
    shares.to_csv(paths.TOPICS, sep="\t", index=False)
    labels.to_csv(paths.TOPIC_LABELS, sep="\t", index=False)
    print(f"wrote {paths.TOPICS.relative_to(paths.ROOT)} ({len(shares)} films x {k} topics) "
          f"and {paths.TOPIC_LABELS.relative_to(paths.ROOT)}")


if __name__ == "__main__":
    main()
