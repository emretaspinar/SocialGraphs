"""STEP 1 - Clean the film plots and remove the character names.

Why: two plots that both mention "Sarah" look similar to TF-IDF and to the sentence encoders, even if the
stories have nothing in common. So every character name is replaced before any similarity is computed.

Input  (from build_network.py)
    network/films_nodes.tsv    1744 films, defines the row order
    network/film_plots.zip     one Wikipedia plot text per film (~630 words on average)

Output (nlp/data/)
    plots_clean.tsv            one row per film (1744 rows), same order as films_nodes.tsv
        node_id                the film
        text_masked            the original plot, every character name replaced by "PERSON"
                               -> input for the sentence encoders, which need real sentences
        tokens                 lowercased lemmas, without stopwords, punctuation, numbers and names
                               -> input for TF-IDF and LDA
        tokens_names_kept      the same tokens, but with names kept (robustness check)
        n_names_masked         how many name tokens were masked in this plot
    person_mentions.tsv        every PERSON entity spaCy found (for inspection)
    nlp/validation/ner_gold_sample.tsv
                               40 random plot sentences for hand-labelling the names. Written only if the
                               file does not exist, so hand labels are never overwritten.

How a word is recognised as a name (three rules, each catches what the previous one missed)
    Rule 1  spaCy's named-entity recogniser (en_core_web_sm) tags it PERSON.
            Only PERSON: places and organisations ("FBI", "Vietnam") are story content and stay.
    Rule 2  Spreading within one plot. Once "Dom Cobb" is found, the words "Dom" and "Cobb" are masked
            everywhere else in the same plot, also where spaCy tagged them ORG or missed them.
    Rule 3  A name lexicon across all plots. spaCy's small model often tags a character ORG for a whole
            plot. So a proper noun is also masked if, across all 1744 plots, it is usually a name.
    Exceptions: titles and roles ("Agent", "Captain", "Mom") are kept, and a few PERSON strings that are
    not characters ("God", "Santa") are never masked. See TITLES, DROP and NOT_NAMES below.

Validation: notebooks/03_nlp.ipynb section 1.4 scores the masking against the 40 hand-labelled
sentences (precision 0.97, recall 0.88). docs/ner_model_choice.md explains why the small spaCy model is used.

Usage (from the repository root):
    python nlp/scripts/clean_plots.py
The spaCy parse takes ~6 minutes the first time; it is cached in nlp/cache/, so later runs take seconds.
"""
import hashlib
import re
import time
import urllib.parse
import zipfile
from collections import Counter

import numpy as np
import pandas as pd
import spacy
from spacy.tokens import DocBin

import paths

# ==================================================================================================
# SETTINGS
# ==================================================================================================
MODEL = "en_core_web_sm"     # spaCy pipeline (tokeniser, part-of-speech tagger, lemmatiser, NER), as in week 7
PLACEHOLDER = "PERSON"       # what a masked name is replaced with in text_masked
GOLD_N, GOLD_SEED = 40, 42   # size and random seed of the hand-labelling sample

# Rule 3 thresholds
PLACE_LABELS = {"GPE", "LOC", "FAC", "NORP"}   # spaCy labels for countries/cities, locations, buildings, nationalities
PLACE_VOTE = 3               # a word stays a name if place-labelled films outnumber PERSON-labelled films at most 3:1
LEXICON_MAX_LOWER = 0.1      # ... and it is written in lowercase at most 10% as often as capitalised ("Rose" fails)
STOPWORD_NAME_FILMS = 10     # a name that is also a stopword ("Will") must be tagged PERSON in at least 10 films
ROMAN = re.compile(r"^[IVXLC]+$")   # Roman numerals ("World War II") are never names

# Clean-up of entity text, from the week 7 NER exercise (week7_ner.py)
LEAD = re.compile(r"^(?:the|The)\s+")    # "the Joker" -> "Joker"
TAIL = re.compile(r"(?:’s|'s|'|’)$")     # "Cobb's"   -> "Cobb"

# Titles, roles and kinship words are KEPT even inside a name ("Agent Smith" -> "Agent PERSON"),
# because they carry story information.
TITLES = {
    "Mr", "Mrs", "Ms", "Miss", "Dr", "Doctor", "Prof", "Professor", "Sir", "Lady", "Lord", "Madam",
    "Agent", "Officer", "Detective", "Det", "Sergeant", "Sgt", "Lieutenant", "Lt", "Captain", "Capt",
    "Commander", "Colonel", "Col", "General", "Gen", "Major", "Admiral", "Private", "Corporal",
    "Chief", "Sheriff", "Deputy", "Judge", "Senator", "President", "Governor", "Mayor", "Director",
    "King", "Queen", "Prince", "Princess", "Duke", "Duchess", "Emperor", "Empress", "Count", "Baron",
    "Father", "Mother", "Brother", "Sister", "Uncle", "Aunt", "Grandma", "Grandpa", "Mom", "Dad",
    "Saint", "St", "Reverend", "Rev", "Pastor", "Bishop", "Pope", "Coach", "Nurse", "Jr", "Sr",
}

# PERSON strings that are not characters, never masked. Chosen once by reading the most frequent PERSON
# strings (notebook section 1.3); not tuned on the hand-labelled sample.
DROP = {
    "God", "Jesus", "Jesus Christ", "Christ", "Satan", "Lucifer", "Santa", "Santa Claus",
    "Christmas", "Hanukkah", "Thanksgiving", "Halloween", "Ford Mustang",
}

# Words that rule 3 would otherwise pick up but that are brands, places, gods or plain words.
# Chosen once by reading the 150 words most often masked only by rule 3; not tuned afterwards.
NOT_NAMES = {
    "Jesus", "Christ", "God", "Christian", "Facebook", "Twitter", "Google", "Apple", "Ford", "Ferrari",
    "Mustang", "Delta", "Stanford", "Olympic", "Olympics", "Apollo", "Mars", "Venus", "Phoenix",
    "Salem", "Seoul", "Brazil", "Florence", "Dallas", "Diego", "Orlando", "Mont", "Costa", "Hills",
    "Oz", "Show", "El", "La", "Le", "Los", "Las", "San", "Santa", "Madame", "Abu",
}


# ==================================================================================================
# 1. LOAD AND PARSE THE PLOTS
# ==================================================================================================

def load_plots():
    """Read every plot from the zip.

    Returns
        ids    list of 1744 node_ids, in films_nodes.tsv order (this order is used for every output)
        plots  dict node_id -> plot text (str)
    """
    nodes = pd.read_csv(paths.FILMS_NODES, sep="\t", comment="#", quoting=3)
    plots = {}
    with zipfile.ZipFile(paths.PLOTS_ZIP) as z:
        for f in z.namelist():
            if f.endswith(".txt") and "README" not in f:
                # file name = URL-encoded node_id, e.g. "Mad_Max%3A_Fury_Road.txt" -> "Mad_Max:_Fury_Road"
                node_id = urllib.parse.unquote(f.split("/")[-1][:-4])
                plots[node_id] = z.read(f).decode("utf-8")
    ids = list(nodes.node_id)
    assert set(ids) == set(plots), "films_nodes.tsv and film_plots.zip disagree"
    return ids, plots


def parse(ids, plots, model=MODEL):
    """Run spaCy on every plot, or load the cached result.

    Returns
        docs   list of 1744 spaCy Doc objects, in ids order. A Doc is the parsed plot: a sequence of
               tokens (~700 per plot), each with .text, .lemma_, .pos_ (part of speech), .is_stop, and
               the named entities in doc.ents (each with .label_, e.g. PERSON, ORG, GPE).
        nlp    the loaded spaCy pipeline (used for its stopword list)

    The cache file name contains the model, its version and a hash of all plot texts, so a changed plot
    or model automatically triggers a new parse.
    """
    nlp = spacy.load(model)
    key = hashlib.sha1("\n".join(plots[i] for i in ids).encode()).hexdigest()[:12]
    cache = paths.CACHE / f"docs-{model}-{nlp.meta['version']}-spacy{spacy.__version__}-{key}.spacy"
    if cache.exists():
        print(f"cache hit: {cache.name}")
        return list(DocBin().from_disk(cache).get_docs(nlp.vocab)), nlp

    t0 = time.time()
    docs = list(nlp.pipe((plots[i] for i in ids), batch_size=64))   # nlp.pipe parses in batches (faster)
    paths.CACHE.mkdir(parents=True, exist_ok=True)
    DocBin(docs=docs).to_disk(cache)
    print(f"spaCy parse: {time.time() - t0:.0f}s, cached to {cache.name}")
    return docs, nlp


# ==================================================================================================
# 2. FIND THE NAMES (rules 1-3)
# ==================================================================================================

def clean_name(text):
    """Entity text without a leading "the" and a trailing possessive: "the Joker's" -> "Joker"."""
    return TAIL.sub("", LEAD.sub("", text.strip()))


def person_entities(doc):
    """The PERSON entities of one plot, without the DROP list ("God", "Santa")."""
    return [e for e in doc.ents if e.label_ == "PERSON" and clean_name(e.text) not in DROP]


def person_words(doc):
    """The single words inside this plot's PERSON entities, e.g. {"Dom", "Cobb", "Arthur"}.

    Titles and one-letter words are left out. Used to build the rule-3 lexicon.
    """
    return {t.text for e in person_entities(doc)
            for t in e if t.is_alpha and t.text not in TITLES and len(t.text) > 1}


def name_lexicon(docs, stopwords):
    """RULE 3: the set of words that, across all plots, are usually character names.

    Counted over all 1744 plots (each film counts once per word):
        person[w]  number of films where w is inside a PERSON entity
        place[w]   number of films where w is inside a place entity (GPE, LOC, FAC, NORP)
        case[w]    number of times w occurs written exactly like this (case-sensitive)

    A word w is in the lexicon if all of these hold:
        - spaCy tagged it PERSON in at least one film
        - place labels do not outvote that: place[w] <= PLACE_VOTE * person[w]
          ("Olivia" stays a name, "Paris" is a city)
        - it is not in NOT_NAMES or DROP, and not a Roman numeral
        - it is rarely written in lowercase: case["rose"] <= 0.1 * case["Rose"] fails, so "Rose" is out.
          Exception: stopwords such as "Will" are allowed if tagged PERSON in >= 10 films; name_mask only
          masks them where spaCy tags them as a proper noun.
    ORG is NOT a vote against a name: the small model tags many characters ORG ("Sophie": ORG 171x, PERSON 1x).

    Returns a set of words (~7,700).
    """
    person, place, case = Counter(), Counter(), Counter()
    for doc in docs:
        person.update(person_words(doc))
        place.update({t.text for e in doc.ents if e.label_ in PLACE_LABELS for t in e})
        case.update(t.text for t in doc if t.is_alpha)

    lexicon = set()
    for w, n_person in person.items():
        if place[w] > PLACE_VOTE * n_person:
            continue                                   # mostly used as a place
        if w in NOT_NAMES | DROP or ROMAN.match(w):
            continue                                   # known non-name
        rarely_lowercase = case[w.lower()] <= LEXICON_MAX_LOWER * case[w]
        frequent_stopword_name = w.lower() in stopwords and n_person >= STOPWORD_NAME_FILMS
        if rarely_lowercase or frequent_stopword_name:
            lexicon.add(w)
    return lexicon


def name_mask(doc, stopwords, lexicon=frozenset(), spread_names=True):
    """Decide for every token of one plot whether it is (part of) a character name.

    Arguments
        doc           one parsed plot
        stopwords     spaCy's stopword set (lowercase words)
        lexicon       the rule-3 word set from name_lexicon(); empty = rules 1-2 only
        spread_names  False = rule 1 only (used in the notebook to show what each rule adds)

    Returns
        mask     numpy bool array, shape (number of tokens in the plot,); True = name token
        persons  the PERSON entities of the plot (list of spaCy spans)
    """
    persons = person_entities(doc)
    mask = np.zeros(len(doc), dtype=bool)
    spread = set()                                     # name words to look for elsewhere (rule 2)

    # Rule 1: every word inside a PERSON entity, except titles ("Agent Smith" -> only "Smith")
    for e in persons:
        for t in e:
            if t.is_alpha and t.text not in TITLES:
                mask[t.i] = True
                if len(t.text) > 1:
                    spread.add(t.text)
    if not spread_names:
        return mask, persons

    for t in doc:
        # Rule 2: the same word elsewhere in this plot, whatever spaCy labelled it there.
        # A name that is also a stopword ("Will") is only masked where tagged as a proper noun (PROPN),
        # so the verb "will" survives.
        if t.text in spread and (t.text.lower() not in stopwords or t.pos_ == "PROPN"):
            mask[t.i] = True
        # Rule 3: a proper noun that is in the corpus-wide lexicon, but never inside a date ("April 3")
        elif t.text in lexicon and t.pos_ == "PROPN" and t.ent_type_ not in {"DATE", "TIME"}:
            mask[t.i] = True
    return mask, persons


# ==================================================================================================
# 3. BUILD THE OUTPUT COLUMNS
# ==================================================================================================

def masked_text(doc, mask):
    """The original text with every run of name tokens replaced by one PLACEHOLDER.

    "Dom Cobb meets Saito." -> "PERSON meets PERSON."   (two name tokens in a row become one PERSON)
    """
    out = []
    for t in doc:
        if mask[t.i] and t.i > 0 and mask[t.i - 1]:
            out[-1] = PLACEHOLDER + t.whitespace_          # continue the previous PERSON
        elif mask[t.i]:
            out.append(PLACEHOLDER + t.whitespace_)
        else:
            out.append(t.text_with_ws)                     # the token plus the space after it
    return "".join(out)


def content_tokens(doc, mask=None):
    """The words TF-IDF and LDA see, as one space-separated string.

    Keeps a token if it is alphabetic, not a stopword, at least 2 letters long, and not masked.
    Each kept token is replaced by its lowercase lemma ("escaped" -> "escape", "thieves" -> "thief").
    mask=None keeps the names (the tokens_names_kept column).
    """
    return " ".join(t.lemma_.lower() for t in doc
                    if t.is_alpha and not t.is_stop and len(t.text) >= 2
                    and (mask is None or not mask[t.i]))


def gold_sample(ids, docs):
    """GOLD_N random plot sentences (at least 6 words) for hand-labelling the names.

    Returns a DataFrame with GOLD_N rows: node_id, start, end (character offsets in the plot),
    sentence, and an empty persons column to fill in by hand ("Name A | Name B").
    """
    candidates = [(nid, s.start_char, s.end_char, s.text.strip())
                  for nid, doc in zip(ids, docs) for s in doc.sents if len(s.text.split()) >= 6]
    rng = np.random.default_rng(GOLD_SEED)
    picked = sorted(rng.choice(len(candidates), size=GOLD_N, replace=False))
    gold = pd.DataFrame([candidates[i] for i in picked], columns=["node_id", "start", "end", "sentence"])
    gold["persons"] = ""
    return gold


# ==================================================================================================
# MAIN
# ==================================================================================================

def main():
    t0 = time.time()

    # 1. load and parse: ids (1744 node_ids), docs (1744 parsed plots, same order)
    ids, plots = load_plots()
    docs, nlp = parse(ids, plots)
    stopwords = nlp.Defaults.stop_words
    lexicon = name_lexicon(docs, stopwords)
    print(f"name lexicon (rule 3): {len(lexicon)} words")

    # 2. mask the names in every plot and build one output row per film
    rows, mentions = [], []
    for nid, doc in zip(ids, docs):
        mask, _ = name_mask(doc, stopwords, lexicon)       # bool array, one value per token
        rows.append({"node_id": nid,
                     "text_masked": masked_text(doc, mask),
                     "tokens": content_tokens(doc, mask),
                     "tokens_names_kept": content_tokens(doc),
                     "n_names_masked": int(mask.sum())})
        mentions += [(nid, e.start_char, e.end_char, e.text, clean_name(e.text) in DROP)
                     for e in doc.ents if e.label_ == "PERSON"]

    # 3. write the files. The TSVs use no quoting (as the course files), so fields must not contain tabs
    #    or newlines.
    paths.DATA.mkdir(parents=True, exist_ok=True)
    clean = pd.DataFrame(rows)                             # 1744 rows x 5 columns
    for col in ["text_masked", "tokens", "tokens_names_kept"]:
        clean[col] = clean[col].str.replace(r"[\t\r\n]+", " ", regex=True)
    clean.to_csv(paths.PLOTS_CLEAN, sep="\t", index=False, quoting=3)
    pd.DataFrame(mentions, columns=["node_id", "start", "end", "text", "dropped"]).to_csv(
        paths.PERSON_MENTIONS, sep="\t", index=False, quoting=3)

    if not paths.NER_GOLD.exists():                        # never overwrite hand labels
        paths.VALIDATION.mkdir(parents=True, exist_ok=True)
        gold_sample(ids, docs).to_csv(paths.NER_GOLD, sep="\t", index=False, quoting=3)
        print(f"wrote {paths.NER_GOLD.name} (fill in the persons column by hand)")
    else:
        print(f"kept existing {paths.NER_GOLD.name}")

    vocab = {w for t in clean.tokens for w in t.split()}
    print(f"{len(clean)} films | names masked per plot: mean {clean.n_names_masked.mean():.1f}, "
          f"median {clean.n_names_masked.median():.0f} | vocabulary {len(vocab)} | {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
