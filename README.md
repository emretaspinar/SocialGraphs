# Film collaboration network vs. plot similarity

Our notebooks have already been executed and contain all the results and outputs.
However, if you would like to run them again, please follow the instructions in the guide below.

The data is generated from `films_final.csv`. Use Python 3.12 or 3.13.

## 1. Install

    python3.12 -m venv ~/socialgraphs-env
    source ~/socialgraphs-env/bin/activate
    pip install -r requirements.txt
    python -m ipykernel install --sys-prefix

Activate the environment in every new terminal, and choose it as kernel in the notebooks.

## 2. Generate the data (run in this order, from this folder)

    python build_network.py                    # network and weighting schemes
    python network_tools.py rewire --n 100     # 100 rewired networks
    python nlp/scripts/clean_plots.py          # clean the plots, mask names
    python nlp/scripts/build_similarity.py --only tfidf tfidf_names_kept minilm mpnet   # similarities (about 15 min, downloads two models)
    python nlp/scripts/write_topics.py         # topic tables
    python community_tools.py partitions       # communities

## 3. Run the notebooks

1. `network/preprocess.ipynb`
2. `nlp/notebooks/03_nlp.ipynb`
3. `nlp/notebooks/03b_nlp_analyses.ipynb`
4. `notebooks/02_network_analysis.ipynb` (slow: 1,000 rewired networks)
5. `notebooks/yaman_task.ipynb`¨'


Figures and tables are saved in `results/`.

The fitted topic model (`nlp/representations/lda.npz`, `lda_topics.tsv`) is included, because refitting LDA with other
package versions gives other topics than the ones we labelled. All other data is generated.

`nlp/validation/`: it holds our hand-made ratings