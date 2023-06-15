"""Various tools to explore or export the data and results

The _summary functions generate dataframes similar to the tables presented in the publication
"""
from srs.lib.utils.generators import generate_all_docmodels
from srs.lib.utils.io_utils import save_json
from srs.config import DOCMODELS_PATH, RESULTS_PATH

import pandas as pd
from pathlib import Path


def export_doc_refs_json(save_path):
    """Saves corpus metadata as a json file. See DocModel.metadata_to_dict for details"""

    docs = {}
    for dm in generate_all_docmodels(DOCMODELS_PATH):
        docs[dm.id] = dm.metadata_to_dict()

    save_json(save_path, docs)


# The following functions output dataframes corresponding to the different tables present in the publications
# Data can be printed or exported, for example using .to_csv()
def cluster_summary(n_topics: int = 5):
    """Builds a dataframe with the number of docs per cluster, as well as the top-n topics"""

    cluster_series = load_cluster_series()#.value_counts()
    dt_df = load_doc_topics_df()

    dt_df['main_topic'] = dt_df.idxmax(axis=1)
    dt_df['cluster'] = cluster_series

    d = {}
    for cluster, n_docs in cluster_series.value_counts().items():
        top_topics = dt_df[dt_df['cluster'] == cluster]['main_topic'].value_counts().nlargest(n_topics).index
        d[cluster] = [n_docs] + list(top_topics)

    cluster_summary_df = pd.DataFrame.from_dict(d, orient='index')
    return cluster_summary_df


def topic_summary(n_words: int = 5):
    """Builds a dataframe with the top words for each topic"""

    tw_df = load_topic_words_df()
    topics_df = pd.DataFrame.from_dict({t: tw_df.loc[t].nlargest(n_words).index for t in tw_df.index}, orient='index')
    return topics_df


def cooc_summary(word_list, n_coocs: int = 20):
    """Makes a summary of the top-n cooccurring words for each word in word_list

    Words in word_list MUST be cooc df column names"""

    cc_df = load_cooc_df()

    top_cooc_df = pd.DataFrame.from_dict({word: cc_df[word].nlargest(n_coocs).index for word in word_list})
    return top_cooc_df


# Shortcuts to load the different results dataframes
def load_docterm_df():
    return pd.read_pickle(RESULTS_PATH / 'abstracts_docterm_df.p')


def load_cooc_df():
    return pd.read_pickle(RESULTS_PATH / 'cooc_df_corpus.p')


def load_doc_topics_df():
    return pd.read_pickle(RESULTS_PATH / 'doc_topics_df.p')


def load_topic_words_df():
    return pd.read_pickle(RESULTS_PATH / 'topic_words_df.p')


def load_cluster_series():
    return pd.read_pickle(RESULTS_PATH / 'doc_cluster_series.p')


def print_dfs():

    print('DOCTERM DF')
    print(load_docterm_df())

    print('\n\nCOOC DF')
    print(load_cooc_df())

    print('\n\nDOC TOPICS DF')
    print(load_doc_topics_df())

    print('\n\nTOPIC WORDS DF')
    print(load_topic_words_df())

    print('\n\nDOC CLUSTER SERIES')
    print(load_cluster_series())


if __name__ == '__main__':
    # export_doc_refs_json(RESULTS_PATH / 'corpus_metadata.json')
    print(cluster_summary())
    print(topic_summary())
    # cooc_summary(COOC_REFS_TERMS)
    pass
