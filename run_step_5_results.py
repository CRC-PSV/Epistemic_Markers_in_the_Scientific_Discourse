from srs.config import RESULTS_PATH
import pandas as pd


def make_corrs_df(lexcounts_df_path, doc_ids=None):

    # load LexCounter
    lc_df = pd.read_pickle(lexcounts_df_path)

    if doc_ids is not None:
        lc_df = lc_df.loc[[para for para in lc_df.index if para.split('_')[0] in doc_ids]] 

    # Make corrs df and normalize
    corrs_df = lc_df.corr()
    corrs_df = corrs_df.applymap(lambda x: 0 if x == 1 else x)

    return corrs_df


def make_means_series(lexcounts_df_path):
    
    lc_df = pd.read_pickle(lexcounts_df_path)
    return lc_df.mean()


def results_main(lexcounts_df_path):
    # Saves all results data in RESULTS_PATH

    # Save average word counts across all paragraphs, as a pandas series
    make_means_series(lexcounts_df_path).to_pickle(RESULTS_PATH / 'word_counts_means_series.p')
    
    # Save correlation matrix for whole corpus
    make_corrs_df(lexcounts_df_path).to_pickle(RESULTS_PATH / 'lex_corrs_df_corpus.p')

    # Load cluster data and save cluster-level correlation matrix
    cluster_series = pd.read_pickle(RESULTS_PATH / 'doc_cluster_series.p')
    clusters = cluster_series.unique()

    for cluster in clusters:
        doc_ids = list(cluster_series[cluster_series == cluster].index)
        corrs_df = make_corrs_df(lexcounts_df_path, doc_ids)
        corrs_df.to_pickle(RESULTS_PATH / f'lex_corrs_df_{cluster}.p')


if __name__ == '__main__':
    results_main(RESULTS_PATH / 'test_lc_df.p')