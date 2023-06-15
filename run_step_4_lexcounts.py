from srs.lib.models.lexcount import LexCounter
from srs.lib.utils.generators import generate_para_lemmas
from srs.lib.utils.io_utils import make_list_mapping_from_csv_path
from srs.config import DOCMODELS_PATH, LEXICON_PATH, RESULTS_PATH



def run_lexcounts(lexcount_df_save_path, lexcount_model_save_path=None):

    # Load lexicon from csv fil as a {'category_name': ['words']} mapping
    lexicon = make_list_mapping_from_csv_path(LEXICON_PATH)

    # Initiate the LexCounts object with the lexicon.
    # Will throw an error or a warning if a problem is detected with the lexicon, i.e. if some categories contain no words
    lc = LexCounter(lex_mapping=lexicon)

    # Iterate through the DocModels and call .update() for each paragraph
    # doc_para_id consists of the doc id stored in each DocModel and the paragraph number, split by an underscore: '[doc_id]_[para_num]'
    # lemmas is a list of lemmas (str) within each paragraph
    for doc_para_id, lemmas in generate_para_lemmas(DOCMODELS_PATH):
        lc.update(doc_para_id, lemmas)


    # If a path was specified, store the LexCounter object as a pickle
    if lexcount_model_save_path is not None:
        lc.to_pickle(lexcount_model_save_path)

    # Exports the LexCount results as a pandas DataFrame and stor as pickle
    # Represents the number of occurrences of reach word of the lexicon (columns) in each paragraph (rows)
    # Unless specified otherwise, words (columns) belogning to the same lexical category will be merged 
    lc_df = lc.as_df(merge_categories=True)
    lc_df.to_pickle(lexcount_df_save_path)
    print(lc_df)
    print(lc_df.sum())


if __name__ == '__main__':
    run_lexcounts(RESULTS_PATH / 'LEXCOUNTS_DF.p')

