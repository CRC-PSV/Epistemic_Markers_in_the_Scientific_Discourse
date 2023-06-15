""""""

from srs.lib.utils.io_utils import load_csv_values_as_single_list
from srs.config import LEXICON_PATH, DOCMODELS_PATH, RESULTS_PATH, RND_SEED
from srs.lib.nlp_params import TT_NVA_TAGS
from srs.lib.utils.generators import generate_ids_text_tags_filtered
from srs.lib.models.coocs import CoocsModel


def step_2_main(window: int = 5):
    """Runs step 2. Pretty straight forward since everything is handled by the CoocsModel"""

    # Load lexicon, init and update CoocsModel
    print('Starting step 2: corpus-wide cooccurrences')
    lexicon = load_csv_values_as_single_list(LEXICON_PATH)
    print(f'Lexicon loaded, cooccurrences will be computed on {len(lexicon)} words with a window of {window}...')
    cm = CoocsModel(lexicon, window=window, tag_attr='lemma')
    for para_id, tags in generate_ids_text_tags_filtered(DOCMODELS_PATH, filter_fct=lambda x: x.pos in TT_NVA_TAGS, flatten=False):
        cm.update(para_id, tags)

    # Save model, export and save df
    cm.shuffle_refs(rnd_seed=RND_SEED)
    # cm.to_pickle(RESULTS_PATH / 'cooc_model_corpus.p')
    cm_df = cm.as_df()
    cm_df.to_pickle(RESULTS_PATH / 'cooc_df_corpus.p')
    print('Cooccurrences computed, cooc dataframe saved in results directory.')
    print('Step 2 done!')


if __name__ == '__main__':
    step_2_main()

