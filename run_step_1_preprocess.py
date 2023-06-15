import sys
import os

from srs.lib.utils.io_utils import read_y_n_input, load_json
from srs.config import LEGACY_MODE, DOCMODELS_PATH, CORPUS_PATH, RESULTS_PATH, LEGACY_IDS_PATH, LEGACY_DOCTERM_LABELS
from srs.lib.preprocess.extraction import extract_and_tag_docmodel_texts, create_docmodels_from_xml_corpus
from srs.lib.utils.generators import generate_all_docmodels, generate_ids_abs_tags
from srs.lib.nlp_params import TT_NVA_TAGS, SPECIAL_CHARACTERS_BASE, TRASH_SECTIONS, LEGACY_TRASH_SECTIONS
from srs.lib.models.tagcounts import TagCountsModel
from srs.lib.models.docterm import DocTermModel


def step_1_setup():
    """Confirm settings and prompt y/n to proceed/exit"""

    print('Starting extraction and preprocessing (step 1)')
    print(f'Make sure file paths and other project settings in charting_config.py are set correctly before proceeding.')
    if len(os.listdir(DOCMODELS_PATH)) > 0:
        print('WARNING! DocModels directory is NOT empty or might contain hidden files. Proceed at your own risk.')

    if LEGACY_MODE:
        print(f'LEGACY_MODE is currently set to {LEGACY_MODE}. See readme for details')
    else:
        print(f'LEGACY_MODE is currently set to {LEGACY_MODE}. See readme for details')

    if not read_y_n_input('Continue? (y/n): '):
        print('Cancelling...')
        sys.exit()


def step_1_extraction():
    """Build DocModels from raw xml corpus"""

    print('Starting extraction step.')
    print('This will create and pickle DocModel objects from the source XML files.')
    create_docmodels_from_xml_corpus(CORPUS_PATH, DOCMODELS_PATH)


def step_1_tagging(legacy: bool):
    """Update DocModels by extracting and tagging textual content"""

    print('Starting tagging step')
    print('This will load and update DocModels, extracting the textual contents and generating tags')
    trash_sections = LEGACY_TRASH_SECTIONS if legacy else TRASH_SECTIONS
    extract_and_tag_docmodel_texts(DOCMODELS_PATH, trash_sections)

    # If the TT bug persists, on legacy mode use a mapping to transform the problematic lemmas directly on the DocModels


def step_1_filtering(legacy: bool, min_abs_len: int = 150, min_text_len: int = 2000):
    """Filters DocModels based on abstract/text text length and deletes unwanted files
y
    If legacy mode is enabled, will filter based on a list of legacy ids instead of textual length
    """

    print('\nStarting filtering step')
    if legacy:
        print('Using legacy mode, DocModels will be filtered using loaded data')
        legacy_ids = load_json(LEGACY_IDS_PATH)
        filter_fct = lambda x: x.id in legacy_ids
    else:
        print('NOT using legacy mode, DocModels will be filtered based on the 150/2000 abstract and text minimum length.')
        filter_fct = lambda x: x.abs_words >= min_abs_len and x.text_words >= min_text_len
    print('Warning: Filtered DocModels will be deleted!')
    print('Filtering docmodels...')
    files_to_delete = []
    for dm in generate_all_docmodels(DOCMODELS_PATH):
        if not filter_fct(dm):
            files_to_delete.append(dm.filename)
    print('Deleting filtered docmodels...')
    for f in files_to_delete:
        os.remove(DOCMODELS_PATH / f)
    print(f'Deleted {len(files_to_delete)} docmodels, {len(os.listdir(DOCMODELS_PATH))} were kept')


def step_1_docterm(legacy: bool):
    """Builds a docterm matrix based on the DocModels' abstracts

    If legacy mode is enabled, the matrix' rows and columns will be reordered to match the original configuration.
    """

    if legacy:
        # Load legacy vocab to reproduce results
        labels = load_json(LEGACY_DOCTERM_LABELS)
        vocab = labels['columns']
        dt = DocTermModel(update_filter_fct=lambda x: x.pos in TT_NVA_TAGS)
        for doc_id, tags in generate_ids_abs_tags(DOCMODELS_PATH):
            dt.update(doc_id, tags)

    else:
        # TODO use named functions so object can be pickled
        tc = TagCountsModel(update_filter_fct=lambda x: x.pos in TT_NVA_TAGS)
        dt = DocTermModel(update_filter_fct=lambda x: x.pos in TT_NVA_TAGS)

        for doc_id, tags in generate_ids_abs_tags(DOCMODELS_PATH):
            tc.update(tags)
            dt.update(doc_id, tags)

        # Make word list
        tc.filter_values(lambda x: (len(x) >= 3) and (not any(char in SPECIAL_CHARACTERS_BASE for char in x)))
        tc_df = tc.as_df()  # cols: total_counts article_counts
        tc_df = tc_df[(tc_df['article_counts'] <= 0.3*len(tc_df)) & (tc_df['article_counts'] >= 50)]  # TODO check filters

        # Make vocab
        vocab = tc_df.index

        # Save TagCountsModel for future reference
        # tc.to_pickle(RESULTS_PATH / 'abstracts_tagcounts_model.p')


    # Filter DocTermModel to only keep vocab words before building the dataframe
    dt.filter_words(lambda x: x in vocab)

    # Save DocTermModel for future reference
    # dt.to_pickle(RESULTS_PATH / 'abstracts_docterm_model.p')

    # Build, normalize and save docterm matrix. If legacy, reorder labels to match original configuration
    dt_df = dt.as_df(log_norm=True)
    if legacy:
        dt_df = dt_df.reindex(index=labels['index'], columns=labels['columns'])

    dt_df.to_pickle(RESULTS_PATH / 'abstracts_docterm_df.p')


def step_1_main():

    step_1_setup()
    step_1_extraction()
    step_1_tagging(LEGACY_MODE)
    step_1_filtering(LEGACY_MODE)
    print('Done extracting the data and building the working corpus.')

    print('Building the docterm matrix from the abstracts')
    step_1_docterm(LEGACY_MODE)
    print('Docterm matrix built and saved to results folder. Step 1 complete!')


if __name__ == '__main__':
    step_1_main()
