from pathlib import Path

# Project path settings

# Raw corpus (bmc xml files) directory. Should contain corpus files ONLY to avoid problem.
# CORPUS_PATH = Path('D:/articles')
CORPUS_PATH = Path('D:/charting_test/testcorpus')

# DocModels will be saved in / loaded from DOCMODELS_PATH. Directory should initially be empty and contain ONLY dataframes to avoid problems.
# This will take a fair amount of storage space.
DOCMODELS_PATH = Path('D:/docmodels')
# Various results (mostly pickled dataframes and json files) will be saved to / loaded from RESULTS_PATH
RESULTS_PATH = Path('D:/results')

# General project paths, should be left as is
CHARTING_PATH = Path(__file__).parent
DATA_PATH = CHARTING_PATH / 'data'
LEXICON_PATH = DATA_PATH / 'lexicon.csv'

# Set this to True to load the legacy working corpus instead of running step 1 from scratch, in order to ensure full reproducibility of the published results
# If using legacy mode, document ids and docterm labels lists should be accessible as json files (these files are ignored if legacy mode is disabled)
# See readme for details
LEGACY_MODE = False
LEGACY_IDS_PATH = DATA_PATH / 'legacy/legacy_ids.json'
LEGACY_DOCTERM_LABELS = DATA_PATH / 'legacy/legacy_docterm_labels.json'

# Other settings
# Random seed (set to 2112 to reproduce the original results)
RND_SEED = 2112
