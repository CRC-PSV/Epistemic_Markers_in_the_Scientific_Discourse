""""""
import os
import treetaggerwrapper
import xml.etree.ElementTree as ET
from pathlib import Path

from srs.lib.docmodel import DocModel


def create_docmodels_from_xml_corpus(srs_path: Path, save_path: Path, extract_metadata: bool = True) -> None:
    """Reads XMLs and creates DocModel objects. Also extracts metadata unless specified otherwise.

    Creates a DocModel for each file in the source folder and pickles it to the destination folder. All files in the
    source folder should be BioMed xml files, and the destination folder should be empty.

    Args:
        srs_path: Path to the folder holding the source XML files
        save_path: Folder in which to save the pickled docmodels
        extract_metadata: Whether to extract metadata on docmodel init

    Returns:

    """

    print(f'Starting to parse xml files at {srs_path}...')
    for i, filename in enumerate(os.listdir(srs_path)):
        try:
            DocModel(filename, ET.parse(srs_path / filename), save_path, extract_metadata_on_init=extract_metadata)
        except:
            print(f'Error on {filename}')
        if (i+1) % 10000 == 0:
            print(f'Parsed {i+1} files...')
    print("Save path : {}".format(save_path))


def extract_and_tag_docmodel_texts(path: Path, trash_sections) -> None:
    """Loads and updates all DocModels in a dir by extracting and tagging abstracts and texts.

    Should be called after creating DocModels from XMLs to complete the extraction / tokenization / tagging process.
    Also updates abs_words, abs_tokens, text_words and text_tokens metadata.

    Args:
        path: Folder holding the pickled DocModels

    Returns:

    """

    tagger = treetaggerwrapper.TreeTagger(TAGLANG='en')
    print(f'Starting to extract and tag texts from docmodels at {path}...')
    for i, dm in enumerate(DocModel.docmodel_generator(path)):
        dm.extract_abstract(trash_sections)
        dm.extract_text(trash_sections)
        dm.treetag_abstract(tagger)
        dm.treetag_text(tagger)

        # Add token counts as metadata
        dm.make_token_counts()

        dm.to_pickle()
        if (i+1) % 10000 == 0: print(f'Processed {i+1} docmodels...')
    print('Done!')

