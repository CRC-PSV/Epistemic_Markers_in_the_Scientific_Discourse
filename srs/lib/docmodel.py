"""Defines the DocModel class, used to represent documents.

These objects hold the metadata and text data for each documents, so the original corpus files should not be needed once the DocModels are generated.
All "extract_..." methods were built specifically to work with BioMed XML files, and will need tweeking to work with different source material.
"""

import pickle
import os
from treetaggerwrapper import make_tags


from srs.lib.nlp_params import TT_EXCLUDED_TAGS


class DocModel:
    def __init__(self, origin_file, tree, save_path, save_on_init=True, extract_metadata_on_init=True):
        # Since the project was iterative by design, the data from each different step is saved to facilitate experimentation
        # For this reason, the ElementTree object is stored here
        # However, those are no longer needed after the tagging step and could be deleted to keep the files lighter
        # To do so, simply update the DocModels and set self.tree to None
        self.tree = tree
        self.origin_file = origin_file
        self.filename = origin_file[:-4] + '.p'
        self.file_path = save_path / self.filename

        # Metadata
        self.id = origin_file[:-4]
        self.year = None
        self.title = None
        self.source = None
        self.doctype = None
        self.issn = None
        self.keywords = None
        self.authors = None
        self.collab = None
        self.doi = None
        self.url = None
        self.volume = None
        self.issue = None
        self.page = None

        self.citation = None

        # From mappings
        self.doctype_cat = None
        self.primary_subjects = None
        self.secondary_subjects = None

        # Exracted raw text content
        self.raw_text_paragraphs = None
        self.raw_abs_paragraphs = None

        # TreeTagger tags
        self.tt_text_paragraphs = None
        self.tt_abs_paragraphs = None

        # Length, total tokens and word tokens
        # _tokens include all tokens, _words includes only those with word POS tags
        self.abs_tokens = None
        self.text_tokens = None
        self.abs_words = None
        self.text_words = None


        # extract and save on creation
        if extract_metadata_on_init:
            self.extract_all_metadata()
        if save_on_init:
            self.to_pickle()

    ### Getters ###
    # Most properties can also be accessed directly (e.g. 'docmodel.id will work the same as docmodel.get_id()')

    def get_id(self):
        return self.id

    def get_year(self):
        return self.year

    def get_title(self):
        return self.title

    def get_source(self):
        return self.source

    def get_doctype(self):
        return self.doctype

    def get_issn(self):
        return self.issn

    def get_keywords(self):
        return self.keywords

    def get_doctype_cat(self):
        return self.doctype_cat

    def get_primary_subjects(self):
        return self.primary_subjects

    def get_secondary_subjects(self):
        return self.secondary_subjects

    def get_raw_text(self):
        """Get text as a list of paragraphs ['para1 text', 'word_list 2 text']"""

        return self.raw_text_paragraphs

    def get_raw_abs(self):
        """Get abstract text as a list of paragraphs ['para1 text', 'word_list 2 text']"""

        return self.raw_abs_paragraphs

    def get_text_tags(self, flatten=False):
        """Get text tags as 2d list [[para1 Tags], [para2 Tags], ...]"""

        return self.tt_text_paragraphs if not flatten else sum(self.tt_text_paragraphs, [])

    def get_abs_tags(self, flatten=False):
        """Get abstract tags as 2d list [[para1 Tags], [para2 Tags], ...]"""

        return self.tt_abs_paragraphs if not flatten else sum(self.tt_abs_paragraphs, [])

    def get_text_sentences_tags(self, *args, **kwargs):
        s = []
        for tag in self.get_text_tags(flatten=True):
            if tag.pos != 'SENT':
                s.append(tag)
            else:
                yield s
                s = []

    ### Extractors ###
    # Extractor function to get all the relevant data from the tree object and set the docmodel properties

    def extract_id(self):
        try:
            id = self.tree.getroot()[0].text
            if id != self.id:
                print(f'id mismatch on doc from {self.origin_file}, using xml id')
            self.id = id
        except:
            self.id = 'error'
            print(f'Error updating id on doc from {self.origin_file}')

    def extract_title(self):
        try:
            self.title = ''.join(self.tree.getroot()[2].find("bibl").find("title")[0].itertext())
        except:
            self.title = 'error'
            print(f'Error updating title on {self.id}')

    def extract_doctype(self):
        try:
            d = self.tree.getroot()[2][0].text
            self.doctype = d.lower().strip()
        except:
            self.doctype = 'error'
            print(f'Error updating doctype on {self.id}')

    def extract_year(self):
        self.year = self._bibl_metadata_extractor('pubdate')

    def extract_source(self):
        self.source = self._bibl_metadata_extractor('source')

    def extract_issn(self):
        self.issn = self._bibl_metadata_extractor('issn')

    def extract_url(self):
        self.url = self._bibl_metadata_extractor('url')

    def extract_volume(self):
        self.volume = self._bibl_metadata_extractor('volume')

    def extract_issue(self):
        self.issue = self._bibl_metadata_extractor('issue')

    def extract_page(self):
        self.page = self._bibl_metadata_extractor('fpage')

    def extract_doi(self):

        self.doi = None
        try:
            for pubid in self.tree.getroot()[2][1].find('xrefbib').iter('pubid'):
                if pubid.attrib['idtype'] == 'doi':
                    self.doi = pubid.text.lower().strip()
        except:
            print(f'Error updating doi on doc {self.id}')
            self.doi = 'error'

    def extract_authors(self):
        try:
            self.authors = [(au.find('snm').text, au.find('fnm').text)
                            for au in self.tree.getroot()[2][1].find('aug').iter('au')
                            if (au.find('snm') is not None) and (au.find('fnm') is not None)] #'type' not in au.attrib]
        except:
            print(f'Error extracting authors on doc {self.id}')
            self.authors = []

    def extract_collab(self):
        try:
            self.collab = self.tree.getroot()[2].find('cpyrt').find('collab').text.strip().split(';')[0]
        except:
            self.collab = 'error'
            print(f'Error updating collab on doc {self.id}')

    def extract_keywords(self):
        try:
            self.keywords = [kwd.text.lower() for kwd in self.tree.getroot()[2].find('kwdg').iter('kwd')]
        except:
            self.keywords = []
            # print(f'Error updating keywords on {self.id}')

    def extract_all_metadata(self):
        self.extract_id()
        self.extract_year()
        self.extract_title()
        self.extract_source()
        self.extract_doctype()
        self.extract_issn()
        self.extract_url()
        self.extract_volume()
        self.extract_issue()
        self.extract_page()
        self.extract_doi()
        self.extract_authors()
        self.extract_collab()
        self.extract_keywords()
        self.extract_citation()

    def extract_citation(self):
        try:
            self.citation = ' '.join([
                self.collab,
                f'({self.year}).',
                self.title + '.',
                self.source.title() + ',',
                self.volume + '(' + self.issue + '),',
                self.page + '.',
                f'https://doi.org/{self.doi}' if self.doi != 'error' else ''
            ])
        except:
            print(f'Error extracting citation for doc {self.id}')
            self.citation = 'error'

    # Metadata from metadata + mappings
    def extract_doctype_cat(self, doctype_cats_mapping):
        try:
            self.doctype_cat = doctype_cats_mapping[self.doctype]
        except:
            self.doctype_cat = 'error'
            print(f'Error updating doctype cat on {self.id}. Base doctype: {self.doctype}')

    def extract_primary_subjects(self, primary_subjects_mapping):
        try:
            self.primary_subjects = primary_subjects_mapping[self.source]
        except:
            self.primary_subjects = []
            print(f'Error updating primary subjects on {self.id}. Source: {self.source}. Issn: {self.issn}')

    def extract_secondary_subjects(self, secondary_subjects_mapping):
        try:
            self.secondary_subjects = secondary_subjects_mapping[self.source]
        except:
            self.secondary_subjects = []
            print(f'Error updating secondary subjects on {self.id}. Source: {self.source}. Issn: {self.issn}')

    # Text and tags
    def extract_abstract(self, trash_sections):
        self.raw_abs_paragraphs = self.extract_content_paragraphs('.fm/abs', trash_sections)

    def extract_abstract_test(self, trash_sections):
        return self.extract_content_paragraphs('.fm/abs', trash_sections)

    def extract_text(self, trash_sections):
        self.raw_text_paragraphs = self.extract_content_paragraphs('bdy', trash_sections)

    def treetag_abstract(self, tagger):
        self.tt_abs_paragraphs = self.treetag_paragraphs(self.raw_abs_paragraphs, tagger)

    def treetag_text(self, tagger):
        self.tt_text_paragraphs = self.treetag_paragraphs(self.raw_text_paragraphs, tagger)

    def make_token_counts(self):

        self.abs_tokens = len(self.get_abs_tags(flatten=True))
        self.text_tokens = len(self.get_text_tags(flatten=True))

        self.abs_words = len([t for t in self.get_abs_tags(True) if t.pos not in TT_EXCLUDED_TAGS])
        self.text_words = len([t for t in self.get_text_tags(True) if t.pos not in TT_EXCLUDED_TAGS])

    ### work and process methods ###


    def extract_content_paragraphs(self, work_section, trash_sections, min_para_len=5):
        bdy = self.tree.find(work_section)
        try:

            for sec in trash_sections:
                for element in bdy.iter(sec):
                    tail = element.tail
                    element.clear()
                    element.tail = tail

            para_list = list(filter(lambda x: len(x) > min_para_len, [''.join(t for t in para.itertext()) for para in bdy.iter('p')]))
        except:
            print(f'Error processing {work_section} on {self.filename}')
            para_list = ['error']
            success = False
        if len(para_list) == 0:
            # print(f'No {work_section} for file {self.id}')
            para_list = ['no text']
        return para_list

    ### TreeTagger preprocess and parse ###

    # Process chaque paragraphe avec TreeTagger pour les transformer en listes de tags
    # Prend une liste [str, str, str,str]
    # Retourne une liste [ [tag, tag, tag], [tag, tag, tag] ]
    def treetag_paragraphs(self, paragraphs, tagger):
        try:
            tt_tags = [make_tags(tagger.tag_text(para.lower()), exclude_nottags=True) for para in paragraphs]
        except:
            print(f'Treetagging error on id: {self.id}')
            tt_tags = []
        return tt_tags

    def _bibl_metadata_extractor(self, tag: str):
        try:
            return self.tree.getroot()[2][1].find(tag).text.lower().strip()
        except Exception:
            print(f'Error extracting {tag} on doc {self.id}')
            return 'error'

    def to_pickle(self, destination=None):
        save_to = destination if destination else self.file_path
        pickle.dump(self, open(save_to, 'wb'))

    def metadata_to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'source': self.source,
            'doctype': self.doctype,
            'year': self.year,
            'issn': self.issn,
            'authors': self.authors,
            'collab': self.collab,
            'doi': self.doi,
            'url': self.url,
            'volume': self.volume,
            'issue': self.issue,
            'page': self.page,
            'citation': self.citation
            # 'keywords': self.keywords,
            # 'abstract_paras': self.raw_abs_paragraphs,
            # 'text_paras': self.raw_text_paragraphs,
        }

    def __str__(self):
        return f'DocModel {self.id} - {self.title}'

    def __repr__(self):
        return {
            'id': self.id,
            'title': self.title,
            'source': self.source,
            'doctype': self.doctype,
            'year': self.year,
            'issn': self.issn,
            'authors': self.authors,
            'collab': self.collab,
            'doi': self.doi,
            'url': self.url,
            'volume': self.volume,
            'issue': self.issue,
            'page': self.page,
            'citation': self.citation
            # 'keywords': self.keywords,
            # 'abstract_paras': self.raw_abs_paragraphs,
            # 'text_paras': self.raw_text_paragraphs,
        }

    @classmethod
    def read_pickle(cls, path):
        return pickle.load(open(path, 'rb'))

    @classmethod
    def docmodel_generator(cls, path, vocal=True, conditions=None):
        for i, filename in enumerate(os.listdir(path)):
            if filename.split('.')[1] != 'p':
                print(f'Ignored [{filename}] due to wrong file extension')
                continue
            with open(path / filename, 'rb') as f:
                try:
                    dm = pickle.load(f)
                    if not conditions or conditions(dm):
                        yield dm
                except EOFError:
                    print(f'Generator error on file: {filename}')
            if vocal and i % 5000 == 0:
                print(f'Generating {i}th docmodel')


if __name__ == '__main__':
    pass
    # test_path = SUB_K_DOCMODELS_DIR
    # srs_path = SUB_K_CORPUS_DIR
    # from mempy3.config import DOCMODELS_PATH
    # from mempy3.config import BASE_STORAGE_PATH
    # test_1 = '1465-9921-8-16.p'
    # test_1 = '1297-9686-44-13.p'
    #test_1 = '0778-7367-67-1-15.p'
    #dm = pickle.load(open(DOCMODELS_PATH / test_1, 'rb'))

    # for dm in DocModel.docmodel_generator(DOCMODELS_PATH):
    #     dm.extract_issn()
    #     dm.extract_keywords()
    #     dm.save_to_pickle(DOCMODELS_PATH / dm.filename)

