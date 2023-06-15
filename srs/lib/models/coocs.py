"""Coocs!"""

from typing import Callable, Iterable, Optional
from collections import defaultdict, Counter
import pandas as pd
import pickle
import random

from srs.lib.docmodel import DocModel
from srs.lib.utils.io_utils import save_json


#  TODO update docstrings, added self.tag_attr and changed update() to take [tag] instead of [str]


class CoocsModel:
    """Object used to count word cooccurrences across a series of texts.

    Works from a vocabulary list. Counts all words cooccurring with each vocab word, within the specified window.
    Also tracks which combinations of vocab word cooccur at least once in each document, making it easy to retrieve the
    ids of all documents in which any specific combination of vocab words were found in the same cooccurrence.

    Ref tracking might consume a lot of memory on large corpus / vocabularies. Consider updating coocs only if it
    becomes a problem.

    Attributes
    ----------
    vocab: Iterable[str]
        The list of targeted words to count cooccurrences on.
    window: int
        How many words to consider in each direction when counting cooccurrences for a targeted word. The value is
        inclusive.
    coocs: defaultdict[Counter]
        Variable used to track the cooccurrences. Dict mapping each vocab word to a Counter tracking its cooccurring
        terms.
    refs: set[tuple]
        Collection of tuples tracking cooccurrence references.
    word_occs: Counter
        Tracks how many times each vocab word was found.

    """

    def __init__(self, vocab: list[str], window: int, tag_attr: str = 'lemma'):
        """CoocsCounter constructor,

        Parameters
        ----------
        vocab: Iterable[str]
            The list of targeted words to count cooccurrences on.
        window: int
            The cooccurrence window (inclusive).
        """

        self.tag_attr = tag_attr
        self.vocab = vocab
        self.window = window
        self.coocs = defaultdict(Counter)
        self.word_occs = Counter()
        self.pairs = [
            tuple(sorted([w1, w2])) for i, w1 in enumerate(self.vocab) for j, w2 in enumerate(self.vocab[i+1:])
        ]
        # Keys: (word_a, word_b) tuple. Word pairs are
        # values: para_id, n coocs for each pair. Counts are doubled since registered both for word1 and word2
        self.refs = defaultdict(Counter)

        # Will hold the shuffled ref ids for each coocs. Keys will be term pairs (tuple) and values list of unique ids
        # Build after updating with .shuffle_refs()
        self.shuffled_refs = {}

    def update(self, doc_id: str, tag_list: Iterable[str],
               update_coocs: Optional[bool] = True, update_refs: Optional[bool] = True):
        """Updates cooccurrence values and references with passed values.

        If update coocs: For each vocab word in the passed word_list, gets the words within the window and updates the
        counter.
        If update refs: If two or more vocab word are found within the same window, the doc reference will be recorded.

        Parameters
        ----------
        doc_id
            Unique identifier of the document. If working on paragraphs, paragraph number should be appended to doc id
            to make sure each one has a unique id.
        word_list: list-like of str
            List of strings representing the document's words.
        update_coocs: bool
            Whether to update cooccurrence counts
        update_refs: bool
            Whether to update vocab words cooccurrence references
        """

        for i, tag in enumerate(tag_list):
            word = getattr(tag, self.tag_attr)
            if word in self.vocab:
                self.word_occs.update([word])
                beg = max(i - self.window, 0)
                end = i + self.window + 1
                sequence = [getattr(w, self.tag_attr) for w in tag_list[beg:end] if getattr(w, self.tag_attr) != word]

                if update_coocs:
                    self.coocs[word].update(sequence)

                if update_refs:
                    # Update refs if at least 2 vocab words are found
                    # if sum(vals := [w in sequence for w in self.vocab]) >= 2:

                    for cooc in sequence:
                        if cooc in self.vocab:
                            self.refs[tuple(sorted([word, cooc]))].update([doc_id])

    def update_coocs_only(self, doc_id: str, tag_list: Iterable[any]):
        """Calls update with coocs only (id, word_list, True, False). Might be cleaner in some cases."""

        self.update(doc_id, tag_list, True, False)

    def update_refs_only(self, doc_id: str, tag_list: Iterable[any]):
        """Calls update with refs only (id, word_list, False, True). Might be cleaner in some cases."""

        self.update(doc_id, tag_list, False, True)

    def shuffle_refs(self, rnd_seed: int = 2112):
        random.seed(rnd_seed)

        for pair, counter in self.refs.items():
            para_ids = list(counter.keys())
            random.shuffle(para_ids)
            self.shuffled_refs[pair] = para_ids

    def as_df(self, filter_fct: Optional[Callable[[str], bool]] = None):
        """Returns a DataFrame with cooccurrence results

        Columns are vocab words (as specified on init) that were found at least once in update texts.
        Index are all words with at least one cooccurrence with a vocab word.
        """

        if filter_fct is not None:
            filtered_cooc_terms = list(filter(filter_fct, {word for counter in self.coocs.values() for word in counter.keys()}))
            return pd.DataFrame(self.coocs, index=filtered_cooc_terms)
        else:
            return pd.DataFrame(self.coocs)

    def export_ref_samples(self, dm_path, save_path, n_samples=20, words_to_sample: Optional[list] = None):
        """Saves cooc samples references as json

        word_to_sample: Optional, if provided, only coocs containing at least one word from the list will be considered
        """

        ref_samples = {}
        for pair, refs in self.shuffled_refs.items():
            if words_to_sample is None or any(word in pair for word in words_to_sample):
                data = [CoocsModel.make_ref_dict(ref, pair, dm_path) for ref in refs[:n_samples]]
                pair_name = f'{pair[0]}_{pair[1]}'
                ref_samples[pair_name] = data
        save_json(save_path, ref_samples)

    def to_pickle(self, path):
        """Pickles the LexCounter object at the specified location."""

        pickle.dump(self, open(path, 'wb'))

    @classmethod
    def read_pickle(cls, path):
        return pickle.load(open(path, 'rb'))

    @classmethod
    def make_ref_dict(cls, ref, words, dm_path):

        # Assumes ref is the article id and para num split by an underscore
        if '_' in ref:
            doc_id, para_num = ref.split('_')
            para_num = int(para_num)
            flatten = False
        else:
            doc_id = ref
            para_num = None
            flatten = True

        dm = DocModel.read_pickle(dm_path / f'{doc_id}.p')
        return {
            'id': doc_id,
            'words': words,
            'para_num': para_num,
            'tot_paras': len(dm.get_raw_text()),
            'title': dm.title,
            'collab': dm.collab,
            'source': dm.source,
            'year': dm.year,
            'citation': dm.citation,
            'para_text': '' if flatten else dm.get_raw_text()[para_num],
            'abs_text': dm.get_raw_abs()
        }



