from typing import Optional, Mapping, Iterable
from collections import Counter
import pandas as pd
import pickle


class LexCounter:
    """Object used to count the occurrences of words belonging to specific lexical categories across the corpus

    Uses a mapping that associates category names to lists of words. Will count the occurrences of each of these words
    in the passed documents (via update_lex_counts), and the totals can then be summed for each category.

    Attributes
    ----------
    lex_mapping: Mapping[str, Iterable[str]]
        A mapping (dict) representing the different categories and their associated words. Has category names as keys
        and list of words as values.
    lex_words: list[str]
        List of all the words across the different categories. Used internally.
    lex_counts: dict
        Dict holding the results, updated when calling update_lex_counts(). Has doc ids as keys and word counts as
        values (list[int], same size as lex_words).

    Methods
    -------
    update_lex_cats(doc_id, word_list)
        Updates lex_counts from passed doc id and word list
    as_df(merge_categories=True)
        Returns the lex counts as a dataframe, where index are the doc ids passed when updating, columns are the words
        in the lexicon and values are the number of occurrences of each word in each doc. If merge categories is true,
        columns belonging to the same lexical category will be summed.
    to_pickle(path)
        Pickles the LexCounter object.
    """

    def __init__(self, lex_mapping: Mapping[str, Iterable[str]]):
        """LexCounter constructor, must set the lexicon by passing a mapping.

        Parameters
        ----------
        lex_mapping: Mapping[str, Iterable[str]]
            A mapping (dict) representing the different categories and their associated words. Has category names as
            keys and list of words as values.
        """

        self.lex_mapping = lex_mapping
        self.lex_words = [word for words in self.lex_mapping.values() for word in words]
        self.lex_counts = {}
        self._check_lex_mapping()

    def update(self, doc_id: str, word_list: Iterable[str]):
        """Updates lex_counts from a doc id and a word list

        Counts the occurrences of lexicon words in word_list and updates lex_counts[doc_id]. If a doc with the same id
        was already processed, new values will replace the old ones.

        Parameters
        ----------
        doc_id
            Unique identifier of the document. If working on paragraphs, paragraph number should be appended to doc id
            to make sure each one has a unique id.
        word_list: list-like of str
            List of strings representing the document's words.
        """

        c = Counter(word_list)
        self.lex_counts.update({doc_id: [c[word] for word in self.lex_words]})

    def as_df(self, merge_categories: Optional[bool] = True, sort_columns: Optional[bool] = True):
        """Returns the lex counts as a dataframe, with or without merging words belonging to the same category.

        Index are the doc ids passed when updating, columns are the words in the lexicon and values are the number of
        occurrences of each word in each doc. If merge categories is true, columns belonging to the same lexical
        category will be summed.

        Parameters
        ----------
        merge_categories: Optional[bool], default: True
            Whether to sum columns belonging to the same category
        sort_columns: Optional[bool], default:True
            Whether to alphabetically sort the columns.
        Returns
        -------
        pandas.DataFrame
            The lexical counts as a dataframe, as described above.
        """

        df = pd.DataFrame.from_dict(self.lex_counts, orient='index', columns=self.lex_words, dtype='UInt16')

        if merge_categories:
            for cat, words in self.lex_mapping.items():
                df[cat] = sum(df[w] for w in words if w in df)
            df.drop([col for col in df.columns if col not in self.lex_mapping.keys()], axis=1, inplace=True)

        return df.reindex(sorted(df.columns), axis=1) if sort_columns else df

    def to_pickle(self, path):
        """Pickles the LexCounter object at the specified location."""

        pickle.dump(self, open(path, 'wb'))

    def _check_lex_mapping(self):
        """Private method, tests the lexicon on init to avoid problems down the line

        Throws an error if a category has no associated words.
        Throws an error if a word is found more than once, either in the same category or different ones.
        Prints a warning for each category name not found in its own word list.
        """

        assert not any(len(words) < 1 for words in self.lex_mapping.values()), \
            'Error, lex mapping has categories with no words!'

        c = Counter(self.lex_words)
        print(f'Duplicate words: {[w for w, n in c.items() if n > 1]}')
        #assert len(self.lex_words) == len(set(self.lex_words)), \
        #    'Error, lexicon mapping contains duplicate words (either within the same category or different ones)'

        for cat, words in self.lex_mapping.items():
            if cat not in words:
                print(f'Warning, word \"{cat}\" was not found in category \"{cat}\", perhaps it should be added?')

    @classmethod
    def read_pickle(cls, path):
        return pickle.load(open(path, 'rb'))
