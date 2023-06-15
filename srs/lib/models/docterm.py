"""Contains stuff to make docterm matrix"""

from typing import Optional, Iterable, Callable
from collections import Counter
import pandas as pd
import numpy as np
import pickle


class DocTermModel:
    def __init__(self, tag_attr: str = 'lemma', update_filter_fct: Optional[Callable[[any], bool]] = None):
        self.doc_word_counts = {}
        self.unique_words = set()
        self.tag_attr = tag_attr
        self.filter_fct = update_filter_fct if update_filter_fct is not None else lambda x: True
        self.total_updates = 0

    def update(self,
               doc_id: str,
               tag_list: Iterable[any],
               ) -> None:
        c = Counter(getattr(tag, self.tag_attr) for tag in tag_list if self.filter_fct(tag))
        self.unique_words.update(c.keys())
        self.doc_word_counts.update({doc_id: c})
        self.total_updates += 1

    def filter_words(self, filter_fct: Callable[[any], bool]):

        self.unique_words = {w for w in self.unique_words if filter_fct(w)}

    def as_df(self, log_norm: bool = False):
        df = pd.DataFrame.from_dict(self.doc_word_counts, orient='index',
                                    columns=list(self.unique_words), dtype='UInt16').fillna(0)
        if log_norm:
            df = df.apply(lambda x: np.log(x + 1))
        return df

    def to_pickle(self, path):
        """Pickles the DocTermCounter object at the specified location."""

        pickle.dump(self, open(path, 'wb'))

    @classmethod
    def read_pickle(cls, path):
        return pickle.load(open(path, 'rb'))


if __name__ == '__main__':

    pass

