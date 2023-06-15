from collections import Counter, defaultdict
from typing import Callable, Iterable, Optional, Any
import pandas as pd
import pickle


class TagCountsModel:
    """Util object for counting lexical occurrences in lists of tags

    Holds two counters (collections -> Counter), one which tracks total instances of each value in the tag lists, and
    the other the total number of lists in which each value is found at least once. Each list must be fed to one ot the
    update methods. Values can be accessed as counters, or returned as pandas dataframes or csv str.

    Also allows to track a secondary attribute. If secondary_attr is specified on init, a defaultdict(Counter) will be
    used to count the secondary attribute values for each values. For example, specifying 'pos' as secondary_attr allows
    to track the pos counts for each individual value. These counts will be added as columns in the dataframe or csv,
    where each different value will be represented in a column. Secondary values should therefore have a limited set of
    possible values, for example counting POS tags for words or lemmas, lemmas for each word, etc.

    PROBLEM: secondary wont work for lemmas. Should instead add a number of cols == to the most values possible, and
    then just add a sorted list for each and padd with none.

    Attributes
    ----------
    total_counts: Counter
        A Counter object tracking the total instances of each value in the tag lists.
    presence_counts: Counter
        A Counter object tracking the total number of tag lists in which each value is found at least once. For example,
        if each tag list represents a document, this will count the number of docs in which each tag (or word or lemma)
        is found at least once.
    secondary_counts: defaultdict[Counter]
        A defaultdict tracking the secondary counts for each value.
    tag_attr: str, optional
        The tag attribute from which to get the values to count. Assumes tags have named attributes,
        e.g. if using TreeTagger tags are named tuples with attributes such as 'lemma' and 'word'. (default is 'lemma')
    secondary_attr: str, optional
        The attribute for the secondary values to count. Works similar to tag_attr but the count will be performed
        for each different value of tag_attr. (default is None)

    Methods
    -------
    update_counts(tag_list, transform_fct=lambda x: x, filter_fct=None)
        Processes a tag list and updates counters

    filter_values(filter_fct)
        Filters the values (keys) in the counters, should usually be called after counting.

    as_df()
        Returns the counters as a pandas dataframe

    as_csv()
        Returns the counters as a csv
    """

    def __init__(self, tag_attr: str = 'lemma', secondary_attr: str = None,
                 update_filter_fct: Optional[Callable[[Any], bool]] = None,
                 tranform_fct: Optional[Callable[[str], str]] = None):
        self.total_counts = Counter()
        self.presence_counts = Counter()
        self.tag_attr = tag_attr
        self.total_updates = 0

        self.secondary_counts = defaultdict(Counter)
        self.secondary_attr = secondary_attr

        self.filter_fct = update_filter_fct if update_filter_fct is not None else lambda x: True
        self.transform_fct = tranform_fct if tranform_fct is not None else lambda x: x

    def update(self,
               tag_list: Iterable[any],
               ) -> None:
        """Processes a tag list and updates the counters

        Takes a tag list (or an iterable of tags) and updates the counters with the values from tag_attr for each tag.
        Can pass a transform function to transform these values (e.g. lower case) or to filter (e.g. on another
        attribute like 'pos' or for min length).

        Parameters
        ----------
        tag_list: iterable
            Iterable of tag objects (named tuples if using Treetagger) with named attributes. Each object must have a
            named attribute corresponding to tag_attr, which is set on init.
        transform_fct: callable
            Function to transform values before counting. Values will be passed as an argument, and the function must
            return the new value. Ex to transform to lower case: lambda x: x.lower() (default is None)
        filter_fct: callable returns bool
            Function to filter tags before counting them. Tags are passed as argument and will only be counted if the
            function returns True. Ex: lambda x: x.pos in ACCEPTED_POS_TAGS (default is None)
        """

        vals = []
        for tag in tag_list:
            if self.filter_fct(tag):
                value = self.transform_fct(getattr(tag, self.tag_attr))
                vals.append(value)
                if self.secondary_attr is not None:
                    self.secondary_counts[value].update({getattr(tag, self.secondary_attr): 1})

        self.total_counts.update(vals)
        self.presence_counts.update(set(vals))
        self.total_updates += 1

    def filter_values(self, filter_fct: Callable[[any], bool]) -> None:
        """Filters values (keys) in counters

        Tests each value (key) in the counters against the passed function. Each value is passed to the function, and
        only those returning true are kept. In other words, it filters values based on some criterion. Should usually be
        called after updating. Does a similar job as filter_fct in update_counts(), but is only called once for each
        unique value instead of once per tag, which makes it faster for heavier operations.

        Parameters
        ----------
        filter_fct: Callable[[any], bool]
            The function to filter values with. Should take the values as input and return a bool. Only values returning
            True will be kept
        """

        for value in list(self.total_counts.keys()):
            if not filter_fct(value):
                del self.total_counts[value]
                del self.presence_counts[value]
                if self.secondary_attr is not None:
                    del self.secondary_counts[value]

    def as_df(self, max_sec_cols: int = 30) -> pd.DataFrame:
        """Returns the counters as a pandas dataframe

        Makes a dataframe from the counters. Will use the values (counters' keys) as index and have two columns
        corresponding to total_counts and presence_counts. If a secondary_attr was specified, will add a column for
        each possible value in secondary_counts. This may take a good while to run on larger datasets.

        Parameters
        ----------
        max_sec_cols: int
            The max number of columns to add before condensing secondary attribute data. If the total number of
            different values in self.secondary_counts is lower than max_sec_col, each value will have its own column
            showing how frequent are each secondary values for each primary values. Else, the columns will only list
            the different secondary values for each primary value, without showing the totals.

            For example, if the primary values are words and secondary values are POS tags:
            If there are 30 or less different POS tags across the data, a column will be added for each one, with the
            value of each cell representing the number of times each word was fond with each tag.
            Else, if there are more than 30 different POS tags, the new columns will list which tags were found at least
            once for each word. The number of columns equals to the maximum number of different tags found for a word.
            For words with fewer pos tags, extra columns are filled wiht an empty string.

        Returns
        -------
        pandas.DataFrame
            Dataframe with values as index, and counts as columns ('total_counts' and 'presence_counts'). More columns
            will be added if a secondary attribute was specified, see above for details.
        """

        df = pd.DataFrame({'total_counts': pd.Series(self.total_counts),
                           'article_counts': pd.Series(self.presence_counts)})

        if self.secondary_attr is not None:
            unique_sec_vals = sorted({v for c in self.secondary_counts.values() for v in c.keys()})
            if len(unique_sec_vals) > max_sec_cols:
                most_sec_vals = max(len(c) for c in self.secondary_counts.values())
                cols = [f'{self.secondary_attr}_{i}' for i in range(most_sec_vals)]
                d = {v: list(c.keys()) + [''] * (most_sec_vals - len(c.keys()))
                     for v, c in self.secondary_counts.items()}
                sf = pd.DataFrame.from_dict(d, orient='index', columns=cols)

            else:
                # d = {k: [v[col] for col in unique_sec_vals] for k, v in self.secondary_counts.items()}
                sf = pd.DataFrame.from_dict({k: [v[col] for col in unique_sec_vals]
                                             for k, v in self.secondary_counts.items()},
                                            columns=unique_sec_vals, orient='index')
            df = pd.concat([df, sf], axis=1)
        return df

    def as_csv(self):
        """Returns the counters as csv

        Csv is made by calling df.to_csv() on a DataFrame created by as_df().

        Returns
        -------
        str
            a string representing the data as a csv.
        """

        return self.as_df().to_csv()

    def to_pickle(self, path):
        """Pickles the TagCounter object at the specified location."""

        pickle.dump(self, open(path, 'wb'))

    @classmethod
    def read_pickle(cls, path):
        return pickle.load(open(path, 'rb'))

