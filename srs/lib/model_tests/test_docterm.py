"""Unit tests for DocTermModel"""
import unittest

from charting.lib.models import DocTermModel
from charting.lib.utils.test_texts import OFFICE_TEST_SENTENCES
from charting.lib.utils.examples_tests_utils import generate_test_id_tags


class DocTermTests(unittest.TestCase):

    @classmethod
    def setUpClass(self) -> None:
        self.dt = DocTermModel(tag_attr='lemma', )

        for s_id, tags in generate_test_id_tags(OFFICE_TEST_SENTENCES):
            self.dt.update(s_id, tags)

        self.base_df = self.dt.as_df()
        self.base_unique_words = self.dt.unique_words

        self.removed_words = ['the', 'you', 'asd']
        self.dt.filter_words(lambda x: x not in self.removed_words)
        self.filtered_df = self.dt.as_df()
        self.filtered_unique_words = self.dt.unique_words

    def test_general(self):
        # Check unique words before and after filter
        self.assertEqual(len(self.filtered_unique_words), len(self.base_unique_words) - len(self.removed_words), f'Filter should have removed {len(self.removed_words)} words from unique_words')

        # Check for class metadata and df mismatches
        self.assertEqual(self.dt.total_updates, len(OFFICE_TEST_SENTENCES), 'Values of total_updates and len(TEST_DATA) should be equal')
        self.assertEqual(len(self.base_df), len(OFFICE_TEST_SENTENCES) - 1, 'Values of len(base_df) should be equal to the number of non empty docs in TEST_DATA')
        self.assertEqual(len(self.base_unique_words), len(self.base_df.columns), 'The len of base_unique_words and base_df.columns should be equal')
        self.assertEqual(len(self.filtered_unique_words), len(self.filtered_df.columns), 'The len of filtered_unique_words and filtered_df.columns should be equal')

        # Check if empty text is in df (should not be)
        self.assertFalse('doc_24' in self.base_df.index, 'Empty doc \'doc_24\' should not have a row in the df')

    def test_doc_total_words(self):
        # Check total word counts within individual docs
        self.assert_row_total('doc_0', 10, 10)
        self.assert_row_total('doc_1', 8, 6)
        self.assert_row_total('doc_5', 9, 6)
        self.assert_row_total('doc_17', 1, 1)
        self.assert_row_total('doc_18', 6, 3)
        self.assert_row_total('doc_20', 4, 0)

    def test_doc_unique_words(self):
        # Check number of unique words within individual docs
        self.assert_row_unique('doc_0', 9, 9)
        self.assert_row_unique('doc_1', 7, 6)
        self.assert_row_unique('doc_5', 8, 6)
        self.assert_row_unique('doc_17', 1, 1)
        self.assert_row_unique('doc_18', 2, 1)
        self.assert_row_unique('doc_20', 1, 0)

    def test_word_total_uses(self):
        # Check the total occurrences of a word across all docs
        self.assert_col_total('dont', 2, 2)
        self.assert_col_total('i', 12, 12)
        self.assert_col_total('asd', 11, 0)
        self.assert_col_total('you', 4, 0)
        self.assert_col_total('weirduniqueword', 1, 1)

    def test_word_unique_docs(self):
        # Check the number of docs in which words appears at least once
        self.assert_col_unique('dont', 2, 2)
        self.assert_col_unique('i', 10, 10)
        self.assert_col_unique('asd', 3, 0)
        self.assert_col_unique('you', 2, 0)
        self.assert_col_unique('weirduniqueword', 1, 1)

    def assert_row_total(self, row_name, base_n, filtered_n):
        self.assertEqual(self.base_df.loc[row_name].sum(), base_n, f'Base doc \'{row_name}\' should have {base_n} total words')
        if filtered_n > 0:
            self.assertEqual(self.filtered_df.loc[row_name].sum(), filtered_n, f'Base doc \'{row_name}\' should have {filtered_n} total words')
        else:
            self.assertFalse(row_name in self.filtered_df.index)

    def assert_row_unique(self, row_name, base_n, filtered_n):
        self.assertEqual((self.base_df.loc[row_name] > 0).sum(), base_n, f'Base doc \'{row_name}\' should have {base_n} unique words')
        if filtered_n > 0:
            self.assertEqual((self.filtered_df.loc[row_name] > 0).sum(), filtered_n, f'Base doc \'{row_name}\' should have {filtered_n} unique words')
        else:
            self.assertFalse(row_name in self.filtered_df.index)

    def assert_col_total(self, col_name, base_n, filtered_n):
        self.assertEqual(self.base_df[col_name].sum(), base_n, f'Word \'{col_name}\' should have been found {base_n} times in base df')
        if filtered_n > 0:
            self.assertEqual(self.filtered_df[col_name].sum(), filtered_n, f'Word \'{col_name}\' should have been found {filtered_n} times in filtered df')
        else:
            self.assertFalse(col_name in self.filtered_df.columns)

    def assert_col_unique(self, col_name, base_n, filtered_n):
        self.assertEqual((self.base_df[col_name] > 0).sum(), base_n, f'Word \'{col_name}\' should have been present in {base_n} docs in base df')
        if filtered_n > 0:
            self.assertEqual((self.filtered_df[col_name] > 0).sum(), filtered_n, f'Word \'{col_name}\' should have been present in {filtered_n} docs in base df')
        else:
            self.assertFalse(col_name in self.filtered_df.columns)


if __name__ == '__main__':
    unittest.main()

