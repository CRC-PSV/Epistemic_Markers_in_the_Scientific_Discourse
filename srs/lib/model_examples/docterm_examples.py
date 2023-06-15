from srs.lib.utils.examples_tests_utils import generate_test_id_tags
from srs.lib.models import DocTermModel
from srs.lib.utils.test_texts import OFFICE_TEST_SENTENCES


def docterm_example():
    # Init the DocTermModel
    # Will count the values of the 'lemma' attribute, except those starting with the letter 'w'
    dt = DocTermModel(tag_attr='lemma', update_filter_fct=lambda x: x.starts_with != 'w')

    # Iters trough the corpus and calls update() on each doc, passing a doc id and a list of tags
    for d, s in generate_test_id_tags(OFFICE_TEST_SENTENCES):
        dt.update(d, s)

    # Print the vocabulary (each word found at least once)
    print(dt.unique_words)

    # Print the docterm matrix as a pandas dataframe
    print(dt.as_df())

    # Filter the vocabulary. Words will be removed from unique_words and the resulting df, but not from the model itself.
    # In this example, all words starting with a vowel are removed
    dt.filter_words(lambda x: x[0] not in ['a', 'e', 'i', 'o', 'u', 'y'])

    # Prints the updated unique_words and df
    print(dt.unique_words)
    print(dt.as_df())


if __name__ == '__main__':
    docterm_example()

