from dataclasses import dataclass, field
from typing import Iterable, Tuple
import re

from charting.lib.utils.test_texts import OFFICE_TEST_SENTENCES


@dataclass
class TestTag:
    """Example Tag class to use for example/test purpose

    Simply init by passing it a string (a word), e.g. t = TestTag('word').
    The lemma property is the word converted to lower case with non-letters removed, or '#NA' if no letters left.
    Also has the starts_with property (first character), which can be used as a filter value in tests or examples.
    """

    word: str
    lemma: str = field(init=False)
    starts_with: str = field(init=False)

    def __post_init__(self):
        self.lemma = re.sub(r'[^\w]', '', self.word.lower())
        if self.lemma == '':
            self.lemma = '#NA'
        self.starts_with = self.word[0] if len(self.word) > 0 else ''

    def __str__(self):
        return f'TestTag({self.word}, {self.lemma}, {self.starts_with})'


def tag_words(words: Iterable[str]) -> list[TestTag]:
    """Takes a list of words and converts them to TestTags.

    Args:
        words: A list of words

    Returns:
        A list of TestTags
    """

    return [TestTag(w) for w in words if w != '']


def tag_text(text: str) -> list[TestTag]:
    """Take a text (str) and converts it to a list of TestTags by calling str.split(' ')

    Args:
        text: A text or sentence, as a string with words split by spaces.

    Returns:
        The text as a list of TestTags
    """

    return tag_words(text.split(' '))


def generate_test_id_tags(texts: Iterable[str], id_prefix: str = 'doc_') -> Tuple[str, list[TestTag]]:
    """Generator, yields a text id and a tagged text (as a lists of TestTags) from a list of texts or sentences.

    Expects regular text, i.e. strings in which words are separated by spaces. All non alphanumeric characters are
    removed.

    Args:
        texts: A list of texts or sentences
        id_prefix: The doc id prefix, will be followed by the i value from enumerate()

    Returns:
        Yields two values, for each text: an id ("sent_N") and the text as a list of TestTags
    """

    for i, s in enumerate(texts):
        yield f'{id_prefix}{i}', tag_text(s)


if __name__ == '__main__':
    tagged_sents = [s for s in generate_test_id_tags(OFFICE_TEST_SENTENCES)]
    print(tagged_sents[0])
    print(tagged_sents[-1])

