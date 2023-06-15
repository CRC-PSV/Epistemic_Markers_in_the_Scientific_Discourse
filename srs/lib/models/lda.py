from sklearn.decomposition import LatentDirichletAllocation

import pandas as pd
import pickle


class LdaModel(LatentDirichletAllocation):
    """Wrapper class for sklearn's LDA"""

    def __init__(
            self,
            model_name,
            docterm_df,
            n_components,
            doc_topic_prior,  # a
            topic_word_prior,  # b
            max_iter,
            learning_decay,
            random_state,
            learning_method
    ):

        #  Instance vars from constructor
        self.model_name = model_name
        self.n_topics = n_components
        self.docterm_df = docterm_df
        self.num_docs, self.num_words = self.docterm_df.shape
        self.is_fitted = False

        super().__init__(
            n_components=n_components,
            doc_topic_prior=doc_topic_prior,
            topic_word_prior=topic_word_prior,
            max_iter=max_iter,
            learning_decay=learning_decay,
            learning_method=learning_method,
            random_state=random_state
        )

    def fit(self, y=None, **kwargs):
        super().fit(self.docterm_df)
        self.is_fitted = True

    def get_topic_words_df(self, normalize=True):
        assert self.is_fitted, \
            'Error, trying to get topics words df from unfitted model! Run model.fit() and try again.'

        df = pd.DataFrame(
            self.components_,
            index=[f'topic_{i}' for i in range(self.n_topics)],
            columns=self.docterm_df.columns
        )

        return df.apply(lambda x: x / df.sum(axis=1)) if normalize else df

    def get_doc_topics_df(self, normalize=True):
        assert self.is_fitted, \
            'Error, trying to get doc topics df from unfitted model! Run model.fit() and try again.'

        df = pd.DataFrame(
            self.transform(self.docterm_df),
            index=self.docterm_df.index,
            columns=[f'topic_{i}' for i in range(self.n_topics)],
        )

        return df.apply(lambda x: x / df.sum(axis=1)) if normalize else df

    def get_word_weights_csv(self, num_words=100):
        topic_words_df = self.get_topic_words_df()
        csv = ''
        for topic in topic_words_df.index:
            csv += ', '.join(
                [word for word in topic_words_df.loc[topic].sort_values(ascending=False)[:num_words].keys()]) + '\n'
            csv += ', '.join([str(weight) for weight in
                              topic_words_df.loc[topic].sort_values(ascending=False)[:num_words].values]) + '\n'
        return csv

    def to_pickle(self, path):
        """Pickles the LdaModel object at the specified location."""

        pickle.dump(self, open(path, 'wb'))

    @classmethod
    def read_pickle(cls, path):
        return pickle.load(open(path, 'rb'))
