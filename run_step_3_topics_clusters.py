from sklearn.cluster import MiniBatchKMeans
import pandas as pd

from srs.config import RESULTS_PATH, RND_SEED
from srs.lib.models.lda import LdaModel


def step_3_lda_clusters():

    print('Running LDA topic modeling and Kmeans clustering from the abstracts docterm matrix.')
    # Load docterm
    dt_df = pd.read_pickle(RESULTS_PATH / 'abstracts_docterm_df.p')

    print('DocTerm matrix loaded, proceeding to topic modeling.')
    # create LdaModel: params
    lda_model = LdaModel(
        'lda_model',
        dt_df,
        n_components=80,
        doc_topic_prior=0.2,  # alpha
        topic_word_prior=0.02,  # beta
        max_iter=100,
        learning_decay=0.9,
        random_state=RND_SEED,
        learning_method='batch',
    )

    lda_model.fit()
    # lda_model.to_pickle(RESULTS_PATH / 'lda/lda_model.p')

    doc_topics_df = lda_model.get_doc_topics_df()
    topic_words_df = lda_model.get_topic_words_df()

    doc_topics_df.to_pickle(RESULTS_PATH / 'doc_topics_df.p')
    topic_words_df.to_pickle(RESULTS_PATH / 'topic_words_df.p')
    print('LDA topic model dataframes saved to results, proceeding to Kmeans clustering...')
    #csv = lda.get_word_weights_csv()
    #with open(LDA_PATH / 'topic_word_probs.csv', 'wb') as f:
    #   f.write(csv.encode('utf-8'))

    # Cluster
    k = MiniBatchKMeans(n_clusters=7, random_state=RND_SEED).fit(doc_topics_df)
    clusters = k.predict(doc_topics_df)

    doc_cluster_series = pd.Series(index=doc_topics_df.index, data=clusters)
    doc_cluster_series = doc_cluster_series.map(lambda x: f'cluster_{x}')
    doc_cluster_series.to_pickle(RESULTS_PATH / 'doc_cluster_series.p')
    print('Clustering series saved to results')


def step_3_main():

    print('Starting step 3: disciplinary cluster analysis')
    step_3_lda_clusters()


if __name__ == '__main__':
    step_3_main()

