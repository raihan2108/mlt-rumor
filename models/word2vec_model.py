import numpy as np

from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize

from utils import load_all_cascades


def tag_docs(tweet, label):
    pass


def baseline_model(clf_name='nb'):
    all_conv = load_all_cascades('../cascades')
    '''corpus = Corpus()
    data_np, lengths_np, labels_np = corpus.tokenize(all_conv)'''
    tweets = [c['text'] for c in all_conv]
    stance_labels = [c['stance_label'] for c in all_conv]
    rumor_labels = [c['rumor_label'] for c in all_conv]

    tagged_tweet = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(tweets)]
    X_train, X_test, stance_train, stance_test, rumor_train, rumor_test = \
        train_test_split(tagged_tweet, stance_labels, rumor_labels, test_size=0.33)

    model = Doc2Vec(vector_size=256, alpha=0.025, min_alpha=0.025, window=5,
                            min_count=10, workers=10, epochs=10)

    print('model initiated')
    model.build_vocab(tagged_tweet)
    print('vocabulary construction complete, start training')
    total_tokens = sum([len(sentence[0]) for sentence in X_train])
    for epoch in range(10):
        model.train(X_train, total_examples=len(X_train),
                    total_words=total_tokens, epochs=model.iter)
        model.alpha -= 0.002  # decrease the learning rate
        model.min_alpha = model.alpha
        print('end of epoch ' + str(epoch + 1))
    X = []

    for sentence in X_test:
        tokens = sentence[0]
        doc_vec = model.infer_vector(tokens)
        X.append(doc_vec)
    X = np.asarray(X)
    y_rumor = np.asarray(rumor_test)
    y_stance = np.asarray(stance_test)

    print(X.shape, y_rumor.shape, y_stance.shape)

    print('start training the classifier')
    if clf_name == 'nb':
        clf_rumor = MultinomialNB()
        clf_stance = MultinomialNB()
    elif clf_name == 'svm':
        clf_rumor = LinearSVC()
        clf_stance = LinearSVC()
    else:
        clf_rumor = LogisticRegression()
        clf_stance = LogisticRegression()

    # ovr_rumor = OneVsRestClassifier(clf_rumor).fit(X, y_rumor)
    clf_rumor.fit(X, y_rumor)
    y_pred_rumor = clf_rumor.predict(X)

    # ovr_stance = OneVsRestClassifier(clf_stance).fit(X, y_stance)
    clf_stance.fit(X, y_stance)
    y_pred_stance = clf_stance.predict(X)

    print('rumor: micro f1 score: %0.4f' % f1_score(y_rumor, y_pred_rumor, average='micro'))
    print('rumor: macro f1 score: %0.4f' % f1_score(y_rumor, y_pred_rumor, average='macro'))

    print('stance: micro f1 score: %0.4f' % f1_score(y_stance, y_pred_stance, average='micro'))
    print('stance: macro f1 score: %0.4f' % f1_score(y_stance, y_pred_stance, average='macro'))


if __name__ == '__main__':
    baseline_model('lr')