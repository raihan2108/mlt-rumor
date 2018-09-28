from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import load_all_cascades


def baseline_model(clf_name='nb'):
    all_conv = load_all_cascades('../cascades')
    '''corpus = Corpus()
    data_np, lengths_np, labels_np = corpus.tokenize(all_conv)'''
    tweets = [c['text'] for c in all_conv]
    stance_labels = [c['stance_label'] for c in all_conv]
    rumor_labels = [c['rumor_label'] for c in all_conv]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(tweets)
    print(X.shape)
    X_train, X_test, stance_train, stance_test, rumor_train, rumor_test = \
        train_test_split(X, stance_labels, rumor_labels, test_size=0.33)
    if clf_name == 'nb':
        clf_rumor = MultinomialNB()
        clf_stance = MultinomialNB()
    elif clf_name == 'svm':
        clf_rumor = LinearSVC()
        clf_stance = LinearSVC()
    else:
        clf_rumor = LogisticRegression()
        clf_stance = LogisticRegression()

    clf_rumor.fit(X_train, rumor_train)
    clf_stance.fit(X_train, stance_train)

    pred_rumor = clf_rumor.predict(X_test)
    pred_stance = clf_stance.predict(X_test)

    f1_micro_rumor = f1_score(rumor_test, pred_rumor, average='micro')
    f1_macro_rumor = f1_score(rumor_test, pred_rumor, average='macro')
    acc_rumor = accuracy_score(rumor_test, pred_rumor)
    f1_micro_stance = f1_score(stance_test, pred_stance, average='micro')
    f1_macro_stance = f1_score(stance_test, pred_stance, average='macro')
    acc_stance = accuracy_score(stance_test, pred_stance)

    print('test: micro f1 rumor:{:0.3f} macro f1 rumor:{:0.3f} accuracy rumor: {:0.3f}'.
          format(f1_micro_rumor, f1_macro_rumor, acc_rumor))
    print('test: micro f1 stance:{:0.3f} macro f1 stance:{:0.3f} accuracy stance: {:0.3f}'.
          format(f1_micro_stance, f1_macro_stance, acc_stance))


if __name__ == '__main__':
    baseline_model('svm')