import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import IsolationForest
import STRING
from sklearn.preprocessing import StandardScaler


def isolation_forest(x, y, contamination=0.1, n_estimators=50, bootstrap=True, max_features=0.33, validation=[]):

    if contamination == 'auto':
        contamination = y.mean()
        print('Contamination Automatized to: %.2f\n' % contamination)

    db = IsolationForest(n_estimators=n_estimators, max_samples=500,
                         bootstrap=bootstrap, verbose=1, random_state=42,
                         contamination=contamination, max_features=max_features)
    db.fit(x)

    labels = db.predict(x)
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    print('CLUSTER NUMBERS ', n_clusters_)
    print(labels)
    labels = pd.DataFrame(labels, columns=['outliers'], index=y.index)
    labels.loc[labels['outliers'] == 1, 'outliers'] = 0
    labels.loc[labels['outliers'] == -1, 'outliers'] = 1

    precision = metrics.precision_score(y.values, labels.values)
    recall = metrics.recall_score(y.values, labels.values)
    fbeta = metrics.fbeta_score(y.values, labels.values, beta=2)

    print('PRECISION %.4f' % precision)
    print('RECALL %.4f' % recall)
    print('FB SCORE %.4f' % fbeta)

    if validation:
        assert validation[0].shape[1] > validation[1].shape[1], 'X valid has less columns than Y valid'
        predict_valid = db.predict(validation[0])
        predict_valid = pd.DataFrame(predict_valid, columns=['outliers'])
        predict_valid.loc[predict_valid['outliers'] == 1, 'outliers'] = 0
        predict_valid.loc[predict_valid['outliers'] == -1, 'outliers'] = 1

        print('PRECISION VALID %.4f' % metrics.precision_score(validation[1].values, predict_valid.values))
        print('RECALL VALID %.4f' % metrics.recall_score(validation[1].values, predict_valid.values))
        print('F1 SCORE VALID %.4f' % metrics.f1_score(validation[1].values, predict_valid.values))
    
    return labels, precision, recall, fbeta


if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)

    normal = pd.read_csv(STRING.normal_file, sep=';', encoding='latin1')
    anormal = pd.read_csv(STRING.anormal_file, sep=';', encoding='latin1')

    df = pd.concat([normal, anormal], axis=0)

    # NORMALIZE DATA
    normalize = pd.concat([normal, anormal], axis=0)
    for i in normalize.drop(['oferta_id', 'target'], axis=1).columns.values.tolist():
        normalize[i] = normalize[i].map(float)
        normalize[i] = StandardScaler().fit_transform(normalize[i].values.reshape(-1, 1))

    df = normalize.copy()

    df = df.reset_index(drop=True)
    df['target'] = df['target'].map(float)
    y = df[['oferta_id', 'target']]
    df = df.drop(['oferta_id', 'target'], axis=1)

    # WE GET THE BEST LABELS OF ISOLATION FOREST
    labels, _, _, _ = isolation_forest(df, y.drop('oferta_id', axis=1), contamination='auto', validation=False,
                                       n_estimators=500,
                                       bootstrap=False, max_features=1.0)
