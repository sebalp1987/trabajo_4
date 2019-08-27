import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import IsolationForest
import STRING
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plot

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
    sample = 2
    normal = pd.read_csv(STRING.normal_file, sep=';', encoding='latin1')
    anormal = pd.read_csv(STRING.anormal_file, sep=';', encoding='latin1')

    df = pd.concat([normal, anormal], axis=0)

    df = df.drop(['oferta_sim_siniestro_5_anio_culpa', 'oferta_sim_anios_asegurado',
                        'oferta_sim_antiguedad_cia_actual', 'oferta_sim_siniestro_1_anio_culpa',
                        'oferta_bonus_simulacion_perc'], axis=1)

    # Final test file, we remove the cases from x
    test_sample_1 = pd.read_csv(STRING.test_sample_1, sep=';')
    test_sample_2 = pd.read_csv(STRING.test_sample_2, sep=';')

    test_sample_1 = test_sample_1[['oferta_id', 'target']]
    test_sample_2 = test_sample_2[['oferta_id', 'target']]

    test_sample_1 = df[df['oferta_id'].isin(test_sample_1['oferta_id'].values.tolist())]
    test_sample_2 = df[df['oferta_id'].isin(test_sample_2['oferta_id'].values.tolist())]

    if sample == 1:
        df = df[-df['oferta_id'].isin(test_sample_1['oferta_id'].values.tolist())]
    if sample == 2:
        df = df[-df['oferta_id'].isin(test_sample_2['oferta_id'].values.tolist())]
    df = df.reset_index(drop=True)
    df['target'] = df['target'].map(float)

    y = df[['oferta_id', 'target']]
    df = df.drop(['oferta_id', 'target'], axis=1)
    y_test_sample_1 = test_sample_1[['oferta_id', 'target']]
    x_test_sample_1 = test_sample_1.drop(['oferta_id', 'target'], axis=1)
    y_test_sample_2 = test_sample_2[['oferta_id', 'target']]
    x_test_sample_2 = test_sample_2.drop(['oferta_id', 'target'], axis=1)

    # NORMALIZE DATA
    for i in df.columns.values.tolist():
        scaler = StandardScaler()
        scaler.fit(df[i].values.reshape(-1, 1))
        df[i] = scaler.transform(df[i].values.reshape(-1, 1))
        x_test_sample_1[i] = scaler.transform(x_test_sample_1[i].values.reshape(-1, 1))
        x_test_sample_2[i] = scaler.transform(x_test_sample_2[i].values.reshape(-1, 1))

    # WE GET THE BEST LABELS OF ISOLATION FOREST
    labels, _, _, _ = isolation_forest(df, y.drop('oferta_id', axis=1), contamination=0.06, validation=[x_test_sample_2,
                                                                                                        y_test_sample_2.drop(
                                                                                                            'oferta_id',
                                                                                                            axis=1)],
                                       n_estimators=500,
                                       bootstrap=False, max_features=1.0)
