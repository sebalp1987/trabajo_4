import STRING
import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sns
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, f1_score, precision_score, recall_score, fbeta_score

sns.set()

metric_save = pd.read_csv(STRING.metric_save, sep=';')
df_prob_1 = pd.read_csv(STRING.path_db_extra + 'probability_save_1.csv', sep=';', dtype={'oferta_id': int}).dropna()
df_prob_2 = pd.read_csv(STRING.path_db_extra + 'probability_save_2.csv', sep=';', dtype={'oferta_id': int}).dropna()
test_sample_1 = pd.read_csv(STRING.test_sample_1, sep=';', dtype={'oferta_id': int})
test_sample_2 = pd.read_csv(STRING.test_sample_2, sep=';', dtype={'oferta_id': int})
df_prob_1 = pd.merge(df_prob_1, test_sample_1[['oferta_id', 'target']], how='left', on='oferta_id')
df_prob_2 = pd.merge(df_prob_2, test_sample_2[['oferta_id', 'target']], how='left', on='oferta_id')


metric_save = metric_save[['model', 'threshold', 'fbscore', 'sample']]
metric_save_1 = metric_save[metric_save['sample'] == 1]
metric_save_2 = metric_save[metric_save['sample'] == 2]

dict_thresh_1 = {}
dict_thresh_2 = {}

for col in df_prob_1.drop(['oferta_id', 'target'], axis=1).columns.values.tolist():
    print(col)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(df_prob_1[col].values.reshape(-1, 1))
    df_prob_1[col] = scaler.transform(df_prob_1[col].values.reshape(-1, 1))
    print(metric_save_1.loc[metric_save_1['model'] == col, 'threshold'])
    thresh = metric_save_1.loc[metric_save_1['model'] == col, 'threshold'].values[0]
    thresh = scaler.transform(thresh.reshape(-1, 1))
    dict_thresh_1[col] = thresh

for col in df_prob_2.drop(['oferta_id', 'target'], axis=1).columns.values.tolist():
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(df_prob_2[col].values.reshape(-1, 1))
    df_prob_2[col] = scaler.transform(df_prob_2[col].values.reshape(-1, 1))
    thresh = metric_save_2.loc[metric_save_2['model'] == col, 'threshold'].values[0]
    thresh = scaler.transform(thresh.reshape(-1, 1))
    dict_thresh_2[col] = thresh

df_prob_1.to_csv(STRING.test_sample_1_normalize, sep=';', index=False)
df_prob_2.to_csv(STRING.test_sample_2_normalize, sep=';', index=False)

colors = ['lightcoral', 'seagreen', 'green']
print(df_prob_1['target'])
for model in df_prob_1.drop(['oferta_id', 'target'], axis=1).columns.values.tolist():
    print(model)
    auc = roc_auc_score(df_prob_1[['target']].values, df_prob_1[[model]].values)
    print('AUC: %.3f' % auc)
    fpr, tpr, thresholds = roc_curve(df_prob_1[['target']].values, df_prob_1[[model]].values,
                                     drop_intermediate=True)
    if model in ['nn', 'ic', 'rc']:
        linestyle = ':'
        color = colors[2]
    elif model == 'ert':
        linestyle = '-.'
        color = colors[1]
    else:
        linestyle = '-'
        color = colors[0]
    plot.plot(fpr, tpr, color=color, lw=2,
              label='ROC curve of class {0} (area = {1:0.2f})'''.format(model, auc), linestyle=linestyle)

plot.plot([0, 1], [0, 1], 'k--', lw=2)
plot.xlim([0.0, 1.0])
plot.ylim([0.0, 1.05])
plot.xlabel('False Positive Rate')
plot.ylabel('True Positive Rate')
plot.legend(loc="lower right")
plot.savefig(STRING.img_path + 'roc_curve_1.png')
plot.show()
print(dict_thresh_1)
for model in df_prob_1.drop(['oferta_id', 'target'], axis=1).columns.values.tolist():
    threshold = dict_thresh_1.get(model)[0][0]
    y_hat = (df_prob_1[model].values > threshold).astype(int)
    print(model)
    print('FBETA', fbeta_score(y_pred=y_hat, y_true=df_prob_1.target.values, beta=1))

    f1 = metric_save_1.loc[metric_save_1['model'] == model, 'fbscore'].values[0]
    prec = []
    rec = []
    tresholds = np.linspace(0.001, 1.0, 1000)
    for trh in tresholds:
        y_hat = (df_prob_1[model].values > trh).astype(int)
        prec.append(precision_score(y_pred=y_hat, y_true=df_prob_1.target.values))
        rec.append(recall_score(y_pred=y_hat, y_true=df_prob_1.target.values))
    print(rec)
    print(prec)
    if model in ['nn', 'ic', 'rc']:
        linestyle = ':'
        color = colors[2]
    elif model == 'ert':
        linestyle = '-.'
        color = colors[1]
    elif model == 'vae':
        linestyle = '-'
    elif model == 'ae':
        linestyle = '-.'
        color = colors[0]
    plot.plot(rec, prec, color=color, lw=2,
              label='Precision-Recall curve of class {0} (area = {1:0.2f})'''.format(model, f1), linestyle=linestyle)

plot.title('Recall vs Precision')
plot.xlabel('Recall')
plot.ylabel('Precision')
plot.legend(loc="lower right")
plot.savefig(STRING.img_path + 'pecrec_curve_1.png')
plot.show()
