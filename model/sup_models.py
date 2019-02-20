import pandas as pd
import STRING
import numpy as np
import matplotlib.pyplot as plot
import seaborn as sns

from resources.sampling import over_sampling, under_sampling

from model.neural_net import NeuralNetwork
from model.inception_model import InceptionModel
from model.residual_connection_model import ResidualConnection

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import recall_score, precision_score, fbeta_score, confusion_matrix, roc_auc_score, roc_curve, \
    precision_recall_curve
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Files
label = 'target'
train = pd.read_csv(STRING.train, sep=';', encoding='latin1')
valid = pd.read_csv(STRING.valid, sep=';', encoding='latin1')
test = pd.read_csv(STRING.test, sep=';', encoding='latin1')
test = pd.concat([train, valid, test], axis=0).reset_index(drop=True)
test = test.drop_duplicates(subset=['oferta_id'])

# Final test file, we remove the cases from x
test_sample_1 = pd.read_csv(STRING.test_sample_1, sep=';')
test_sample_2 = pd.read_csv(STRING.test_sample_2, sep=';')

test_sample_1 = test_sample_1[['oferta_id', 'target']]
test_sample_2 = test_sample_2[['oferta_id', 'target']]

# X, Y
x = test.drop([label] + ['oferta_id'], axis=1)
y = test[[label, 'oferta_id']]

# Parameters

seed = 42
np.random.seed(seed)
sns.set()
tresholds = np.linspace(0.001, 1.0, 1000)
scores = []
learning_rate = 0.001
epochs = 1000
batch_size = None
sampling = None
prob_dropout = 0.3
activation = 'tanh'
model = 'ert'
node_size = 100
steps_per_epoch = 15
validation_steps = 10
loss_function = 'binary_crossentropy'
sparsity_const = 10e-5
feature_range = (-1, 1)

# Variance Reduction
selection = VarianceThreshold(threshold=0.0)
selection.fit(x)
features = selection.get_support(indices=True)
features = x.columns[features]
x = x[features]

for i in x.columns.values.tolist():
    x[i] = x[i].map(float)
    scaler = MinMaxScaler(feature_range=(feature_range))
    scaler.fit(x[i].values.reshape(-1, 1))
    x[i] = scaler.transform(x[i].values.reshape(-1, 1))

cols = x.shape[1]
fileNames = np.array(x.columns.values)
columns = x.columns.values

# Models
models = {'ert': ExtraTreesClassifier(n_estimators=500, max_depth=10, bootstrap=True, oob_score=True,
                                      class_weight='balanced_subsample', max_features='sqrt', random_state=42,
                                      verbose=True),
          'nn': NeuralNetwork(n_cols=cols, node_size=[node_size, node_size], activation=activation,
                              prob_dropout=prob_dropout, sparsity_const=sparsity_const),
          'rc': ResidualConnection(n_cols=cols, activation=activation, prob_dropout=prob_dropout,
                                   number_layers=5, node_size=node_size, nodes_range=[node_size, node_size],
                                   sparsity_const=sparsity_const),
          'ic': InceptionModel(n_cols=cols, activation=activation, node_size=node_size, branch_number=3,
                               prob_dropout=prob_dropout, sparsity_const=sparsity_const)
          }

fileModel = models.get(model)

# We define the stratify folds
y_pred_score = np.empty(shape=[0, 2])
predicted_index = np.empty(shape=[0, ])
skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=False)

# For each Fold
k = 0
feature_importance_list = []
feature_name_k = pd.DataFrame(columns=['names', 'index'])
for train_index, test_index in skf.split(x.values, y[[label]].values):
    k += 1
    x_train, x_test = x.loc[train_index].values, x.loc[test_index].values
    y_train, y_test = y.drop('oferta_id', axis=1).loc[train_index].values, y.drop('oferta_id', axis=1).loc[
        test_index].values
    print(train_index, test_index)
    if sampling is None:
        pass
    elif sampling == 'ALLKNN':
        x_train, y_train = under_sampling(x_train, y_train)
        class_weight = None
    else:
        x_train, y_train = over_sampling(x_train, y_train, model=sampling)
        class_weight = None

    try:
        min_sample_leaf = round(y_train.shape[0] * 0.06)
        min_sample_split = min_sample_leaf * 10
        fileModel.min_samples_leaf = min_sample_leaf
        fileModel.min_samples_split = min_sample_split
        fileModel.fit(x_train, y_train)
        y_pred_score_i = fileModel.predict_proba(x_test)

        # Feature Importance
        featureImportance = fileModel.feature_importances_
        feature_importance_list.append(featureImportance)
        sorted_idx_k = np.argsort(featureImportance)
        fi_k = featureImportance[sorted_idx_k]
        fi_k = fi_k[-10:]
        fileNames_k = fileNames[sorted_idx_k]
        fileNames_k = fileNames_k[-10:]
        fileNames_k = pd.DataFrame(fileNames_k, columns=['names'])
        fileNames_k = fileNames_k.reset_index(drop=False).sort_values(by='index', ascending=False).drop('index', axis=1).reset_index(drop=True).reset_index(drop=False)
        feature_name_k = pd.concat([feature_name_k, fileNames_k], axis=0)
        print(feature_name_k)
        print(featureImportance)
        if k == 5:
            # FEATURE IMPORTANCE
            feature_importance_list = np.array(feature_importance_list)
            featureImportance = np.average(feature_importance_list, axis=0)
            featureImportance = featureImportance / featureImportance.max()
            sorted_idx = np.argsort(featureImportance)
            fi = featureImportance[sorted_idx]
            fi = fi[-10:]
            barPos = np.arange(sorted_idx.shape[0]) + 0.5
            barPos = barPos[-10:]
            plot.barh(barPos, fi, align='center')
            fileNames_1 = fileNames[sorted_idx]
            fileNames_1 = fileNames_1[-10:]
            fileNames_1 = [STRING.configure_names.get(item, item) for item in fileNames_1]
            print(fileNames_1)
            plot.yticks(barPos, fileNames_1, fontsize=8)
            plot.xlabel('Variable Importance')
            plot.savefig(STRING.img_path + 'feature_importance_k_folds.png')
            plot.show()
            plot.close()

            # POSITION NAMES
            feature_name_k = feature_name_k.reset_index(drop=True).rename(columns={'index': 'position'})
            feature_name_k['position'] = feature_name_k['position'].map(int) + 1
            feature_name_k = feature_name_k.groupby(['names']).agg({'names': ['count'], 'position': ['mean']})
            feature_name_k.columns = feature_name_k.columns.droplevel(1)
            feature_name_k = feature_name_k.sort_values(by=['names', 'position'],
                                                        ascending=[False, True])
            feature_name_k = feature_name_k.rename(columns={'names': 'count'})
            feature_name_k = feature_name_k.reset_index(drop=False)
            for k, v in STRING.configure_names.items():
                feature_name_k.loc[feature_name_k['names'] == k, 'names'] = v

            plot.bar(feature_name_k['names'], feature_name_k['count'])
            plot.plot(feature_name_k['names'], feature_name_k['position'])
            plot.xticks(rotation=30, fontsize=8)
            plot.savefig(STRING.img_path + 'feature_importance_ranking.png')
            plot.show()



    except:
        fileModel.fit_model(x_train, y_train, learning_rate=learning_rate, loss_function=loss_function,
                            epochs=epochs,
                            batch_size=batch_size, verbose=False, validation_data=None, validation_split=0.2,
                            steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)
        y_pred_score_i = fileModel.predict_model(x_test)

    y_pred_score = np.append(y_pred_score, y_pred_score_i, axis=0)
    predicted_index = np.append(predicted_index, test_index, axis=0)
    del x_train, x_test, y_train, y_test



# We keep only one class probability
y_pred_score = y_pred_score[:, 1]

# We check the optimal threshold
for treshold in tresholds:
    y_hat = (y_pred_score > treshold).astype(int)
    scores.append([
        recall_score(y_pred=y_hat, y_true=y[label].values),
        precision_score(y_pred=y_hat, y_true=y[label].values),
        fbeta_score(y_pred=y_hat, y_true=y[label].values,
                    beta=1)])

scores = np.array(scores)

# Plot Metrics
plot.plot(tresholds, scores[:, 0], label='$Recall$')
plot.plot(tresholds, scores[:, 1], label='$Precision$')
plot.plot(tresholds, scores[:, 2], label='$F_2$')
plot.ylabel('Score')
plot.xlabel('Threshold')
plot.legend(loc='best')
plot.savefig(STRING.img_path + 'precision_recall_' + model + '.png')
plot.close()

# METRICS: PRECISION-RECALL
final_tresh = tresholds[scores[:, 2].argmax()]
print('Threshold', final_tresh)
y_hat_test = (y_pred_score > final_tresh).astype(int)

precision = precision_score(y[label].values, y_hat_test)
recall = recall_score(y[label].values, y_hat_test)
fbeta = fbeta_score(y[label].values, y_hat_test, beta=1)
print('PRECISION ', precision)
print('RECALL ', recall)
print('FBSCORE ', fbeta)

prob_sample = pd.concat([y, pd.DataFrame(y_pred_score, columns=[model]), pd.DataFrame(y_hat_test, columns=['yhat'])],
                        axis=1)

test_sample_1 = prob_sample[prob_sample['oferta_id'].isin(test_sample_1['oferta_id'].values.tolist())]
test_sample_2 = prob_sample[prob_sample['oferta_id'].isin(test_sample_2['oferta_id'].values.tolist())]

# Confussion Matrix
i = 0
metric_save = []
for test_sample in [test_sample_1, test_sample_2]:
    i += 1
    precision = precision_score(test_sample[label].values, test_sample['yhat'].values)
    recall = recall_score(test_sample[label].values, test_sample['yhat'].values)
    fbeta = fbeta_score(test_sample[label].values, test_sample['yhat'].values, beta=1)
    print('PRECISION ', precision)
    print('RECALL ', recall)
    print('FBSCORE ', fbeta)

    conf_matrix = confusion_matrix(test_sample[label].values, test_sample['yhat'].values)
    plot.figure(figsize=(12, 12))
    sns.heatmap(conf_matrix, xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'], annot=True,
                fmt="d", cmap="Blues", cbar=False)
    plot.title("Confusion matrix")
    plot.ylabel('True class')
    plot.xlabel('Predicted class')
    plot.savefig(STRING.img_path + model + '_confusion_matrix_sample_' + str(i) + '.png')
    plot.close()

    metric_save_i = [model, i, final_tresh, precision, recall, fbeta]
    metric_save.append(metric_save_i)

    # PROBABILITY FILE
    try:
        df_prob = pd.read_csv(STRING.path_db_extra + 'probability_save_' + str(i) + '.csv', sep=';')
        del df_prob[model]
        y_final_test = test_sample.drop('yhat', axis=1)
        df_prob['oferta_id'] = df_prob['oferta_id'].map(int)
        y_final_test['oferta_id'] = y_final_test['oferta_id'].map(int)
        df_prob = pd.merge(df_prob, y_final_test, how='left', on='oferta_id')
        df_prob.to_csv(STRING.path_db_extra + 'probability_save_' + str(i) + '.csv', sep=';', index=False)
        del df_prob
    except FileNotFoundError:
        df_prob = pd.DataFrame(columns=['oferta_id', 'vae', 'ae', 'ert', 'nn', 'ic', 'rc'])
        y_final_test = test_sample.drop('yhat', axis=1)
        df_prob = pd.concat([df_prob, y_final_test], axis=0)
        df_prob.to_csv(STRING.path_db_extra + 'probability_save_' + str(i) + '.csv', sep=';', index=False)

metric_save = pd.DataFrame(metric_save,
                           columns=['model', 'sample', 'threshold', 'precision', 'recall', 'fbscore'])
try:
    df = pd.read_csv(STRING.metric_save, sep=';')
except FileNotFoundError:
    df = pd.DataFrame(
        columns=['model', 'sample', 'threshold', 'precision', 'recall', 'fbscore'])

df = pd.concat([df, metric_save], axis=0)
df = df.drop_duplicates(subset=['model', 'sample'], keep='last')
df.to_csv(STRING.metric_save, sep=';', index=False)


