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
from sklearn.metrics import recall_score, precision_score, fbeta_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

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

test_sample_1 = test[test['oferta_id'].isin(test_sample_1['oferta_id'].values.tolist())]
test_sample_2 = test[test['oferta_id'].isin(test_sample_2['oferta_id'].values.tolist())]

traininig_sample = test[-test['oferta_id'].isin(test_sample_1['oferta_id'].values.tolist() + 
                                                test_sample_2['oferta_id'].values.tolist())]

# We split to half to have the same proportion of anomalies to train: 50%TOTAL1/AN1, 50%TOTAL2/AN2
training_sample_1, training_sample_2 = train_test_split(traininig_sample, test_size=0.5, random_state=42)
training_sample_1 = pd.concat([training_sample_1, test_sample_2], axis=0)
training_sample_2 = pd.concat([training_sample_2, test_sample_1], axis=0)

# Shuffle
test = test.sample(frac=1).reset_index(drop=True)
training_sample_1 = training_sample_1.sample(frac=1).reset_index(drop=True)
training_sample_2 = training_sample_2.sample(frac=1).reset_index(drop=True)

# X, Y
x = test.drop([label] + ['oferta_id'], axis=1)
y = test[[label]]

x_train_sample_1 = training_sample_1.drop([label] + ['oferta_id'], axis=1)
x_train_sample_2 = training_sample_2.drop([label] + ['oferta_id'], axis=1)
y_train_sample_1 = training_sample_1[[label]]
y_train_sample_2 = training_sample_2[[label]]

x_test_sample_1 = test_sample_1.drop([label] + ['oferta_id'], axis=1)
x_test_sample_2 = test_sample_2.drop([label] + ['oferta_id'], axis=1)
y_test_sample_1 = test_sample_1[[label]]
y_test_sample_2 = test_sample_2[[label]]


# Parameters
sns.set()
tresholds = np.linspace(0.1, 1.0, 200)
scores = []
sampling = None
seed = 42
np.random.seed(seed)
model = 'rc'


# Variance Reduction
selection = VarianceThreshold(threshold=0.0)
selection.fit(x)
features = selection.get_support(indices=True)
features = x.columns[features]
x = x[features]
x_train_sample_1 = x_train_sample_1[features]
x_train_sample_2 = x_train_sample_2[features]
x_test_sample_1 = x_test_sample_1[features]
x_test_sample_2 = x_test_sample_2[features]

for i in x.columns.values.tolist():
    x[i] = x[i].map(float)
    scaler = StandardScaler()
    scaler.fit(x[i].values.reshape(-1, 1))
    x[i] = scaler.transform(x[i].values.reshape(-1, 1))
    x_train_sample_1[i] = scaler.transform(x_train_sample_1[i].values.reshape(-1, 1))
    x_train_sample_2[i] = scaler.transform(x_train_sample_2[i].values.reshape(-1, 1))
    x_test_sample_1[i] = scaler.transform(x_test_sample_1[i].values.reshape(-1, 1))
    x_test_sample_2[i] = scaler.transform(x_test_sample_2[i].values.reshape(-1, 1))

cols = x.shape[1]
fileNames = np.array(x.columns.values)

# Models
models = {'ert': ExtraTreesClassifier(n_estimators=500, max_depth=20, bootstrap=True, oob_score=True,
                                      class_weight='balanced_subsample', max_features='auto', random_state=42),
          'nn': NeuralNetwork(n_cols=cols, node_size=[100, 100], activation='relu', prob_dropout=0.2),
          'rc': ResidualConnection(n_cols=cols, activation='relu', prob_dropout=0.2,
                                   number_layers=10, node_size=32, nodes_range=range(1000, 100, -100)),
          'ic': InceptionModel(n_cols=cols, activation='relu', node_size=32, branch_number=6,
                               prob_dropout=0.2)

          }

fileModel = models.get(model)

# We define the stratify folds
y_pred_score = np.empty(shape=[0, 2])
predicted_index = np.empty(shape=[0, ])
skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=False)

# For each Fold
for train_index, test_index in skf.split(x.values, y[label].values):
    x_train, x_test = x.loc[train_index].values, x.loc[test_index].values
    y_train, y_test = y.loc[train_index].values, y.loc[test_index].values
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
        min_sample_leaf = round(y_train.shape[0] * 0.01)
        min_sample_split = min_sample_leaf * 10
        fileModel.min_samples_leaf = min_sample_leaf
        fileModel.min_samples_split = min_sample_split
        fileModel.fit(x_train, y_train)
        y_pred_score_i = fileModel.predict_proba(x_test)
    except:
        fileModel.fit_model(x_train, y_train, learning_rate=0.001, loss_function='mean_squared_error',
                            epochs=500,
                            batch_size=5000, verbose=True, validation_data=None, validation_split=0.2)
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
plot.show()
plot.close()

# METRICS: PRECISION-RECALL
final_tresh = tresholds[scores[:, 2].argmax()]
print('Threshold', final_tresh)
y_hat_test = (y_pred_score > final_tresh).astype(int)

precision = precision_score(y.values, y_hat_test)
recall = recall_score(y.values, y_hat_test)
fbeta = fbeta_score(y.values, y_hat_test, beta=1)
print('PRECISION ', precision)
print('RECALL ', recall)
print('FBSCORE ', fbeta)

# Confussion Matrix
conf_matrix = confusion_matrix(y.values, y_hat_test)
plot.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'], annot=True,
            fmt="d", cmap="Blues", cbar=False)
plot.title("Confusion matrix")
plot.ylabel('True class')
plot.xlabel('Predicted class')
plot.savefig(STRING.img_path + model + 'confusion_matrix.png')
plot.show()

# FINAL TEST
i = 0
fileModel = models.get(model)
for sample in [[x_train_sample_1, y_train_sample_1, x_test_sample_1, y_test_sample_1],
                            [x_train_sample_2, y_train_sample_2, x_test_sample_2, y_test_sample_2]]:
    i += 1
    x = sample[0]
    y = sample[1]
    x_final_test = sample[2]
    y_final_test = sample[3]

    try:
        min_sample_leaf = round(y.shape[0] * 0.01)
        print(min_sample_leaf)
        min_sample_split = min_sample_leaf * 10
        fileModel.min_samples_leaf = min_sample_leaf
        fileModel.min_samples_split = min_sample_split
        fileModel.fit(x, y)
        y_pred_score = fileModel.predict_proba(x_final_test)
    except:
        fileModel.fit_model(x, y, learning_rate=0.001, loss_function='mean_squared_error',
                            epochs=500,
                            batch_size=2500, verbose=True, validation_data=None, validation_split=0.2)
        y_pred_score = fileModel.predict_model(x_final_test)

    scores = []

    for treshold in tresholds:
        y_hat = (y_pred_score[:, 1] > treshold).astype(int)
        scores.append([recall_score(y_pred=y_hat, y_true=y_final_test.values),
                       precision_score(y_pred=y_hat, y_true=y_final_test.values),
                       fbeta_score(y_pred=y_hat, y_true=y_final_test.values,
                                   beta=1)])

    scores = np.array(scores)

    y_pred_score = y_pred_score[:, 1]
    y_hat_test = (y_pred_score > final_tresh).astype(int)
    precision = precision_score(y_final_test.values, y_hat_test)
    recall = recall_score(y_final_test.values, y_hat_test)
    fbeta = fbeta_score(y_final_test.values, y_hat_test, beta=1)
    print(model)
    print('PRECISION ', precision)
    print('RECALL ', recall)
    print('FBSCORE ', fbeta)

    # Confussion Matrix
    conf_matrix = confusion_matrix(y_final_test.values, y_hat_test)
    plot.figure(figsize=(12, 12))
    sns.heatmap(conf_matrix, xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'], annot=True,
                fmt="d", cmap="Blues", cbar=False)
    plot.title("Confusion matrix")
    plot.ylabel('True class')
    plot.xlabel('Predicted class')
    plot.savefig(STRING.img_path + model + '_confusion_matrix_sample_' + str(i) + '.png')
    plot.show()

    auc = roc_auc_score(y_final_test.values, y_pred_score)
    print('AUC: %.3f' % auc)

    # calculate roc curve
    fpr, tpr, thresholds = roc_curve(y_final_test.values, y_pred_score,
                                     drop_intermediate=False)
    # plot no skill
    plot.plot([0, 1], [0, 1], linestyle='--')
    # plot the roc curve for the model
    plot.plot(fpr, tpr, marker='.')
    # show the plot
    plot.savefig(STRING.img_path +  model + '_roc_curve_sample_' + str(i) + '.png')
    plot.show()

    metric_save = [[model, i, final_tresh, precision, recall, fbeta, fpr.tolist(), tpr.tolist(), auc]]
    metric_save = pd.DataFrame(metric_save,
                               columns=['model', 'sample', 'threshold', 'precision', 'recall', 'fbscore',
                                        'fpr', 'tpr', 'auc'])
    try:
        df = pd.read_csv(STRING.metric_save, sep=';')
    except FileNotFoundError:
        df = pd.DataFrame(
            columns=['model', 'sample', 'threshold', 'precision', 'recall', 'fbscore', 'fpr', 'tpr', 'auc'])

    df = pd.concat([df, metric_save], axis=0)
    df.to_csv(STRING.metric_save, sep=';', index=False)

