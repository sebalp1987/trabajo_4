import pandas as pd
import STRING
import numpy as np
import matplotlib.pyplot as plot
import seaborn as sns

from resources.sampling import over_sampling, under_sampling
from model.neural_net import NeuralNetwork

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import recall_score, precision_score, fbeta_score, confusion_matrix

# Files
label = 'target'
train = pd.read_csv(STRING.train, sep=';', encoding='latin1')
valid = pd.read_csv(STRING.valid, sep=';', encoding='latin1')
test = pd.read_csv(STRING.test, sep=';', encoding='latin1')
test = pd.concat([train, valid, test], axis=0).reset_index()
x = test.drop([label] + ['oferta_id'], axis=1)
y = test[[label]]


# Parameters
sns.set()
tresholds = np.linspace(0.1, 1.0, 200)
scores = []
sampling = None
cols = x.shape[1]
# Models
fileModel1 = ExtraTreesClassifier(n_estimators=100, max_depth=50, bootstrap=False, oob_score=False,
                                 class_weight='balanced_subsample')
fileModel = NeuralNetwork(n_cols=cols, node_size=[100], activation='relu', prob_dropout=0.2)


# We define the stratify folds
y_pred_score = np.empty(shape=[0, 2])
predicted_index = np.empty(shape=[0, ])
skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=False)

# For each Fold
for train_index, test_index in skf.split(x.values, y[label].values):
    print('1')

    x_train, x_test = x.loc[train_index].values, x.loc[test_index].values
    y_train, y_test = y.loc[train_index].values, y.loc[test_index].values

    if sampling is None:
        pass
    elif sampling == 'ALLKNN':
        x_train, y_train = under_sampling(x_train, y_train)
        class_weight = None
    else:
        x_train, y_train = over_sampling(x_train, y_train, model=sampling)
        class_weight = None

    min_sample_leaf = round(y_train.shape[0] * 0.01)
    min_sample_split = min_sample_leaf * 10
    max_features = round(x_train.shape[1] / 3)

    try:
        fileModel.fit(x_train, y_train)
        y_pred_score_i = fileModel.predict_proba(x_test)
    except:
        fileModel.fit_model(x_train, y_train, learning_rate=0.001, loss_function='cosine_proximity',
                            epochs=100,
                            batch_size=1000, verbose=True, validation_data=None)
        y_pred_score_i = fileModel.predict_model(x_test)
        print(y_pred_score_i)

    y_pred_score = np.append(y_pred_score, y_pred_score_i, axis=0)
    predicted_index = np.append(predicted_index, test_index, axis=0)
    del x_train, x_test, y_train, y_test

# We keep only one class probability
y_pred_score = np.delete(y_pred_score, 0, axis=1)

# We check the optimal threshold
for treshold in tresholds:
    y_hat = (y_pred_score > treshold).astype(int)
    y_hat = y_hat.tolist()
    y_hat = [item for sublist in y_hat for item in sublist]

    scores.append([
            recall_score(y_pred=y_hat, y_true=test[label].values),
            precision_score(y_pred=y_hat, y_true=test[label].values),
            fbeta_score(y_pred=y_hat, y_true=test[label].values,
                        beta=1)])

scores = np.array(scores)

# Plot Metrics
plot.plot(tresholds, scores[:, 0], label='$Recall$')
plot.plot(tresholds, scores[:, 1], label='$Precision$')
plot.plot(tresholds, scores[:, 2], label='$F_2$')
plot.ylabel('Score')
plot.xlabel('Threshold')
plot.legend(loc='best')
plot.show()
plot.close()

# METRICS: PRECISION-RECALL
final_tresh = tresholds[scores[:, 2].argmax()]
print('Threshold', final_tresh)
y_hat_test = (y_pred_score > final_tresh).astype(int)
y_hat_test = y_hat_test.tolist()
y_hat_test = [item for sublist in y_hat_test for item in sublist]

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
            fmt="d", cmap="Blues")
plot.title("Confusion matrix")
plot.ylabel('True class')
plot.xlabel('Predicted class')
plot.show()
