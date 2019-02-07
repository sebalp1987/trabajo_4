from sklearn import ensemble
import pandas as pd
from resources.confussion_matrix import plot_csm
from sklearn.metrics import fbeta_score, precision_score, recall_score, confusion_matrix, accuracy_score
import numpy as np
from resources.sampling import under_sampling, over_sampling
import pylab as plot
import STRING
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict

class ExtremeRandomize(object):

    def __init__(self, n_estimators: int = 200,
                 max_depth: int = 50, bootstrap=False,
                 oob_score: bool = False, class_weight='balanced_subsample',
                 sampling=None, final_threshold=0.403, beta=2, metric_weight='binary'):

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.class_weight = class_weight
        self.sampling = sampling
        self.final_threshold = final_threshold
        self.beta = beta
        self.metric_weight = metric_weight

    def threshold(self, x_train, y_train, x_valid, y_valid, plot_graph=True):
        """
        Obtain optimal threshold using FBeta as parameter using a range (0.1, 1.0, 200) for 
        evaluation
        """

        if self.sampling is None:
            class_weight = self.class_weight

        elif self.sampling == 'ALLKNN':
            x_train, y_train = under_sampling(x_train, y_train)
            class_weight = None

        else:
            x_train, y_train = over_sampling(x_train, y_train, model=self.sampling)
            class_weight = None

        if isinstance(x_train, pd.DataFrame):
            x_train = x_train.values
        if isinstance(y_train, (pd.DataFrame, pd.Series)):
            y_train = y_train.values
        if isinstance(x_valid, pd.DataFrame):
            x_valid = x_valid.values
        if isinstance(y_valid, (pd.DataFrame, pd.Series)):
            y_valid = y_valid.values

        min_sample_leaf = round(x_train.shape[0] * 0.01)
        min_sample_split = min_sample_leaf * 10
        max_features = None

        file_model = ensemble.ExtraTreesClassifier(criterion='gini', bootstrap=self.bootstrap,
                                                   min_samples_leaf=min_sample_leaf,
                                                   min_samples_split=min_sample_split,
                                                   n_estimators=self.n_estimators,
                                                   max_depth=self.max_depth, max_features=max_features,
                                                   oob_score=self.oob_score,
                                                   random_state=531, verbose=1, class_weight=class_weight,
                                                   n_jobs=1)
        cv = StratifiedKFold(n_splits=10, random_state=None)
        file_model.fit(x_train, y_train)

        thresholds = np.linspace(0.1, 1.0, 200)

        scores = []

        y_pred_score = cross_val_predict(file_model, x_valid,
                                         y_valid, cv=cv, method='predict_proba')

        y_pred_score = np.delete(y_pred_score, 0, axis=1)

        for threshold in thresholds:
            y_hat = (y_pred_score > threshold).astype(int)
            y_hat = y_hat.tolist()
            y_hat = [item for sublist in y_hat for item in sublist]

            scores.append([
                recall_score(y_pred=y_hat, y_true=y_valid),
                precision_score(y_pred=y_hat, y_true=y_valid),
                fbeta_score(y_pred=y_hat, y_true=y_valid,
                            beta=self.beta, average=self.metric_weight)])

        scores = np.array(scores)

        if plot_graph:
            plot.plot(thresholds, scores[:, 0], label='$Recall$')
            plot.plot(thresholds, scores[:, 1], label='$Precision$')
            plot.plot(thresholds, scores[:, 2], label='$F_2$')
            plot.ylabel('Score')
            plot.xlabel('Threshold')
            plot.legend(loc='best')
            plot.close()

        self.final_threshold = thresholds[scores[:, 2].argmax()]
        print(self.final_threshold)
        return self.final_threshold

    def evaluation(self, x_train, y_train, x_test, y_test, plot_graph=True):
        """
        Evaluate the performance of the ERT model using Recall, Precision and FBeta using the
        optimal threshold. Also we can get a Confusion Matrix and Importance Feature ploting.
        """

        columns = None
        label = None

        if self.sampling is None:
            class_weight = self.class_weight

        elif self.sampling == 'ALLKNN':
            x_train, y_train = under_sampling(x_train, y_train)
            class_weight = None

        else:
            x_train, y_train = over_sampling(x_train, y_train, model=self.sampling)
            class_weight = None

        if isinstance(x_train, pd.DataFrame):
            columns = x_train.columns.values
            x_train = x_train.values
        if isinstance(y_train, (pd.DataFrame, pd.Series)):
            label = y_train.columns.values
            y_train = y_train.values
        if isinstance(x_test, pd.DataFrame):
            x_test = x_test.values
        if isinstance(y_test, (pd.DataFrame, pd.Series)):
            y_test = y_test.values

        min_sample_leaf = round(x_train.shape[0] * 0.01)
        min_sample_split = min_sample_leaf * 10
        max_features = 'sqrt'

        file_model = ensemble.ExtraTreesClassifier(criterion='entropy', bootstrap=self.bootstrap,
                                                   min_samples_leaf=min_sample_leaf,
                                                   min_samples_split=min_sample_split,
                                                   n_estimators=self.n_estimators,
                                                   max_depth=self.max_depth, max_features=max_features,
                                                   oob_score=self.oob_score,
                                                   random_state=531, verbose=1, class_weight=class_weight,
                                                   n_jobs=1)

        file_model.fit(x_train, y_train)

        y_hat_test = file_model.predict_proba(x_test)

        y_hat_test = np.delete(y_hat_test, 0, axis=1)
        print(self.final_threshold)
        y_hat_test = (y_hat_test > self.final_threshold).astype(int)
        y_hat_test = y_hat_test.tolist()
        y_hat_test = [item for sublist in y_hat_test for item in sublist]

        print('Final threshold: %.3f' % self.final_threshold)
        print('Test Recall Score: %.3f' % recall_score(y_pred=y_hat_test, y_true=y_test))
        print('Test Precision Score: %.3f' % precision_score(y_pred=y_hat_test, y_true=y_test))
        print('Test F2 Score: %.3f' % fbeta_score(y_pred=y_hat_test, y_true=y_test, beta=self.beta, average=self.metric_weight))

        # PLOTS
        if plot_graph:
            # CONFUSION MATRIX
            cnf_matrix = confusion_matrix(y_test, y_hat_test)
            plot_csm(cnf_matrix, classes=['No' + str(label), str(label)], title='Confusion matrix')
            
            # FEATURE IMPORTANCE
            if columns is not None:
                feature_importance = file_model.feature_importances_
                feature_importance = feature_importance / feature_importance.max()
                sorted_idx = np.argsort(feature_importance)
                bar_position = np.arange(sorted_idx.shape[0]) + 0.5
                plot.barh(bar_position, feature_importance[sorted_idx], align='center')
                plot.yticks(bar_position, columns[sorted_idx])
                plot.xlabel('Variable Importance')
                plot.show()

    def over_fitting(self, x_train, y_train, x_test, y_test, n_estimator_range=range(1, 301, 10)):
        """
        Calculate overfitting using accuracy for ERT Class model. We get the training Acc and
        the test Acc, and also plot it.
        """

        accuracy_test_list = []
        accuracy_train_list = []

        if self.sampling is None:
            class_weight = self.class_weight

        elif self.sampling == 'ALLKNN':
            x_train, y_train = under_sampling(x_train, y_train)
            class_weight = None

        else:
            x_train, y_train = over_sampling(x_train, y_train, model=self.sampling)
            class_weight = None

        if isinstance(x_train, pd.DataFrame):
            x_train = x_train.values
        if isinstance(y_train, (pd.DataFrame, pd.Series)):
            y_train = y_train.values
        if isinstance(x_test, pd.DataFrame):
            x_test = x_test.values
        if isinstance(y_test, (pd.DataFrame, pd.Series)):
            y_test = y_test.values

        min_sample_leaf = round(x_train.shape[0] * 0.01)
        min_sample_split = min_sample_leaf * 10
        max_features = 'sqrt'

        for iter in n_estimator_range:
            print('iter nº: ', iter)

            file_model = ensemble.ExtraTreesClassifier(criterion='entropy', bootstrap=self.bootstrap,
                                                       min_samples_leaf=min_sample_leaf,
                                                       min_samples_split=min_sample_split,
                                                       n_estimators=iter,
                                                       max_depth=self.max_depth, max_features=max_features,
                                                       oob_score=self.oob_score,
                                                       random_state=531, verbose=1, class_weight=class_weight,
                                                       n_jobs=1)
            file_model.fit(x_train, y_train)

            predictions = file_model.predict_proba(x_test)
            predictions = np.delete(predictions, 0, axis=1)
            predictions = (predictions > self.final_threshold).astype(int)
            accuracy_test_list.append(accuracy_score(y_test, predictions))

            predictions = file_model.predict_proba(x_train)
            predictions = np.delete(predictions, 0, axis=1)
            predictions = (predictions > self.final_threshold).astype(int)
            accuracy_train_list.append(accuracy_score(y_train, predictions))

        plot.figure()
        plot.plot(n_estimator_range, accuracy_train_list, label='Training Set Accuracy')
        plot.plot(n_estimator_range, accuracy_test_list, label='Test Set Accuracy')
        plot.legend(loc='upper right')
        plot.xlabel('Number of Trees in Ensamble')
        plot.ylabel('Accuracy')
        plot.show()

    def applied(self, x_train, y_train, x_predicted):
        """
        The applied definitive model, getting just probabilities.
        """

        if self.sampling is None:
            class_weight = self.class_weight

        elif self.sampling == 'ALLKNN':
            x_train, y_train = under_sampling(x_train, y_train)
            class_weight = None

        else:
            x_train, y_train = over_sampling(x_train, y_train, model=self.sampling)
            class_weight = None

        if isinstance(x_train, pd.DataFrame):
            x_train = x_train.values
        if isinstance(y_train, (pd.DataFrame, pd.Series)):
            y_train = y_train.values

        min_sample_leaf = round(x_train.shape[0] * 0.01)
        min_sample_split = min_sample_leaf * 10
        max_features = 'sqrt'

        file_model = ensemble.ExtraTreesClassifier(criterion='entropy', bootstrap=self.bootstrap,
                                                   min_samples_leaf=min_sample_leaf,
                                                   min_samples_split=min_sample_split,
                                                   n_estimators=self.n_estiators,
                                                   max_depth=self.max_depth, max_features=max_features,
                                                   oob_score=self.oob_score,
                                                   random_state=531, verbose=1, class_weight=class_weight,
                                                   n_jobs=1)

        file_model.fit(x_train, y_train)
        y_hat_test = file_model.predict_proba(x_predicted)
        y_hat_test = np.delete(y_hat_test, 0, axis=1)
        y_hat_test = (y_hat_test > self.final_threshold).astype(int)

        return y_hat_test


if __name__ == '__main__':

    seed = 42
    np.random.seed(seed)

    train = pd.read_csv(STRING.train, sep=';', encoding='latin1')
    test = pd.read_csv(STRING.test, sep=';', encoding='latin1')
    valid = pd.read_csv(STRING.valid, sep=';', encoding='latin1')

    train['target'] = train['target'].map(float)
    y_train = train[['oferta_id', 'target']]

    train = train.drop(['oferta_id', 'target'], axis=1)

    y_valid = valid[['target']]
    y_valid['target'] = y_valid['target'].map(float)

    valid = valid.drop(['oferta_id', 'target'], axis=1)

    y_test = test[['target']]
    y_test['target'] = y_test['target'].map(float)
    test = test.drop(['oferta_id', 'target'], axis=1)

    ert = ExtremeRandomize(max_depth=None, n_estimators=300, beta=1, class_weight='balanced_subsample', sampling=None,
                           metric_weight='binary')
    ert.threshold(train, y_train.drop('oferta_id', axis=1), test, y_test)
    ert.evaluation(train, y_train.drop('oferta_id', axis=1), valid, y_valid)
    ert.over_fitting(train, y_train.drop('oferta_id', axis=1), test, y_test)

